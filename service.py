"""The resident index, and the local HTTP service over it.

Holds the L2-normalised library as a memory-mapped float32 matrix and
answers "which reference spectra are closest to this one" with an exact
search. No approximate index: at this size a full GEMV is about 10-20 ms,
and a space-partitioning tree degenerates to a linear scan above roughly
20 dimensions, so a tree would be slower *and* approximate.

Memory-mapping means startup is instant and several workers share the same
physical pages.

    POST /search   {"wavelengths": [...], "values": [...],
                    "k": 10, "metric": "sam",
                    "source": ["usgs_splib07"],
                    "category": ["mineral"],
                    "min_overlap": 0.5,
                    "min_overlap_bands": 112}
    GET  /health

Results carry the OpenSpecLib spectrum id, so a hit can be resolved back to
its full record in the library the index was built from.

Four metrics, in two families. ``sam`` and ``cosine`` compare reflectance as
stored. ``cr_sam`` compares continuum-removed band depth and ``gd1`` the
first derivative, so for those the library is scored in the same derived
form, from its own matrix set and its own support mask -- a query
transformed on its own and compared against raw reflectance is not a
comparison, only a number shaped like an angle. Those sets come from
derive.py, and a metric whose set an index does not carry is refused, with
the command that would build it, rather than answered from the reflectance
matrices; ``GET /health`` lists the metrics an index can actually answer.

Only the bands present in BOTH the query and a candidate are compared, and
a candidate whose overlap is too thin is rejected rather than scored: an
angle measured across a gap the sensor never sampled is not a measurement.
The threshold takes a fraction and an absolute number of grid points, and
the stricter of the two decides: the fraction is measured against what the
query itself covers, so on its own it loosens exactly where the query is
gappiest and the guarantee is needed most.
Spectral gaps are carried as no-support on both sides -- see gaps.py -- so
"overlap" counts bands that were really measured, and a query that drops the
water-vapour windows is not compared across them.

:class:`Library` is what i.hyper.lib_server's own query path uses directly,
without going through HTTP: a module already running in the same process
has no reason to serialise a spectrum to itself.
"""
import json
import math
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

import gaps
import transforms

METRICS = transforms.ALL


class Library:
    """The resident index: a normalised matrix plus its metadata."""

    def __init__(self, path):
        with open(os.path.join(path, "index.json")) as fh:
            self.info = json.load(fh)
        with open(os.path.join(path, "metadata.json")) as fh:
            self.meta = json.load(fh)
        n, nb = self.info["n_spectra"], self.info["n_bands"]
        self.grid = np.arange(
            self.info["grid_lo"],
            self.info["grid_hi"] + self.info["grid_step"] / 2,
            self.info["grid_step"])

        self.path, self.n, self.nb = path, n, nb

        # Three matrices so a shared-support cosine is four GEMVs and never
        # slices rows: values (zero-filled), their squares, and a 0/1
        # support mask. Slicing the library per query copied ~200 MB and
        # cost two orders of magnitude more than the arithmetic.
        #
        # One such set per metric family, because a transformed metric
        # compares a derived quantity and has its own support. Mapping is
        # lazy: a set costs nothing until a query asks for that metric.
        self._sets = {}
        by_metric = self.info.get("matrices_by_metric") or {}
        if not by_metric:
            # An index derived before the per-metric sets existed. Only the
            # untransformed metrics can be served from it.
            by_metric = {m: ["values.f32", "values_sq.f32", "support.f32"]
                         for m in transforms.IDENTITY}
        self.matrices_by_metric = by_metric
        self.metrics = tuple(m for m in transforms.ALL if m in by_metric)
        self.sources = np.array([m["source"] for m in self.meta])
        self.categories = np.array([m["category"] for m in self.meta])
        self.source_names = sorted(set(self.sources.tolist()))
        self.category_names = sorted(set(self.categories.tolist()))

    def _matrices(self, metric):
        """The (values, squares, support) this metric scores against.

        Refused rather than substituted when the set is missing: scoring a
        transformed query against untransformed reflectance returns a
        number shaped like an angle and ordered by nothing in particular,
        which is worse than an error.
        """
        names = self.matrices_by_metric.get(metric)
        if not names:
            raise ValueError(
                f"the index at {self.path} has no matrices for metric "
                f"{metric!r}; derive them with "
                f"'i.hyper.lib_server operation=derive index={self.path} "
                f"metric={metric}'")
        key = tuple(names)
        if key not in self._sets:
            for name in names:
                full = os.path.join(self.path, name)
                if not os.path.exists(full):
                    raise ValueError(
                        f"metric {metric!r} needs {name}, which is missing "
                        f"from {self.path}; derive it with "
                        f"'i.hyper.lib_server operation=derive "
                        f"index={self.path} metric={metric}'")
            self._sets[key] = tuple(
                np.memmap(os.path.join(self.path, name), dtype=np.float32,
                          mode="r", shape=(self.n, self.nb))
                for name in names)
        return self._sets[key]

    def search(self, q_wl, q_val, k=10, metric="sam", min_overlap=0.5,
               min_overlap_bands=0, sources=None, categories=None):
        """Return the k closest spectra, exactly.

        :param q_wl: query wavelengths in nm
        :param q_val: query values, any scale (the angle is scale-free)
        :param k: how many to return
        :param metric: one of METRICS
        :param min_overlap: reject candidates covering less than this
            fraction of the query's own in-grid bands
        :param min_overlap_bands: reject candidates covering fewer than this
            many grid points, whatever the fraction works out to. The two
            are independent requirements and the stricter one decides: the
            fraction alone is measured against a denominator that shrinks
            when the query has gaps, so the same fraction silently buys a
            weaker guarantee on a gappy query than on a complete one
        :param sources: restrict to these source.library values
        :param categories: restrict to these material.category values
        :return: list of result dicts, closest first
        """
        q_wl = np.asarray(q_wl, dtype=np.float64)
        q_val = np.asarray(q_val, dtype=np.float64)
        good = np.isfinite(q_wl) & np.isfinite(q_val)
        q_wl, q_val = q_wl[good], q_val[good]
        order = np.argsort(q_wl)
        q_wl, q_val = q_wl[order], q_val[order]

        # Project the query onto the canonical grid, within each contiguous
        # segment only: the grid points the sensor skipped stay NaN and so
        # fall out of the support mask below, rather than being bridged with
        # a straight line and then compared against candidates that have
        # real structure there.
        q, query_gaps = gaps.project(self.grid, q_wl, q_val)
        if int(np.isfinite(q).sum()) < 2:
            raise ValueError("query does not overlap the library grid")

        vals, sq, sup = self._matrices(metric)
        q = transforms.apply(q, self.grid, metric)

        qmask = ~np.isnan(q)
        qn = int(qmask.sum())
        qz = np.nan_to_num(q, nan=0.0).astype(np.float32)
        qm = qmask.astype(np.float32)
        q2 = (qz * qz).astype(np.float32)

        # Four whole-matrix products, no row extraction. The support matrix
        # is read once for two of them by stacking the vectors into a
        # 2-column GEMM.
        num = vals @ qz                           # sum v*q
        cand_sq = sq @ qm                         # sum v^2 over the overlap
        both = sup @ np.stack([q2, qm], axis=1)
        query_sq, overlap = both[:, 0], both[:, 1]

        denom = np.sqrt(cand_sq) * np.sqrt(query_sq)
        with np.errstate(invalid="ignore", divide="ignore"):
            cos = np.where(denom > 0, num / denom, -1.0)
        cos = np.clip(cos, -1.0, 1.0)

        # A candidate that shares too few bands with the query is rejected,
        # not scored: an angle measured over a handful of bands, or across
        # a gap the sensor never sampled, is not a measurement.
        need = max(2.0, min_overlap * qn, float(min_overlap_bands))
        cos = np.where(overlap >= need, cos, -np.inf)
        if sources:
            cos = np.where(np.isin(self.sources, list(sources)), cos, -np.inf)
        if categories:
            cos = np.where(np.isin(self.categories, list(categories)), cos,
                           -np.inf)

        finite = int(np.isfinite(cos).sum())
        if finite == 0:
            return [], query_gaps, need
        kk = min(k, finite)
        top = np.argpartition(-cos, kk - 1)[:kk]
        top = top[np.argsort(-cos[top])]

        out = []
        for t in top:
            gi = int(t)
            m = self.meta[gi]
            out.append({
                "id": m["id"],
                "name": m["name"],
                "source": m["source"],
                "category": m["category"],
                "subcategory": m["subcategory"],
                "formula": m["formula"],
                "spectrum_type": m["spectrum_type"],
                "native_range_nm": [m["lo_nm"], m["hi_nm"]],
                "angle_rad": float(np.arccos(cos[gi])),
                "cosine": float(cos[gi]),
                "overlap_bands": int(overlap[gi]),
                "query_bands": qn,
            })
        return out, query_gaps, need


def make_handler(lib):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *a):          # quiet by default
            pass

        def _send(self, code, payload):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/") == "/health":
                self._send(200, {"status": "ok", "n_spectra": lib.n,
                                 "n_bands": lib.nb,
                                 "grid": [lib.info["grid_lo"],
                                          lib.info["grid_hi"],
                                          lib.info["grid_step"]],
                                 "metrics": list(lib.metrics),
                                 "gap_aware": True,
                                 "absolute_overlap_floor": True,
                                 "library": lib.info.get("library"),
                                 "sources": lib.info.get("sources", {}),
                                 "categories": lib.category_names})
            else:
                self._send(404, {"error": "use GET /health or POST /search"})

        def do_POST(self):
            if self.path.rstrip("/") != "/search":
                self._send(404, {"error": "use POST /search"})
                return
            try:
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n) or b"{}")
            except (ValueError, TypeError) as exc:
                self._send(400, {"error": f"bad JSON: {exc}"})
                return
            metric = req.get("metric", "sam")
            if metric not in METRICS:
                self._send(400, {"error": f"metric must be one of {METRICS}"})
                return
            try:
                t0 = time.perf_counter()
                hits, query_gaps, need = lib.search(
                    req["wavelengths"], req["values"],
                    k=int(req.get("k", 10)), metric=metric,
                    min_overlap=float(req.get("min_overlap", 0.5)),
                    min_overlap_bands=int(req.get("min_overlap_bands", 0)),
                    sources=req.get("source"),
                    categories=req.get("category"))
                dt = time.perf_counter() - t0
            except KeyError as exc:
                self._send(400, {"error": f"missing field {exc}"})
                return
            except ValueError as exc:
                self._send(400, {"error": str(exc)})
                return
            self._send(200, {"metric": metric, "elapsed_ms": round(dt*1000, 2),
                             "n_candidates": lib.n,
                             "overlap_required": int(math.ceil(need)),
                             "query_gaps_nm": [[round(a, 2), round(b, 2)]
                                               for a, b in query_gaps],
                             "results": hits})
    return Handler


def serve(index, host="127.0.0.1", port=8977, log=print):
    """Load an index and serve it until interrupted.

    :param index: index directory
    :param host: address to bind
    :param port: port to bind
    :param log: called with one line of startup information
    :return: the Library that was served, once the server stops
    """
    t0 = time.perf_counter()
    lib = Library(index)
    # Nothing is mapped until a query asks for a metric, so the size
    # reported is what each metric's set will cost when it is first used.
    per_set = 3 * lib.n * lib.nb * 4 / 1e6
    log(f"loaded {lib.n:,} spectra x {lib.nb} bands in "
        f"{(time.perf_counter() - t0) * 1000:.0f} ms; metrics "
        f"{', '.join(lib.metrics)} over "
        f"{len({tuple(v) for v in lib.matrices_by_metric.values()})} "
        f"matrix set(s) of {per_set:.0f} MB, mapped on first use")
    missing = [m for m in transforms.ALL if m not in lib.metrics]
    if missing:
        log(f"not served from this index: {', '.join(missing)} "
            f"(derive them with 'i.hyper.lib_server operation=derive "
            f"index={index} metric={','.join(missing)}')")
    srv = ThreadingHTTPServer((host, port), make_handler(lib))
    log(f"listening on http://{host}:{port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return lib
