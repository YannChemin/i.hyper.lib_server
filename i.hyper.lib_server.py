#!/usr/bin/env python
##############################################################################
# MODULE:    i.hyper.lib_server
# AUTHOR(S): Spectral Feature Extraction and Interpretation Engine
# PURPOSE:   Build, serve and query a resident nearest-neighbour index over
#            the OpenSpecLib spectral library that the i.hyper.lib_*
#            harvesters feed, so that spectral identification against the
#            whole library is a GRASS operation rather than a side script.
# SPDX-License-Identifier: Unlicense
##############################################################################

# %module
# % description: Build, serve and query a resident nearest-neighbour index over an OpenSpecLib spectral library
# % keyword: imagery
# % keyword: hyperspectral
# % keyword: spectral library
# % keyword: spectral angle
# % keyword: server
# %end

# %option
# % key: operation
# % type: string
# % required: yes
# % options: build,derive,serve,status,query
# % description: build: resample the library onto the canonical grid; derive: write the per-metric matrices the search scores against; serve: run the local HTTP service; status: report what an index or a running service holds; query: find the closest reference spectra to a measured one
# %end

# %option G_OPT_M_DIR
# % key: index
# % required: no
# % description: Index directory (default: $HOME/grassdata/specserve)
# % guisection: Index
# %end

# %option G_OPT_M_DIR
# % key: library
# % required: no
# % description: OpenSpecLib library to index, the directory holding spectra/ (default: $HOME/grassdata/openspeclib)
# % guisection: Index
# %end

# %option
# % key: step
# % type: double
# % required: no
# % answer: 10.0
# % description: Canonical grid step in nm
# % guisection: Index
# %end

# %option
# % key: lo
# % type: double
# % required: no
# % answer: 350.0
# % description: First wavelength of the canonical grid, in nm
# % guisection: Index
# %end

# %option
# % key: hi
# % type: double
# % required: no
# % answer: 2500.0
# % description: Last wavelength of the canonical grid, in nm
# % guisection: Index
# %end

# %option
# % key: min_coverage
# % type: double
# % required: no
# % answer: 0.30
# % options: 0.0-1.0
# % description: Skip library spectra covering less than this fraction of the canonical grid
# % guisection: Index
# %end

# %option
# % key: batch
# % type: integer
# % required: no
# % answer: 2000
# % description: Library rows read per Parquet batch (bounds peak memory, not speed)
# % guisection: Index
# %end

# %option
# % key: metric
# % type: string
# % required: no
# % multiple: yes
# % options: sam,cosine,cr_sam,gd1
# % description: Metrics to derive, or the single metric to query with. sam and cosine compare reflectance as stored; cr_sam compares continuum-removed band depth and gd1 the first derivative, each from its own matrix set
# % guisection: Search
# %end

# %option G_OPT_F_INPUT
# % key: input
# % required: no
# % description: Query spectrum: a two-column text file of wavelength (nm) and value, comma, semicolon, tab or space separated, with an optional header line
# % guisection: Search
# %end

# %option G_OPT_F_OUTPUT
# % key: output
# % required: no
# % description: Write the query results here instead of to standard output
# % guisection: Search
# %end

# %option
# % key: k
# % type: integer
# % required: no
# % answer: 10
# % description: Number of matches to return
# % guisection: Search
# %end

# %option
# % key: source
# % type: string
# % required: no
# % multiple: yes
# % description: Restrict matches to these source libraries (e.g. usgs_splib07,ecosis)
# % guisection: Search
# %end

# %option
# % key: category
# % type: string
# % required: no
# % multiple: yes
# % description: Restrict matches to these material categories (e.g. mineral,vegetation)
# % guisection: Search
# %end

# %option
# % key: min_overlap
# % type: double
# % required: no
# % answer: 0.5
# % options: 0.0-1.0
# % description: Reject a candidate covering less than this fraction of the query's own in-grid bands
# % guisection: Search
# %end

# %option
# % key: min_overlap_bands
# % type: integer
# % required: no
# % answer: 0
# % description: Reject a candidate covering fewer than this many grid points, whatever the fraction works out to; the stricter of the two requirements decides
# % guisection: Search
# %end

# %option
# % key: format
# % type: string
# % required: no
# % answer: plain
# % options: plain,csv,json
# % description: Output format for query and status
# % guisection: Search
# %end

# %option
# % key: host
# % type: string
# % required: no
# % answer: 127.0.0.1
# % description: Address the service binds to, or the one status queries
# % guisection: Service
# %end

# %option
# % key: port
# % type: integer
# % required: no
# % answer: 8977
# % description: Port the service binds to, or the one status queries
# % guisection: Service
# %end

# %option
# % key: url
# % type: string
# % required: no
# % description: Query a service already running at this URL instead of opening the index in this process
# % guisection: Service
# %end

# %flag
# % key: s
# % description: Query the running service at host/port rather than opening the index directly
# % guisection: Service
# %end

"""Build, serve and query the spectral library index from GRASS.

The i.hyper.lib_* harvesters fill one shared store, openspeclib combines it
into a validated library, and this module turns that library into a resident
index and answers nearest-neighbour queries against it. The search engine
itself lives beside this script and is installed next to it: indexing.py,
derive.py, service.py, transforms.py and gaps.py.

Long passes report through gs.percent(), which the libmmgrass.so shim
forwards to MetaMonitor (https://github.com/YannChemin/MetaMonitor); the
module writes no slot of its own, as that would report the same work twice.
"""

import json
import os
import sys

import grass.script as gs

DEFAULT_INDEX = os.path.join(os.path.expanduser("~"), "grassdata", "specserve")
DEFAULT_LIBRARY = os.path.join(
    os.path.expanduser("~"), "grassdata", "openspeclib"
)

# Separators a two-column spectrum file may use, tried in this order on the
# first data line and then kept for the whole file.
_SEPARATORS = (",", ";", "\t", None)


def _engine_dir():
    """Where indexing.py and friends live, installed or in the source tree.

    Returns:
        The directory, or None when the search engine cannot be found.
    """
    from grass.script.utils import get_lib_path

    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [get_lib_path(modname="i.hyper.lib_server"), here]
    for base in (os.environ.get("GISBASE"), os.environ.get("GRASS_ADDON_BASE")):
        if base:
            candidates.append(os.path.join(base, "etc", "i.hyper.lib_server"))
    for path in candidates:
        if path and os.path.isfile(os.path.join(path, "indexing.py")):
            return path
    return None


def _load_engine():
    """Put the search engine on sys.path and return its modules.

    Returns:
        The (indexing, derive, service, transforms) modules.
    """
    path = _engine_dir()
    if path is None:
        gs.fatal(
            _(
                "Cannot find the search engine (indexing.py). It is installed "
                "beside the module, under etc/i.hyper.lib_server."
            )
        )
    if path not in sys.path:
        sys.path.insert(0, path)
    try:
        import derive
        import indexing
        import service
        import transforms
    except ImportError as exc:
        gs.fatal(
            _("Cannot load the search engine from {path}: {err}").format(
                path=path, err=exc
            )
        )
    return indexing, derive, service, transforms


def _percent_reporter():
    """A progress callback that feeds gs.percent().

    Returns:
        A function taking (done, total).
    """

    def report(done, total):
        gs.percent(done, max(total, 1), 1)

    return report


def read_spectrum(path):
    """Read a two-column wavelength/value text file.

    A header line is tolerated: a first line whose fields do not parse as
    numbers is skipped rather than failing, because every spectrometer
    export writes one and refusing them would make the module unusable on
    real files.

    Args:
        path: File to read.

    Returns:
        A (wavelengths, values) pair of float lists.
    """
    wavelengths, values = [], []
    separator = None
    with open(path) as handle:
        for number, line in enumerate(handle):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = None
            for candidate in (separator,) if wavelengths else _SEPARATORS:
                parts = line.split(candidate)
                if len(parts) >= 2:
                    try:
                        pair = (float(parts[0]), float(parts[1]))
                    except ValueError:
                        continue
                    fields, separator = pair, candidate
                    break
            if fields is None:
                if number == 0 or not wavelengths:
                    continue  # Header, or a comment the file did not mark.
                gs.warning(
                    _("Ignoring unreadable line {n} of <{path}>").format(
                        n=number + 1, path=path
                    )
                )
                continue
            wavelengths.append(fields[0])
            values.append(fields[1])
    if len(wavelengths) < 2:
        gs.fatal(
            _("<{path}> holds no usable wavelength/value pairs").format(
                path=path
            )
        )
    return wavelengths, values


def _write(text, output):
    """Write one block of output to a file or to standard output.

    Args:
        text: What to write.
        output: Destination path, or an empty string for standard output.
    """
    if output:
        with open(output, "w") as handle:
            handle.write(text + "\n")
        gs.message(_("Written to <{path}>").format(path=output))
    else:
        print(text)


def _format_hits(payload, fmt):
    """Render a search response.

    Args:
        payload: The response dict, as the service would return it.
        fmt: One of plain, csv, json.

    Returns:
        The rendered text.
    """
    if fmt == "json":
        return json.dumps(payload, indent=2)
    hits = payload["results"]
    if fmt == "csv":
        lines = [
            "rank,id,name,source,category,angle_deg,cosine,overlap_bands"
        ]
        for rank, hit in enumerate(hits, 1):
            lines.append(
                "{},{},{},{},{},{:.4f},{:.6f},{}".format(
                    rank,
                    hit["id"],
                    str(hit["name"]).replace(",", " "),
                    hit["source"],
                    hit["category"],
                    hit["angle_rad"] * 180.0 / 3.141592653589793,
                    hit["cosine"],
                    hit["overlap_bands"],
                )
            )
        return "\n".join(lines)
    lines = [
        "{:>3}  {:>7}  {:>6}  {:<18} {:<12} {}".format(
            "#", "angle", "bands", "source", "category", "name"
        )
    ]
    for rank, hit in enumerate(hits, 1):
        lines.append(
            "{:>3}  {:>6.2f}d  {:>6}  {:<18} {:<12} {}".format(
                rank,
                hit["angle_rad"] * 180.0 / 3.141592653589793,
                hit["overlap_bands"],
                hit["source"],
                hit["category"] or "-",
                hit["name"],
            )
        )
        lines.append("     {}".format(hit["id"]))
    return "\n".join(lines)


def do_build(options):
    """Resample a whole library onto the canonical grid.

    Args:
        options: Parsed module options.
    """
    indexing, _derive, _service, _transforms = _load_engine()
    library = options["library"] or DEFAULT_LIBRARY
    index = options["index"] or DEFAULT_INDEX
    if not os.path.isdir(library):
        gs.fatal(_("Library <{path}> does not exist").format(path=library))

    gs.message(
        _("Indexing <{lib}> into <{idx}>...").format(lib=library, idx=index)
    )
    try:
        info = indexing.build_index(
            library=library,
            out=index,
            step=float(options["step"]),
            lo=float(options["lo"]),
            hi=float(options["hi"]),
            min_coverage=float(options["min_coverage"]),
            batch=int(options["batch"]),
            progress=_percent_reporter(),
            log=gs.verbose,
        )
    except indexing.OpenSpecLibMissing as exc:
        gs.fatal(str(exc))
    except FileNotFoundError as exc:
        gs.fatal(str(exc))
    gs.percent(1, 1, 1)
    gs.message(
        _("{kept} spectra indexed over {bands} bands, {skipped} skipped").format(
            kept=info["n_spectra"],
            bands=info["n_bands"],
            skipped=info["skipped"],
        )
    )
    gs.message(
        _("Derive the search matrices next: operation=derive index={idx}").format(
            idx=index
        )
    )


def do_derive(options):
    """Write the per-metric matrices the search scores against.

    Args:
        options: Parsed module options.
    """
    _indexing, derive, _service, transforms = _load_engine()
    index = options["index"] or DEFAULT_INDEX
    if not os.path.isfile(os.path.join(index, "index.json")):
        gs.fatal(
            _(
                "<{idx}> holds no index. Build one first: "
                "operation=build library=<dir> index={idx}"
            ).format(idx=index)
        )
    metrics = [m for m in (options["metric"] or "").split(",") if m]
    if not metrics:
        metrics = ["sam"] + list(transforms.TRANSFORMED)
    gs.message(
        _("Deriving matrices for {metrics} in <{idx}>...").format(
            metrics=", ".join(metrics), idx=index
        )
    )
    try:
        info = derive.derive_matrices(
            index, metrics, progress=_percent_reporter(), log=gs.verbose
        )
    except (ValueError, OSError) as exc:
        gs.fatal(str(exc))
    gs.percent(1, 1, 1)
    gs.message(
        _("Index <{idx}> now answers: {metrics}").format(
            idx=index, metrics=", ".join(sorted(info["matrices_by_metric"]))
        )
    )


def do_serve(options):
    """Run the HTTP service in the foreground until interrupted.

    Args:
        options: Parsed module options.
    """
    _indexing, _derive, service, _transforms = _load_engine()
    index = options["index"] or DEFAULT_INDEX
    if not os.path.isfile(os.path.join(index, "index.json")):
        gs.fatal(_("<{idx}> holds no index").format(idx=index))
    gs.message(
        _("Serving <{idx}> on {host}:{port}; interrupt to stop").format(
            idx=index, host=options["host"], port=options["port"]
        )
    )
    try:
        service.serve(
            index,
            host=options["host"],
            port=int(options["port"]),
            log=gs.message,
        )
    except OSError as exc:
        gs.fatal(
            _("Cannot bind {host}:{port}: {err}").format(
                host=options["host"], port=options["port"], err=exc
            )
        )


def _service_url(options):
    """The service address a query or status should talk to.

    Args:
        options: Parsed module options.

    Returns:
        The base URL.
    """
    if options["url"]:
        return options["url"].rstrip("/")
    return "http://{host}:{port}".format(
        host=options["host"], port=options["port"]
    )


def do_status(options, flags):
    """Report what an index holds, or what a running service answers.

    Args:
        options: Parsed module options.
        flags: Parsed module flags.
    """
    index = options["index"] or DEFAULT_INDEX
    if flags["s"] or options["url"]:
        import urllib.error
        import urllib.request

        url = _service_url(options) + "/health"
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                payload = json.loads(response.read().decode())
        except (urllib.error.URLError, OSError, ValueError) as exc:
            gs.fatal(
                _("No service answering at {url}: {err}").format(
                    url=url, err=exc
                )
            )
    else:
        path = os.path.join(index, "index.json")
        if not os.path.isfile(path):
            gs.fatal(_("<{idx}> holds no index").format(idx=index))
        with open(path) as handle:
            info = json.load(handle)
        payload = {
            "index": index,
            "n_spectra": info["n_spectra"],
            "n_bands": info["n_bands"],
            "grid": [info["grid_lo"], info["grid_hi"], info["grid_step"]],
            "metrics": sorted(info.get("matrices_by_metric") or {}),
            "library": info.get("library"),
            "sources": info.get("sources", {}),
            "skipped": info.get("skipped"),
        }
    if options["format"] == "json":
        _write(json.dumps(payload, indent=2), options["output"])
        return
    lines = []
    for key in ("index", "library", "n_spectra", "n_bands", "grid", "metrics"):
        if key in payload:
            value = payload[key]
            if isinstance(value, list):
                value = ", ".join(str(v) for v in value)
            lines.append("{:<12} {}".format(key, value))
    for name, count in sorted((payload.get("sources") or {}).items()):
        lines.append("{:<12} {:>8}".format("  " + name, count))
    _write("\n".join(lines), options["output"])


def do_query(options, flags):
    """Find the library spectra closest to a measured one.

    Args:
        options: Parsed module options.
        flags: Parsed module flags.
    """
    if not options["input"]:
        gs.fatal(_("operation=query needs input=<spectrum file>"))
    metrics = [m for m in (options["metric"] or "sam").split(",") if m]
    if len(metrics) != 1:
        gs.fatal(_("operation=query takes exactly one metric"))
    metric = metrics[0]
    wavelengths, values = read_spectrum(options["input"])
    gs.verbose(
        _("Query: {n} bands, {lo:.1f}-{hi:.1f} nm").format(
            n=len(wavelengths), lo=min(wavelengths), hi=max(wavelengths)
        )
    )
    sources = [s for s in (options["source"] or "").split(",") if s]
    categories = [c for c in (options["category"] or "").split(",") if c]

    if flags["s"] or options["url"]:
        payload = _query_service(
            options, metric, wavelengths, values, sources, categories
        )
    else:
        payload = _query_index(
            options, metric, wavelengths, values, sources, categories
        )
    if not payload["results"]:
        gs.warning(
            _(
                "No candidate shares enough bands with the query "
                "({need} grid points required)"
            ).format(need=payload["overlap_required"])
        )
    _write(_format_hits(payload, options["format"]), options["output"])


def _query_index(options, metric, wavelengths, values, sources, categories):
    """Search by opening the index in this process.

    Args:
        options: Parsed module options.
        metric: The single metric to score with.
        wavelengths: Query wavelengths in nm.
        values: Query values.
        sources: Source libraries to restrict to, possibly empty.
        categories: Material categories to restrict to, possibly empty.

    Returns:
        The same payload shape the HTTP service returns.
    """
    import math
    import time

    _indexing, _derive, service, _transforms = _load_engine()
    index = options["index"] or DEFAULT_INDEX
    if not os.path.isfile(os.path.join(index, "index.json")):
        gs.fatal(_("<{idx}> holds no index").format(idx=index))
    library = service.Library(index)
    start = time.perf_counter()
    try:
        hits, query_gaps, need = library.search(
            wavelengths,
            values,
            k=int(options["k"]),
            metric=metric,
            min_overlap=float(options["min_overlap"]),
            min_overlap_bands=int(options["min_overlap_bands"]),
            sources=sources or None,
            categories=categories or None,
        )
    except ValueError as exc:
        gs.fatal(str(exc))
    elapsed = time.perf_counter() - start
    for lo, hi in query_gaps:
        gs.verbose(
            _("Query gap {lo:.1f}-{hi:.1f} nm carried as no-support").format(
                lo=lo, hi=hi
            )
        )
    return {
        "metric": metric,
        "elapsed_ms": round(elapsed * 1000, 2),
        "n_candidates": library.n,
        "overlap_required": int(math.ceil(need)),
        "query_gaps_nm": [[round(a, 2), round(b, 2)] for a, b in query_gaps],
        "results": hits,
    }


def _query_service(options, metric, wavelengths, values, sources, categories):
    """Search by posting to a service that is already running.

    Args:
        options: Parsed module options.
        metric: The single metric to score with.
        wavelengths: Query wavelengths in nm.
        values: Query values.
        sources: Source libraries to restrict to, possibly empty.
        categories: Material categories to restrict to, possibly empty.

    Returns:
        The service's response payload.
    """
    import urllib.error
    import urllib.request

    url = _service_url(options) + "/search"
    request = {
        "wavelengths": wavelengths,
        "values": values,
        "k": int(options["k"]),
        "metric": metric,
        "min_overlap": float(options["min_overlap"]),
        "min_overlap_bands": int(options["min_overlap_bands"]),
    }
    if sources:
        request["source"] = sources
    if categories:
        request["category"] = categories
    body = json.dumps(request).encode()
    try:
        post = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(post, timeout=120) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        gs.fatal(
            _("Service refused the query ({code}): {detail}").format(
                code=exc.code, detail=detail
            )
        )
    except (urllib.error.URLError, OSError, ValueError) as exc:
        gs.fatal(
            _("No service answering at {url}: {err}").format(url=url, err=exc)
        )


def main(options, flags):
    """Dispatch on the requested operation.

    Args:
        options: Parsed module options.
        flags: Parsed module flags.

    Returns:
        The process exit status.
    """
    operation = options["operation"]
    if operation == "build":
        do_build(options)
    elif operation == "derive":
        do_derive(options)
    elif operation == "serve":
        do_serve(options)
    elif operation == "status":
        do_status(options, flags)
    else:
        do_query(options, flags)
    return 0


if __name__ == "__main__":
    opts, flgs = gs.parser()
    sys.exit(main(opts, flgs))
