"""Build the resident search index from an OpenSpecLib library.

OpenSpecLib is the canonical store: ``<library>/spectra/<source>.parquet``
holds one row per spectrum, and the wavelength axes live once each in
``<library>/spectra/wavelengths.parquet``, referenced by an int32 grid id.
Comparing a query against the library row by row costs an np.interp per
candidate -- about 9 us, so 2.5 s for 250,000 candidates, per query. This
resamples every spectrum once onto a canonical grid, L2-normalises it, and
writes a float32 memmap plus a parallel metadata table. A spectral angle is
then one GEMV over the whole library.

Normalising offline is what makes the angle cheap: with unit-norm rows and a
unit-norm query, cos(theta) is a plain dot product, so there is no per-row
norm to recompute.

The grid deduplication pays off twice. Wavelengths are read once rather than
once per row, and the projection onto the canonical grid is a fixed linear
map per grid, so it is precomputed as gather indices plus weights and each
spectrum costs two gathers instead of a search. A row whose values are not
all finite falls back to np.interp, because dropping a bad value changes
that row's support and so cannot use the shared plan.

Called by i.hyper.lib_server, which passes a progress callback so that the
module reports through ``gs.percent()``. The percentage is spectra read over
spectra to read, not elapsed time. Nothing here writes a MetaMonitor slot:
inside a GRASS module the ``G_percent()`` stream is already forwarded by the
libmmgrass.so shim (https://github.com/YannChemin/MetaMonitor), and a second
report of the same work would show up as a second task.
"""
import glob
import json
import os
import tempfile

import numpy as np
import pyarrow.parquet as pq

import gaps


class OpenSpecLibMissing(RuntimeError):
    """openspeclib is not importable, so no unit conversion is available."""


def _openspeclib():
    """Import what the index needs from openspeclib, or fail loudly.

    Guessing the unit conversions would be a silent factor-of-1000 error on
    every wavenumber axis in the library, so there is no fallback.
    """
    try:
        from openspeclib.models import WavelengthUnit
        from openspeclib.storage import WAVELENGTHS_FILENAME
        from openspeclib.units import convert_wavelength_array
    except ImportError as exc:
        raise OpenSpecLibMissing(
            "Cannot import openspeclib ({}). The index is built from an "
            "OpenSpecLib library and uses its own unit conversions. Install "
            "it with 'pip install -e ~/dev/openspeclib', or run GRASS with "
            "PYTHONPATH=$HOME/dev/openspeclib/src.".format(exc)) from exc
    return WavelengthUnit, WAVELENGTHS_FILENAME, convert_wavelength_array


# Columns the index needs. Wavelengths come from the grid registry, not from
# here, and the values are the only heavy column left.
COLS = [
    "id",
    "name",
    "source.library",
    "material.category",
    "material.subcategory",
    "material.formula",
    "spectral_data.type",
    "spectral_data.wavelength_grid_id",
    "spectral_data.values",
]


def to_nm(wl, unit):
    """Convert a wavelength axis to nm using OpenSpecLib's own converters.

    Called once per unique grid, not per spectrum, so the per-value Python
    loop inside convert_wavelength_array costs nothing. Wavenumbers invert
    the axis, which is why every caller sorts afterwards.
    """
    WavelengthUnit, _, convert_wavelength_array = _openspeclib()
    try:
        u = WavelengthUnit(unit)
    except ValueError:
        return None                        # Unknown unit: skip the grid.
    return np.asarray(
        convert_wavelength_array(list(wl), u, WavelengthUnit.NANOMETERS),
        dtype=np.float64)


class GridPlan:
    """The fixed projection of one native wavelength axis onto the grid.

    ``sel`` reorders a row's native values to ascending wavelength and drops
    non-finite axis entries; ``inside`` marks the canonical bands the axis
    actually covers; ``lo``/``w`` are the linear-interpolation gather index
    and weight for each of those bands.

    ``inside`` excludes the grid points that fall in the axis's own spectral
    gaps (see gaps.py). Without that, a record whose instrument skipped the
    water-vapour windows -- which most field spectrometer records do --
    claims support there anyway, and the query is scored against a straight
    line. That is the same defect on the library side as bridging the query,
    and it is worse, because it is baked into the support matrix for every
    record at index time. Costing nothing per row is what makes it free to
    fix: the gap test is a property of the axis, so it is evaluated once per
    unique wavelength grid rather than once per spectrum.
    """

    __slots__ = ("sel", "inside", "lo", "w", "wl_good", "lo_nm", "hi_nm",
                 "n_native")

    def __init__(self, sel, inside, lo, w, wl_good, lo_nm, hi_nm, n_native):
        self.sel, self.inside, self.lo, self.w = sel, inside, lo, w
        self.wl_good = wl_good             # Only the fallback path reads it.
        self.lo_nm, self.hi_nm, self.n_native = lo_nm, hi_nm, n_native

    def project(self, values, grid, nb):
        """Resample one row's values onto the canonical grid.

        Returns NaN outside the axis's own coverage rather than
        extrapolating. Non-finite values fall back to np.interp over the
        row's own good samples, since they change its support.
        """
        v = np.asarray(values, dtype=np.float64)
        if v.size != self.n_native:
            return None
        v = v[self.sel]
        row = np.full(nb, np.nan, dtype=np.float32)
        if np.isfinite(v).all():
            row[self.inside] = (v[self.lo] * (1.0 - self.w)
                                + v[self.lo + 1] * self.w)
            return row
        good = np.isfinite(v)
        if good.sum() < 2:
            return None
        # Dropping non-finite values can open gaps this row has but its axis
        # does not, so its own coverage is recomputed rather than reusing
        # the plan's: interpolating over the surviving samples alone would
        # bridge a run of missing values the same way bridging a gap does.
        x, y = self.wl_good[good], v[good]
        sub = self.inside & gaps.covered_mask(grid, x)
        if not sub.any():
            return None
        row[sub] = np.interp(grid[sub], x, y)
        return row


def build_plans(path, grid, nb, min_coverage):
    """Precompute one GridPlan per wavelength grid in the registry.

    Returns (plans, unusable) where plans maps grid_id to a GridPlan or to
    None for a grid that cannot be indexed, and unusable counts those.
    """
    tab = pq.read_table(path)
    plans, unusable = {}, 0
    for row in tab.to_pylist():
        gid = int(row["grid_id"])
        wl = to_nm(row["wavelengths"], row["wavelength_unit"])
        if wl is None or wl.size < 2:
            plans[gid] = None
            unusable += 1
            continue
        finite = np.flatnonzero(np.isfinite(wl))
        sel = finite[np.argsort(wl[finite], kind="stable")]
        wl = wl[sel]
        # Strictly increasing: a duplicated wavelength would divide by zero
        # in the weight below, and the second copy carries no new information.
        keep = np.concatenate(([True], np.diff(wl) > 0))
        sel, wl = sel[keep], wl[keep]
        if wl.size < 2:
            plans[gid] = None
            unusable += 1
            continue
        inside = gaps.covered_mask(grid, wl)
        if inside.sum() < min_coverage * nb:
            plans[gid] = None
            unusable += 1
            continue
        # Every g is inside some segment, so the bracketing pair found here
        # lies within that same segment: no weight is ever computed across a
        # gap.
        g = grid[inside]
        lo = np.clip(np.searchsorted(wl, g, side="right") - 1, 0, wl.size - 2)
        w = (g - wl[lo]) / (wl[lo + 1] - wl[lo])
        plans[gid] = GridPlan(sel, inside, lo, np.clip(w, 0.0, 1.0), wl,
                              float(wl[0]), float(wl[-1]),
                              int(row["n_points"]))
    return plans, unusable




def build_index(library, out, step=10.0, lo=350.0, hi=2500.0,
                min_coverage=0.30, batch=2000, progress=None, log=print):
    """Resample a whole OpenSpecLib library onto one canonical grid.

    One pass, appending to a temporary raw file and reshaping at the end,
    so memory stays flat whatever the library's size.

    :param library: OpenSpecLib library directory (the one holding spectra/)
    :param out: index directory to write library.f32, metadata.json and
        index.json into
    :param step: canonical grid step in nm
    :param lo: first grid wavelength in nm
    :param hi: last grid wavelength in nm
    :param min_coverage: skip spectra covering less than this fraction of
        the canonical grid
    :param batch: rows per Parquet read
    :param progress: called as progress(done, total) as spectra are read
    :param log: called with one line of human-readable progress
    :return: the index.json content, as a dict
    :raises OpenSpecLibMissing: openspeclib is not importable
    :raises FileNotFoundError: the directory is not an OpenSpecLib library
    """
    _, wavelengths_filename, _ = _openspeclib()
    grid = np.arange(lo, hi + step / 2, step)
    nb = grid.size
    os.makedirs(out, exist_ok=True)

    spectra = os.path.join(library, "spectra")
    wl_path = os.path.join(spectra, wavelengths_filename)
    if not os.path.isfile(wl_path):
        raise FileNotFoundError(
            "Not an OpenSpecLib library: {} is missing. Build one with "
            "'openspeclib combine --output <dir>'.".format(wl_path))
    files = sorted(p for p in glob.glob(os.path.join(spectra, "*.parquet"))
                   if os.path.basename(p) != wavelengths_filename)
    if not files:
        raise FileNotFoundError(
            "No source parquet files under {}".format(spectra))

    plans, unusable_grids = build_plans(wl_path, grid, nb, min_coverage)
    log("{} sources, {} wavelength grids ({} outside the canonical range), "
        "grid {:.0f}-{:.0f} nm at {:.0f} nm = {} bands".format(
            len(files), len(plans), unusable_grids, lo, hi, step, nb))

    # Footer-only read, so the row count costs nothing and progress is a
    # real fraction of the spectra rather than of the files.
    handles = [(p, pq.ParquetFile(p)) for p in files]
    total_rows = sum(h.metadata.num_rows for _, h in handles)
    log("{:,} spectra to index".format(total_rows))

    raw = tempfile.NamedTemporaryFile(dir=out, suffix=".f32", delete=False)
    meta = []
    kept = skipped = seen = 0
    per_source = {}

    try:
        for path, pf in handles:
            name = os.path.basename(path)[:-8]
            for chunk in pf.iter_batches(batch_size=batch, columns=COLS):
                d = chunk.to_pydict()
                for j in range(chunk.num_rows):
                    seen += 1
                    plan = plans.get(d["spectral_data.wavelength_grid_id"][j])
                    if plan is None:
                        skipped += 1
                        continue
                    row = plan.project(d["spectral_data.values"][j], grid, nb)
                    if row is None:
                        skipped += 1
                        continue
                    sup = ~np.isnan(row)
                    v = row[sup]
                    n = np.linalg.norm(v)
                    if not np.isfinite(n) or n <= 0:
                        skipped += 1
                        continue
                    row[sup] = v / n       # Unit norm over its own support.

                    raw.write(row.tobytes())
                    # The source comes from the row, never from the file
                    # name: a mislabelled row would silently attribute a
                    # spectrum to the wrong library.
                    src = d["source.library"][j]
                    per_source[src] = per_source.get(src, 0) + 1
                    meta.append({
                        "id": d["id"][j],
                        "name": d["name"][j],
                        "source": src,
                        "category": d["material.category"][j],
                        "subcategory": d["material.subcategory"][j],
                        "formula": d["material.formula"][j],
                        "spectrum_type": d["spectral_data.type"][j],
                        "lo_nm": plan.lo_nm, "hi_nm": plan.hi_nm,
                        "n_native_bands": plan.n_native,
                    })
                    kept += 1
                if progress is not None:
                    progress(seen, total_rows)
            log("  {}: {:,}/{:,} read, kept {:,} skipped {:,}".format(
                name, seen, total_rows, kept, skipped))
    except Exception:
        raw.close()
        os.unlink(raw.name)
        raise

    raw.close()
    mat_path = os.path.join(out, "library.f32")
    os.replace(raw.name, mat_path)

    info = {"n_spectra": kept, "n_bands": nb, "grid_lo": lo, "grid_hi": hi,
            "grid_step": step, "dtype": "float32",
            "nan": "outside native coverage",
            "normalisation": "L2 over each spectrum's own support",
            "skipped": skipped, "unusable_grids": unusable_grids,
            "library": library, "sources": per_source}
    with open(os.path.join(out, "metadata.json"), "w") as fh:
        json.dump(meta, fh)
    with open(os.path.join(out, "index.json"), "w") as fh:
        json.dump(info, fh, indent=2)

    log("{:,} spectra indexed, {:,} skipped".format(kept, skipped))
    for src in sorted(per_source):
        log("  {:<18} {:>8,}".format(src, per_source[src]))
    log("  {}  {:.1f} MB".format(mat_path, os.path.getsize(mat_path) / 1e6))
    return info
