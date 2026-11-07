"""Spectral gaps, and projection onto the canonical grid that respects them.

A spectrum's wavelength axis is rarely contiguous. Airborne and drone
products routinely drop both water-vapour windows, and field spectrometer
records often have the same regions removed as noise. Resampling such an
axis onto the canonical grid with one np.interp over [first, last] draws a
straight line across each gap and so claims support the instrument never
measured -- for the query, inventing structure to compare, and for a library
record, inventing structure to be compared against.

Measured on one real C2RD drone spectrum (HySpex VS620, 408 bands,
411.4-2449.7 nm, gaps at 1329.0-1482.6 nm and 1784.5-1984.0 nm) against the
251,753-spectrum index: bridging fabricates 36 of the 203 in-range grid
points, 17.7%, inflates every angle in the top 10 by 2.0-2.6 degrees, and
reorders 3 of them. The bias is conservative -- it makes good matches look
worse rather than inventing matches -- but it is systematic and it scales
with how much of a sensor's range is missing.

Both sides of the comparison use this module, so "overlap" counts real
shared bands on both. The gap test is deliberately the same relative one
i.hyper.spectroscopy's ``_spectral_segments()`` uses, so a cube segments
identically there and here.
"""
import numpy as np

# A step is a gap when it exceeds this many nm, or this multiple of the
# axis's own median spacing, whichever is larger. The relative term is what
# makes one test work for a 10 nm multispectral axis and a 1 nm laboratory
# one; the absolute floor keeps ordinary jitter from splitting an axis.
GAP_MIN_NM = 50.0
GAP_SPACING_FACTOR = 5.0


def segments(wavelengths):
    """Contiguous-coverage segment index for each wavelength.

    Returns an array of the same length as *wavelengths*, numbering the
    contiguous blocks of coverage from 0 upwards; the index increases at
    each spectral gap. Expects ascending input: on unsorted input no gap is
    detected and the whole axis is reported as one segment.
    """
    wl = np.asarray(wavelengths, dtype=float)
    if wl.size < 2:
        return np.zeros(wl.size, dtype=np.int32)
    spacing = np.diff(wl)
    threshold = max(GAP_MIN_NM, GAP_SPACING_FACTOR * float(np.median(spacing)))
    return np.concatenate(([0], np.cumsum(spacing > threshold))).astype(np.int32)


def gap_spans(wavelengths):
    """The (lo_nm, hi_nm) of each gap in an ascending wavelength axis."""
    wl = np.asarray(wavelengths, dtype=float)
    seg = segments(wl)
    return [(float(wl[i]), float(wl[i + 1]))
            for i in np.flatnonzero(np.diff(seg) > 0)]


def covered_mask(grid, wavelengths):
    """Which canonical grid points an axis really covers.

    True only for grid points inside one of the axis's contiguous segments,
    so the points an instrument skipped stay unsupported instead of being
    bridged. A one-point segment covers the single nearest grid point when
    it falls within half a grid step, which is an assignment rather than an
    interpolation across anything.
    """
    grid = np.asarray(grid, dtype=float)
    wl = np.asarray(wavelengths, dtype=float)
    mask = np.zeros(grid.size, dtype=bool)
    if wl.size == 0:
        return mask
    step = float(np.median(np.diff(grid))) if grid.size > 1 else 0.0
    seg = segments(wl)
    for s in range(int(seg.max()) + 1):
        sub = wl[seg == s]
        if sub.size == 1:
            j = int(np.argmin(np.abs(grid - sub[0])))
            if step and abs(grid[j] - sub[0]) <= step / 2.0:
                mask[j] = True
        else:
            mask |= (grid >= sub[0]) & (grid <= sub[-1])
    return mask


def project(grid, wavelengths, values):
    """Resample a spectrum onto the canonical grid, honouring its gaps.

    Interpolates within each contiguous segment only and leaves every other
    grid point NaN, which is what makes it fall out of the support mask the
    scoring already builds. Returns (row, gaps) with *row* float32 of
    ``grid``'s length and *gaps* the (lo, hi) spans that were left empty.
    """
    grid = np.asarray(grid, dtype=float)
    wl = np.asarray(wavelengths, dtype=float)
    val = np.asarray(values, dtype=float)
    row = np.full(grid.size, np.nan, dtype=np.float32)
    if wl.size == 0:
        return row, []
    step = float(np.median(np.diff(grid))) if grid.size > 1 else 0.0
    seg = segments(wl)
    for s in range(int(seg.max()) + 1):
        sel = seg == s
        x, y = wl[sel], val[sel]
        if x.size == 1:
            j = int(np.argmin(np.abs(grid - x[0])))
            if step and abs(grid[j] - x[0]) <= step / 2.0:
                row[j] = y[0]
            continue
        inside = (grid >= x[0]) & (grid <= x[-1])
        if inside.any():
            row[inside] = np.interp(grid[inside], x, y)
    return row, gap_spans(wl)


def finite_runs(mask):
    """[(start, stop), ...] index ranges of each True run in a 1D mask.

    Used to apply a continuum or a derivative within one contiguous block of
    support at a time: a hull drawn over two blocks, or a gradient taken
    across the boundary between them, would bridge the gap that *project*
    just took care to leave open.
    """
    m = np.asarray(mask, dtype=bool).astype(np.int8)
    if m.size == 0:
        return []
    edges = np.diff(np.concatenate(([0], m, [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(),
                    np.flatnonzero(edges == -1).tolist()))
