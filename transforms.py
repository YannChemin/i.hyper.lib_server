"""Metric-specific transforms of a gridded spectrum.

Shared by the index deriver and the server, which is the point of the
module: a transform applied to the query but not to the library compares
two different physical quantities and returns a number that looks like an
angle. Both sides import from here so they cannot drift apart.

Each transform is applied within one contiguous block of support at a time.
A convex hull drawn over two blocks would span the gap between them, and a
gradient taken across their boundary would be the slope of a line that was
never measured -- the same bridging ``gaps.project`` takes care to avoid,
one step later. A block too short to carry a continuum or a derivative
(under three points) is dropped to NaN rather than left as raw reflectance,
which would mix two quantities in one vector; the support mask that goes
with the transform therefore differs from the untransformed one, which is
why each metric needs its own matrices rather than a shared set.

Scale does not matter. The library rows are L2-normalised over their own
support, and both transforms below commute with a per-row scale factor --
``y/continuum`` exactly, a derivative up to that same factor -- while the
angle itself is scale-free. So transforming the normalised row gives the
same comparison as transforming the raw one.
"""
import numpy as np

import gaps

# "sam" and "cosine" compare reflectance as stored, so they need no
# transform and share one set of matrices.
IDENTITY = ("sam", "cosine")
TRANSFORMED = ("cr_sam", "gd1")
ALL = IDENTITY + TRANSFORMED


def suffix(metric):
    """The file-name suffix of the matrix set a metric scores against.

    Untransformed metrics keep the original unsuffixed names, so an index
    derived before the transformed sets existed still serves ``sam``.
    """
    return "" if metric in IDENTITY else "_" + metric


def apply(row, grid, metric):
    """Return *row* in the form *metric* compares, NaN outside support.

    ``cr_sam`` returns continuum-removed **band depth**, ``1 - R/C``, not
    the ratio itself. The distinction decides whether the metric works at
    all: every continuum-removed spectrum is a vector of ones with shallow
    dips in it, so any two of them are nearly parallel and the angle between
    them is dominated by the constant component they share rather than by
    the absorptions that distinguish them. Subtracting the continuum instead
    of dividing by it leaves only the absorptions, which is the quantity
    mineral identification actually rests on. A featureless spectrum becomes
    the zero vector and so matches nothing, which is the honest answer.

    ``gd1`` returns the first derivative with respect to wavelength.
    """
    if metric in IDENTITY:
        return row
    if metric not in TRANSFORMED:
        raise ValueError(f"unknown metric {metric!r}")
    out = np.asarray(row, dtype=np.float32).copy()
    for lo, hi in gaps.finite_runs(~np.isnan(out)):
        x, y = np.asarray(grid[lo:hi], dtype=np.float64), out[lo:hi]
        if x.size < 3:
            out[lo:hi] = np.nan
            continue
        if metric == "cr_sam":
            cont = _upper_hull(x, y)
            # A non-positive continuum means the upper envelope of the
            # block sits at or below zero, so the readings under it are not
            # reflectances at all but an atmospheric-correction artefact.
            # Those points lose their support rather than being called
            # featureless, which a depth of zero would quietly claim.
            with np.errstate(invalid="ignore", divide="ignore"):
                depth = 1.0 - y / cont
            out[lo:hi] = np.where(cont > 0, depth, np.nan)
        else:
            out[lo:hi] = np.gradient(y.astype(np.float64), x)
    return out


def _upper_hull(x, y):
    """The upper convex hull of (x, y), interpolated back onto x.

    This is the local continuum: the taut line laid over the top of the
    block, touching it wherever the spectrum is not absorbing.
    """
    hull = [0]
    for i in range(1, len(x)):
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            cross = (x[b] - x[a]) * (y[i] - y[a]) - (y[b] - y[a]) * (x[i] - x[a])
            if cross >= 0:
                hull.pop()
            else:
                break
        hull.append(i)
    return np.interp(x, x[hull], y[hull])
