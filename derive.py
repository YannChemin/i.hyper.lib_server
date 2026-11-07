"""Derive the auxiliary matrices the search needs for copy-free scoring.

A shared-support cosine needs three sums per candidate, over the bands the
query and that candidate both have:

    numerator      = sum v*q          -> values      @ q
    candidate norm = sum v^2          -> values_sq   @ query_mask
    query norm     = sum q^2          -> support     @ q_squared
    overlap count  = sum 1            -> support     @ query_mask

Each is a GEMV over a whole matrix, so none of them needs to slice rows out
of the library. Slicing was the prototype's bottleneck: fancy indexing a
memmap copies roughly 200 MB per query, which cost ~740 ms against the ~7 ms
the arithmetic actually takes.

Values are stored zero-filled (NaN means "band not measured", and zero
contributes nothing to any of the four sums), with the support mask kept
separately so a real zero reading is never mistaken for a missing band.

One set of three matrices per metric family. The transformed metrics compare
a derived quantity -- continuum-removed band depth for ``cr_sam``, a first
derivative for ``gd1`` -- and a query in that form scored against
untransformed reflectance is not a comparison at all: it returns a number
shaped like an angle and ordered by nothing in particular. The transform
also changes the support, because a block of fewer than three points carries
no continuum and no derivative and is dropped, so the mask cannot be shared
either. ``sam`` and ``cosine`` need no transform and keep the original
unsuffixed file names.

Run after the index is built; reads library.f32 and writes beside it.
Progress goes back to the caller as a callback, which i.hyper.lib_server
turns into ``gs.percent()``; the libmmgrass.so shim forwards that to
MetaMonitor (https://github.com/YannChemin/MetaMonitor) on its own, so
nothing here writes a slot.
"""
import json
import os

import numpy as np

import transforms

STEP = 20000  # Rows per chunk: peak memory stays tens of MB.


def derive_one(out, metric, src, grid, n, nb, progress):
    """Write the three matrices one metric scores against.

    :param out: index directory
    :param metric: the metric whose family to write
    :param src: the (n, nb) reflectance memmap
    :param grid: the canonical wavelength grid
    :param n: number of spectra
    :param nb: number of bands
    :param progress: called as progress(rows_done, metric)
    :return: the three file names written
    """
    sfx = transforms.suffix(metric)
    names = [f"values{sfx}.f32", f"values_sq{sfx}.f32", f"support{sfx}.f32"]

    def mapped(name):
        return np.memmap(os.path.join(out, name), dtype=np.float32,
                         mode="w+", shape=(n, nb))

    vals, sq, sup = (mapped(f) for f in names)
    for i in range(0, n, STEP):
        blk = np.asarray(src[i:i + STEP])
        if metric in transforms.TRANSFORMED:
            blk = np.stack([transforms.apply(r, grid, metric) for r in blk])
        m = ~np.isnan(blk)
        v = np.where(m, blk, 0.0).astype(np.float32)
        vals[i:i + STEP] = v
        sq[i:i + STEP] = v * v
        sup[i:i + STEP] = m.astype(np.float32)
        progress(min(i + STEP, n), metric)
    for a in (vals, sq, sup):
        a.flush()
    del vals, sq, sup
    return names


def derive_matrices(index_dir, metrics=None, progress=None, log=print):
    """Write one matrix set per metric family of *metrics*.

    :param index_dir: directory holding index.json and library.f32
    :param metrics: metrics to derive; all known ones by default
    :param progress: called as progress(done, total) over rows x families
    :param log: called with one line of human-readable progress
    :return: the updated index.json content, as a dict
    :raises ValueError: an unknown metric was asked for
    """
    metrics = list(metrics or (("sam",) + transforms.TRANSFORMED))
    for m in metrics:
        if m not in transforms.ALL:
            raise ValueError("unknown metric {!r}; known: {}".format(
                m, ", ".join(transforms.ALL)))

    with open(os.path.join(index_dir, "index.json")) as fh:
        info = json.load(fh)
    n, nb = info["n_spectra"], info["n_bands"]
    grid = np.arange(info["grid_lo"],
                     info["grid_hi"] + info["grid_step"] / 2,
                     info["grid_step"])
    src = np.memmap(os.path.join(index_dir, "library.f32"), dtype=np.float32,
                    mode="r", shape=(n, nb))

    families = []
    for m in metrics:
        sfx = transforms.suffix(m)
        if sfx not in families:
            families.append(sfx)
    total = n * len(families)
    log("{:,} spectra x {} bands, {} matrix set(s): {}".format(
        n, nb, len(families), ", ".join(metrics)))

    written = dict(info.get("matrices_by_metric") or {})
    done_rows = [0]

    def on_rows(rows_in_metric, metric):
        if progress is not None:
            progress(done_rows[0] + rows_in_metric, total)

    seen = set()
    for m in metrics:
        sfx = transforms.suffix(m)
        if sfx in seen:
            continue
        seen.add(sfx)
        names = derive_one(index_dir, m, src, grid, n, nb, on_rows)
        done_rows[0] += n
        written[m] = names
        size = sum(os.path.getsize(os.path.join(index_dir, f))
                   for f in names) / 1e6
        log("  {}: wrote {} ({:.0f} MB)".format(m, ", ".join(names), size))

    # "cosine" scores against the same reflectance matrices as "sam".
    if "sam" in written:
        for m in transforms.IDENTITY:
            written[m] = written["sam"]

    info["matrices_by_metric"] = written
    info["matrices"] = written.get("sam", info.get("matrices"))
    info["layout"] = ("zero-filled values, their squares, and a 0/1 support "
                      "mask, one set per metric family; cr_sam is continuum-"
                      "removed band depth, gd1 a first derivative")
    with open(os.path.join(index_dir, "index.json"), "w") as fh:
        json.dump(info, fh, indent=2)

    sets = {tuple(v) for v in written.values()}
    grand = sum(os.path.getsize(os.path.join(index_dir, f))
                for names in sets for f in names) / 1e6
    log("wrote {} matrix set(s), {:.0f} MB total".format(len(sets), grand))
    return info
