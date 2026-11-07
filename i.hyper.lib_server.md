# i.hyper.lib_server

## NAME

*i.hyper.lib_server* - Build, serve and query a resident
nearest-neighbour index over an OpenSpecLib spectral library.

## SYNOPSIS

**i.hyper.lib_server**
**operation**=*build|derive|serve|status|query*
[**index**=*dir*] [**library**=*dir*]
[**step**=*value*] [**lo**=*value*] [**hi**=*value*]
[**min_coverage**=*value*] [**batch**=*value*]
[**metric**=*string*[,*string*,...]] [**input**=*file*] [**output**=*file*]
[**k**=*value*] [**source**=*string*[,*string*,...]]
[**category**=*string*[,*string*,...]]
[**min_overlap**=*value*] [**min_overlap_bands**=*value*]
[**format**=*plain|csv|json*]
[**host**=*string*] [**port**=*value*] [**url**=*string*] [**-s**]

## DESCRIPTION

*i.hyper.lib_server* is the search end of the `i.hyper.lib_*` family. The
harvesters (*i.hyper.lib_ecosis*, *i.hyper.lib_gsti*, *i.hyper.lib_ossl*,
*i.hyper.lib_relab*, *i.hyper.lib_usgs*) fill one shared store;
`openspeclib` combines that store into a validated library of per-source
Parquet files; this module turns that library into a resident index and
answers *which reference spectra are closest to this measured one*.

The question it answers is the same one *i.hyper.speclookup* asks of the
raw store, but at a different scale and with a different guarantee. A
linear scan over the store re-interpolates every candidate for every
query -- about 9 microseconds each, so seconds per query over a few hundred
thousand spectra. Here every library spectrum is resampled **once** onto a
canonical grid, L2-normalised, and written as a `float32` memory map, so a
spectral angle against the whole library is a single matrix-vector product:
tens of milliseconds, exactly, with no approximation anywhere.

### Why an exact search

No approximate index is used, and that is a measurement rather than a
preference. At this size a full GEMV takes 10-20 ms, while a
space-partitioning tree degenerates to a linear scan above roughly 20
dimensions, so a tree would be both slower and approximate. The matrices
are memory-mapped, so start-up is immediate and several processes share the
same physical pages.

### Operations

| operation | What it does |
|---|---|
| `build` | Resamples every spectrum of **library** onto the canonical grid and writes `library.f32`, `metadata.json` and `index.json` into **index** |
| `derive` | Writes the three matrices each **metric** family scores against, beside the index |
| `serve` | Runs the local HTTP service over **index** in the foreground, until interrupted |
| `status` | Reports what an index holds, or what a running service answers (**-s**) |
| `query` | Returns the **k** closest library spectra to the spectrum in **input** |

`build` then `derive` is the normal sequence: the index alone answers
nothing until its matrices exist, and `status` lists the metrics an index
can actually serve.

### Metrics

Four, in two families. `sam` and `cosine` compare reflectance as stored.
`cr_sam` compares continuum-removed **band depth** and `gd1` the first
derivative, and for those the library is scored in the same derived form,
from its own matrix set and its own support mask.

This is why each family needs its own matrices rather than sharing one set.
A query transformed on its own and compared against untransformed
reflectance is not a comparison: it returns a number shaped like an angle,
ordered by nothing in particular. A metric whose matrix set an index does
not carry is therefore **refused**, with the command that would build it,
rather than quietly answered from the reflectance matrices.

`cr_sam` returns band depth, `1 - R/C`, not the ratio. Every
continuum-removed spectrum is a vector of ones with shallow dips in it, so
any two of them are nearly parallel and the angle between them is dominated
by the constant they share rather than by the absorptions that distinguish
them. Subtracting the continuum leaves only the absorptions, which is the
quantity mineral identification rests on.

### Spectral gaps, on both sides

A wavelength axis is rarely contiguous: drone and airborne products
routinely drop both water-vapour windows, and field spectrometer records
often have the same regions removed as noise. Resampling such an axis with
one interpolation over its full span draws a straight line across each gap
and so claims support the instrument never measured.

Both the query and the library are projected gap by gap, so "overlap"
counts bands that were really measured on both sides. Measured on one
C2RD drone spectrum (HySpex VS620, 408 bands, 411.4-2449.7 nm, gaps at
1329.0-1482.6 nm and 1784.5-1984.0 nm) against a 251 753-spectrum index,
bridging fabricates 36 of the 203 in-range grid points, 17.7 %, inflates
every angle in the top 10 by 2.0-2.6 degrees, and reorders 3 of them.

A candidate sharing too few bands with the query is **rejected rather than
scored**: an angle measured over a handful of bands is not a measurement.
Two thresholds apply and the stricter decides. **min_overlap** is a
fraction of what the query itself covers, so on its own it loosens exactly
where the query is gappiest and the guarantee is needed most;
**min_overlap_bands** is an absolute floor in grid points.

### Querying without a server

**operation**=*query* opens the index in the module's own process by
default, which needs no service running: a module already holding the
matrices has no reason to serialise a spectrum to itself. Pass **-s**, or a
**url**, to send the query to a service instead -- worth it when many
queries follow one another, since the service keeps the matrices mapped
between them.

### Progress

Long passes report through `gs.percent()`, as a real fraction of the
spectra read, which the `libmmgrass.so` shim forwards to MetaMonitor
(<https://github.com/YannChemin/MetaMonitor>). The module writes no
MetaMonitor slot of its own; that would report the same work twice.

## NOTES

The index is built from an **OpenSpecLib** library, not from the shared
harvest store directly, because the unit conversions are openspeclib's own.
`openspeclib` must therefore be importable; if it is not, the module says
so and stops rather than guessing, since guessing a wavenumber axis is a
silent factor-of-1000 error.

The canonical grid defaults to 350-2500 nm at 10 nm, which is the
reflective range. Library spectra outside it -- OSSL's mid-infrared
absorbance, for one -- are skipped, and `index.json` records how many.

`library.f32` is `n_spectra x n_bands x 4` bytes and each metric family
adds three matrices of the same size, so a 360 000-spectrum index on the
default grid is roughly 310 MB for the reflectance set and the same again
per transformed metric. They are mapped on first use, not at start-up.

## EXAMPLES

### Build an index and the matrices it answers from

```sh
i.hyper.lib_server operation=build \
    library=$HOME/grassdata/openspeclib \
    index=$HOME/grassdata/specserve \
    step=10 lo=350 hi=2500 min_coverage=0.30

i.hyper.lib_server operation=derive \
    index=$HOME/grassdata/specserve metric=sam,cr_sam,gd1
```

### Ask what an index holds

```sh
i.hyper.lib_server operation=status index=$HOME/grassdata/specserve
```

### Identify a measured spectrum

```sh
i.hyper.lib_server operation=query \
    input=leaf.csv k=10 metric=sam \
    source=ecosis,gsti category=vegetation
```

With a mineral target, the continuum-removed metric is usually the one that
discriminates, and a hard floor on the overlap keeps a short SWIR window
from matching on three bands:

```sh
i.hyper.lib_server operation=query \
    input=outcrop.csv k=5 metric=cr_sam \
    category=mineral min_overlap=0.6 min_overlap_bands=40 \
    format=csv output=matches.csv
```

### Sample a cube and identify what is there

`r3.what` reads only the voxels asked for, so a point spectrum comes out of
a hyperspectral cube in well under a second:

```sh
r3.what input=mosaic_final coordinates=412340,4805210 -n \
    | tr ' ' '\n' > pixel.csv
i.hyper.lib_server operation=query input=pixel.csv k=10 metric=cr_sam
```

### Run the service, then query it

```sh
i.hyper.lib_server operation=serve index=$HOME/grassdata/specserve port=8977 &
i.hyper.lib_server operation=status -s port=8977
i.hyper.lib_server operation=query -s input=leaf.csv k=10
```

## SEE ALSO

*[i.hyper.lib_ecosis](i.hyper.lib_ecosis.md)*,
*[i.hyper.lib_gsti](i.hyper.lib_gsti.md)*,
*[i.hyper.lib_ossl](i.hyper.lib_ossl.md)*,
*[i.hyper.lib_relab](i.hyper.lib_relab.md)*,
*[i.hyper.lib_usgs](i.hyper.lib_usgs.md)*,
*[i.hyper.speclookup](i.hyper.speclookup.md)*,
*[i.hyper.specresamp](i.hyper.specresamp.md)*,
*[r3.what](r3.what.md)*

## AUTHOR

Spectral Feature Extraction and Interpretation Engine
