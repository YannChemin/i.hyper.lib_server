# i.hyper.lib_server

A GRASS GIS addon that builds, serves and queries a resident
nearest-neighbour index over an [OpenSpecLib](https://github.com/YannChemin/openspeclib)
spectral library.

The `i.hyper.lib_*` harvesters (`i.hyper.lib_ecosis`, `i.hyper.lib_gsti`,
`i.hyper.lib_ossl`, `i.hyper.lib_relab`, `i.hyper.lib_usgs`) fill one shared
store; `openspeclib` combines that store into a validated library; this
module turns the library into a `float32` memory map and answers *which
reference spectra are closest to this measured one* in tens of
milliseconds, exactly.

```sh
i.hyper.lib_server operation=build  library=~/grassdata/openspeclib index=~/grassdata/specserve
i.hyper.lib_server operation=derive index=~/grassdata/specserve metric=sam,cr_sam,gd1
i.hyper.lib_server operation=query  input=leaf.csv k=10 metric=cr_sam category=vegetation
i.hyper.lib_server operation=serve  index=~/grassdata/specserve port=8977
```

See `i.hyper.lib_server.md` for the manual: the metrics and why each needs
its own matrices, how spectral gaps are carried on both sides of the
comparison, and the two overlap thresholds that decide when a candidate is
rejected rather than scored.

## Layout

| File | Role |
|---|---|
| `i.hyper.lib_server.py` | The GRASS module: options, dispatch, spectrum reading, output |
| `indexing.py` | Resamples a library onto the canonical grid and writes `library.f32` |
| `derive.py` | Writes the three matrices each metric family scores against |
| `service.py` | The resident `Library` and the local HTTP service over it |
| `transforms.py` | Continuum removal and first derivative, shared by deriver and search |
| `gaps.py` | Spectral gaps, and projection onto the grid that respects them |

Everything but the module script is installed into
`etc/i.hyper.lib_server/` and found there at run time.

## Requirements

`numpy`, `pyarrow`, and `openspeclib` importable (`pip install -e
~/dev/openspeclib`, or run GRASS with
`PYTHONPATH=$HOME/dev/openspeclib/src`). The index is built from an
OpenSpecLib library and uses its unit conversions rather than guessing
them.

## Build

```sh
ln -s $HOME/dev/i.hyper.lib_server $HOME/dev/grass-addons/src/imagery/i.hyper.lib_server
make -C $HOME/dev/grass-addons/src/imagery/i.hyper.lib_server MODULE_TOPDIR=$HOME/dev/grass
```

## Tests

```sh
PYTHONPATH=$HOME/dev/openspeclib/src python3 -m pytest tests
```

## Licence

Public domain, see [LICENSE](LICENSE).
