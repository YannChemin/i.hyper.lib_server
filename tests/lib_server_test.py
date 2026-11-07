"""Tests for i.hyper.lib_server's engine and its spectrum reader.

The module itself is a GRASS script, so the parts that do not need a GRASS
session are exercised directly: the index builder and the search are driven
against a small synthetic library written to a temporary directory, and the
module's own file reader is loaded with a stub for ``grass.script``.
"""

import builtins
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ADDON = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ADDON))

import derive  # noqa: E402
import indexing  # noqa: E402
import service  # noqa: E402
import transforms  # noqa: E402

GRID_LO, GRID_HI, STEP = 400.0, 900.0, 10.0


def _module():
    """Load the GRASS script with grass.script stubbed out."""
    stub = types.ModuleType("grass.script")
    for name in ("message", "warning", "verbose", "percent", "debug"):
        setattr(stub, name, lambda *a, **k: None)

    def fatal(msg, *a, **k):
        raise SystemExit(msg)

    stub.fatal = fatal
    stub.parser = lambda: ({}, {})
    utils = types.ModuleType("grass.script.utils")
    utils.get_lib_path = lambda modname, libname=None: None
    stub.utils = utils
    grass = types.ModuleType("grass")
    grass.script = stub
    sys.modules.setdefault("grass", grass)
    sys.modules["grass.script"] = stub
    sys.modules["grass.script.utils"] = utils
    builtins._ = lambda s: s
    path = ADDON / "i.hyper.lib_server.py"
    mod = types.ModuleType("i_hyper_lib_server")
    mod.__file__ = str(path)
    exec(compile(path.read_text(), str(path), "exec"), mod.__dict__)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _module()


def _library(tmp_path, spectra):
    """Write a minimal OpenSpecLib library holding the given spectra.

    Args:
        tmp_path: Directory to build it in.
        spectra: (id, name, wavelengths, values) tuples.

    Returns:
        The library directory.
    """
    pq = pytest.importorskip("pyarrow.parquet")
    pa = pytest.importorskip("pyarrow")
    out = tmp_path / "lib" / "spectra"
    out.mkdir(parents=True)

    grids, grid_of = [], {}
    rows = {k: [] for k in
            ("id", "name", "source.library", "material.category",
             "material.subcategory", "material.formula", "spectral_data.type",
             "spectral_data.wavelength_grid_id", "spectral_data.values")}
    for sid, name, wl, values in spectra:
        key = tuple(wl)
        if key not in grid_of:
            grid_of[key] = len(grids)
            grids.append({"grid_id": len(grids), "wavelengths": list(wl),
                          "wavelength_unit": "nm", "n_points": len(wl)})
        rows["id"].append(sid)
        rows["name"].append(name)
        rows["source.library"].append("usgs_splib07")
        rows["material.category"].append("mineral")
        rows["material.subcategory"].append(None)
        rows["material.formula"].append(None)
        rows["spectral_data.type"].append("reflectance")
        rows["spectral_data.wavelength_grid_id"].append(grid_of[key])
        rows["spectral_data.values"].append(list(values))
    pq.write_table(pa.table(rows), out / "usgs_splib07.parquet")
    pq.write_table(pa.table({
        "grid_id": [g["grid_id"] for g in grids],
        "wavelengths": [g["wavelengths"] for g in grids],
        "wavelength_unit": [g["wavelength_unit"] for g in grids],
        "n_points": [g["n_points"] for g in grids],
    }), out / "wavelengths.parquet")
    return out.parent


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Build and derive a small index once for the search tests."""
    pytest.importorskip("openspeclib")
    tmp = tmp_path_factory.mktemp("index")
    wl = list(np.arange(400.0, 901.0, 5.0))
    flat = [0.4] * len(wl)
    sloped = [0.2 + 0.0004 * (w - 400.0) for w in wl]
    dipped = [0.4 - 0.25 * np.exp(-((w - 700.0) / 30.0) ** 2) for w in wl]
    library = _library(tmp, [
        ("lib:flat", "Flat", wl, flat),
        ("lib:sloped", "Sloped", wl, sloped),
        ("lib:dipped", "Dipped at 700", wl, dipped),
    ])
    index = tmp / "index"
    info = indexing.build_index(
        library=str(library), out=str(index), step=STEP, lo=GRID_LO,
        hi=GRID_HI, min_coverage=0.3, batch=2, log=lambda *a: None)
    derive.derive_matrices(str(index), ["sam", "cr_sam", "gd1"],
                           log=lambda *a: None)
    return index, info, wl, dipped


class TestBuild:
    def test_every_spectrum_is_indexed(self, built):
        _index, info, _wl, _dipped = built
        assert info["n_spectra"] == 3
        assert info["skipped"] == 0
        assert info["n_bands"] == int((GRID_HI - GRID_LO) / STEP) + 1

    def test_rows_are_unit_norm_over_their_support(self, built):
        index, info, _wl, _dipped = built
        mat = np.memmap(index / "library.f32", dtype=np.float32, mode="r",
                        shape=(info["n_spectra"], info["n_bands"]))
        for row in mat:
            good = row[~np.isnan(row)]
            assert np.linalg.norm(good) == pytest.approx(1.0, abs=1e-5)

    def test_progress_reaches_the_total(self, tmp_path):
        pytest.importorskip("openspeclib")
        wl = list(np.arange(400.0, 901.0, 5.0))
        library = _library(tmp_path, [
            (f"lib:{i}", f"S{i}", wl, [0.3 + 0.01 * i] * len(wl))
            for i in range(5)])
        seen = []
        indexing.build_index(
            library=str(library), out=str(tmp_path / "idx"), step=STEP,
            lo=GRID_LO, hi=GRID_HI, batch=2,
            progress=lambda done, total: seen.append((done, total)),
            log=lambda *a: None)
        assert seen[-1] == (5, 5)

    def test_a_directory_that_is_not_a_library_is_refused(self, tmp_path):
        pytest.importorskip("openspeclib")
        with pytest.raises(FileNotFoundError):
            indexing.build_index(library=str(tmp_path),
                                 out=str(tmp_path / "idx"), log=lambda *a: None)


class TestDerive:
    def test_each_family_gets_its_own_matrices(self, built):
        index, _info, _wl, _dipped = built
        info = json.loads((index / "index.json").read_text())
        by_metric = info["matrices_by_metric"]
        assert set(by_metric) == set(transforms.ALL)
        # sam and cosine share one set; the transformed metrics do not.
        assert by_metric["sam"] == by_metric["cosine"]
        assert by_metric["cr_sam"] != by_metric["sam"]
        assert by_metric["gd1"] not in (by_metric["sam"], by_metric["cr_sam"])

    def test_unknown_metric_is_refused(self, built):
        index, _info, _wl, _dipped = built
        with pytest.raises(ValueError, match="unknown metric"):
            derive.derive_matrices(str(index), ["nonsense"],
                                   log=lambda *a: None)


class TestSearch:
    def test_a_library_spectrum_matches_itself_first(self, built):
        index, _info, wl, dipped = built
        lib = service.Library(str(index))
        hits, _gaps, _need = lib.search(wl, dipped, k=3)
        assert hits[0]["id"] == "lib:dipped"
        assert hits[0]["angle_rad"] == pytest.approx(0.0, abs=1e-3)

    def test_a_filter_restricts_the_candidates(self, built):
        index, _info, wl, dipped = built
        lib = service.Library(str(index))
        hits, _gaps, _need = lib.search(wl, dipped, k=3, sources=["ecosis"])
        assert hits == []

    def test_a_metric_without_matrices_is_refused_not_substituted(
        self, tmp_path, built
    ):
        """Scoring a transformed query against reflectance is worse than
        an error, so the index must refuse it."""
        index, _info, wl, dipped = built
        info = json.loads((index / "index.json").read_text())
        stripped = tmp_path / "stripped"
        stripped.mkdir()
        (stripped / "metadata.json").write_bytes(
            (index / "metadata.json").read_bytes())
        info["matrices_by_metric"] = {"sam": info["matrices_by_metric"]["sam"]}
        (stripped / "index.json").write_text(json.dumps(info))
        for name in info["matrices_by_metric"]["sam"] + ["library.f32"]:
            (stripped / name).write_bytes((index / name).read_bytes())
        lib = service.Library(str(stripped))
        with pytest.raises(ValueError, match="no matrices for metric"):
            lib.search(wl, dipped, k=1, metric="cr_sam")

    def test_a_thin_overlap_is_rejected_rather_than_scored(self, built):
        index, _info, _wl, _dipped = built
        lib = service.Library(str(index))
        # Three bands at the very blue end: far too little to measure an
        # angle over, whatever the cosine works out to.
        hits, _gaps, need = lib.search([400.0, 405.0, 410.0], [0.4, 0.4, 0.4],
                                       k=3, min_overlap=0.5,
                                       min_overlap_bands=20)
        assert hits == []
        assert need >= 20


class TestSpectrumReader:
    def _write(self, tmp_path, text):
        path = tmp_path / "spectrum.csv"
        path.write_text(text)
        return str(path)

    def test_comma_separated_with_header(self, mod, tmp_path):
        path = self._write(tmp_path, "wavelength,reflectance\n400,0.1\n410,0.2\n")
        assert mod.read_spectrum(path) == ([400.0, 410.0], [0.1, 0.2])

    def test_whitespace_separated(self, mod, tmp_path):
        path = self._write(tmp_path, "400 0.1\n410 0.2\n420 0.3\n")
        wl, values = mod.read_spectrum(path)
        assert wl == [400.0, 410.0, 420.0]
        assert values == [0.1, 0.2, 0.3]

    def test_semicolon_and_comments(self, mod, tmp_path):
        path = self._write(tmp_path, "# exported\n400;0.1\n410;0.2\n")
        assert mod.read_spectrum(path) == ([400.0, 410.0], [0.1, 0.2])

    def test_a_file_with_no_pairs_is_fatal(self, mod, tmp_path):
        path = self._write(tmp_path, "nothing here\n")
        with pytest.raises(SystemExit):
            mod.read_spectrum(path)
