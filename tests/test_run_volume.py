# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy", "tifffile", "tqdm", "mbo_utilities", "lbm_suite2p_python"]
# ///
import pytest
import os
from pathlib import Path
import logging
import mbo_utilities.log
import mbo_utilities as mbo
import lbm_suite2p_python as lsp

HERE = os.path.dirname(__file__)
TEMP_DIR = os.path.join(HERE, "_tmp")
PRIVATE_DIR = os.path.join(HERE, "data", "private")
PUBLIC_DIR = os.path.join(HERE, "data", "public")

URL = "http://localhost:8386/"  # TEMP_DIR


def skip(key: str, default: bool) -> bool:
    """Return if environment variable is set and true."""
    return os.getenv(key, default) in {True, 1, "1"}


@pytest.fixture
def fake_data(request, tmp_path):
    from mbo_utilities import read_scan, save_as

    raw_path = Path(request.config.getoption("--data-path"))
    assert raw_path.exists(), f"Data path does not exist: {raw_path}"

    scan = read_scan(raw_path)
    assembled_dir = tmp_path / "assembled"
    save_as(scan, assembled_dir, ext="tif")

    assembled_files = sorted(assembled_dir.rglob("*.tif"))
    return assembled_files, tmp_path


def pytest_addoption(parser):
    parser.addoption(
        "--data-path", action="store", default=None, help="Path to test TIFF folder"
    )


@pytest.fixture
def fake_data(tiff_paths, tmp_path):
    paths, tempdir = tiff_paths, Path(tmp_path)
    return paths, tempdir


def test_run_plane(fake_data):
    files, tempdir = fake_data
    file = files[0]
    metadata = mbo.get_metadata(file)
    ops = mbo.params_from_metadata(metadata)
    ops["nplanes"] = 1  # for safety
    result = lsp.run_plane(
        file,
        save_path=tempdir,
        ops=ops,
        keep_raw=True,
        keep_reg=True,
        force_reg=True,
        force_detect=True,
    )
    assert isinstance(result, dict)
    assert (Path(result["save_path"]) / "ops.npy").exists()


def test_run_volume(fake_data):
    files, tempdir = fake_data
    metadata = mbo.get_metadata(files[0])
    ops = mbo.params_from_metadata(metadata)
    ops["nplanes"] = len(files)
    results = lsp.run_volume(files, save_path=tempdir, ops=ops)
    assert isinstance(results, list)
    assert all(Path(p).exists() for p in results)
