import pytest
from pathlib import Path
import mbo_utilities as mbo
import lbm_suite2p_python as lsp

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
    result = lsp.run_plane(file, save_path=tempdir, ops=ops, keep_raw=True, keep_reg=True, force_reg=True, force_detect=True)
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
