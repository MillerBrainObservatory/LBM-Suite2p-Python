# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pytest", "tifffile", "mbo_utilities", "lbm_suite2p_python"]
# ///
"""mbo studio integration: RegisteredStreamArray opens a binary-less suite2p
plane dir (ops.npy + reg_outputs.npy, no .bin) as a lazy registered array.

Layered like test_streaming.py: can_open / dispatch routing need no suite2p; the
end-to-end frame check (register synthetic data, open via mbo.imread, verify the
registered frames) is gated on suite2p.
"""

from pathlib import Path

import numpy as np
import pytest

from lbm_suite2p_python.streaming import StreamingBinaryFile
from lbm_suite2p_python.streaming_array import RegisteredStreamArray

try:
    import suite2p  # noqa: F401

    _HAS_SUITE2P = True
except Exception:
    _HAS_SUITE2P = False


def _make_plane_dir(tmp_path, *, with_binary=False, with_reg=True, data_path=None):
    """Fabricate a suite2p-shaped plane dir with controllable file presence."""
    d = tmp_path / "plane"
    d.mkdir(parents=True, exist_ok=True)
    ops = {"Ly": 8, "Lx": 6, "nframes": 5, "plane": 1}
    if data_path is not None:
        ops["data_path"] = str(data_path)
    np.save(d / "ops.npy", ops)
    if with_reg:
        np.save(d / "reg_outputs.npy", {
            "yoff": np.zeros(5, np.int64), "xoff": np.zeros(5, np.int64),
            "yoff1": None, "xoff1": None, "bidiphase": 0,
        })
    if with_binary:
        np.zeros((5, 8, 6), np.int16).tofile(d / "data.bin")
    return d


def test_can_open_matches_only_binaryless_reg_dir_with_raw(tmp_path):
    raw = tmp_path / "raw.tif"
    raw.write_bytes(b"x")  # just needs to exist for _resolve_raw_path

    ok = _make_plane_dir(tmp_path / "a", with_binary=False, with_reg=True, data_path=raw)
    assert RegisteredStreamArray.can_open(ok) is True
    # also accepts the ops.npy path itself
    assert RegisteredStreamArray.can_open(ok / "ops.npy") is True


def test_can_open_rejects_binary_present(tmp_path):
    raw = tmp_path / "raw.tif"; raw.write_bytes(b"x")
    d = _make_plane_dir(tmp_path / "b", with_binary=True, with_reg=True, data_path=raw)
    # a real binary is present -> defer to Suite2pArray
    assert RegisteredStreamArray.can_open(d) is False


def test_can_open_rejects_missing_pieces(tmp_path):
    raw = tmp_path / "raw.tif"; raw.write_bytes(b"x")
    no_reg = _make_plane_dir(tmp_path / "c", with_binary=False, with_reg=False, data_path=raw)
    assert RegisteredStreamArray.can_open(no_reg) is False          # no reg_outputs.npy
    no_raw = _make_plane_dir(tmp_path / "d", with_binary=False, with_reg=True, data_path=None)
    assert RegisteredStreamArray.can_open(no_raw) is False          # no data_path
    missing_raw = _make_plane_dir(
        tmp_path / "e", with_binary=False, with_reg=True, data_path=tmp_path / "gone.tif"
    )
    assert RegisteredStreamArray.can_open(missing_raw) is False     # data_path not on disk
    assert RegisteredStreamArray.can_open(tmp_path / "raw.tif") is False  # a plain file
    assert RegisteredStreamArray.can_open(tmp_path / "does_not_exist") is False


def test_dispatch_routes_streaming_dir_here_binary_dir_to_suite2p(tmp_path):
    from mbo_utilities.lazy_array import _dispatch

    raw = tmp_path / "raw.tif"; raw.write_bytes(b"x")
    streaming = _make_plane_dir(tmp_path / "s", with_binary=False, with_reg=True, data_path=raw)
    binary = _make_plane_dir(tmp_path / "bin", with_binary=True, with_reg=True, data_path=raw)

    assert _dispatch(streaming).__name__ == "RegisteredStreamArray"
    # no regression: a real binary suite2p dir still routes to Suite2pArray
    assert _dispatch(binary).__name__ == "Suite2pArray"


def _synthetic_movie(N=300, Ly=96, Lx=96, seed=1):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:Ly, 0:Lx]
    base = np.zeros((Ly, Lx), np.float32)
    for cy, cx, amp in [(28, 30, 1800), (60, 58, 2400), (45, 72, 2000)]:
        base += amp * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 6.0 ** 2)))
    base += 300
    sh = rng.integers(-2, 3, size=(N, 2))
    mov = np.empty((N, Ly, Lx), np.int16)
    for i, (dy, dx) in enumerate(sh):
        mov[i] = np.roll(base, (dy, dx), axis=(0, 1)).astype(np.int16)
    return mov


@pytest.mark.skipif(not _HAS_SUITE2P, reason="suite2p (and cv2) required")
def test_imread_opens_registered_view_and_frames_match(tmp_path):
    """Register synthetic data via streaming, then mbo.imread the plane dir and
    confirm it returns a registered view whose frames equal the reconstitution."""
    import tifffile
    import torch
    import mbo_utilities as mbo
    from lbm_suite2p_python.default_ops import default_ops
    from lbm_suite2p_python.run_lsp import run_plane_stream

    mov = _synthetic_movie()
    N, Ly, Lx = mov.shape
    # keep raw and outputs in separate trees (as a real run does)
    raw_dir = tmp_path / "raw"; raw_dir.mkdir()
    raw_tif = raw_dir / "raw.tif"
    tifffile.imwrite(raw_tif, mov)
    arr = mbo.imread(raw_tif)  # 5D TiffArray (N,1,1,Ly,Lx)

    plane_dir = tmp_path / "out" / "zplane01"
    plane_dir.mkdir(parents=True)
    ops = default_ops()
    ops.update({
        "Ly": Ly, "Lx": Lx, "nframes": N, "nframes_chan1": N,
        "save_path": str(plane_dir), "ops_path": str(plane_dir / "ops.npy"),
        "do_registration": 1, "roidetect": 0, "fs": 10.0, "tau": 1.0,
        "block_size": [32, 32], "torch_device": "cpu", "plane": 1,
        "data_path": str(raw_tif),
    })
    run_plane_stream(ops, arr, 0)

    # imread dispatches the plane dir to the registered view
    assert type(mbo.imread(plane_dir)).__name__ == "RegisteredStreamArray"

    # pin the device for the frame comparison (imread auto-picks cuda when
    # available, and cross-device nonrigid interpolation differs by ~1 px).
    from lbm_suite2p_python.streaming_array import RegisteredStreamArray
    rsa = RegisteredStreamArray(plane_dir, device=torch.device("cpu"))
    assert rsa._shape5d() == (N, 1, 1, Ly, Lx)
    assert rsa.dtype == np.int16
    assert rsa.metadata.get("fs") == 10.0
    assert rsa.nt == N and rsa.nz == 1

    # frames equal the direct reconstitution, and differ from the raw movie
    ref = StreamingBinaryFile.from_run(arr, plane_dir, plane_index=0, device=torch.device("cpu"))
    ref_full = ref[:]
    assert not np.array_equal(ref_full, mov)

    # __getitem__ follows numpy 5D semantics against a reference built from the
    # registered stack (the GUI squeezes the singleton C/Z for display).
    ref5 = ref_full[:, None, None, :, :]
    assert np.array_equal(rsa[10], ref5[10])                      # (1, 1, Ly, Lx)
    assert np.array_equal(rsa[2:8], ref5[2:8])
    assert np.array_equal(rsa[2:8, 0, 0, 10:20, 5:15], ref5[2:8, 0, 0, 10:20, 5:15])
    assert np.array_equal(rsa[7, 0, 0], ref5[7, 0, 0])
    assert rsa[7, 0, 0].shape == (Ly, Lx)

    # the z-stats tab indexes with a strided time slice (this was the GUI crash);
    # phasecorr-view-style (t_slice, c, z, :, :) must also work.
    assert np.array_equal(rsa[::25, 0, 0, :, :], ref5[::25, 0, 0, :, :])
    assert rsa[::25, 0, 0, :, :].shape[0] == len(range(0, N, 25))
