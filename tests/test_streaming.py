# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pytest", "lbm_suite2p_python"]
# ///
"""Tests for streaming mbo lazy arrays through suite2p (no data_raw.bin/data.bin).

Two layers:
- The raw-streaming surface (StreamingBinaryFile read path) is validated against
  a synthetic 5D array with no suite2p dependency — runs anywhere.
- The registered reconstitution is validated to be *byte-identical* to what
  suite2p's registration writes to data.bin. That needs suite2p installed and is
  skipped otherwise.
"""

import numpy as np
import pytest

from lbm_suite2p_python.streaming import StreamingBinaryFile

try:
    import suite2p  # noqa: F401

    _HAS_SUITE2P = True
except Exception:
    _HAS_SUITE2P = False


class _Arr5D:
    """Minimal stand-in for an mbo lazy array (5D TCZYX, numpy-backed)."""

    def __init__(self, data5d):
        self._d = np.asarray(data5d)

    def _shape5d(self):
        return self._d.shape

    @property
    def shape(self):
        return self._d.shape

    def __getitem__(self, key):
        return self._d[key]


def _make_arr(T=20, C=1, Z=3, Ly=8, Lx=6, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.integers(-100, 4000, size=(T, C, Z, Ly, Lx)).astype(np.int16)
    return _Arr5D(data), data


def test_raw_stream_shape_and_dtype():
    arr, data = _make_arr()
    f = StreamingBinaryFile(arr, plane_index=2, channel_index=0, n_frames=20)
    assert f.shape == (20, 8, 6)
    assert f.Ly == 8 and f.Lx == 6 and f.n_frames == 20
    assert f[:].dtype == np.int16


def test_raw_stream_slice_matches_source():
    arr, data = _make_arr()
    z, c = 1, 0
    f = StreamingBinaryFile(arr, plane_index=z, channel_index=c, n_frames=data.shape[0])
    assert np.array_equal(f[3:9], data[3:9, c, z])
    assert np.array_equal(f[:], data[:, c, z])


def test_raw_stream_fancy_and_scalar_index():
    arr, data = _make_arr()
    z, c = 2, 0
    f = StreamingBinaryFile(arr, plane_index=z, channel_index=c, n_frames=data.shape[0])
    # non-contiguous fancy index (registration samples a reference this way)
    idx = np.array([0, 5, 5, 11, 2])
    assert np.array_equal(f[idx], data[idx, c, z])
    # scalar index squeezes the time axis, like BinaryFile
    assert np.array_equal(f[7], data[7, c, z])
    assert f[7].shape == (8, 6)


def test_raw_stream_frame_indices_tmap():
    arr, data = _make_arr(T=20)
    z, c = 0, 0
    sel = [1, 3, 5, 7, 9]  # 0-based source timepoints (strided selection)
    f = StreamingBinaryFile(
        arr, plane_index=z, channel_index=c, frame_indices=sel
    )
    assert f.n_frames == len(sel)
    assert np.array_equal(f[:], data[sel, c, z])
    assert np.array_equal(f[1:3], data[[3, 5], c, z])


def test_setitem_is_discarded():
    arr, data = _make_arr()
    f = StreamingBinaryFile(arr, plane_index=0, channel_index=0, n_frames=data.shape[0])
    before = f[:].copy()
    f[0:3] = np.zeros((3, 8, 6), np.int16)  # registration "writes" — must no-op
    assert np.array_equal(f[:], before)


def test_sampled_mean_runs():
    arr, data = _make_arr()
    f = StreamingBinaryFile(arr, plane_index=0, channel_index=0, n_frames=data.shape[0])
    m = f.sampled_mean()
    assert m.shape == (8, 6) and m.dtype == np.float32


def test_registered_empty_and_oob_reads_dont_crash(tmp_path):
    """A window function / stats sampler can index an empty slice or past the
    end; those must return empty, never hand 0 frames to shift_frames (the GUI
    crash was torch.stack on an empty list)."""
    arr, data = _make_arr(T=10, Ly=8, Lx=6)
    N = data.shape[0]
    np.save(tmp_path / "reg_outputs.npy", {
        "yoff": np.zeros(N, np.int64), "xoff": np.zeros(N, np.int64),
        "yoff1": None, "xoff1": None, "bidiphase": 0,
    })
    f = StreamingBinaryFile(
        arr, plane_index=0, channel_index=0, n_frames=N, registered=True,
        save_path=tmp_path, nonrigid=False,  # rigid-only: no suite2p needed here
    )
    assert f[N:N + 3].shape == (0, 8, 6)   # slice entirely past the end
    assert f[5:5].shape == (0, 8, 6)       # empty slice
    assert f[N].shape == (0, 8, 6)         # out-of-range scalar (guarded, no crash)


def _synthetic_registered_movie(N=300, Ly=96, Lx=96, max_shift=3, seed=1):
    """A base image with blobs, translated by known integer shifts per frame.

    Frames must exceed the nonrigid block size or suite2p's own registration
    degenerates to a single block (real LBM planes are far larger); the test
    shrinks block_size so 96x96 still tiles into multiple blocks.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:Ly, 0:Lx]
    base = np.zeros((Ly, Lx), np.float32)
    for cy, cx, amp in [(0.3, 0.35, 1500), (0.6, 0.6, 2200), (0.45, 0.75, 1800)]:
        base += amp * np.exp(
            -(((yy - cy * Ly) ** 2 + (xx - cx * Lx) ** 2) / (2 * 6.0 ** 2))
        )
    base += 200
    shifts = rng.integers(-max_shift, max_shift + 1, size=(N, 2))
    frames = np.empty((N, Ly, Lx), np.int16)
    for i, (dy, dx) in enumerate(shifts):
        frames[i] = np.roll(base, (dy, dx), axis=(0, 1)).astype(np.int16)
    return frames


@pytest.mark.skipif(not _HAS_SUITE2P, reason="suite2p (and cv2) required")
def test_registered_reconstitution_is_byte_identical(tmp_path):
    """StreamingBinaryFile(registered=True) must reproduce, frame-for-frame, what
    suite2p's registration writes to data.bin — same raw source, same shifts,
    same block layout, same int16 cast."""
    import torch
    from suite2p import default_settings
    from suite2p.registration.register import registration_wrapper

    device = torch.device("cpu")  # deterministic; no GPU needed for the test
    settings = default_settings()
    reg = settings["registration"]
    reg["block_size"] = [32, 32]  # tile the small test frames into many blocks

    raw = _synthetic_registered_movie()
    N, Ly, Lx = raw.shape
    arr = _Arr5D(raw[:, None, None, :, :])  # (N, 1, 1, Ly, Lx)

    # f_raw is the raw stream (exactly what run_plane_stream feeds registration);
    # f_reg is an in-RAM ndarray that captures the registered frames suite2p
    # writes — the data.bin ground truth.
    f_raw_stream = StreamingBinaryFile(arr, plane_index=0, channel_index=0, n_frames=N)
    f_reg = np.zeros((N, Ly, Lx), np.int16)
    reg_outputs = registration_wrapper(
        f_reg, f_raw=f_raw_stream, save_path=str(tmp_path), settings=reg, device=device
    )
    # mimic pipeline_s2p saving the shifts right after registration.
    np.save(tmp_path / "reg_outputs.npy", reg_outputs)

    stream = StreamingBinaryFile(
        arr, plane_index=0, channel_index=0, n_frames=N, registered=True,
        save_path=tmp_path, block_size=reg["block_size"],
        nonrigid=reg["nonrigid"], device=device,
    )

    recon = stream[:]
    assert recon.shape == f_reg.shape
    assert recon.dtype == np.int16
    assert np.array_equal(recon, f_reg), (
        f"reconstituted frames differ from registration output: "
        f"{int((recon != f_reg).sum())} / {recon.size} pixels"
    )

    # batched reads must match the full read (detection/extraction read in chunks)
    assert np.array_equal(stream[100:175], f_reg[100:175])
    # single-frame read (GUI scrub) must keep the (Ly, Lx) frame shape
    assert stream[123].shape == (Ly, Lx)
    assert np.array_equal(stream[123], f_reg[123])


@pytest.mark.skipif(not _HAS_SUITE2P, reason="suite2p (and cv2) required")
def test_run_plane_stream_writes_no_binaries(tmp_path):
    """End-to-end: run_plane_stream runs registration + detection + extraction
    straight off the lazy array, writing the usual .npy outputs and reg_outputs
    but never data_raw.bin / data.bin."""
    from lbm_suite2p_python.default_ops import default_ops
    from lbm_suite2p_python.run_lsp import run_plane_stream

    rng = np.random.default_rng(0)
    N, Ly, Lx = 300, 96, 96
    yy, xx = np.mgrid[0:Ly, 0:Lx]
    base = np.zeros((Ly, Lx), np.float32)
    for cy, cx, amp in [(28, 30, 1800), (60, 58, 2400), (45, 72, 2000)]:
        base += amp * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 5.0 ** 2)))
    base += 300
    sh = rng.integers(-2, 3, size=(N, 2))
    mov = np.empty((N, Ly, Lx), np.int16)
    for i, (dy, dx) in enumerate(sh):
        mov[i] = (np.roll(base, (dy, dx), axis=(0, 1))
                  + rng.normal(0, 15, (Ly, Lx))).astype(np.int16)
    arr = _Arr5D(mov[:, None, None, :, :])

    ops = default_ops()
    ops.update({
        "Ly": Ly, "Lx": Lx, "nframes": N, "nframes_chan1": N,
        "save_path": str(tmp_path), "ops_path": str(tmp_path / "ops.npy"),
        "do_registration": 1, "roidetect": 1, "fs": 10.0, "tau": 1.0,
        "block_size": [32, 32], "torch_device": "cpu",
    })

    assert run_plane_stream(ops, arr, 0) is True
    for name in ("reg_outputs.npy", "ops.npy", "stat.npy", "F.npy", "iscell.npy"):
        assert (tmp_path / name).exists(), f"missing {name}"
    assert not (tmp_path / "data_raw.bin").exists()
    assert not (tmp_path / "data.bin").exists()

    # the saved shifts can be re-applied to the raw array via from_run, reading
    # block_size / nonrigid back from the run's ops.npy.
    import torch

    reg = StreamingBinaryFile.from_run(
        arr, tmp_path, plane_index=0, device=torch.device("cpu")
    )
    assert reg._registered and reg.n_frames == N
    out = reg[:]
    assert out.shape == (N, Ly, Lx) and out.dtype == np.int16
    # shifts were applied: registered frames differ from the raw source
    assert not np.array_equal(out, mov)
