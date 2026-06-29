"""Stream mbo lazy arrays through suite2p without writing data_raw.bin / data.bin.

suite2p's ``pipeline()`` consumes ``f_reg`` / ``f_raw`` purely by duck typing —
its docstring allows ``numpy.ndarray or BinaryFile``. Registration reads
``f_raw`` and writes registered frames into ``f_reg``; detection and extraction
only ever *read* ``f_reg``. ``StreamingBinaryFile`` is a drop-in for that
surface, backed by a single (channel, plane) of a 5D TCZYX mbo lazy array:

- As ``f_raw`` (read-only source): ``__getitem__`` returns int16 frames
  ``arr[t, c, z]`` — byte-identical to what mbo's ``imwrite`` would put in
  ``data_raw.bin``.
- As ``f_reg`` with ``registered=True``: writes are discarded (registered
  frames are never materialized); reads reconstitute registered frames on the
  fly from the shifts suite2p saves to ``<save_path>/reg_outputs.npy``, by
  replaying suite2p's own ``bidiphase.shift`` + ``register.shift_frames``
  (rigid ``torch.roll`` + nonrigid ``transform_data``). Only the tiny
  ``reg_outputs.npy`` ever touches disk.

Reconstituted frames are identical to ``data.bin`` because the raw source, the
stored shifts, the recomputed ``make_blocks(Ly, Lx, block_size)`` layout, and
the int16 cast are all the same ones registration used.

Single (functional) channel only; ``arr`` must support 5D ``[t, c, z, :, :]``
indexing (every mbo ``LazyArray`` does).
"""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path

import numpy as np


class StreamingBinaryFile:
    """suite2p ``BinaryFile``-compatible view of one plane of an mbo lazy array.

    Parameters
    ----------
    arr : mbo lazy array
        5D TCZYX source. Indexed as ``arr[t, channel_index, plane_index, :, :]``.
    plane_index : int
        0-based Z index to expose.
    channel_index : int, default 0
        0-based channel index (functional channel).
    n_frames : int, optional
        Number of timepoints to expose. Defaults to the source T (or the length
        of ``frame_indices`` when given).
    frame_indices : list[int], optional
        0-based source timepoints this stream maps onto (for truncation or
        strided ``timepoints=`` selections). Stream index ``i`` reads source
        timepoint ``frame_indices[i]``. None means identity.
    registered : bool, default False
        If True, reads reconstitute registered frames from ``reg_outputs.npy``.
        If False, reads return raw frames.
    save_path : str or Path, optional
        Plane directory holding ``reg_outputs.npy`` (required when registered).
    block_size : tuple[int, int], default (128, 128)
        Nonrigid block size — must match the value registration used so the
        recomputed block layout is identical.
    nonrigid : bool, default True
        Whether nonrigid shifts were computed. When False, only rigid shifts
        are applied even if ``reg_outputs.npy`` carries nonrigid arrays.
    device : torch.device, optional
        Device for the reconstitution (rigid roll + nonrigid grid_sample).
        Use the same device registration ran on so frames match bit-for-bit.
    """

    def __init__(
        self,
        arr,
        plane_index,
        *,
        channel_index=0,
        n_frames=None,
        frame_indices=None,
        registered=False,
        save_path=None,
        block_size=(128, 128),
        nonrigid=True,
        device=None,
    ):
        self._arr = arr
        self._z = int(plane_index)
        self._c = int(channel_index)

        if hasattr(arr, "_shape5d"):
            s5 = arr._shape5d()
        else:  # natural-rank fallback (T, Y, X)
            s5 = (arr.shape[0], 1, 1, arr.shape[-2], arr.shape[-1])
        self.Ly, self.Lx = int(s5[3]), int(s5[4])

        # 0-based source timepoints this stream exposes (identity if None).
        self._tmap = (
            np.asarray(frame_indices, dtype=np.int64)
            if frame_indices is not None
            else None
        )
        if self._tmap is not None:
            self.n_frames = int(len(self._tmap))
        elif n_frames is not None:
            self.n_frames = int(n_frames)
        else:
            self.n_frames = int(s5[0])

        self.dtype = np.int16
        self.write = False

        self._registered = bool(registered)
        self._save_path = Path(save_path) if save_path is not None else None
        self._block_size = (int(block_size[0]), int(block_size[1]))
        self._nonrigid = bool(nonrigid)
        self._device = device
        self._shifts = None  # lazily loaded reg_outputs (yoff/xoff/.../blocks)

    @classmethod
    def from_run(
        cls,
        arr,
        plane_dir,
        *,
        plane_index=0,
        channel_index=0,
        n_frames=None,
        frame_indices=None,
        device=None,
        validate=True,
    ):
        """Registered view of ``arr`` reconstructed from a completed run's shifts.

        Reads ``plane_dir/reg_outputs.npy`` (the shifts) and ``plane_dir/ops.npy``
        (the ``block_size`` / ``nonrigid`` / ``torch_device`` / source frame map
        the run used) and returns a registered ``StreamingBinaryFile`` — indexing
        it yields the registered frames on the fly, applying the saved shifts to
        ``arr``. Works for any run type (streaming or binary); only the saved
        shifts matter.

        Parameters
        ----------
        arr : mbo lazy array
            The raw 5D TCZYX source the shifts were computed from.
        plane_dir : str or Path
            Plane output directory containing reg_outputs.npy and ops.npy.
        plane_index, channel_index : int
            Which plane/channel of ``arr`` the shifts correspond to.
        n_frames, frame_indices : optional
            Frame selection. When both are None, ``frame_indices`` is restored
            from ``ops['stream_frame_indices']`` (so a strided/offset run maps
            shift ``i`` back onto source frame ``frame_indices[i]``), and
            ``n_frames`` defaults to the shift count.
        device : torch.device or str, optional
            Reconstitution device. None honors the run's recorded
            ``ops['torch_device']`` so frames match the data.bin it would have
            written (cross-device nonrigid interpolation differs by ~1 px).
        validate : bool or {"dims"}, default True
            True checks raw dimensions AND a content fingerprint, raising if the
            source no longer matches what was registered. "dims" checks
            dimensions only; False skips all checks.

        Examples
        --------
        >>> import mbo_utilities as mbo
        >>> from lbm_suite2p_python import StreamingBinaryFile
        >>> raw = mbo.imread("D:/data/raw")               # 5D TCZYX
        >>> reg = StreamingBinaryFile.from_run(raw, "D:/out/zplane01_tp00001-01000")
        >>> frames = reg[:1000]    # registered int16 frames, reconstituted lazily
        """
        plane_dir = Path(plane_dir)
        ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
        block_size = _resolve_block_size(ops)
        # restore the run's source frame map so a strided/offset selection applies
        # shift i to source frame frame_indices[i] (not source frame i).
        if frame_indices is None:
            fi = ops.get("stream_frame_indices")
            if fi is not None:
                frame_indices = np.asarray(fi, dtype=np.int64)
        # one shift entry per registered frame; default the exposed length to that
        # so a truncated run lines up without the caller remembering the count.
        if n_frames is None and frame_indices is None:
            reg_outputs = np.load(plane_dir / "reg_outputs.npy", allow_pickle=True).item()
            n_frames = int(len(np.asarray(reg_outputs["yoff"])))
        device = _resolve_replay_device(ops.get("torch_device"), device)
        if validate:
            _validate_raw_against_ops(
                arr,
                ops,
                n_frames=n_frames,
                plane_index=plane_index,
                channel_index=channel_index,
                frame_indices=frame_indices,
                mode=validate,
            )
        return cls(
            arr,
            plane_index,
            channel_index=channel_index,
            n_frames=n_frames,
            frame_indices=frame_indices,
            registered=True,
            save_path=plane_dir,
            block_size=block_size,
            nonrigid=bool(ops.get("nonrigid", True)),
            device=device,
        )

    # ---- BinaryFile surface ------------------------------------------------
    @property
    def shape(self):
        return (self.n_frames, self.Ly, self.Lx)

    @property
    def size(self):
        return int(self.n_frames) * int(self.Ly) * int(self.Lx)

    def __len__(self):
        return self.n_frames

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        pass

    def __setitem__(self, key, value):
        # Registered frames are virtual (reconstituted on read) and the raw
        # source is read-only — registration's writes have nothing to persist.
        return

    def __getitem__(self, key):
        idx, squeeze = self._time_indices(key)
        raw = np.asarray(self._read_raw(idx)) if idx.size else np.empty(
            (0, self.Ly, self.Lx), dtype=np.int16
        )
        # guard empty reads (empty slice, or a window function indexing past the
        # end): never hand 0 frames to shift_frames (torch.stack would raise).
        if self._registered and len(raw):
            out = self._apply_shifts(raw, idx)
        else:
            out = raw.astype(np.int16, copy=False)
        return out[0] if (squeeze and len(out)) else out

    @property
    def data(self):
        return self[:]

    def sampled_mean(self):
        """Mean image from up to 1000 evenly spaced frames (mirrors BinaryFile)."""
        nsamps = min(self.n_frames, 1000)
        inds = np.linspace(0, self.n_frames, 1 + nsamps).astype(np.int64)[:-1]
        return self[inds].astype(np.float32).mean(axis=0)

    # ---- internals ---------------------------------------------------------
    def _time_indices(self, key):
        """Resolve an axis-0 key into (stream_indices, squeeze_scalar)."""
        if isinstance(key, slice):
            return np.arange(*key.indices(self.n_frames), dtype=np.int64), False
        if isinstance(key, (int, np.integer)):
            i = int(key)
            if i < 0:
                i += self.n_frames
            return np.array([i], dtype=np.int64), True
        return np.asarray(key, dtype=np.int64), False

    def _read_raw(self, idx):
        """Read source frames for stream indices ``idx`` -> (N, Ly, Lx) int16."""
        src = self._tmap[idx] if self._tmap is not None else idx
        # contiguous run -> single slice read (fast path); else gather frame-wise
        # (robust to backends that don't support fancy time indexing).
        if np.array_equal(src, np.arange(src[0], src[0] + len(src))):
            sl = slice(int(src[0]), int(src[0]) + len(src))
            out = np.asarray(self._arr[sl, self._c, self._z, :, :])
        else:
            out = np.stack(
                [np.asarray(self._arr[int(i), self._c, self._z, :, :]) for i in src]
            )
        return np.ascontiguousarray(out)

    def _load_shifts(self):
        if self._shifts is not None:
            return self._shifts
        from suite2p.registration import nonrigid as s2p_nonrigid

        reg = np.load(self._save_path / "reg_outputs.npy", allow_pickle=True).item()
        yoff1 = reg.get("yoff1") if self._nonrigid else None
        has_nr = yoff1 is not None
        self._shifts = {
            "yoff": np.asarray(reg["yoff"]),
            "xoff": np.asarray(reg["xoff"]),
            "yoff1": np.asarray(reg["yoff1"]) if has_nr else None,
            "xoff1": np.asarray(reg["xoff1"]) if has_nr else None,
            "bidiphase": int(reg.get("bidiphase", 0) or 0),
            "blocks": (
                s2p_nonrigid.make_blocks(self.Ly, self.Lx, self._block_size)
                if has_nr
                else None
            ),
        }
        return self._shifts

    def _apply_shifts(self, raw, idx):
        """Reconstitute registered frames by replaying suite2p's shift application."""
        import torch
        from suite2p.registration import bidiphase as bidi, register

        sh = self._load_shifts()
        fr = torch.from_numpy(np.ascontiguousarray(raw))
        if self._device is not None:
            fr = fr.to(self._device)
        if sh["bidiphase"] != 0:
            fr = bidi.shift(fr, sh["bidiphase"])

        yoff = sh["yoff"][idx].astype(int)
        xoff = sh["xoff"][idx].astype(int)
        if sh["yoff1"] is not None:
            yoff1, xoff1 = sh["yoff1"][idx], sh["xoff1"][idx]
        else:
            yoff1 = xoff1 = None

        dev = self._device if self._device is not None else torch.device("cpu")
        out = np.asarray(
            register.shift_frames(fr, yoff, xoff, yoff1, xoff1, sh["blocks"], device=dev)
        )
        # suite2p's nonrigid transform_data squeezes the batch axis for a single
        # frame; keep the (N, Ly, Lx) contract so __getitem__'s squeeze is correct.
        if out.ndim == 2:
            out = out[np.newaxis]
        return out


def _shape5d_of(arr):
    """(T, C, Z, Ly, Lx) for an mbo lazy array or a TYX/TCZYX ndarray."""
    if hasattr(arr, "_shape5d"):
        return tuple(int(v) for v in arr._shape5d())
    s = arr.shape
    if len(s) == 5:
        return tuple(int(v) for v in s)
    return (int(s[0]), 1, 1, int(s[-2]), int(s[-1]))


def _resolve_block_size(ops):
    """Nonrigid block size the run used (ops override, else upstream default)."""
    bs = ops.get("block_size")
    if not bs:
        try:
            from suite2p import default_settings

            bs = default_settings()["registration"]["block_size"]
        except Exception:
            bs = (128, 128)
    return (int(bs[0]), int(bs[1]))


def _resolve_replay_device(recorded, device=None):
    """Device for reconstitution: explicit > the run's recorded device > auto.

    Reconstituting on the device registration ran on keeps frames bit-identical
    to the data.bin it would have written; cross-device nonrigid ``grid_sample``
    can flip int16 block-edge pixels by ~1. Falls back to cpu (with a warning)
    when a recorded cuda device is unavailable.
    """
    try:
        import torch
    except Exception:
        return None
    if device is not None:
        return torch.device(device) if isinstance(device, str) else device
    if recorded:
        try:
            dev = torch.device(recorded)
        except Exception:
            dev = None
        if dev is not None:
            if dev.type == "cuda" and not torch.cuda.is_available():
                warnings.warn(
                    f"run registered on {recorded!r} but cuda is unavailable; "
                    "reconstituting on cpu (may differ by ~1 px at nonrigid block "
                    "edges from the original data.bin).",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return torch.device("cpu")
            try:
                if dev.type == "cuda":
                    torch.zeros(1, device=dev)
                return dev
            except Exception:
                pass
    try:
        if torch.cuda.is_available():
            torch.zeros(1, device="cuda")
            return torch.device("cuda")
    except Exception:
        pass
    return torch.device("cpu")


def raw_fingerprint(
    arr,
    plane_index,
    channel_index=0,
    *,
    frame_indices=None,
    n_samples=3,
    sample_indices=None,
):
    """Content fingerprint of the raw frames a plane was registered from.

    Hashes a few evenly-spaced source frames of ``(channel_index, plane_index)``
    so a later replay can prove the raw source still produces the same pixels the
    shifts were computed on — a dimension check alone misses a re-assembled or
    re-pointed source of the same shape. ``sample_indices`` (stream indices)
    pins the exact frames to re-hash on validation. Returns an npy-safe dict.
    """
    f = StreamingBinaryFile(
        arr, plane_index, channel_index=channel_index, frame_indices=frame_indices
    )
    n = f.n_frames
    if sample_indices is None:
        sample_indices = np.unique(
            np.linspace(0, max(n - 1, 0), min(int(n_samples), n)).astype(np.int64)
        )
    else:
        sample_indices = np.asarray(sample_indices, dtype=np.int64)
    hashes = [
        hashlib.blake2b(
            np.ascontiguousarray(np.asarray(f[int(i)], dtype=np.int16)).tobytes(),
            digest_size=16,
        ).hexdigest()
        for i in sample_indices
    ]
    return {
        "sample_indices": [int(i) for i in sample_indices],
        "hashes": hashes,
        "n_frames": int(n),
        "Ly": int(f.Ly),
        "Lx": int(f.Lx),
    }


def _validate_raw_against_ops(
    arr, ops, *, n_frames, plane_index, channel_index, frame_indices, mode=True
):
    """Raise ValueError unless ``arr`` is the source the shifts were fit on.

    ``mode`` True checks dimensions then the stored ``raw_fingerprint``; "dims"
    stops after dimensions (apply shifts to a different same-shape array); False
    is a no-op. A missing fingerprint warns rather than raises, so legacy plane
    dirs (recorded before fingerprinting) still open.
    """
    if not mode:
        return
    T, _C, Z, Ly, Lx = _shape5d_of(arr)
    if "Ly" in ops and (Ly, Lx) != (int(ops["Ly"]), int(ops["Lx"])):
        raise ValueError(
            f"raw is {Ly}x{Lx} but the run registered "
            f"{int(ops['Ly'])}x{int(ops['Lx'])} — dimensions do not fit."
        )
    need = (
        int(np.max(frame_indices)) + 1
        if frame_indices is not None
        else int(n_frames or 0)
    )
    if T < need:
        raise ValueError(
            f"raw has {T} frames but replay needs {need}; re-point to the full "
            "source or open the plane with its data.bin."
        )
    if Z <= int(plane_index):
        raise ValueError(
            f"raw has {Z} plane(s) but plane_index={int(plane_index)} was requested."
        )
    if mode == "dims":
        return
    fp = ops.get("raw_fingerprint")
    if not fp:
        warnings.warn(
            "no raw_fingerprint recorded for this run; cannot verify the raw "
            "source is the one that was registered. Pass validate='dims' to "
            "silence, or re-run to record a fingerprint.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    actual = raw_fingerprint(
        arr,
        plane_index,
        channel_index,
        frame_indices=frame_indices,
        sample_indices=fp.get("sample_indices"),
    )
    if actual["hashes"] != list(fp.get("hashes", [])):
        raise ValueError(
            "raw source content does not match what was registered (the source "
            "was re-assembled, re-pointed, or read with different options). "
            "Replaying shifts here would produce wrong frames. Re-point to the "
            "exact source, or pass validate='dims'/False to apply anyway."
        )


def _open_raw_source(ops, n_needed):
    """imread the run's raw source as the FULL assembled volume, or None.

    ``data_path`` records only the first raw file; imread of that single file can
    return fewer/un-assembled frames than the run processed. Tries, in order: the
    recorded ``raw_source``, the containing directory of ``data_path`` (which
    reconstructs the assembled series), then ``data_path`` itself — preferring the
    first whose timepoint count covers ``n_needed``. Reads with the run's recorded
    ``reader_kwargs`` so frames match what the shifts were computed on.
    """
    from mbo_utilities import imread

    cands = []
    rs = ops.get("raw_source")
    if rs:
        cands.append(Path(rs))
    dp = ops.get("data_path")
    if dp:
        p = Path(dp)
        if p.suffix and p.parent != p:
            cands.append(p.parent)
        cands.append(p)

    reader_kwargs = ops.get("reader_kwargs") or {}
    seen, fallback = set(), None
    for c in cands:
        s = str(c)
        if s in seen or not c.exists():
            continue
        seen.add(s)
        try:
            arr = imread(c, **reader_kwargs)
        except Exception:
            continue
        try:
            nt = _shape5d_of(arr)[0]
        except Exception:
            nt = None
        if nt is None or nt >= n_needed:
            return arr
        fallback = fallback or arr
    return fallback


def apply_shifts(
    raw, plane_dir, *, plane_index=None, channel_index=0, device=None, validate=True
):
    """Apply a completed run's registration shifts to a raw movie, lazily.

    Replays the shifts saved in ``plane_dir`` (reg_outputs.npy + ops.npy) against
    ``raw`` and returns a registered, BinaryFile-like view whose int16 frames are
    reconstituted on read — the data.bin suite2p would have written, without
    materializing it. Run-type and array-type agnostic: it works for streaming or
    binary runs and for any source whose dimensions match the run.

    Parameters
    ----------
    raw : mbo lazy array, ndarray, or path-like
        The raw source. A path is imread with the run's recorded ``reader_kwargs``
        and assembled to the full volume. A 3D ``(T, Y, X)`` ndarray is treated as
        a single plane/channel; otherwise the source must expose 5D
        ``[t, c, z, :, :]`` indexing.
    plane_dir : str or Path
        Plane output directory with reg_outputs.npy and ops.npy.
    plane_index : int, optional
        0-based Z index of ``raw`` the shifts belong to. None uses the run's
        recorded plane (``ops['plane'] - 1``) when ``raw`` is volumetric, else 0.
    channel_index : int, default 0
        Functional channel.
    device : torch.device or str, optional
        Reconstitution device. None honors the run's recorded ``torch_device``.
    validate : bool or {"dims"}, default True
        Raw-source safety. True checks dimensions and a content fingerprint
        (refusing a drifted/wrong source); "dims" checks dimensions only (apply
        shifts to a different same-shape array); False skips all checks.

    Returns
    -------
    StreamingBinaryFile
        Registered view; index like an array (``reg[t]``, ``reg[a:b]``) for int16
        frames, or ``reg[:]`` for the whole registered movie.
    """
    plane_dir = Path(plane_dir)
    if isinstance(raw, (str, Path)):
        ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
        reg = np.load(plane_dir / "reg_outputs.npy", allow_pickle=True).item()
        n_needed = int(len(np.asarray(reg["yoff"])))
        opened = _open_raw_source({**ops, "raw_source": str(raw)}, n_needed)
        if opened is None:
            raise FileNotFoundError(f"could not read raw source {raw!r}")
        raw = opened
    elif isinstance(raw, np.ndarray) and raw.ndim == 3:
        raw = raw[:, np.newaxis, np.newaxis]
    if plane_index is None:
        ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
        nplanes = _shape5d_of(raw)[2]
        plane_index = (int(ops.get("plane", 1) or 1) - 1) if nplanes > 1 else 0
    return StreamingBinaryFile.from_run(
        raw,
        plane_dir,
        plane_index=plane_index,
        channel_index=channel_index,
        device=device,
        validate=validate,
    )
