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
    ):
        """Registered view of ``arr`` reconstructed from a completed run's shifts.

        Reads ``plane_dir/reg_outputs.npy`` (the shifts) and ``plane_dir/ops.npy``
        (the ``block_size`` / ``nonrigid`` the run used) and returns a
        registered ``StreamingBinaryFile`` — indexing it yields the registered
        frames on the fly, applying the saved shifts to ``arr``. Works whether
        or not the run used streaming; only the saved shifts matter.

        Parameters
        ----------
        arr : mbo lazy array
            The raw 5D TCZYX source the shifts were computed from.
        plane_dir : str or Path
            Plane output directory containing reg_outputs.npy and ops.npy.
        plane_index, channel_index : int
            Which plane/channel of ``arr`` the shifts correspond to.
        n_frames, frame_indices : optional
            Frame selection — must match the frames the run registered (the
            shift arrays have one entry per registered frame).
        device : torch.device, optional
            Device for the shift application.

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
        block_size = ops.get("block_size") or (128, 128)
        # the shift arrays have exactly one entry per registered frame; default
        # the exposed length to that so a truncated run (e.g. num_timepoints)
        # lines up without the caller having to remember the count. A strided
        # `timepoints=` run needs the same `frame_indices` passed explicitly.
        if n_frames is None and frame_indices is None:
            reg_outputs = np.load(plane_dir / "reg_outputs.npy", allow_pickle=True).item()
            n_frames = int(len(np.asarray(reg_outputs["yoff"])))
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
