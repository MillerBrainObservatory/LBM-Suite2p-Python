"""mbo studio integration: open a binary-less suite2p plane directory as a lazy
registered array (raw frames + saved shifts, reconstituted on the fly).

Registered with mbo_utilities via the ``mbo_utilities.lazy_arrays`` entry point
(see pyproject), so ``mbo <plane_dir>`` / ``imread(<plane_dir>)`` returns a
``RegisteredStreamArray`` when the directory has:

- ``ops.npy`` + ``reg_outputs.npy`` (the registration shifts), and
- **no** ``data.bin`` / ``data_raw.bin`` (a streamed run, or a binary run whose
  ``.bin`` files were cleaned up by the default ``keep_reg``/``keep_raw=False``),
  and
- a locatable raw source (``ops['data_path']``).

It carries ``PRIORITY`` above ``Suite2pArray`` (100) so it claims those dirs
first; when a real binary is present, ``can_open`` returns False and
``Suite2pArray`` reads it directly. The class is intentionally isolated from
``streaming.py`` (the pipeline-critical module): if the mbo base classes ever
move, only the entry-point load fails — mbo skips it and the pipeline is
unaffected.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mbo_utilities.lazy_array import LazyArray
from mbo_utilities.arrays._base import ReductionMixin, _normalize_key

from lbm_suite2p_python.streaming import StreamingBinaryFile


def _recorded_raw_exists(ops: dict) -> bool:
    """True if the exact raw source recorded by the run is on disk.

    Used by ``can_open`` — deliberately strict (exact path, no parent guessing)
    so a stale/moved source declines cleanly instead of matching an unrelated
    directory. ``raw_source`` (the canonical full source the run read) is
    preferred; ``data_path`` (the first raw file) is the legacy fallback.
    """
    for key in ("raw_source", "data_path"):
        v = ops.get(key)
        if v and Path(v).exists():
            return True
    return False


def _open_raw(ops: dict, n_needed: int):
    """imread the raw source as the FULL assembled volume.

    ``data_path`` records only the first raw file; ``imread`` of that single
    file can return fewer/un-assembled frames than the run processed. So try, in
    order: the recorded ``raw_source``, the containing directory of
    ``data_path`` (which reconstructs the assembled series), then ``data_path``
    itself — and prefer the first whose timepoint count covers the registered
    frames. Returns the opened array, or None when nothing is usable.
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
            cands.append(p.parent)  # series dir -> assembled volume
        cands.append(p)

    seen, fallback = set(), None
    for c in cands:
        s = str(c)
        if s in seen or not c.exists():
            continue
        seen.add(s)
        try:
            arr = imread(c, **(ops.get("reader_kwargs") or {}))
        except Exception:
            continue
        try:
            nt = int(arr._shape5d()[0])
        except Exception:
            nt = None
        if nt is None or nt >= n_needed:
            return arr
        fallback = fallback or arr  # short, but better than nothing
    return fallback


def _has_binary(plane_dir: Path) -> bool:
    return (plane_dir / "data.bin").exists() or (plane_dir / "data_raw.bin").exists()


def _pick_device():
    """cuda if usable, else cpu, else None (torch absent)."""
    try:
        import torch
    except Exception:
        return None
    try:
        if torch.cuda.is_available():
            torch.zeros(1, device="cuda")
            return torch.device("cuda")
    except Exception:
        pass
    return torch.device("cpu")


class RegisteredStreamArray(ReductionMixin, LazyArray):
    """Lazy registered view of one suite2p plane, rebuilt from raw + shifts.

    Presents a 5D ``(T, 1, 1, Ly, Lx)`` array whose frames are the registered
    frames suite2p would have written to ``data.bin`` — reconstituted on demand
    from the raw source and ``reg_outputs.npy`` via ``StreamingBinaryFile``.
    """

    PRIORITY = 150  # above Suite2pArray (100): claim binary-less reg dirs first

    @classmethod
    def can_open(cls, path) -> bool:
        try:
            p = Path(path)
            if p.name == "ops.npy":
                p = p.parent
            if not p.is_dir():
                return False
            if not ((p / "ops.npy").exists() and (p / "reg_outputs.npy").exists()):
                return False
            if _has_binary(p):
                return False  # real binary present -> Suite2pArray reads it directly
            ops = np.load(p / "ops.npy", allow_pickle=True).item()
            return _recorded_raw_exists(ops)
        except Exception:
            return False

    def __init__(self, path, device=None):
        from mbo_utilities import imread
        from lbm_suite2p_python.utils import _get_num_planes

        p = Path(path)
        if p.name == "ops.npy":
            p = p.parent
        self._plane_dir = p
        self._ops = np.load(p / "ops.npy", allow_pickle=True).item()

        reg_outputs = np.load(p / "reg_outputs.npy", allow_pickle=True).item()
        self._nt = int(len(np.asarray(reg_outputs["yoff"])))

        # re-read raw (with the run's reader_kwargs) as the full assembled volume.
        self._raw = _open_raw(self._ops, self._nt)
        if self._raw is None:
            raise FileNotFoundError(
                f"RegisteredStreamArray: raw source not found for {p}. "
                f"raw_source={self._ops.get('raw_source')!r}, "
                f"data_path={self._ops.get('data_path')!r} are not on disk. "
                "Re-point the raw data or open the plane with its data.bin."
            )

        plane = int(self._ops.get("plane", 1) or 1)
        self._plane_index = (plane - 1) if _get_num_planes(self._raw) > 1 else 0
        self._device = device if device is not None else _pick_device()

        self._stream = StreamingBinaryFile.from_run(
            self._raw, p, plane_index=self._plane_index,
            n_frames=self._nt, device=self._device,
        )
        self._Ly, self._Lx = self._stream.Ly, self._stream.Lx
        self.filenames = [p / "ops.npy"]

    # ---- LazyArray contract ------------------------------------------------
    def _shape5d(self):
        return (self._nt, 1, 1, self._Ly, self._Lx)

    @property
    def dtype(self):
        return np.dtype(np.int16)

    @property
    def metadata(self) -> dict:
        return dict(self._ops)

    @property
    def source_path(self):
        return self._plane_dir

    def __len__(self):
        return self._nt

    def __array__(self, dtype=None, copy=None):
        """Representative registered frame (for vmin/vmax preview)."""
        frame = np.asarray(self._stream[0])
        return frame.astype(dtype) if dtype is not None else frame

    def __getitem__(self, key):
        key = _normalize_key(key, 5)
        key = key + (slice(None),) * (5 - len(key))
        t_key, c_key, z_key, y_key, x_key = key

        frames = np.asarray(self._stream[t_key])  # (N,Ly,Lx) or (Ly,Lx) for int t
        t_is_int = isinstance(t_key, (int, np.integer))
        if frames.ndim == 2:
            frames = frames[np.newaxis]           # normalize to (N,Ly,Lx)

        # promote to canonical (T, C, Z, Y, X) with singleton C/Z, then index
        # C/Z/Y/X (integer keys squeeze, slices keep) and finally drop T if it
        # was an integer index — matching numpy basic-indexing semantics.
        out = frames[:, np.newaxis, np.newaxis, :, :]
        out = out[:, c_key, z_key, y_key, x_key]
        if t_is_int and len(out):
            out = out[0]
        return out
