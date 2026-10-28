"""Read-side variants of suite2p's BinaryFile."""

from suite2p.io.binary import BinaryFile


class ClippedBinaryFile(BinaryFile):
    """Read-only BinaryFile that exposes only the first ``n_frames`` frames.

    Upstream infers the frame count from the file size when reading, so a raw
    chan1 binary longer than chan2 trips registration's equal-length check.
    Clipping the view, rather than the file, leaves the input untouched.
    """

    def __init__(self, Ly, Lx, filename, n_frames):
        self._n_frames = None  ## the base constructor reads n_frames
        super().__init__(Ly=Ly, Lx=Lx, filename=str(filename))
        self._n_frames = min(int(n_frames), super().n_frames)
        self.file = self.file[: self._n_frames]

    @property
    def n_frames(self):
        return super().n_frames if self._n_frames is None else self._n_frames
