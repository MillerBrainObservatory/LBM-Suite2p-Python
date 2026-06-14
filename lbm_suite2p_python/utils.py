import os
import numpy as np
from pathlib import Path


# mbo_utilities >= 4.0 exposes a single LazyArray base; isinstance covers
# every built-in and any third-party plugin. The class-name tuple is the
# fallback for mbo_utilities < 4.0, which has no shared base.
try:
    from mbo_utilities import LazyArray as _LazyArray
except ImportError:  # mbo_utilities < 4.0
    _LazyArray = None

_LAZY_ARRAY_TYPES = (
    "ScanImageArray",
    "LBMArray",
    "PiezoArray",
    "SinglePlaneArray",
    "Suite2pArray",
    "MBOTiffArray",
    "MboRawArray",
    "TiffArray",
    "ZarrArray",
    "H5Array",
    "NumpyArray",
    "BinArray",
)


def _is_lazy_array(obj):
    """Check if obj is an mbo_utilities lazy array type."""
    if _LazyArray is not None and isinstance(obj, _LazyArray):
        return True
    return type(obj).__name__ in _LAZY_ARRAY_TYPES


def _get_num_planes(arr):
    """
    Get number of z-planes from a lazy array.

    mbo_utilities arrays are always 5D TCZYX, so Z is at shape[2].
    Falls back to legacy 4D TZYX (Z at shape[1]) and other heuristics
    for non-mbo arrays.

    Parameters
    ----------
    arr : array-like
        Input array, typically from mbo_utilities.

    Returns
    -------
    int
        Number of z-planes (1 if no Z dimension).
    """
    # mbo_utilities Shape5DMixin
    if hasattr(arr, "nz"):
        return arr.nz
    if hasattr(arr, "num_planes") and arr.num_planes is not None:
        return arr.num_planes
    shape = arr.shape
    if len(shape) == 5:
        return shape[2]  # 5D TCZYX
    if len(shape) == 4:
        return shape[1]  # legacy 4D TZYX
    return 1


def _resize_masks_fit_crop(mask, target_shape):
    """Centers a mask within the target shape, cropping if too large or padding if too small."""
    sy, sx = mask.shape
    ty, tx = target_shape

    # If mask is larger, crop it
    if sy > ty or sx > tx:
        start_y = (sy - ty) // 2
        start_x = (sx - tx) // 2
        return mask[start_y : start_y + ty, start_x : start_x + tx]

    # If mask is smaller, pad it
    resized_mask = np.zeros(target_shape, dtype=mask.dtype)
    start_y = (ty - sy) // 2
    start_x = (tx - sx) // 2
    resized_mask[start_y : start_y + sy, start_x : start_x + sx] = mask
    return resized_mask


def get_common_path(ops_files: list | tuple):
    """
    Find the common parent path of all files.

    Parameters
    ----------
    ops_files : list or tuple
        List of file paths.

    Returns
    -------
    Path
        Common parent directory of all files.
    """
    if not isinstance(ops_files, (list, tuple)):
        ops_files = [ops_files]
    if len(ops_files) == 1:
        path = Path(ops_files[0]).parent
        while (
            path.exists() and len(list(path.iterdir())) <= 1
        ):  # Traverse up if only one item exists
            path = path.parent
        return path
    else:
        return Path(os.path.commonpath(ops_files))


def estimate_peak_memory(ops, Ly, Lx, n_frames, device="cuda", workers=1):
    """
    Estimate peak memory for one Suite2p plane from its parameters.

    Registration and detection run sequentially in suite2p's pipeline, so
    the peak for a plane is ``max(registration, detection)``, not the sum.
    The two stages load different pools:

    - Registration compute runs on ``device``. When ``device`` is cuda the
      per-batch float32/FFT buffers live in VRAM, scaled by
      ``batch_size * Ly * Lx``. The reference-image correlation
      (``pick_initial_reference``) and the binary read stay on host.
    - Detection's binned movie is a host numpy array, and the default
      ``sparsery`` / ``sourcery`` detectors run on CPU. ``device`` is only
      used by the cellpose path, which sees the 2D meanImg / max_proj, not
      the movie. So the binned movie never enters VRAM.

    Host RAM therefore peaks during detection (binned movie plus the
    high-pass / sparsery copies, ~2.5x); VRAM peaks during registration
    (or cellpose inference, when enabled). Neither VRAM term scales with
    ``n_frames``; host detection plateaus once ``n_frames // bin_size``
    exceeds ``nbins``.

    Parameters
    ----------
    ops : dict
        Flat Suite2p ops. Reads ``nimg_init``, ``batch_size`` (registration
        batch), ``nbins``, ``bin_size``, ``tau``, ``fs``, ``nchannels``, and
        ``anatomical_only``. Missing keys fall back to suite2p defaults.
    Ly, Lx : int
        Frame height and width in pixels. The detection crop (yrange/xrange)
        is unknown before registration, so full ``Ly``/``Lx`` are used as an
        upper bound.
    n_frames : int
        Number of frames in the plane.
    device : str, optional (default "cuda")
        Torch device. VRAM terms are reported only when this starts with
        "cuda".
    workers : int, optional (default 1)
        Concurrent plane workers. Per-plane peaks are multiplied by this for
        the ``*_total`` fields.

    Returns
    -------
    dict
        Bytes for ``host_per_plane``, ``vram_per_plane``, ``host_total``,
        ``vram_total``. VRAM fields are 0 when ``device`` is not cuda.

    Notes
    -----
    The 2.5x detection multiplier and the cpsam VRAM constant are rough; the
    real values depend on data and hardware. Calibrate with
    ``torch.cuda.max_memory_allocated()`` and an RSS sample around the
    registration / detection calls for tight bounds.
    """
    cuda = str(device).startswith("cuda")

    nimg_init = min(int(ops.get("nimg_init", 400)), n_frames)
    reg_host = nimg_init * Ly * Lx * 10  # int16 frames + float64 ref corr (CPU)

    nbins = int(ops.get("nbins", 5000))
    bin_size = ops.get("bin_size") or max(
        1, n_frames // nbins, round(ops.get("tau", 1.0) * ops.get("fs", 10.0))
    )
    nbinned = min(nbins, n_frames // bin_size)
    detect_host = int(2.5 * nbinned * Ly * Lx * 4)  # binned movie + hp/sparsery copies (CPU)

    host_peak = max(reg_host, detect_host)

    vram_peak = 0
    if cuda:
        reg_batch = int(ops.get("batch_size", 100))
        nchan = int(ops.get("nchannels", 1))
        reg_vram = nchan * 8 * reg_batch * Ly * Lx * 4  # ~8 float32/FFT buffers per batch
        cpsam_vram = 1_500_000_000 if ops.get("anatomical_only", 0) else 0  # cpsam weights + activations
        vram_peak = max(reg_vram, cpsam_vram)

    return {
        "host_per_plane": host_peak,
        "vram_per_plane": vram_peak,
        "host_total": host_peak * max(1, workers),
        "vram_total": vram_peak * max(1, workers),
    }


def bin1d(X, bin_size, axis=0):
    """
    Mean bin over `axis` of `X` with bin `bin_size`.

    Parameters
    ----------
    X : np.ndarray
        Input array to be binned.
    bin_size : int
        Size of the bin. If <=0, no binning is performed.
    axis : int, optional
        Axis along which to bin. Default is 0.

    Returns
    -------
    np.ndarray
        Binned array with reduced size along the specified axis.
    """
    if bin_size > 0:
        size = list(X.shape)
        Xb = X.swapaxes(0, axis)
        size_new = Xb.shape
        Xb = (
            Xb[: size[axis] // bin_size * bin_size]
            .reshape((size[axis] // bin_size, bin_size, *size_new[1:]))
            .mean(axis=1)
        )
        Xb = Xb.swapaxes(axis, 0)
        return Xb
    else:
        return X
