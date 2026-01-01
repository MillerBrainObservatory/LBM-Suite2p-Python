import os
import numpy as np
from pathlib import Path


# Supported lazy array types from mbo_utilities
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
    return type(obj).__name__ in _LAZY_ARRAY_TYPES


def _get_num_planes(arr):
    """
    Get number of z-planes from a lazy array.

    Parameters
    ----------
    arr : array-like
        Input array, typically from mbo_utilities.

    Returns
    -------
    int
        Number of z-planes (1 for 3D arrays, Z dimension for 4D).
    """
    if hasattr(arr, "num_planes"):
        return arr.num_planes
    if hasattr(arr, "num_channels"):
        return arr.num_channels
    shape = arr.shape
    if len(shape) == 4:
        return shape[1]  # TZYX format
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
