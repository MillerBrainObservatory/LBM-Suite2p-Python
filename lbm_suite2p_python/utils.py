import os
import subprocess
import numpy as np
from pathlib import Path

import tifffile


def smooth_video(input_path, output_path, target_fps=60):
    filter_str = (
        f"minterpolate=fps={target_fps}:mi_mode=mci:mc_mode=aobmc:me=umh:vsbmc=1"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-vf",
        filter_str,
        "-fps_mode",
        "cfr",
        "-r",
        str(target_fps),
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "slow",
        output_path,
    ]

    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if result.returncode != 0:
        print("FFmpeg error:", result.stderr)
        raise subprocess.CalledProcessError(
            result.returncode, cmd, output=result.stdout, stderr=result.stderr
        )


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


def convert_to_rgba(zstack):
    """
    Converts a grayscale Z-stack (14x500x500) to an RGBA format (14x500x500x4).

    Parameters
    ----------
    zstack : np.ndarray
        Input grayscale Z-stack with shape (num_slices, height, width).

    Returns
    -------
    np.ndarray
        RGBA Z-stack with shape (num_slices, height, width, 4).
    """
    # Normalize grayscale values to [0,1] range
    normalized = (zstack - zstack.min()) / (zstack.max() - zstack.min())

    # Convert to RGB (repeat grayscale across RGB channels)
    rgba_stack = np.zeros((*zstack.shape, 4), dtype=np.float32)
    rgba_stack[..., :3] = np.repeat(normalized[..., np.newaxis], 3, axis=-1)

    # Set alpha channel to fully opaque (1.0)
    rgba_stack[..., 3] = 1.0

    return rgba_stack


def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def get_common_path(ops_files: list | tuple):
    """
    Find the common path of all files in `ops_files`.
    If there is a single file or no common path, return the first non-empty path.
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


def combine_tiffs(files: list[str | Path]) -> np.ndarray:
    """
    Combines multiple TIFF files into a single stacked TIFF.

    Parameters
    ----------
    files : list of str or Path
        List of file paths to the TIFF files to be combined.

    Returns
    -------
    np.ndarray
        A 3D NumPy array representing the concatenated TIFF stack.

    Notes
    -----
    - Input TIFFs should have identical spatial dimensions (`Y x X`).
    - The output shape will be `(T_total, Y, X)`, where `T_total` is the sum of all input time points.
    """
    first_file = files[0]
    first_tiff = tifffile.imread(first_file)
    num_files = len(files)
    num_frames, height, width = first_tiff.shape

    new_tiff = np.zeros((num_frames * num_files, height, width), dtype=first_tiff.dtype)

    for i, f in enumerate(files):
        tiff = tifffile.imread(f)
        new_tiff[i * num_frames : (i + 1) * num_frames] = tiff

    return new_tiff


def bin1d(X, bin_size, axis=0):
    """
    Mean bin over `axis` of `X` with bin `bin_size`.

    Directly from [rastermap](https://github.com/MouseLand/rastermap/blob/main/rastermap/utils.py).

    Parameters
    ----------
    X : np.ndarray
        Input array to be binned.
    bin_size : int
        Size of the bin. If <=0, no binning is performed.
    axis : int, optional
        Axis along which to bin. Default is 0.
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


