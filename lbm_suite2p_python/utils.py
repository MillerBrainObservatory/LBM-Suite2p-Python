import os
import subprocess

import numpy as np
from pathlib import Path

import tifffile

from scipy.ndimage import percentile_filter, gaussian_filter1d, uniform_filter1d


def smooth_video(input_path, output_path, target_fps=60):
    filter_str = f"minterpolate=fps={target_fps}:mi_mode=mci:mc_mode=aobmc:me=umh:vsbmc=1"
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-vf", filter_str,
        "-fps_mode", "cfr",
        "-r", str(target_fps),
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "slow",
        output_path
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print("FFmpeg error:", result.stderr)
        raise subprocess.CalledProcessError(result.returncode, cmd, output=result.stdout, stderr=result.stderr)


def _resize_masks_fit_crop(mask, target_shape):
    """Centers a mask within the target shape, cropping if too large or padding if too small."""
    sy, sx = mask.shape
    ty, tx = target_shape

    # If mask is larger, crop it
    if sy > ty or sx > tx:
        start_y = (sy - ty) // 2
        start_x = (sx - tx) // 2
        return mask[start_y:start_y + ty, start_x:start_x + tx]

    # If mask is smaller, pad it
    resized_mask = np.zeros(target_shape, dtype=mask.dtype)
    start_y = (ty - sy) // 2
    start_x = (tx - sx) // 2
    resized_mask[start_y:start_y + sy, start_x:start_x + sx] = mask
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


def dff_percentile(f_trace, window_size=300, percentile=8):
    """
    Compute ΔF/F₀ using a rolling percentile baseline.

    Parameters:
    -----------
    f_trace : np.ndarray
        (N_neurons, N_frames) fluorescence traces.
    window_size : int
        Size of the rolling window (in frames).
    percentile : int
        Percentile to use for baseline F₀ estimation.

    Returns:
    --------
    dff : np.ndarray
        (N_neurons, N_frames) ΔF/F₀ traces.
    """
    f0 = np.array([
        percentile_filter(f, percentile, size=window_size, mode='nearest')
        for f in f_trace
    ])
    return (f_trace - f0) / (f0 + 1e-6)  # 1e-6 to avoid division by zero


def dff_maxmin(f_trace, fps, smooth_window=5):
    """Compute DF/F₀ using a 5s Gaussian filter followed by rolling max-min ('maxmin')."""
    window_size = int(5 * fps)

    # Step 1: Apply Gaussian filter for initial baseline estimation
    f_smoothed = gaussian_filter1d(f_trace, sigma=window_size, axis=1)

    # Step 2: Rolling max-min baseline
    f_min = uniform_filter1d(f_smoothed, size=window_size, axis=1, mode="nearest")
    f_max = uniform_filter1d(f_trace, size=window_size, axis=1, mode="nearest")
    f_baseline = (f_min + f_max) / 2  # Approximate rolling baseline

    # Step 3: Compute ΔF/F₀
    dff = (f_trace - f_baseline) / (f_baseline + 1e-8)

    # Step 4: Normalize 0 to 1 for visualization
    dff_n = (dff - np.min(dff, axis=1, keepdims=True)) / (np.max(dff, axis=1, keepdims=True) - np.min(dff, axis=1, keepdims=True) + 1e-8)
    dff_smooth = uniform_filter1d(dff_n, size=smooth_window, axis=1)

    return dff_smooth


def dff_shot_noise(dff, fr):
    """
    Estimate the shot noise level of calcium imaging traces.

    This metric quantifies the noise level based on frame-to-frame differences,
    assuming slow calcium dynamics compared to the imaging frame rate. It was
    introduced by Rupprecht et al. (2021) [1] as a standardized method for comparing
    noise levels across datasets with different acquisition parameters.

    The noise level :math:`\\nu` is computed as:

    .. math::

        \\nu = \\frac{\\mathrm{median}_t\\left( \\left| \\Delta F/F_{t+1} - \\Delta F/F_t \\right| \\right)}{\\sqrt{f_r}}

    where
      - :math:`\\Delta F/F_t` is the fluorescence trace at time :math:`t`
      - :math:`f_r` is the imaging frame rate (in Hz).

    Parameters
    ----------
    dff : np.ndarray
        Array of shape (n_neurons, n_frames), containing raw :math:`\\Delta F/F` traces
        (percent units, **without neuropil subtraction**).
    fr : float
        Frame rate of the recording in Hz.

    Returns
    -------
    np.ndarray
        Noise level :math:`\\nu` for each neuron, expressed in %/√Hz units.

    Notes
    -----
    - The metric relies on the slow dynamics of calcium signals compared to frame rate.
    - Higher values of :math:`\\nu` indicate higher shot noise.
    - Units are % divided by √Hz, and while unconventional, they enable comparison across frame rates.

    References
    ----------
    [1] Rupprecht et al., "Large-scale calcium imaging & noise levels",
        A Neuroscientific Blog (2021).
        https://gcamp6f.com/2021/10/04/large-scale-calcium-imaging-noise-levels/
    """
    return np.median(np.abs(np.diff(dff, axis=1)), axis=1) / np.sqrt(fr)


def get_common_path(ops_files: list | tuple):
    """
    Find the common path of all files in `ops_files`.
    If there is a single file or no common path, return the first non-empty path.
    """
    if not isinstance(ops_files, (list, tuple)):
        ops_files = [ops_files]
    if len(ops_files) == 1:
        path = Path(ops_files[0]).parent
        while path.exists() and len(list(path.iterdir())) <= 1:  # Traverse up if only one item exists
            path = path.parent
        return path
    else:
        return Path(os.path.commonpath(ops_files))


def combine_tiffs(files):
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
        new_tiff[i * num_frames:(i + 1) * num_frames] = tiff

    return new_tiff
