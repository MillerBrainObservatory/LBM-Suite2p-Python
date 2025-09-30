import os
import subprocess
import numpy as np
from pathlib import Path

import tifffile
from scipy.ndimage import percentile_filter
import json


def ops_to_json(ops: dict | str | Path, outpath=None, indent=2):
    """
    Convert a Suite2p ops.npy file (or dict) to JSON.

    Parameters
    ----------
    ops : dict or str or Path
        Either a loaded ops dictionary or the path to an ops.npy file.
    outpath : str or Path, optional
        Output path for ops.json. If None, saves as 'ops.json' in the same
        directory as the input ops.npy (or current directory if ops is a dict).
    indent : int
        Indentation for JSON formatting.

    Returns
    -------
    Path
        Path to the written ops.json file.
    """
    # Load dict if given a path
    if isinstance(ops, (str, Path)):
        ops_path = Path(ops)
        if ops_path.is_dir():
            ops_path = ops_path / "ops.npy"
        if ops_path.suffix != ".npy":
            raise ValueError(f"Expected .npy file, got {ops_path}")
        ops_dict = np.load(ops_path, allow_pickle=True).item()
        base_dir = ops_path.parent
    elif isinstance(ops, dict):
        ops_dict = ops
        base_dir = Path.cwd()
    else:
        raise TypeError(f"`ops` must be dict, str, or Path, not {type(ops)}")

    # Convert numpy types to JSON serializable
    def _serialize(obj):
        if isinstance(obj, (np.generic, np.bool_)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Unserializable type {type(obj)}")

    # Decide output path
    if outpath is None:
        outpath = base_dir / "ops.json"
    else:
        outpath = Path(outpath)

    with open(outpath, "w") as f:
        json.dump(ops_dict, f, indent=indent, default=_serialize)

    print(f"Saved {outpath}")
    return outpath


def normalize_traces(F, mode="per_neuron"):
    """
    Normalize fluorescence traces F to [0, 1] range.
    Parameters
    ----------
    F : ndarray
        2d array of fluorescence traces (n_neurons x n_timepoints).
    mode : str
        Normalization mode, either "per_neuron" or "percentile".

    Returns
    -------
    F_norm : ndarray
        Normalized fluorescence traces in [0, 1] range.

    Notes
    -----
    - "per_neuron": scales each neuron's trace based on its own min and max.
    - "percentile": scales each neuron's trace based on its 1st and 99th percentiles.
    - If min == max for each cell, the trace is set to all zeros to avoid division by zero.
    """
    F_norm = np.zeros_like(F, dtype=float)

    if mode == "per_neuron":
        for i in range(F.shape[0]):
            f = F[i]
            fmax = np.max(f)
            fmin = np.min(f)
            if fmax > fmin:
                F_norm[i] = (f - fmin) / (fmax - fmin)
            else:
                F_norm[i] = f * 0
    elif mode == "percentile":
        for i in range(F.shape[0]):
            f = F[i]
            fmin = np.percentile(f, 1)
            fmax = np.percentile(f, 99)
            if fmax > fmin:
                F_norm[i] = (f - fmin) / (fmax - fmin)  # noqa
            else:
                F_norm[i] = f * 0
    return F_norm


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


def dff_rolling_percentile(f_trace, window_size=300, percentile=8):
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
    if not isinstance(f_trace, np.ndarray):
        raise TypeError("f_trace must be a numpy array")
    if f_trace.ndim != 2:
        raise ValueError("f_trace must be a 2D array with shape (N_neurons, N_frames)")
    if f_trace.shape[0] == 0 or f_trace.shape[1] == 0:
        raise ValueError("f_trace must not be empty")

    floor = np.median(f_trace, axis=1, keepdims=True) * 0.01

    f0 = np.array(
        [
            percentile_filter(f, percentile, size=window_size, mode="nearest")
            for f in f_trace
        ]
    )

    f0 = np.maximum(f0, floor)
    return (f_trace - f0) / (f0 + 1e-6)  # 1e-6 to avoid division by zero


def dff_median_filter(f_trace):
    """
    Compute ΔF/F₀ using a rolling median filter baseline.

    Parameters:
    -----------
    f_trace : np.ndarray
        (N_neurons, N_frames) fluorescence traces.

    Returns:
    --------
    dff : np.ndarray
        (N_neurons, N_frames) ΔF/F₀ traces.
    """
    if not isinstance(f_trace, np.ndarray):
        raise TypeError("f_trace must be a numpy array")
    if f_trace.ndim != 2:
        raise ValueError("f_trace must be a 2D array with shape (N_neurons, N_frames)")
    if f_trace.shape[0] == 0 or f_trace.shape[1] == 0:
        raise ValueError("f_trace must not be empty")

    f0 = np.median(f_trace, axis=1, keepdims=True) * 0.01
    return (f_trace - f0) / (f0 + 1e-6)  # 1e-6 to avoid division by zero


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


def load_planar_results(ops: dict | str | Path, z_plane: list | int = None) -> dict:
    """
    Load stat, iscell, spks files and return as a dict. Does NOT filter by valid cells, array contain both
    accepted and rejected neurons. Filter for accepted-only via f[iscell] or fneue[iscell] if needed.

    Parameters
    ----------
    ops : dict, str or Path
        Dict of or path to the ops.npy file. Can be a fully qualified path or a directory containing ops.npy.
    z_plane : int or None, optional
        the z-plane index for this file. If provided, it is stored in the output.

    Returns
    -------
    dict
        dictionary with keys:
        - 'F': fluorescence traces loaded from F.npy,
        - 'Fneu': neuropil fluorescence traces loaded from Fneu.npy,
        - 'spks': spike traces loaded from spks.npy,
        - 'stat': stats loaded from stat.npy,
        - 'iscell': boolean array from iscell.npy,
        - 'cellprob': cell probability from classifier.
        - 'z_plane': an array (of shape [n_neurons,]) with the provided z_plane index.

    See Also
    --------
    lbm_suite2p_python.load_ops
    lbm_suite2p_python.load_traces
    """
    if isinstance(ops, list):
        raise ValueError(f"Input should not be a list!")
    if isinstance(ops, (str, Path)):
        if Path(ops).is_dir():
            ops = Path(ops).joinpath("ops.npy")
            if not ops.exists():
                raise FileNotFoundError(f"ops.npy not found in given directory: {ops}")
    output_ops = load_ops(ops)

    save_path = Path(output_ops["save_path"])

    F = np.load(save_path.joinpath("F.npy"))
    Fneu = np.load(save_path.joinpath("Fneu.npy"))
    spks = np.load(save_path.joinpath("spks.npy"))
    stat = np.load(save_path.joinpath("stat.npy"), allow_pickle=True)
    iscell = np.load(save_path.joinpath("iscell.npy"), allow_pickle=True)[:, 0].astype(
        bool
    )
    cellprob = np.load(save_path.joinpath("iscell.npy"), allow_pickle=True)[:, 1]
    # model = np.load(save_path.joinpath("model.npy"), allow_pickle=True).item()

    n_neurons = spks.shape[0]
    if z_plane is None:
        z_plane_arr = output_ops.get("plane", np.zeros(n_neurons, dtype=int))
    else:
        z_plane_arr = np.full(n_neurons, z_plane, dtype=int)
    return {
        "F": F,
        "Fneu": Fneu,
        "spks": spks,
        "stat": stat,
        "iscell": iscell,
        "cellprob": cellprob,
        "z_plane": z_plane_arr,
        # "rastermap_model": model,
    }


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


def load_traces(ops: dict | str | Path):
    """
    Return (accepted-only) fluorescence traces, neuropil traces and spike traces from ops file.

    Parameters
    ----------
    ops : str, Path or dict
        Path to the ops.npy file or a dict containing the ops data.

    Returns
    -------
    tuple
        A tuple containing three arrays **filtered to contain only accepted neurons**:
        - F: Fluorescence traces (2D array, shape [n_neurons, n_timepoints])
        - Fneu: Neuropil fluorescence traces (2D array, shape [n_neurons, n_timepoints])
        - spks: Spike traces (2D array, shape [n_neurons, n_timepoints])

    See Also
    --------
    lbm_suite2p_python.load_ops
    lbm_suite2p_python.load_planar_results
    """
    output_ops = load_ops(ops)
    save_path = Path(output_ops["save_path"])

    F = np.load(save_path.joinpath("F.npy"))
    Fneu = np.load(save_path.joinpath("Fneu.npy"))
    spks = np.load(save_path.joinpath("spks.npy"))
    iscell = np.load(save_path.joinpath("iscell.npy"), allow_pickle=True)[:, 0].astype(
        bool
    )
    return F[iscell], Fneu[iscell], spks[iscell]


def save_ops(ops: dict, path: Path | str) -> None:
    """Save ops dict to a npy file. Ensure parent directory exists."""
    path = Path(path)
    path.parent.mkdir(exist_ok=True)
    np.save(str(path), ops, allow_pickle=True)


def load_ops(ops_input: str | Path | list[str | Path]) -> dict:
    """Simple utility load a suite2p npy file"""
    if isinstance(ops_input, (str, Path)):
        return np.load(ops_input, allow_pickle=True).item()
    elif isinstance(ops_input, dict):
        return ops_input
    print("Warning: No valid ops file provided, returning empty dict.")
    return {}
