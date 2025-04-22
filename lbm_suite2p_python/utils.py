import os
import subprocess

import numpy as np
from pathlib import Path

import tifffile
from matplotlib import patches

import matplotlib.pyplot as plt
import matplotlib as mpl

import suite2p
from scipy.ndimage import percentile_filter, gaussian_filter1d, uniform_filter1d

mpl.rcParams.update({
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': False,
    'figure.subplot.wspace': .01,
    'figure.subplot.hspace': .01,
    'figure.figsize': (18, 13),
    'ytick.major.left': True,
})
jet = mpl.cm.get_cmap('jet')
jet.set_bad(color='k')


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


def load_ops(ops_input: str | Path | list[str | Path]):
    """ Simple utility load a suite2p npy file"""
    if isinstance(ops_input, (str, Path)):
        return np.load(ops_input, allow_pickle=True).item()
    elif isinstance(ops_input, dict):
        return ops_input


def resize_to_max_proj(mask, target_shape):
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


def load_traces(ops):
    """
    Load fluorescence traces and related data from an ops file directory and return valid cells.

    This function loads the raw fluorescence traces, neuropil traces, and spike data from the directory
    specified in the ops dictionary. It also loads the 'iscell' file and returns only the traces corresponding
    to valid cells (i.e. where iscell is True).

    Parameters
    ----------
    ops : dict
        Dictionary containing at least the key 'save_path', which specifies the directory where the following
        files are stored: 'F.npy', 'Fneu.npy', 'spks.npy', and 'iscell.npy'.

    Returns
    -------
    F_valid : ndarray
        Array of fluorescence traces for valid cells (n_valid x n_timepoints).
    Fneu_valid : ndarray
        Array of neuropil fluorescence traces for valid cells (n_valid x n_timepoints).
    spks_valid : ndarray
        Array of spike data for valid cells (n_valid x n_timepoints).

    Notes
    -----
    The 'iscell.npy' file is expected to be an array where the first column (iscell[:, 0]) contains
    boolean values indicating valid cells.
    """
    save_path = Path(ops['save_path'])
    F = np.load(save_path.joinpath('F.npy'))
    Fneu = np.load(save_path.joinpath('Fneu.npy'))
    spks = np.load(save_path.joinpath('spks.npy'))
    iscell = np.load(save_path.joinpath('iscell.npy'), allow_pickle=True)[:, 0].astype(bool)

    F_valid = F[iscell]
    Fneu_valid = Fneu[iscell]
    spks_valid = spks[iscell]

    return F_valid, Fneu_valid, spks_valid


def plot_projection(
        ops,
        savepath=None,
        fig_label=None,
        vmin=None,
        vmax=None,
        add_scalebar=False,
        proj="meanImg",
        display_masks=False,
        accepted_only=False
):
    if proj == "meanImg":
        txt = "Mean-Image"
    elif proj == "max_proj":
        txt = "Max-Projection"
    elif proj == "meanImgE":
        txt = "Mean-Image (Enhanced)"
    else:
        raise ValueError("Unknown projection type. Options are ['meanImg', 'max_proj', 'meanImgE']")

    if savepath:
        savepath = Path(savepath)

    data = ops[proj]
    shape = data.shape
    fig, ax = plt.subplots(figsize=(6, 6), facecolor='black')
    vmin = np.nanpercentile(data, 2) if vmin is None else vmin
    vmax = np.nanpercentile(data, 98) if vmax is None else vmax

    if vmax - vmin < 1e-6:
        vmax = vmin + 1e-6
    ax.imshow(data, cmap='gray', vmin=vmin, vmax=vmax)

    # move projection title higher if masks are displayed to avoid overlap.
    proj_title_y = 1.07 if display_masks else 1.02
    ax.text(0.5, proj_title_y, txt, transform=ax.transAxes,
            fontsize=14, fontweight='bold', fontname="Courier New",
            color='white', ha='center', va='bottom')
    if fig_label:
        fig_label = fig_label.replace("_", " ").replace("-", " ").replace(".", " ")
        ax.set_ylabel(fig_label, color='white', fontweight='bold', fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    if display_masks:
        stats_file = Path(ops['save_path']).joinpath('stat.npy')
        iscell = np.load(Path(ops['save_path']).joinpath('iscell.npy'), allow_pickle=True)[:, 0].astype(bool)
        stats = np.load(stats_file, allow_pickle=True)
        im = suite2p.ROI.stats_dicts_to_3d_array(stats, Ly=ops['Ly'], Lx=ops['Lx'], label_id=True)
        im[im == 0] = np.nan
        accepted_cells = np.sum(iscell)
        rejected_cells = np.sum(~iscell)
        cell_rois = resize_to_max_proj(
            np.nanmax(im[iscell], axis=0) if np.any(iscell) else np.zeros_like(im[0]),
            shape)
        green_overlay = np.zeros((*shape, 4), dtype=np.float32)
        green_overlay[..., 1] = 1
        green_overlay[..., 3] = (cell_rois > 0) * 1.0
        ax.imshow(green_overlay)
        if not accepted_only:
            non_cell_rois = resize_to_max_proj(
                np.nanmax(im[~iscell], axis=0) if np.any(~iscell) else np.zeros_like(im[0]),
                shape)
            magenta_overlay = np.zeros((*shape, 4), dtype=np.float32)
            magenta_overlay[..., 0] = 1
            magenta_overlay[..., 2] = 1
            magenta_overlay[..., 3] = (non_cell_rois > 0) * 0.5
            ax.imshow(magenta_overlay)
        ax.text(0.37, 1.02, f"Accepted: {accepted_cells:03d}", transform=ax.transAxes,
                fontsize=14, fontweight='bold', fontname="Courier New",
                color='lime', ha='right', va='bottom')
        ax.text(0.63, 1.02, f"Rejected: {rejected_cells:03d}", transform=ax.transAxes,
                fontsize=14, fontweight='bold', fontname="Courier New",
                color='magenta', ha='left', va='bottom')
    if add_scalebar and 'dx' in ops:
        pixel_size = ops['dx']
        scale_bar_length = 100 / pixel_size
        scalebar_x = shape[1] * 0.05
        scalebar_y = shape[0] * 0.90
        ax.add_patch(patches.Rectangle(
            (scalebar_x, scalebar_y), scale_bar_length, 5,
            edgecolor='white', facecolor='white'))
        ax.text(scalebar_x + scale_bar_length / 2, scalebar_y - 10,
                "100 μm", color='white', fontsize=10, ha='center', fontweight='bold')

    # remove the spines that will show up as white bars
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()

    if savepath:
        savepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, dpi=300, facecolor='black')
    else:
        plt.show()


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

