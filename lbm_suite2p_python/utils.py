import os
import numpy as np
from pathlib import Path
import tifffile

from scipy.stats import zscore
import math

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from colorsys import hsv_to_rgb

import suite2p

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


def load_ops(ops_input: str | Path | list[str | Path]):
    if isinstance(ops_input, (str, Path)):
        return np.load(ops_input, allow_pickle=True).item()
    elif isinstance(ops_input, dict):
        return ops_input

def plot_fluorescence_grid_auto(f_cells_list, savepath, roi_range=(0, 500), frame_range=(0, 1000), sorted=True):
    """
    Generates an auto-sized square grid plot of Z-score normalized fluorescence heatmaps for multiple f_cells.

    Each row represents consecutive planes (e.g., Row 1 = Planes 1-4, Row 2 = Planes 5-8, etc.).
    The background is black with white text for labels. Optionally, ROIs can be sorted by mean absolute Z-score.

    Parameters
    ----------
    f_cells_list : list of np.ndarray
        List of fluorescence traces with shape (ROIs x Frames).
    savepath : str
        Path to save the generated figure.
    roi_range : tuple of int, optional
        The range of ROIs to include in the plots (default is (0, 500)).
    frame_range : tuple of int, optional
        The range of frames to include in the plots (default is (0, 1000)).
    sorted : bool, optional
        Whether to sort ROIs by mean absolute Z-score (default is False).

    Raises
    ------
    ValueError
        If `f_cells_list` is empty.

    Notes
    -----
    The function dynamically arranges the heatmaps into a square-like grid while ensuring that each row
    is labeled with the corresponding planes it represents. Sorting (if enabled) orders ROIs in descending
    order of mean absolute Z-score.
    """

    n = len(f_cells_list)
    if n == 0:
        raise ValueError("f_cells_list must contain at least one element.")

    grid_size = math.ceil(math.sqrt(n))
    rows, cols = grid_size, grid_size if n > (grid_size - 1) * grid_size else grid_size - 1

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows), facecolor="black")
    axes = np.array(axes).reshape(-1)

    for idx, f_cells in enumerate(f_cells_list):
        ax = axes[idx]

        subset = f_cells[roi_range[0]:roi_range[1], frame_range[0]:frame_range[1]]
        subset_zscore = zscore(subset, axis=1, nan_policy='omit')

        if sorted:
            sorted_indices = np.argsort(np.mean(np.abs(subset_zscore), axis=1))[::-1]
            subset_zscore = subset_zscore[sorted_indices]

        sns.heatmap(subset_zscore, cmap="inferno", center=0, cbar=False, ax=ax)

        if idx % cols == 0:
            plane_start = idx + 1
            plane_end = min(idx + cols, n)
            ax.set_ylabel(f"Planes {plane_start}-{plane_end}", fontsize=24, fontweight="bold", color="white")

        ax.set_facecolor("black")
        ax.spines["bottom"].set_color("white")
        ax.spines["top"].set_color("white")
        ax.spines["left"].set_color("white")
        ax.spines["right"].set_color("white")

        ax.set_xticks([])
        ax.set_yticks([])

    for j in range(idx + 1, rows * cols):
        fig.delaxes(axes[j])

    if sorted:
        plt.suptitle("Fluorescence Traces (Z-Scored, Sorted by Activity)", fontsize=32, fontweight="bold",
                     color="white", y=0.92)
    else:
        plt.suptitle("Fluorescence Traces (Z-Scored)", fontsize=32, fontweight="bold", color="white", y=0.92)

    plt.figtext(0.5, 0.01,
                f"Frame Interval: {frame_range[0]}-{frame_range[1]}, ROI Interval: {roi_range[0]}-{roi_range[1]}",
                fontsize=24, fontweight="bold", color="white", ha="center")

    plt.subplots_adjust(hspace=0.1, wspace=0.1)

    plt.savefig(savepath, dpi=1200, bbox_inches="tight", facecolor="black")
    plt.show()

def plot_roi_maps(ops_list, savepath):
    """
    Generates a 3-row x N-column plot of max projection and ROI maps for multiple ops files.

    The function visualizes max projection images, all detected cell ROIs, and all non-cell ROIs
    for a list of ops files. Each column corresponds to a different ops file (z-plane).

    Parameters
    ----------
    ops_list : list of str or dict
        A list of file paths (str) to ops.npy files or loaded ops dictionaries.
    savepath : str
        The file path to save the generated figure.

    Notes
    -----
    - The function expects each ops.npy file to have an associated `stat.npy` and `iscell.npy` file
      stored in the same `save_path`.
    - Each subplot column corresponds to a different z-plane.
    - The first row contains max projection images.
    - The second row contains all detected cell ROIs.
    - The third row contains all detected non-cell ROIs.
    """

    n_ops = len(ops_list)

    fig, axes = plt.subplots(3, n_ops, figsize=(6 * n_ops, 18))

    for idx, ops in enumerate(ops_list):
        if isinstance(ops, str):
            ops = np.load(ops, allow_pickle=True).item()

        stats_file = Path(ops['save_path']).joinpath('stat.npy')
        iscell = np.load(Path(ops['save_path']).joinpath('iscell.npy'), allow_pickle=True)[:, 0].astype(int)
        stats = np.load(stats_file, allow_pickle=True)

        Lx, Ly = ops["Lx"], ops["Ly"]
        n_cells = len(stats)

        h = np.random.rand(n_cells)
        hsvs = np.zeros((2, Ly, Lx, 3), dtype=np.float32)

        for i, stat in enumerate(stats):
            ypix, xpix, lam = stat['ypix'], stat['xpix'], stat['lam']
            hsvs[iscell[i], ypix, xpix, 0] = h[i]
            hsvs[iscell[i], ypix, xpix, 1] = 1
            hsvs[iscell[i], ypix, xpix, 2] = lam / lam.max()

        rgbs = np.array([hsv_to_rgb(*hsv) for hsv in hsvs.reshape(-1, 3)]).reshape(hsvs.shape)

        ax = axes[0, idx]
        ax.imshow(ops['max_proj'], cmap='gray')
        if idx == 0:
            ax.set_ylabel("Max Projection", fontsize=36, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

        ax = axes[1, idx]
        ax.imshow(rgbs[1])
        if idx == 0:
            ax.set_ylabel("All Cell ROIs", fontsize=36, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

        ax = axes[2, idx]
        ax.imshow(rgbs[0])
        if idx == 0:
            ax.set_ylabel("All Non-Cell ROIs", fontsize=36, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

        axes[0, idx].set_title(f"z-plane {idx + 1}", fontsize=36, fontweight="bold")

    plt.savefig(savepath, dpi=1200)

def plot_execution_time(filepath, savepath):
    """
    Plots the execution time for each processing step per z-plane.

    This function loads execution timing data from a `.npy` file and visualizes the
    runtime of different processing steps as a stacked bar plot with a black background.

    Parameters
    ----------
    filepath : str or Path
        Path to the `.npy` file containing the volume timing stats.
    savepath : str or Path
        Path to save the generated figure.

    Notes
    -----
    - The `.npy` file should contain structured data with `plane`, `registration`,
      `detection`, `extraction`, `classification`, `deconvolution`, and `total_plane_runtime` fields.
    """

    plane_stats = np.load(filepath)

    planes = plane_stats["plane"]
    reg_time = plane_stats["registration"]
    detect_time = plane_stats["detection"]
    extract_time = plane_stats["extraction"]
    total_time = plane_stats["total_plane_runtime"]

    plt.figure(figsize=(10, 6), facecolor="black")
    ax = plt.gca()
    ax.set_facecolor("black")

    plt.xlabel("Z-Plane", fontsize=14, fontweight="bold", color="white")
    plt.ylabel("Execution Time (s)", fontsize=14, fontweight="bold", color="white")
    plt.title("Execution Time per Processing Step", fontsize=16, fontweight="bold", color="white")

    plt.bar(planes, reg_time, label="Registration", alpha=0.8, color="#FF5733")
    plt.bar(planes, detect_time, label="Detection", alpha=0.8, bottom=reg_time, color="#33FF57")
    bars3 = plt.bar(planes, extract_time, label="Extraction", alpha=0.8, bottom=reg_time + detect_time, color="#3357FF")

    for bar, total in zip(bars3, total_time):
        height = bar.get_y() + bar.get_height()
        if total > 1:  # Only label if execution time is large enough to be visible
            plt.text(bar.get_x() + bar.get_width()/2, height + 2, f"{int(total)}",
                     ha="center", va="bottom", fontsize=12, color="white", fontweight="bold")

    plt.xticks(planes, fontsize=12, fontweight="bold", color="white")
    plt.yticks(fontsize=12, fontweight="bold", color="white")

    plt.grid(axis="y", linestyle="--", alpha=0.4, color="white")

    ax.spines["bottom"].set_color("white")
    ax.spines["left"].set_color("white")
    ax.spines["top"].set_color("white")
    ax.spines["right"].set_color("white")

    plt.legend(fontsize=12, facecolor="black", edgecolor="white", labelcolor="white", loc="upper left", bbox_to_anchor=(1, 1))

    plt.savefig(savepath, bbox_inches="tight", facecolor="black")
    plt.show()

def plot_volume_signal(filepath, savepath):
    """
    Plots the mean fluorescence signal per z-plane with standard deviation error bars.

    This function loads signal statistics from a `.npy` file and visualizes the mean
    fluorescence signal per z-plane, with error bars representing the standard deviation.

    Parameters
    ----------
    filepath : str or Path
        Path to the `.npy` file containing the volume stats. The output of `get_volume_stats()`.
    savepath : str or Path
        Path to save the generated figure.

    Notes
    -----
    - The `.npy` file should contain structured data with `plane`, `mean_trace`, and `std_trace` fields.
    - Error bars represent the standard deviation of the fluorescence signal.
    """

    plane_stats = np.load(filepath)

    planes = plane_stats["plane"]
    mean_signal = plane_stats["mean_trace"]
    std_signal = plane_stats["std_trace"]

    plt.figure(figsize=(10, 6), facecolor="black")
    ax = plt.gca()
    ax.set_facecolor("black")

    plt.xlabel("Z-Plane", fontsize=14, fontweight="bold", color="white")
    plt.ylabel("Mean Signal (ΔF/F)", fontsize=14, fontweight="bold", color="white")
    plt.title("Mean Fluorescence Signal per Z-Plane", fontsize=16, fontweight="bold", color="white")

    plt.errorbar(planes, mean_signal, yerr=std_signal, fmt='o-', color="cyan",
                 ecolor="lightblue", elinewidth=2, capsize=4, markersize=6, alpha=0.8, label="Mean ± STD")

    plt.xticks(planes, fontsize=12, fontweight="bold", color="white")
    plt.yticks(fontsize=12, fontweight="bold", color="white")

    plt.grid(axis="y", linestyle="--", alpha=0.4, color="white")

    ax.spines["bottom"].set_color("white")
    ax.spines["left"].set_color("white")
    ax.spines["top"].set_color("white")
    ax.spines["right"].set_color("white")

    plt.legend(fontsize=12, facecolor="black", edgecolor="white", labelcolor="white")

    plt.savefig(savepath, bbox_inches="tight", facecolor="black")
    plt.show()

def plot_volume_stats(filepath, savepath):
    """
    Plots the number of accepted and rejected neurons per z-plane.

    This function loads neuron count data from a `.npy` file and visualizes the
    accepted vs. rejected neurons as a stacked bar plot with a black background.

    Parameters
    ----------
    filepath : str or Path
        Path to the `.npy` file containing the volume stats. The output of get_volume_stats()
    savepath : str or Path
        Path to save the generated figure.

    Notes
    -----
    - The `.npy` file should contain structured data with `plane`, `accepted`, and `rejected` fields.
    """

    plane_stats = np.load(filepath)

    planes = plane_stats["plane"]
    accepted = plane_stats["accepted"]
    rejected = plane_stats["rejected"]

    plt.figure(figsize=(10, 6), facecolor="black")
    ax = plt.gca()
    ax.set_facecolor("black")

    plt.xlabel("Z-Plane", fontsize=14, fontweight="bold", color="white")
    plt.ylabel("Number of Neurons", fontsize=14, fontweight="bold", color="white")
    plt.title("Accepted vs. Rejected Neurons per Z-Plane", fontsize=16, fontweight="bold", color="white")

    bars1 = plt.bar(planes, accepted, label="Accepted Neurons", alpha=0.8, color="#4CAF50")  # Light green
    bars2 = plt.bar(planes, rejected, label="Rejected Neurons", alpha=0.8, bottom=accepted, color="#F57C00")  # Light orange

    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width()/2, height/2, f"{int(height)}",
                     ha="center", va="center", fontsize=12, color="white", fontweight="bold")

    for bar1, bar2 in zip(bars1, bars2):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        if height2 > 0:
            plt.text(bar2.get_x() + bar2.get_width()/2, height1 + height2/2, f"{int(height2)}",
                     ha="center", va="center", fontsize=12, color="white", fontweight="bold")

    plt.xticks(planes, fontsize=12, fontweight="bold", color="white")
    plt.yticks(fontsize=12, fontweight="bold", color="white")

    plt.grid(axis="y", linestyle="--", alpha=0.4, color="white")

    ax.spines["bottom"].set_color("white")
    ax.spines["left"].set_color("white")
    ax.spines["top"].set_color("white")
    ax.spines["right"].set_color("white")

    plt.legend(fontsize=12, facecolor="black", edgecolor="white", labelcolor="white")

    plt.savefig(savepath, bbox_inches="tight", facecolor="black")

def get_volume_stats(ops_files: list[str | Path], overwrite: bool=True):
    """
    Plots the number of accepted and rejected neurons per z-plane.

    This function loads neuron count data from a `.npy` file and visualizes the
    accepted vs. rejected neurons as a stacked bar plot with a black background.

    Parameters
    ----------
    ops_files : list of str or Path
        Each item in the list should be a path pointing to a z-lanes `ops.npy` file.
        The number of items in this list should match the number of z-planes in your session.
    overwrite : bool
        If a file already exists, it will be overwritten. Defaults to True.

    Notes
    -----
    - The `.npy` file should contain structured data with `plane`, `accepted`, and `rejected` fields.
    """
    if ops_files is None:
        print('No ops files found.')
        return None

    plane_stats = {}
    for i, file in enumerate(ops_files):

        output_ops = load_ops(file)
        iscell = np.load(Path(output_ops['save_path']).joinpath('iscell.npy'), allow_pickle=True)[:, 0].astype(bool)
        traces = np.load(Path(output_ops['save_path']).joinpath('F.npy'), allow_pickle=True)
        mean_trace = np.mean(traces)
        std_trace = np.std(traces)
        num_accepted = np.sum(iscell)
        num_rejected = np.sum(~iscell)
        timing = output_ops['timing']
        plane_stats[i + 1] = (num_accepted, num_rejected, mean_trace, std_trace, timing, file)

    common_path = os.path.commonpath(ops_files)
    plane_save = os.path.join(common_path, "volume_stats.npy")
    plane_stats_npy = np.array(
        [(plane, accepted, rejected, mean_trace, std_trace,
          timing["registration"], timing["detection"], timing["extraction"],
          timing["classification"], timing["deconvolution"], timing["total_plane_runtime"], filepath)
         for plane, (accepted, rejected, mean_trace, std_trace, timing, filepath) in plane_stats.items()],
        dtype=[
            ("plane", "i4"),
            ("accepted", "i4"),
            ("rejected", "i4"),
            ("mean_trace", "f8"),
            ("std_trace", "f8"),
            ("registration", "f8"),
            ("detection", "f8"),
            ("extraction", "f8"),
            ("classification", "f8"),
            ("deconvolution", "f8"),
            ("total_plane_runtime", "f8"),
            ("filepath", "U255")
        ]
    )

    if not Path(plane_save).is_file():
        np.save(plane_save, plane_stats_npy)
    elif Path(plane_save).is_file() and overwrite:
        np.save(plane_save, plane_stats_npy)
    else:
        print(f"File {plane_save} already exists. Skipping.")
    return plane_save

def post_process(ops_fname, overwrite=True):
    """
    Runs post-processing functions for suite2p output.

    This function loads an `ops.npy` file and runs a set of predefined plotting functions
    to generate registration, segmentation, and trace visualizations.

    Parameters
    ----------
    ops_fname : str or Path
        Path to the `ops.npy` file.
    overwrite : bool, optional
        Whether to overwrite existing output files (default is True).

    Notes
    -----
    - Generates and saves the following plots in the same directory as `ops.npy`:
      - `registration.png` (reference & registered images)
      - `segmentation.png` (ROI segmentation)
      - `traces.png` (fluorescence & deconvolved traces)
    """
    print(f"Post processing started.")
    filenames = {
        "registration.png": plot_registration,
        "segmentation.png": plot_segmentation,
        "traces.png": plot_traces
    }

    ops_loaded = load_ops(ops_fname)
    for fname, plot_func in filenames.items():
        path = Path(ops_loaded["save_path"]) / fname
        if overwrite or not path.exists():
            plot_func(ops_loaded, str(path))

def plot_registration(ops, savepath):
    """
    Plots registration images from suite2p processing.

    TODO: Allow dict or path to dict

    Parameters
    ----------
    ops : dict
        Dictionary loaded from `ops.npy`, containing suite2p output data.
    savepath : str or Path
        Path to save the generated figure.

    Notes
    -----
    - The figure contains four subplots:
      1. Reference image used for registration.
      2. Max projection of the registered image.
      3. Mean registered image.
      4. High-pass filtered mean registered image.
    """
    plt.figure()
    plt.subplot(1, 4, 1)
    plt.imshow(ops['refImg'], cmap='gray', )
    plt.title("Reference Image for Registration")

    plt.subplot(1, 4, 2)
    plt.imshow(ops['max_proj'], cmap='gray')
    plt.title("Registered Image, Max Projection")

    plt.subplot(1, 4, 3)
    plt.imshow(ops['meanImg'], cmap='gray')
    plt.title("Mean registered image")

    plt.subplot(1, 4, 4)
    plt.imshow(ops['meanImgE'], cmap='gray')
    plt.title("High-pass filtered Mean registered image")
    plt.savefig(savepath, dpi=300)
    print(f'Saved to {savepath}')

def plot_segmentation(ops, savepath):
    """
    Plots segmented ROIs from suite2p processing.

    Parameters
    ----------
    ops : dict
        Dictionary loaded from `ops.npy`, containing suite2p output data.
    savepath : str or Path
        Path to save the generated figure.

    Notes
    -----
    - Uses `stat.npy` and `iscell.npy` to differentiate cell vs. non-cell ROIs.
    - The figure contains four subplots:
      1. Max projection of the registered image.
      2. Overlay of all detected ROIs.
      3. Overlay of non-cell ROIs.
      4. Overlay of cell ROIs.
    """
    stats_file = Path(ops['save_path']).joinpath('stat.npy')
    iscell = np.load(Path(ops['save_path']).joinpath('iscell.npy'), allow_pickle=True)[:, 0].astype(bool)
    stats = np.load(stats_file, allow_pickle=True)

    im = suite2p.ROI.stats_dicts_to_3d_array(stats, Ly=ops['Ly'], Lx=ops['Lx'], label_id=True)
    im[im == 0] = np.nan

    plt.figure()
    plt.subplot(1, 4, 1)
    plt.imshow(ops['max_proj'], cmap='gray')
    plt.title("Registered Image, Max Projection")

    plt.subplot(1, 4, 2)
    plt.imshow(np.nanmax(im, axis=0), cmap='jet')
    plt.title("All ROIs Found")

    plt.subplot(1, 4, 3)
    plt.imshow(np.nanmax(im[~iscell], axis=0, ), cmap='jet')
    plt.title("All Non-Cell ROIs")

    plt.subplot(1, 4, 4)
    plt.imshow(np.nanmax(im[iscell], axis=0), cmap='jet')
    plt.title("All Cell ROIs")
    plt.savefig(savepath, dpi=300)

def plot_traces(ops, savepath, nframes=None, ntraces=None):
    """
    Plots fluorescence and deconvolved traces for randomly selected ROIs.

    Parameters
    ----------
    ops : dict
        Dictionary loaded from `ops.npy`, containing suite2p output data.
    savepath : str or Path
        Path to save the generated figure.
    nframes : int or None
        Number of frames to include.
    ntraces : int or None
        Number of ROIs to include.

    Notes
    -----
    - If `nframes` is None, all frames are used.
    - If `ntraces` is None, up to 20 ROIs are randomly selected.
    - The figure contains multiple subplots, one per ROI, displaying:
      - Raw fluorescence trace.
      - Neuropil fluorescence trace.
      - Deconvolved spikes, normalized to match fluorescence range.
    """
    f_cells = np.load(Path(ops['save_path']).joinpath('F.npy'))
    f_neuropils = np.load(Path(ops['save_path']).joinpath('Fneu.npy'))
    spks = np.load(Path(ops['save_path']).joinpath('spks.npy'))

    total_rois = f_cells.shape[0]
    total_frames = f_cells.shape[1]

    if ntraces is None:
        ntraces = min(20, total_rois)
    else:
        ntraces = min(ntraces, total_rois)

    if nframes is None:
        nframes = total_frames
    else:
        nframes = min(nframes, total_frames)

    rois = np.random.choice(total_rois, ntraces, replace=False)
    subsample_factor = max(1, nframes // 300)
    timepoints = np.arange(0, nframes, subsample_factor)

    fig, axes = plt.subplots(ntraces, 1, figsize=(12, 1.5 * ntraces), sharex=True)
    fig.suptitle("Fluorescence and Deconvolved Traces for Randomly Selected ROIs", fontsize=26, fontweight='bold', fontname="Arial", y=0.98)

    if ntraces == 1:
        axes = [axes]

    for i, roi in enumerate(rois):
        ax = axes[i]

        f = f_cells[roi, :nframes][::subsample_factor]
        f_neu = f_neuropils[roi, :nframes][::subsample_factor]
        sp = spks[roi, :nframes][::subsample_factor]

        fmax = max(f.max(), f_neu.max())
        fmin = min(f.min(), f_neu.min())
        frange = fmax - fmin
        if sp.max() > 0:
            sp = (sp / sp.max()) * frange

        ax.plot(timepoints, f, label="Cell Fluorescence", linewidth=1.5)
        ax.plot(timepoints, f_neu, label="Neuropil Fluorescence", linewidth=1.5)
        ax.plot(timepoints, sp + fmin, label="Deconvolved", linewidth=1.5, linestyle='dashed')

        ax.set_yticks([fmin, fmax])
        ax.set_yticklabels([f"{fmin:.1f}", f"{fmax:.1f}"], fontsize=12, fontweight='bold', fontname="Arial", rotation=45)

        ax.set_ylabel(f"ROI {roi}", rotation=90, labelpad=10, fontsize=14, fontweight='bold', fontname="Arial", va="center", ha="right")

        if i == 0:
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, 2.3), fontsize=18, ncol=3, frameon=False)

    axes[-1].set_xlabel("Frame Index", fontsize=16, fontweight='bold', fontname="Arial")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(savepath, dpi=300)
    plt.close()

def combine_tiffs(files):
    """
    Combines multiple TIFF files into a single stacked TIFF.

    This function concatenates multiple 3D TIFF files (`T x Y x X`) along the time axis
    to create a single output TIFF.

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

def make_subdir_from_list(files: list):
    """
    Moves each file in a list into its own subdirectory.

    This function organizes a list of file paths by moving each file into a
    subdirectory named after its stem.

    Parameters
    ----------
    files : list of Path
        List of file paths to be moved into subdirectories.

    Notes
    -----
    - The function creates a subdirectory named after the file stem and moves the file into it.
    - If the filename contains plane information (e.g., `plane_01`), the plane name is extracted for directory naming.
    """
    for file in files:
        fpath = file.parent / file.stem
        plane_name = fpath.stem.rpartition('_')[:-2][0]
        plane_path = file.parent / plane_name
        plane_path.mkdir(exist_ok=True)
        new_fname = plane_path / file.name
        file.rename(new_fname)

def get_fcells_list(ops_list: list):
    if not isinstance(ops_list, list):
        raise ValueError("`ops_list` must be a list")
    f_cells_list = []
    for ops in ops_list:
        ops = load_ops(ops)
        f_cells = np.load(Path(ops['save_path']).joinpath('F.npy'))
        f_cells_list.append(f_cells)
    return f_cells_list
