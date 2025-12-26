import glob
import subprocess
from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt

from lbm_suite2p_python.utils import get_common_path
from lbm_suite2p_python.postprocessing import load_ops


def update_ops_paths(ops_files: str | list):
    """
    Update save_path, save_path0, and save_folder in an ops dictionary based on its current location. Use after moving an ops_file or batch of ops_files.
    """
    if isinstance(ops_files, (str, Path)):
        ops_files = [ops_files]

    for ops_file in ops_files:
        ops = np.load(ops_file, allow_pickle=True).item()

        ops_path = Path(ops_file)
        plane0_folder = ops_path.parent
        plane_folder = plane0_folder.parent

        ops["save_path"] = str(plane0_folder)
        ops["save_path0"] = str(plane_folder)
        ops["save_folder"] = plane_folder.name
        ops["ops_path"] = ops_path

        np.save(ops_file, ops)


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
    plt.title(
        "Execution Time per Processing Step",
        fontsize=16,
        fontweight="bold",
        color="white",
    )

    plt.bar(planes, reg_time, label="Registration", alpha=0.8, color="#FF5733")
    plt.bar(
        planes,
        detect_time,
        label="Detection",
        alpha=0.8,
        bottom=reg_time,
        color="#33FF57",
    )
    bars3 = plt.bar(
        planes,
        extract_time,
        label="Extraction",
        alpha=0.8,
        bottom=reg_time + detect_time,
        color="#3357FF",
    )

    for bar, total in zip(bars3, total_time):
        height = bar.get_y() + bar.get_height()
        if total > 1:  # Only label if execution time is large enough to be visible
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + 2,
                f"{int(total)}",
                ha="center",
                va="bottom",
                fontsize=12,
                color="white",
                fontweight="bold",
            )

    plt.xticks(planes, fontsize=12, fontweight="bold", color="white")
    plt.yticks(fontsize=12, fontweight="bold", color="white")

    plt.grid(axis="y", linestyle="--", alpha=0.4, color="white")

    ax.spines["bottom"].set_color("white")
    ax.spines["left"].set_color("white")
    ax.spines["top"].set_color("white")
    ax.spines["right"].set_color("white")

    plt.legend(
        fontsize=12,
        facecolor="black",
        edgecolor="white",
        labelcolor="white",
        loc="upper left",
        bbox_to_anchor=(1, 1),
    )

    plt.savefig(savepath, bbox_inches="tight", facecolor="black")
    plt.show()


def plot_volume_signal(zstats, savepath):
    """
    Plots the mean fluorescence signal per z-plane with standard deviation error bars.

    This function loads signal statistics from a `.npy` file and visualizes the mean
    fluorescence signal per z-plane, with error bars representing the standard deviation.

    Parameters
    ----------
    zstats : str or Path
        Path to the `.npy` file containing the volume stats. The output of `get_zstats()`.
    savepath : str or Path
        Path to save the generated figure.

    Notes
    -----
    - The `.npy` file should contain structured data with `plane`, `mean_trace`, and `std_trace` fields.
    - Error bars represent the standard deviation of the fluorescence signal.
    """

    plane_stats = np.load(zstats)

    planes = plane_stats["plane"]
    mean_signal = plane_stats["mean_trace"]
    std_signal = plane_stats["std_trace"]

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="black")
    ax.set_facecolor("black")

    ax.errorbar(
        planes,
        mean_signal,
        yerr=std_signal,
        fmt="o-",
        color="#3498db",
        ecolor="#85c1e9",
        elinewidth=1.5,
        capsize=3,
        markersize=5,
        alpha=0.9,
        label="Mean ± STD",
    )

    ax.set_xlabel("Z-Plane", fontsize=10, fontweight="bold", color="white")
    ax.set_ylabel("Mean Raw Signal", fontsize=10, fontweight="bold", color="white")
    ax.set_title("Mean Fluorescence Signal per Z-Plane", fontsize=11, fontweight="bold", color="white")

    ax.tick_params(colors="white", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("white")
    ax.spines["left"].set_color("white")

    ax.legend(fontsize=8, facecolor="#1a1a1a", edgecolor="white", labelcolor="white")

    savepath = Path(savepath)
    if savepath.is_dir():
        savepath = savepath / "mean_volume_signal.png"
    plt.savefig(savepath, bbox_inches="tight", facecolor="black", dpi=150)
    plt.close(fig)


def plot_volume_neuron_counts(zstats, savepath):
    """
    Plots the number of accepted and rejected neurons per z-plane.

    This function loads neuron count data from a `.npy` file and visualizes the
    accepted vs. rejected neurons as a stacked bar plot with a black background.

    Parameters
    ----------
    zstats : str, Path
        Full path to the zstats.npy file.
    savepath : str or Path
        Path to directory where generated figure will be saved.

    Notes
    -----
    - The `.npy` file should contain structured data with `plane`, `accepted`, and `rejected` fields.
    """

    zstats = Path(zstats)
    if not zstats.is_file():
        raise FileNotFoundError(f"{zstats} is not a valid zstats.npy file.")

    plane_stats = np.load(zstats)
    savepath = Path(savepath)

    planes = plane_stats["plane"]
    accepted = plane_stats["accepted"]
    rejected = plane_stats["rejected"]
    savename = savepath.joinpath(
        f"all_neurons_{accepted.sum()}acc_{rejected.sum()}rej.png"
    )

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="black")
    ax.set_facecolor("black")

    bar_width = 0.8
    bars1 = ax.bar(
        planes, accepted, width=bar_width, label=f"Accepted ({accepted.sum()})",
        alpha=0.85, color="#2ecc71", edgecolor="#27ae60", linewidth=0.5
    )
    bars2 = ax.bar(
        planes, rejected, width=bar_width, bottom=accepted,
        label=f"Rejected ({rejected.sum()})", alpha=0.85, color="#e74c3c",
        edgecolor="#c0392b", linewidth=0.5
    )

    # Add count labels inside bars (only if tall enough)
    for bar in bars1:
        height = bar.get_height()
        if height > 5:  # Only label if tall enough to fit text
            ax.text(
                bar.get_x() + bar.get_width() / 2, height / 2,
                f"{int(height)}", ha="center", va="center",
                fontsize=8, color="white", fontweight="bold"
            )

    for bar1, bar2 in zip(bars1, bars2):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        if height2 > 5:
            ax.text(
                bar2.get_x() + bar2.get_width() / 2, height1 + height2 / 2,
                f"{int(height2)}", ha="center", va="center",
                fontsize=8, color="white", fontweight="bold"
            )

    ax.set_xlabel("Z-Plane", fontsize=10, fontweight="bold", color="white")
    ax.set_ylabel("Number of ROIs", fontsize=10, fontweight="bold", color="white")
    ax.set_title("Accepted vs Rejected ROIs per Z-Plane", fontsize=11, fontweight="bold", color="white")

    ax.tick_params(colors="white", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("white")
    ax.spines["left"].set_color("white")

    ax.legend(fontsize=8, facecolor="#1a1a1a", edgecolor="white", labelcolor="white", loc="upper right")

    plt.savefig(savename, bbox_inches="tight", facecolor="black", dpi=150)
    plt.close(fig)


def get_volume_stats(ops_files: list[str | Path], overwrite: bool = True):
    """
    Given a list of ops.npy files, accumulate common statistics for assessing zplane quality.

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
    if not ops_files:
        print("No ops files found.")
        return None

    plane_stats = {}
    for i, file in enumerate(ops_files):
        output_ops = load_ops(file)
        raw_z = output_ops.get("plane", None)
        if raw_z is None:
            zplane_num = i
        else:
            if isinstance(raw_z, (int, np.integer)):
                zplane_num = int(raw_z)
            else:
                s = str(raw_z)
                digits = "".join([c for c in s if c.isdigit()])
                zplane_num = int(digits) if digits else i

        save_path = Path(output_ops["save_path"])
        timing = output_ops.get("timing", {})

        # Check if required files exist
        iscell_file = save_path / "iscell.npy"
        traces_file = save_path / "F.npy"

        if not iscell_file.exists():
            print(f"Skipping plane {zplane_num}: iscell.npy not found at {save_path}")
            continue

        if not traces_file.exists():
            print(f"Skipping plane {zplane_num}: F.npy not found at {save_path}")
            continue

        # Load files
        try:
            iscell_raw = np.load(iscell_file, allow_pickle=True)
            traces = np.load(traces_file, allow_pickle=True)

            # Validate iscell data - check for _NoValue or other invalid types
            if not isinstance(iscell_raw, np.ndarray) or iscell_raw.size == 0:
                print(f"Skipping plane {zplane_num}: iscell.npy is empty or invalid")
                continue

            # Handle potential _NoValue entries by filtering to valid numeric data
            try:
                iscell = iscell_raw[:, 0].astype(bool)
            except (TypeError, ValueError) as e:
                print(f"Skipping plane {zplane_num}: iscell data conversion failed - {e}")
                continue

            # Validate traces
            if not isinstance(traces, np.ndarray) or traces.size == 0:
                print(f"Skipping plane {zplane_num}: F.npy is empty or invalid")
                continue

        except Exception as e:
            print(f"Skipping plane {zplane_num}: Error loading files - {e}")
            continue

        # Safe stat computations with explicit conversion
        try:
            n_accepted = int(np.sum(iscell))
            n_rejected = int(np.sum(~iscell))
            trace_mean = float(np.nanmean(traces))
            trace_std = float(np.nanstd(traces))
        except (TypeError, ValueError) as e:
            print(f"Skipping plane {zplane_num}: Error computing statistics - {e}")
            continue

        plane_stats[zplane_num] = {
            "accepted": n_accepted,
            "rejected": n_rejected,
            "mean": trace_mean,
            "std": trace_std,
            "registration": timing.get("registration", np.nan),
            "detection": timing.get("detection", timing.get("detect", np.nan)),
            "extraction": timing.get("extraction", np.nan),
            "classification": timing.get("classification", np.nan),
            "deconvolution": timing.get("deconvolution", np.nan),
            "total_runtime": timing.get("total_plane_runtime", np.nan),
            "filepath": str(file),
            "zplane": zplane_num,
        }

    # Check if any planes had valid statistics
    if not plane_stats:
        print("No valid plane statistics collected - all planes skipped or missing files")
        return None

    common = get_common_path(ops_files)
    out = []
    for p, stats in sorted(plane_stats.items()):
        out.append(
            (
                p,
                stats["accepted"],
                stats["rejected"],
                stats["mean"],
                stats["std"],
                stats["registration"],
                stats["detection"],
                stats["extraction"],
                stats["classification"],
                stats["deconvolution"],
                stats["total_runtime"],
                stats["filepath"],
                stats["zplane"],
            )
        )
    dtype = [
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
        ("filepath", "U255"),
        ("zplane", "i4"),
    ]
    arr = np.array(out, dtype=dtype)
    save_path = Path(common) / "zstats.npy"
    if overwrite or not save_path.exists():
        np.save(save_path, arr)
    return str(save_path)


def save_images_to_movie(image_input, savepath, duration=None, format=".mp4"):
    """
    Convert a sequence of saved images into a movie.

    TODO: move to mbo_utilities.

    Parameters
    ----------
    image_input : str, Path, or list
        Directory containing saved segmentation images or a list of image file paths.
    savepath : str or Path
        Path to save the video file.
    duration : int, optional
        Desired total video duration in seconds. If None, defaults to 1 FPS (1 image per second).
    format : str, optional
        Video format: ".mp4" (PowerPoint-compatible), ".avi" (lossless), ".mov" (ProRes). Default is ".mp4".

    Examples
    --------
    >>> import mbo_utilities as mbo
    >>> import lbm_suite2p_python as lsp

    Get all png files autosaved during LBM-Suite2p-Python `run_volume()`
    >>> segmentation_pngs = mbo.get_files("path/suite3d/results/", "segmentation.png", max_depth=3)
    >>> lsp.save_images_to_movie(segmentation_pngs, "path/to/save/segmentation.png", format=".mp4")
    """
    savepath = Path(savepath).with_suffix(format)  # Ensure correct file extension
    temp_video = savepath.with_suffix(".avi")  # Temporary AVI file for MOV conversion
    savepath.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(image_input, (str, Path)):
        image_dir = Path(image_input)
        image_files = sorted(
            glob.glob(str(image_dir / "*.png"))
            + glob.glob(str(image_dir / "*.jpg"))
            + glob.glob(str(image_dir / "*.tif"))
        )
    elif isinstance(image_input, list):
        image_files = sorted(map(str, image_input))
    else:
        raise ValueError(
            "image_input must be a directory path or a list of file paths."
        )

    if not image_files:
        return

    first_image = cv2.imread(image_files[0])
    height, width, _ = first_image.shape
    fps = len(image_files) / duration if duration else 1

    if format == ".mp4":
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        video_path = savepath
    elif format == ".avi":
        fourcc = cv2.VideoWriter_fourcc(*"HFYU")
        video_path = savepath
    elif format == ".mov":
        fourcc = cv2.VideoWriter_fourcc(*"HFYU")
        video_path = temp_video
    else:
        raise ValueError("Invalid format. Use '.mp4', '.avi', or '.mov'.")

    video_writer = cv2.VideoWriter(
        str(video_path), fourcc, max(fps, 1), (width, height)
    )

    for image_file in image_files:
        frame = cv2.imread(image_file)
        video_writer.write(frame)

    video_writer.release()

    if format == ".mp4":
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vcodec",
            "libx264",
            "-acodec",
            "aac",
            "-preset",
            "slow",
            "-crf",
            "18",
            str(savepath),  # Save directly to `savepath`
        ]
        subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"MP4 saved at {savepath}")

    elif format == ".mov":
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(temp_video),
            "-c:v",
            "prores_ks",  # Use Apple ProRes codec
            "-profile:v",
            "3",  # ProRes 422 LT
            "-pix_fmt",
            "yuv422p10le",
            str(savepath),
        ]
        subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        temp_video.unlink()


def consolidate_volume(suite2p_path, merged_dir="merged", overwrite=False):
    """
    Consolidate all plane results into a single merged directory.

    Combines ops.npy, stat.npy, iscell.npy, F.npy, Fneu.npy, and spks.npy
    from all planeXX_stitched folders into a single merged/ folder with
    plane-indexed arrays.

    Parameters
    ----------
    suite2p_path : str or Path
        Path to suite2p directory containing planeXX_stitched folders
    merged_dir : str, optional
        Name of merged directory to create (default: "merged")
    overwrite : bool, optional
        Whether to overwrite existing merged directory (default: False)

    Returns
    -------
    merged_path : Path
        Path to the created merged directory

    Examples
    --------
    >>> import lbm_suite2p_python as lsp
    >>> merged = lsp.consolidate_volume("path/to/suite2p")
    >>> # Load consolidated results
    >>> ops = np.load(merged / "ops.npy", allow_pickle=True).item()
    >>> stat = np.load(merged / "stat.npy", allow_pickle=True)
    >>> iscell = np.load(merged / "iscell.npy")
    """
    suite2p_path = Path(suite2p_path)
    merged_path = suite2p_path / merged_dir

    # Find all plane directories
    plane_dirs = sorted(suite2p_path.glob("plane*_stitched"))
    if not plane_dirs:
        raise ValueError(f"No plane*_stitched directories found in {suite2p_path}")

    print(f"Found {len(plane_dirs)} planes to consolidate")

    # Check if merged directory exists
    if merged_path.exists() and not overwrite:
        print(f"Merged directory already exists: {merged_path}")
        print("Use overwrite=True to recreate")
        return merged_path

    # Create merged directory
    merged_path.mkdir(exist_ok=True)

    # Initialize lists for consolidation
    all_stats = []
    all_iscell = []
    all_F = []
    all_Fneu = []
    all_spks = []
    all_ops = []

    # Track ROI offsets for each plane
    n_cells_per_plane = []

    for plane_dir in plane_dirs:
        plane_num = int(''.join(filter(str.isdigit, plane_dir.name)))
        print(f"  Processing plane {plane_num}: {plane_dir.name}")

        # Load required files
        stat_file = plane_dir / "stat.npy"
        iscell_file = plane_dir / "iscell.npy"
        ops_file = plane_dir / "ops.npy"

        if not all([stat_file.exists(), iscell_file.exists(), ops_file.exists()]):
            print(f"    WARNING: Missing required files, skipping plane {plane_num}")
            continue

        # Load data
        stat = np.load(stat_file, allow_pickle=True)
        iscell = np.load(iscell_file)
        ops = np.load(ops_file, allow_pickle=True).item()

        # Add plane number to each stat entry
        for s in stat:
            s['iplane'] = plane_num

        all_stats.extend(stat)
        all_iscell.append(iscell)
        all_ops.append(ops)
        n_cells_per_plane.append(len(stat))

        # Load optional trace files
        F_file = plane_dir / "F.npy"
        Fneu_file = plane_dir / "Fneu.npy"
        spks_file = plane_dir / "spks.npy"

        if F_file.exists():
            F = np.load(F_file)
            all_F.append(F)
        else:
            print(f"    WARNING: F.npy not found for plane {plane_num}")

        if Fneu_file.exists():
            Fneu = np.load(Fneu_file)
            all_Fneu.append(Fneu)
        else:
            print(f"    WARNING: Fneu.npy not found for plane {plane_num}")

        if spks_file.exists():
            spks = np.load(spks_file)
            all_spks.append(spks)
        else:
            print(f"    WARNING: spks.npy not found for plane {plane_num}")

    # Save consolidated files
    print(f"\nSaving consolidated results to {merged_path}")

    # Save stat.npy
    stat_array = np.array(all_stats, dtype=object)
    np.save(merged_path / "stat.npy", stat_array)
    print(f"  Saved stat.npy: {len(stat_array)} total ROIs")

    # Save iscell.npy
    if all_iscell:
        iscell_array = np.vstack(all_iscell)
        np.save(merged_path / "iscell.npy", iscell_array)
        n_accepted = (iscell_array[:, 0] == 1).sum()
        print(f"  Saved iscell.npy: {n_accepted} accepted, {len(iscell_array) - n_accepted} rejected")

    # Save trace files
    if all_F:
        F_array = np.vstack(all_F)
        np.save(merged_path / "F.npy", F_array)
        print(f"  Saved F.npy: shape {F_array.shape}")

    if all_Fneu:
        Fneu_array = np.vstack(all_Fneu)
        np.save(merged_path / "Fneu.npy", Fneu_array)
        print(f"  Saved Fneu.npy: shape {Fneu_array.shape}")

    if all_spks:
        spks_array = np.vstack(all_spks)
        np.save(merged_path / "spks.npy", spks_array)
        print(f"  Saved spks.npy: shape {spks_array.shape}")

    # Save consolidated ops
    # Use first plane's ops as template and add plane-specific info
    consolidated_ops = all_ops[0].copy() if all_ops else {}
    consolidated_ops['nplanes'] = len(plane_dirs)
    consolidated_ops['n_cells_per_plane'] = n_cells_per_plane
    consolidated_ops['plane_dirs'] = [str(d) for d in plane_dirs]
    consolidated_ops['save_path'] = str(merged_path)

    np.save(merged_path / "ops.npy", consolidated_ops)
    print(f"  Saved ops.npy: {len(plane_dirs)} planes consolidated")

    print(f"\nConsolidation complete!")
    print(f"Total ROIs: {len(stat_array)}")
    print(f"Planes: {len(plane_dirs)}")

    return merged_path


def plot_volume_diagnostics(
    ops_files: list[str | Path],
    save_path: str | Path = None,
    figsize: tuple = (16, 12),
) -> plt.Figure:
    """
    Generate a single-figure diagnostic summary for an entire processed volume.

    Creates a publication-quality figure showing across all z-planes:
    - Row 1: ROI counts (accepted/rejected stacked bars), Mean signal per plane
    - Row 2: SNR distribution per plane, Size distribution per plane
    - Row 3: Compactness vs SNR (all planes), Skewness vs SNR (all planes)

    Parameters
    ----------
    ops_files : list of str or Path
        List of paths to ops.npy files for each z-plane.
    save_path : str or Path, optional
        If provided, save figure to this path.
    figsize : tuple, default (16, 12)
        Figure size in inches.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object.
    """
    from lbm_suite2p_python.postprocessing import load_ops

    if not ops_files:
        fig = plt.figure(figsize=figsize, facecolor="black")
        fig.text(0.5, 0.5, "No ops files provided", ha="center", va="center",
                fontsize=16, fontweight="bold", color="white")
        return fig

    # Collect data from all planes
    plane_data = []
    all_snr = []
    all_npix = []
    all_compactness = []
    all_skewness = []
    all_plane_ids = []  # track which plane each ROI belongs to

    for ops_file in ops_files:
        ops_file = Path(ops_file)
        ops = load_ops(ops_file)

        plane_dir = ops_file.parent
        raw_plane = ops.get("plane", None)

        # Extract plane number
        if raw_plane is not None:
            if isinstance(raw_plane, (int, np.integer)):
                plane_num = int(raw_plane)
            else:
                s = str(raw_plane)
                digits = "".join([c for c in s if c.isdigit()])
                plane_num = int(digits) if digits else len(plane_data)
        else:
            plane_num = len(plane_data)

        # Load required files
        iscell_file = plane_dir / "iscell.npy"
        stat_file = plane_dir / "stat.npy"
        F_file = plane_dir / "F.npy"
        Fneu_file = plane_dir / "Fneu.npy"

        if not all([iscell_file.exists(), stat_file.exists(), F_file.exists()]):
            continue

        try:
            iscell_raw = np.load(iscell_file, allow_pickle=True)
            if not isinstance(iscell_raw, np.ndarray) or iscell_raw.size == 0:
                continue
            iscell = iscell_raw[:, 0].astype(bool)

            stat = np.load(stat_file, allow_pickle=True)
            F = np.load(F_file, allow_pickle=True)
            Fneu = np.load(Fneu_file, allow_pickle=True) if Fneu_file.exists() else np.zeros_like(F)
        except Exception:
            continue

        n_accepted = int(np.sum(iscell))
        n_rejected = int(np.sum(~iscell))

        # Compute SNR for accepted cells
        F_corr = F - 0.7 * Fneu
        baseline = np.percentile(F_corr, 20, axis=1, keepdims=True)
        baseline = np.maximum(baseline, 1e-6)
        dff = (F_corr - baseline) / baseline

        signal = np.std(dff, axis=1)
        noise = np.median(np.abs(np.diff(dff, axis=1)), axis=1) / 0.6745
        snr = signal / (noise + 1e-6)

        # Extract ROI properties
        npix = np.array([s.get("npix", 0) for s in stat])
        compactness = np.array([s.get("compact", np.nan) for s in stat])
        skewness = np.array([s.get("skew", np.nan) for s in stat])

        # Store per-plane stats
        mean_signal = float(np.nanmean(F))
        std_signal = float(np.nanstd(F))
        mean_snr = float(np.nanmean(snr[iscell])) if n_accepted > 0 else 0.0

        plane_data.append({
            "plane": plane_num,
            "n_accepted": n_accepted,
            "n_rejected": n_rejected,
            "mean_signal": mean_signal,
            "std_signal": std_signal,
            "mean_snr": mean_snr,
        })

        # Collect accepted cell data for scatter plots
        if n_accepted > 0:
            all_snr.extend(snr[iscell])
            all_npix.extend(npix[iscell])
            all_compactness.extend(compactness[iscell])
            all_skewness.extend(skewness[iscell])
            all_plane_ids.extend([plane_num] * n_accepted)

    if not plane_data:
        fig = plt.figure(figsize=figsize, facecolor="black")
        fig.text(0.5, 0.5, "No valid plane data found\n\nCheck that ops.npy, stat.npy, F.npy exist",
                ha="center", va="center", fontsize=14, fontweight="bold", color="white")
        return fig

    # Convert to arrays
    planes = np.array([d["plane"] for d in plane_data])
    n_accepted = np.array([d["n_accepted"] for d in plane_data])
    n_rejected = np.array([d["n_rejected"] for d in plane_data])
    mean_signals = np.array([d["mean_signal"] for d in plane_data])
    std_signals = np.array([d["std_signal"] for d in plane_data])
    mean_snrs = np.array([d["mean_snr"] for d in plane_data])

    all_snr = np.array(all_snr)
    all_npix = np.array(all_npix)
    all_compactness = np.array(all_compactness)
    all_skewness = np.array(all_skewness)
    all_plane_ids = np.array(all_plane_ids)

    # Create figure with 3x2 grid
    fig = plt.figure(figsize=figsize, facecolor="black")
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25,
                          left=0.08, right=0.95, top=0.93, bottom=0.08)

    # Color palette for planes
    n_planes = len(planes)
    cmap = plt.cm.viridis
    plane_colors = {p: cmap(i / max(1, n_planes - 1)) for i, p in enumerate(planes)}

    # Panel 1: ROI counts per plane
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("black")

    bar_width = 0.8
    bars1 = ax1.bar(planes, n_accepted, width=bar_width, label=f"Accepted ({n_accepted.sum()})",
                   alpha=0.85, color="#2ecc71", edgecolor="#27ae60", linewidth=0.5)
    bars2 = ax1.bar(planes, n_rejected, width=bar_width, bottom=n_accepted,
                   label=f"Rejected ({n_rejected.sum()})", alpha=0.85, color="#e74c3c",
                   edgecolor="#c0392b", linewidth=0.5)

    # Labels inside bars
    for bar in bars1:
        h = bar.get_height()
        if h > 5:
            ax1.text(bar.get_x() + bar.get_width()/2, h/2, f"{int(h)}",
                    ha="center", va="center", fontsize=7, color="white", fontweight="bold")
    for b1, b2 in zip(bars1, bars2):
        h1, h2 = b1.get_height(), b2.get_height()
        if h2 > 5:
            ax1.text(b2.get_x() + b2.get_width()/2, h1 + h2/2, f"{int(h2)}",
                    ha="center", va="center", fontsize=7, color="white", fontweight="bold")

    ax1.set_xlabel("Z-Plane", fontsize=9, fontweight="bold", color="white")
    ax1.set_ylabel("Number of ROIs", fontsize=9, fontweight="bold", color="white")
    ax1.set_title("ROI Counts per Plane", fontsize=10, fontweight="bold", color="white")
    ax1.tick_params(colors="white", labelsize=8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["bottom"].set_color("white")
    ax1.spines["left"].set_color("white")
    ax1.legend(fontsize=7, facecolor="#1a1a1a", edgecolor="white", labelcolor="white", loc="upper right")

    # Panel 2: mean signal per plane
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("black")

    ax2.errorbar(planes, mean_signals, yerr=std_signals, fmt="o-",
                color="#3498db", ecolor="#85c1e9", elinewidth=1.5,
                capsize=3, markersize=5, alpha=0.9, label="Mean ± STD")

    ax2.set_xlabel("Z-Plane", fontsize=9, fontweight="bold", color="white")
    ax2.set_ylabel("Mean Raw Signal", fontsize=9, fontweight="bold", color="white")
    ax2.set_title("Fluorescence Signal per Plane", fontsize=10, fontweight="bold", color="white")
    ax2.tick_params(colors="white", labelsize=8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["bottom"].set_color("white")
    ax2.spines["left"].set_color("white")
    ax2.legend(fontsize=7, facecolor="#1a1a1a", edgecolor="white", labelcolor="white")

    # Panel 3: SNR distribution per plane
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor("black")

    if len(all_snr) > 0:
        # Box plot per plane
        snr_by_plane = [all_snr[all_plane_ids == p] for p in planes]
        snr_by_plane = [s[~np.isnan(s)] for s in snr_by_plane]

        bp = ax3.boxplot(snr_by_plane, positions=planes, widths=0.6, patch_artist=True,
                        showfliers=False, medianprops=dict(color="#ffe66d", linewidth=2))
        for patch in bp["boxes"]:
            patch.set_facecolor("#2ecc71")
            patch.set_alpha(0.7)
        for whisker in bp["whiskers"]:
            whisker.set_color("white")
        for cap in bp["caps"]:
            cap.set_color("white")

        # Add mean line
        ax3.plot(planes, mean_snrs, "o--", color="#e74c3c", markersize=4, label="Mean SNR")
    else:
        ax3.text(0.5, 0.5, "No SNR data", ha="center", va="center", fontsize=12, color="white")

    ax3.set_xlabel("Z-Plane", fontsize=9, fontweight="bold", color="white")
    ax3.set_ylabel("SNR", fontsize=9, fontweight="bold", color="white")
    ax3.set_title("SNR Distribution per Plane", fontsize=10, fontweight="bold", color="white")
    ax3.tick_params(colors="white", labelsize=8)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.spines["bottom"].set_color("white")
    ax3.spines["left"].set_color("white")
    if len(all_snr) > 0:
        ax3.legend(fontsize=7, facecolor="#1a1a1a", edgecolor="white", labelcolor="white")

    # Panel 4: size distribution per plane
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor("black")

    if len(all_npix) > 0:
        npix_by_plane = [all_npix[all_plane_ids == p] for p in planes]
        npix_by_plane = [n[n > 0] for n in npix_by_plane]

        bp4 = ax4.boxplot(npix_by_plane, positions=planes, widths=0.6, patch_artist=True,
                         showfliers=False, medianprops=dict(color="#ffe66d", linewidth=2))
        for patch in bp4["boxes"]:
            patch.set_facecolor("#3498db")
            patch.set_alpha(0.7)
        for whisker in bp4["whiskers"]:
            whisker.set_color("white")
        for cap in bp4["caps"]:
            cap.set_color("white")

        # Mean size per plane
        mean_sizes = [np.mean(n) if len(n) > 0 else 0 for n in npix_by_plane]
        ax4.plot(planes, mean_sizes, "o--", color="#e74c3c", markersize=4, label="Mean Size")
    else:
        ax4.text(0.5, 0.5, "No size data", ha="center", va="center", fontsize=12, color="white")

    ax4.set_xlabel("Z-Plane", fontsize=9, fontweight="bold", color="white")
    ax4.set_ylabel("Size (pixels)", fontsize=9, fontweight="bold", color="white")
    ax4.set_title("ROI Size Distribution per Plane", fontsize=10, fontweight="bold", color="white")
    ax4.tick_params(colors="white", labelsize=8)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)
    ax4.spines["bottom"].set_color("white")
    ax4.spines["left"].set_color("white")
    if len(all_npix) > 0:
        ax4.legend(fontsize=7, facecolor="#1a1a1a", edgecolor="white", labelcolor="white")

    # Panel 5: compactness distribution per plane
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.set_facecolor("black")

    if len(all_compactness) > 0:
        compact_by_plane = [all_compactness[all_plane_ids == p] for p in planes]
        compact_by_plane = [c[~np.isnan(c)] for c in compact_by_plane]

        # Only create boxplot if we have data
        valid_compact = [c for c in compact_by_plane if len(c) > 0]
        valid_planes_compact = [p for p, c in zip(planes, compact_by_plane) if len(c) > 0]

        if valid_compact:
            bp5 = ax5.boxplot(valid_compact, positions=valid_planes_compact, widths=0.6, patch_artist=True,
                             showfliers=False, medianprops=dict(color="#ffe66d", linewidth=2))
            for patch in bp5["boxes"]:
                patch.set_facecolor("#9b59b6")  # Purple for compactness
                patch.set_alpha(0.7)
            for whisker in bp5["whiskers"]:
                whisker.set_color("white")
            for cap in bp5["caps"]:
                cap.set_color("white")

            # Mean compactness per plane
            mean_compact = [np.mean(c) if len(c) > 0 else np.nan for c in compact_by_plane]
            valid_mean_compact = [m for m, c in zip(mean_compact, compact_by_plane) if len(c) > 0]
            ax5.plot(valid_planes_compact, valid_mean_compact, "o--", color="#e74c3c", markersize=4, label="Mean")
            ax5.legend(fontsize=7, facecolor="#1a1a1a", edgecolor="white", labelcolor="white")
        else:
            ax5.text(0.5, 0.5, "No compactness data", ha="center", va="center", fontsize=12, color="white")
    else:
        ax5.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12, color="white")

    ax5.set_xlabel("Z-Plane", fontsize=9, fontweight="bold", color="white")
    ax5.set_ylabel("Compactness", fontsize=9, fontweight="bold", color="white")
    ax5.set_title("Compactness Distribution per Plane", fontsize=10, fontweight="bold", color="white")
    ax5.tick_params(colors="white", labelsize=8)
    ax5.spines["top"].set_visible(False)
    ax5.spines["right"].set_visible(False)
    ax5.spines["bottom"].set_color("white")
    ax5.spines["left"].set_color("white")

    # Panel 6: skewness distribution per plane
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.set_facecolor("black")

    if len(all_skewness) > 0:
        skew_by_plane = [all_skewness[all_plane_ids == p] for p in planes]
        skew_by_plane = [s[~np.isnan(s)] for s in skew_by_plane]

        # Only create boxplot if we have data
        valid_skew = [s for s in skew_by_plane if len(s) > 0]
        valid_planes_skew = [p for p, s in zip(planes, skew_by_plane) if len(s) > 0]

        if valid_skew:
            bp6 = ax6.boxplot(valid_skew, positions=valid_planes_skew, widths=0.6, patch_artist=True,
                             showfliers=False, medianprops=dict(color="#ffe66d", linewidth=2))
            for patch in bp6["boxes"]:
                patch.set_facecolor("#e67e22")  # Orange for skewness
                patch.set_alpha(0.7)
            for whisker in bp6["whiskers"]:
                whisker.set_color("white")
            for cap in bp6["caps"]:
                cap.set_color("white")

            # Mean skewness per plane
            mean_skew = [np.mean(s) if len(s) > 0 else np.nan for s in skew_by_plane]
            valid_mean_skew = [m for m, s in zip(mean_skew, skew_by_plane) if len(s) > 0]
            ax6.plot(valid_planes_skew, valid_mean_skew, "o--", color="#e74c3c", markersize=4, label="Mean")
            ax6.legend(fontsize=7, facecolor="#1a1a1a", edgecolor="white", labelcolor="white")
        else:
            ax6.text(0.5, 0.5, "No skewness data", ha="center", va="center", fontsize=12, color="white")
    else:
        ax6.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12, color="white")

    ax6.set_xlabel("Z-Plane", fontsize=9, fontweight="bold", color="white")
    ax6.set_ylabel("Skewness", fontsize=9, fontweight="bold", color="white")
    ax6.set_title("Skewness Distribution per Plane", fontsize=10, fontweight="bold", color="white")
    ax6.tick_params(colors="white", labelsize=8)
    ax6.spines["top"].set_visible(False)
    ax6.spines["right"].set_visible(False)
    ax6.spines["bottom"].set_color("white")
    ax6.spines["left"].set_color("white")

    # Title
    total_accepted = n_accepted.sum()
    total_rejected = n_rejected.sum()
    fig.suptitle(f"Volume Quality Diagnostics: {n_planes} planes, {total_accepted} accepted, {total_rejected} rejected ROIs",
                fontsize=12, fontweight="bold", color="white", y=0.98)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="black")
        plt.close(fig)

    return fig


def plot_orthoslices(
    ops_files: list[str | Path],
    save_path: str | Path = None,
    figsize: tuple = (16, 6),
    use_mean: bool = True,
) -> plt.Figure:
    """
    Generate orthogonal maximum intensity projections (XY, XZ, YZ) of the volume.

    Creates a 3-panel figure showing the volume from three orthogonal views,
    with proper interpolation to isotropic resolution. Axes are displayed
    in micrometers using voxel size metadata.

    Parameters
    ----------
    ops_files : list of str or Path
        List of paths to ops.npy files for each z-plane, ordered by z.
    save_path : str or Path, optional
        If provided, save figure to this path.
    figsize : tuple, default (16, 6)
        Figure size in inches.
    use_mean : bool, default True
        If True, use meanImg. If False, use refImg (registered reference).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object.
    """
    from scipy.ndimage import zoom
    from lbm_suite2p_python.postprocessing import load_ops

    if not ops_files:
        fig = plt.figure(figsize=figsize, facecolor="black")
        fig.text(0.5, 0.5, "No ops files provided", ha="center", va="center",
                fontsize=16, fontweight="bold", color="white")
        return fig

    # get voxel size from first ops file
    first_ops = load_ops(ops_files[0])
    dx_um, dy_um, dz_um = 1.0, 1.0, 15.0  # defaults (15um is typical z-step)
    try:
        from mbo_utilities.metadata import get_voxel_size
        voxel = get_voxel_size(first_ops)
        # use voxel sizes if they're not the default 1.0 placeholder
        if voxel.dx != 1.0:
            dx_um = voxel.dx
        if voxel.dy != 1.0:
            dy_um = voxel.dy
        # dz can be None for LBM stacks (user must supply)
        if voxel.dz is not None and voxel.dz != 1.0:
            dz_um = voxel.dz
    except (ImportError, Exception):
        pass
    # fallback to ops fields if still default
    if dx_um == 1.0:
        pixel_res = first_ops.get("pixel_resolution")
        if pixel_res is not None:
            if isinstance(pixel_res, (int, float)):
                dx_um = dy_um = float(pixel_res)
            elif len(pixel_res) >= 2:
                dx_um, dy_um = float(pixel_res[0]), float(pixel_res[1])
    if dz_um == 15.0:
        dz_from_ops = first_ops.get("dz") or first_ops.get("z_step")
        if dz_from_ops is not None and dz_from_ops != 1.0:
            dz_um = float(dz_from_ops)

    # collect images from all planes
    images = []
    for ops_file in ops_files:
        ops_file = Path(ops_file)
        ops = load_ops(ops_file)

        img_key = "meanImg" if use_mean else "refImg"
        img = ops.get(img_key)
        if img is None or not isinstance(img, np.ndarray):
            img = ops.get("meanImg" if not use_mean else "refImg")
        if img is None or not isinstance(img, np.ndarray):
            continue
        images.append(img)

    if not images:
        fig = plt.figure(figsize=figsize, facecolor="black")
        fig.text(0.5, 0.5, "No valid images found", ha="center", va="center",
                fontsize=16, fontweight="bold", color="white")
        return fig

    # stack into 3D volume (Z, Y, X)
    volume = np.stack(images, axis=0)
    nz, ny, nx = volume.shape

    # compute volume dimensions in microns
    vol_x_um = nx * dx_um
    vol_y_um = ny * dy_um
    vol_z_um = (nz - 1) * dz_um if nz > 1 else dz_um

    # interpolate volume to isotropic resolution for proper orthoslices
    xy_res = (dx_um + dy_um) / 2
    z_zoom = dz_um / xy_res if xy_res > 0 else 1.0
    z_zoom = min(z_zoom, 10.0)  # cap to avoid memory issues
    if z_zoom > 1.1:
        volume_resampled = zoom(volume, (z_zoom, 1, 1), order=1)
    else:
        volume_resampled = volume

    # compute projections
    xy_proj = np.max(volume, axis=0)
    xz_proj = np.max(volume_resampled, axis=1)
    yz_proj = np.max(volume_resampled, axis=2)

    # create figure
    fig = plt.figure(figsize=figsize, facecolor="black")
    gs = fig.add_gridspec(1, 3, wspace=0.15, left=0.05, right=0.95, top=0.88, bottom=0.1)

    # extents for proper axis scaling (all in microns)
    xy_extent = [0, vol_x_um, vol_y_um, 0]
    xz_extent = [0, vol_x_um, vol_z_um, 0]
    yz_extent = [0, vol_z_um, vol_y_um, 0]

    # panel 1: XY projection
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("black")
    im1 = ax1.imshow(xy_proj, cmap="magma", aspect="equal", extent=xy_extent,
                     vmin=np.percentile(xy_proj, 1), vmax=np.percentile(xy_proj, 99.5))
    ax1.set_xlabel("X (μm)", fontsize=10, fontweight="bold", color="white")
    ax1.set_ylabel("Y (μm)", fontsize=10, fontweight="bold", color="white")
    ax1.set_title("XY Projection", fontsize=11, fontweight="bold", color="white")
    ax1.tick_params(colors="white", labelsize=8)
    for spine in ax1.spines.values():
        spine.set_color("white")

    # panel 2: XZ projection
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("black")
    im2 = ax2.imshow(xz_proj, cmap="magma", aspect="auto", extent=xz_extent,
                     vmin=np.percentile(xz_proj, 1), vmax=np.percentile(xz_proj, 99.5),
                     interpolation="bilinear")
    ax2.set_xlabel("X (μm)", fontsize=10, fontweight="bold", color="white")
    ax2.set_ylabel("Z (μm)", fontsize=10, fontweight="bold", color="white")
    ax2.set_title("XZ Projection", fontsize=11, fontweight="bold", color="white")
    ax2.tick_params(colors="white", labelsize=8)
    for spine in ax2.spines.values():
        spine.set_color("white")

    # panel 3: YZ projection
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor("black")
    im3 = ax3.imshow(yz_proj.T, cmap="magma", aspect="auto", extent=yz_extent,
                     vmin=np.percentile(yz_proj, 1), vmax=np.percentile(yz_proj, 99.5),
                     interpolation="bilinear")
    ax3.set_xlabel("Z (μm)", fontsize=10, fontweight="bold", color="white")
    ax3.set_ylabel("Y (μm)", fontsize=10, fontweight="bold", color="white")
    ax3.set_title("YZ Projection", fontsize=11, fontweight="bold", color="white")
    ax3.tick_params(colors="white", labelsize=8)
    for spine in ax3.spines.values():
        spine.set_color("white")

    # colorbar
    cbar = fig.colorbar(im1, ax=[ax1, ax2, ax3], shrink=0.6, pad=0.02, location="right")
    cbar.set_label("Intensity (a.u.)", fontsize=10, color="white")
    cbar.ax.tick_params(colors="white")
    cbar.outline.set_edgecolor("white")

    # title with volume dimensions in microns
    title = f"Orthogonal Projections: {nz} planes, {vol_x_um:.0f}×{vol_y_um:.0f}×{vol_z_um:.0f} μm"
    fig.suptitle(title, fontsize=12, fontweight="bold", color="white", y=0.96)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="black")
        plt.close(fig)

    return fig


def plot_3d_roi_map(
    ops_files: list[str | Path],
    save_path: str | Path = None,
    figsize: tuple = (14, 10),
    color_by: str = "snr",
    show_rejected: bool = False,
) -> plt.Figure:
    """
    Generate a 3D scatter plot of ROI centroids across the volume.

    Creates a 3D visualization showing the spatial distribution of detected
    cells colored by SNR. Axes are displayed in micrometers when valid voxel
    size metadata is available, otherwise in pixels/planes.

    Parameters
    ----------
    ops_files : list of str or Path
        List of paths to ops.npy files for each z-plane.
    save_path : str or Path, optional
        If provided, save figure to this path.
    figsize : tuple, default (14, 10)
        Figure size in inches.
    color_by : str, default "snr"
        How to color the ROIs: "snr", "plane", "size", or "activity".
    show_rejected : bool, default False
        If True, also show rejected ROIs in gray.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object.
    """
    from lbm_suite2p_python.postprocessing import load_ops

    if not ops_files:
        fig = plt.figure(figsize=figsize, facecolor="black")
        fig.text(0.5, 0.5, "No ops files provided", ha="center", va="center",
                fontsize=16, fontweight="bold", color="white")
        return fig

    # Get voxel size from first ops file
    first_ops = load_ops(ops_files[0])
    dx_um, dy_um, dz_um = 1.0, 1.0, 15.0  # defaults (15um is typical z-step)
    try:
        from mbo_utilities.metadata import get_voxel_size
        voxel = get_voxel_size(first_ops)
        # use voxel sizes if they're not the default 1.0 placeholder
        if voxel.dx != 1.0:
            dx_um = voxel.dx
        if voxel.dy != 1.0:
            dy_um = voxel.dy
        # dz can be None for LBM stacks (user must supply)
        if voxel.dz is not None and voxel.dz != 1.0:
            dz_um = voxel.dz
    except (ImportError, Exception):
        pass
    # fallback to ops fields if still default
    if dx_um == 1.0:
        pixel_res = first_ops.get("pixel_resolution")
        if pixel_res is not None:
            if isinstance(pixel_res, (int, float)):
                dx_um = dy_um = float(pixel_res)
            elif len(pixel_res) >= 2:
                dx_um, dy_um = float(pixel_res[0]), float(pixel_res[1])
    if dz_um == 15.0:
        dz_from_ops = first_ops.get("dz") or first_ops.get("z_step")
        if dz_from_ops is not None and dz_from_ops != 1.0:
            dz_um = float(dz_from_ops)

    # Collect ROI data from all planes
    all_x = []
    all_y = []
    all_z = []
    all_colors = []
    all_accepted = []

    # For rejected ROIs
    rej_x, rej_y, rej_z = [], [], []

    for plane_idx, ops_file in enumerate(ops_files):
        ops_file = Path(ops_file)
        ops = load_ops(ops_file)
        plane_dir = ops_file.parent

        # Use enumeration index for z-depth (planes are ordered)
        plane_num = plane_idx

        # Load required files
        stat_file = plane_dir / "stat.npy"
        iscell_file = plane_dir / "iscell.npy"

        if not stat_file.exists() or not iscell_file.exists():
            continue

        try:
            stat = np.load(stat_file, allow_pickle=True)
            iscell_raw = np.load(iscell_file, allow_pickle=True)
            if not isinstance(iscell_raw, np.ndarray) or iscell_raw.size == 0:
                continue
            iscell = iscell_raw[:, 0].astype(bool)
        except Exception:
            continue

        # Get color values based on color_by
        z_um = plane_num * dz_um  # z-depth in microns for this plane
        if color_by == "snr":
            F_file = plane_dir / "F.npy"
            Fneu_file = plane_dir / "Fneu.npy"
            if F_file.exists():
                F = np.load(F_file, allow_pickle=True)
                Fneu = np.load(Fneu_file, allow_pickle=True) if Fneu_file.exists() else np.zeros_like(F)
                F_corr = F - 0.7 * Fneu
                baseline = np.percentile(F_corr, 20, axis=1, keepdims=True)
                baseline = np.maximum(baseline, 1e-6)
                dff = (F_corr - baseline) / baseline
                signal = np.std(dff, axis=1)
                noise = np.median(np.abs(np.diff(dff, axis=1)), axis=1) / 0.6745
                color_vals = signal / (noise + 1e-6)
            else:
                # no F data - use zeros
                color_vals = np.zeros(len(stat))
        elif color_by == "size":
            color_vals = np.array([s.get("npix", 100) for s in stat])
        elif color_by == "activity":
            F_file = plane_dir / "F.npy"
            Fneu_file = plane_dir / "Fneu.npy"
            if F_file.exists():
                F = np.load(F_file, allow_pickle=True)
                Fneu = np.load(Fneu_file, allow_pickle=True) if Fneu_file.exists() else np.zeros_like(F)
                F_corr = F - 0.7 * Fneu
                # compute zscore: (x - mean) / std
                mean_f = np.mean(F_corr, axis=1, keepdims=True)
                std_f = np.std(F_corr, axis=1, keepdims=True)
                std_f = np.maximum(std_f, 1e-6)
                zscore = (F_corr - mean_f) / std_f
                # use max zscore as activity metric
                color_vals = np.max(zscore, axis=1)
            else:
                # no F data - use zeros
                color_vals = np.zeros(len(stat))
        else:  # plane - use z-depth in microns
            color_vals = np.ones(len(stat)) * z_um

        # Extract centroids and convert to microns
        for i, s in enumerate(stat):
            med = s.get("med", [0, 0])
            y_px, x_px = med[0], med[1]
            # Convert pixels to microns
            x_um = x_px * dx_um
            y_um = y_px * dy_um
            # z_um already calculated above as plane_num * dz_um

            if iscell[i]:
                all_x.append(x_um)
                all_y.append(y_um)
                all_z.append(z_um)
                all_colors.append(color_vals[i])
                all_accepted.append(True)
            elif show_rejected:
                rej_x.append(x_um)
                rej_y.append(y_um)
                rej_z.append(z_um)

    if not all_x:
        fig = plt.figure(figsize=figsize, facecolor="black")
        fig.text(0.5, 0.5, "No ROI data found", ha="center", va="center",
                fontsize=16, fontweight="bold", color="white")
        return fig

    # Convert to arrays
    all_x = np.array(all_x)
    all_y = np.array(all_y)
    all_z = np.array(all_z)
    all_colors = np.array(all_colors)

    # Create figure with 3D axis
    fig = plt.figure(figsize=figsize, facecolor="black")
    ax = fig.add_subplot(111, projection="3d", facecolor="black")

    # Set pane colors to dark
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("white")
    ax.yaxis.pane.set_edgecolor("white")
    ax.zaxis.pane.set_edgecolor("white")

    # Plot rejected ROIs first (if enabled)
    if show_rejected and rej_x:
        ax.scatter(rej_x, rej_y, rej_z, c="gray", s=10, alpha=0.3, label="Rejected")

    # Choose colormap based on color_by
    if color_by == "plane":
        cmap = "viridis"
        clabel = "Z-depth (μm)"
    elif color_by == "snr":
        cmap = "plasma"
        clabel = "SNR"
        # Clip extreme values
        vmin, vmax = np.percentile(all_colors, [5, 95])
        all_colors = np.clip(all_colors, vmin, vmax)
    elif color_by == "size":
        cmap = "cividis"
        clabel = "Size (pixels)"
        vmin, vmax = np.percentile(all_colors, [5, 95])
        all_colors = np.clip(all_colors, vmin, vmax)
    else:  # activity
        cmap = "magma"
        clabel = "Z-score"
        vmin, vmax = np.percentile(all_colors, [5, 95])
        all_colors = np.clip(all_colors, vmin, vmax)

    # Plot accepted ROIs
    scatter = ax.scatter(all_x, all_y, all_z, c=all_colors, cmap=cmap,
                        s=15, alpha=0.7, edgecolors="none")

    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label(clabel, fontsize=10, color="white")
    cbar.ax.tick_params(colors="white")
    cbar.outline.set_edgecolor("white")

    # Style axes - always show in microns
    ax.set_xlabel("X (μm)", fontsize=10, fontweight="bold", color="white", labelpad=10)
    ax.set_ylabel("Y (μm)", fontsize=10, fontweight="bold", color="white", labelpad=10)
    ax.set_zlabel("Z (μm)", fontsize=10, fontweight="bold", color="white", labelpad=10)

    ax.tick_params(colors="white", labelsize=8)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.zaxis.label.set_color("white")

    # Set grid color
    ax.xaxis._axinfo["grid"]["color"] = (1, 1, 1, 0.2)
    ax.yaxis._axinfo["grid"]["color"] = (1, 1, 1, 0.2)
    ax.zaxis._axinfo["grid"]["color"] = (1, 1, 1, 0.2)

    # Title with volume dimensions in microns
    n_cells = len(all_x)
    n_planes = len(np.unique(all_z))
    x_range = all_x.max() - all_x.min()
    y_range = all_y.max() - all_y.min()
    z_range = all_z.max() - all_z.min()

    fig.suptitle(
        f"3D ROI Distribution: {n_cells} cells across {n_planes} planes\n"
        f"Volume: {x_range:.0f} × {y_range:.0f} × {z_range:.0f} μm",
        fontsize=12, fontweight="bold", color="white", y=0.95
    )

    if show_rejected and rej_x:
        ax.legend(fontsize=9, facecolor="#1a1a1a", edgecolor="white", labelcolor="white")

    # Adjust view angle for better visualization
    ax.view_init(elev=20, azim=45)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="black")
        plt.close(fig)

    return fig


def plot_3d_rastermap_clusters(
    suite2p_path: str | Path,
    save_path: str | Path = None,
    figsize: tuple = (14, 10),
    n_clusters: int = 40,
    show_rejected: bool = False,
    rastermap_kwargs: dict = None,
) -> plt.Figure:
    """
    Generate a 3D scatter plot of ROI centroids colored by rastermap cluster.

    Loads volumetric suite2p output, runs rastermap clustering on fluorescence
    data, and visualizes ROI positions in 3D colored by cluster assignment.
    Checks for existing rastermap model file before running.

    Parameters
    ----------
    suite2p_path : str or Path
        Path to suite2p directory containing plane*_stitched folders or merged folder.
    save_path : str or Path, optional
        If provided, save figure to this path.
    figsize : tuple, default (14, 10)
        Figure size in inches.
    n_clusters : int, default 40
        Number of rastermap clusters. Ignored if loading existing model.
    show_rejected : bool, default False
        If True, also show rejected ROIs in gray.
    rastermap_kwargs : dict, optional
        Additional kwargs passed to Rastermap(). Ignored if loading existing model.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object.

    Notes
    -----
    Looks for existing rastermap model at:
    - suite2p_path/rastermap_model.npy
    - suite2p_path/merged/rastermap_model.npy

    If not found, runs rastermap on the consolidated fluorescence data
    and saves the model for future use.

    Examples
    --------
    >>> import lbm_suite2p_python as lsp
    >>> fig = lsp.plot_3d_rastermap_clusters("path/to/suite2p")
    >>> fig = lsp.plot_3d_rastermap_clusters("path/to/suite2p", n_clusters=50)
    """
    import logging
    from lbm_suite2p_python.postprocessing import load_ops

    logger = logging.getLogger(__name__)
    suite2p_path = Path(suite2p_path)

    # check if rastermap is available
    try:
        from rastermap import Rastermap
        has_rastermap = True
    except ImportError:
        has_rastermap = False
        logger.warning("rastermap not installed, cannot generate cluster visualization")

    # find plane directories
    plane_dirs = sorted(suite2p_path.glob("plane*_stitched"))
    merged_dir = suite2p_path / "merged"

    if merged_dir.exists() and (merged_dir / "stat.npy").exists():
        # use merged directory
        data_dirs = [merged_dir]
        use_merged = True
    elif plane_dirs:
        data_dirs = plane_dirs
        use_merged = False
    else:
        fig = plt.figure(figsize=figsize, facecolor="black")
        fig.text(0.5, 0.5, "No plane directories or merged folder found",
                ha="center", va="center", fontsize=14, fontweight="bold", color="white")
        return fig

    # check data is volumetric (4D = multiple planes)
    if not use_merged and len(plane_dirs) < 2:
        logger.warning("Only 1 plane found - this visualization is designed for volumetric (4D) data")

    # look for existing rastermap model
    model_paths = [
        suite2p_path / "rastermap_model.npy",
        merged_dir / "rastermap_model.npy",
    ]
    rastermap_model = None
    for mp in model_paths:
        if mp.exists():
            try:
                rastermap_model = np.load(mp, allow_pickle=True).item()
                logger.info(f"Loaded existing rastermap model from {mp}")
                break
            except Exception as e:
                logger.warning(f"Failed to load rastermap model from {mp}: {e}")

    # get voxel size from first ops file
    first_ops_file = None
    for d in data_dirs:
        ops_file = d / "ops.npy"
        if ops_file.exists():
            first_ops_file = ops_file
            break

    dx_um, dy_um, dz_um = 1.0, 1.0, 15.0
    if first_ops_file:
        first_ops = load_ops(first_ops_file)
        try:
            from mbo_utilities.metadata import get_voxel_size
            voxel = get_voxel_size(first_ops)
            # use voxel sizes if they're not the default 1.0 placeholder
            if voxel.dx != 1.0:
                dx_um = voxel.dx
            if voxel.dy != 1.0:
                dy_um = voxel.dy
            # dz can be None for LBM stacks (user must supply)
            if voxel.dz is not None and voxel.dz != 1.0:
                dz_um = voxel.dz
        except (ImportError, Exception):
            pass
        # fallback to ops fields if still default
        if dx_um == 1.0:
            pixel_res = first_ops.get("pixel_resolution")
            if pixel_res is not None:
                if isinstance(pixel_res, (int, float)):
                    dx_um = dy_um = float(pixel_res)
                elif len(pixel_res) >= 2:
                    dx_um, dy_um = float(pixel_res[0]), float(pixel_res[1])
        if dz_um == 15.0:
            dz_from_ops = first_ops.get("dz") or first_ops.get("z_step")
            if dz_from_ops is not None and dz_from_ops != 1.0:
                dz_um = float(dz_from_ops)

    # collect ROI data and fluorescence
    all_x, all_y, all_z = [], [], []
    all_F = []
    all_iscell = []  # track iscell for filtering saved rastermap
    rej_x, rej_y, rej_z = [], [], []

    if use_merged:
        # load from merged directory
        stat_file = merged_dir / "stat.npy"
        iscell_file = merged_dir / "iscell.npy"
        F_file = merged_dir / "F.npy"
        Fneu_file = merged_dir / "Fneu.npy"

        if not all([stat_file.exists(), iscell_file.exists(), F_file.exists()]):
            fig = plt.figure(figsize=figsize, facecolor="black")
            fig.text(0.5, 0.5, "Missing required files in merged directory",
                    ha="center", va="center", fontsize=14, fontweight="bold", color="white")
            return fig

        stat = np.load(stat_file, allow_pickle=True)
        iscell = np.load(iscell_file)[:, 0].astype(bool)
        F = np.load(F_file)
        Fneu = np.load(Fneu_file) if Fneu_file.exists() else np.zeros_like(F)

        for i, s in enumerate(stat):
            med = s.get("med", [0, 0])
            y_px, x_px = med[0], med[1]
            plane_num = s.get("iplane", 0)

            x_um = x_px * dx_um
            y_um = y_px * dy_um
            z_um = plane_num * dz_um

            if iscell[i]:
                all_x.append(x_um)
                all_y.append(y_um)
                all_z.append(z_um)
            elif show_rejected:
                rej_x.append(x_um)
                rej_y.append(y_um)
                rej_z.append(z_um)

        # neuropil correction
        F_corr = F - 0.7 * Fneu
        all_F = F_corr[iscell]
        all_iscell = iscell

    else:
        # load from individual plane directories
        F_list = []
        iscell_list = []

        for plane_idx, plane_dir in enumerate(plane_dirs):
            stat_file = plane_dir / "stat.npy"
            iscell_file = plane_dir / "iscell.npy"
            F_file = plane_dir / "F.npy"
            Fneu_file = plane_dir / "Fneu.npy"

            if not all([stat_file.exists(), iscell_file.exists(), F_file.exists()]):
                continue

            try:
                stat = np.load(stat_file, allow_pickle=True)
                iscell_raw = np.load(iscell_file)
                iscell = iscell_raw[:, 0].astype(bool)
                F = np.load(F_file)
                Fneu = np.load(Fneu_file) if Fneu_file.exists() else np.zeros_like(F)
            except Exception:
                continue

            z_um = plane_idx * dz_um
            F_corr = F - 0.7 * Fneu
            iscell_list.append(iscell)

            for i, s in enumerate(stat):
                med = s.get("med", [0, 0])
                y_px, x_px = med[0], med[1]
                x_um = x_px * dx_um
                y_um = y_px * dy_um

                if iscell[i]:
                    all_x.append(x_um)
                    all_y.append(y_um)
                    all_z.append(z_um)
                elif show_rejected:
                    rej_x.append(x_um)
                    rej_y.append(y_um)
                    rej_z.append(z_um)

            F_list.append(F_corr[iscell])

        if F_list:
            all_F = np.vstack(F_list)
        if iscell_list:
            all_iscell = np.concatenate(iscell_list)

    if len(all_x) == 0:
        fig = plt.figure(figsize=figsize, facecolor="black")
        fig.text(0.5, 0.5, "No accepted ROIs found",
                ha="center", va="center", fontsize=14, fontweight="bold", color="white")
        return fig

    all_x = np.array(all_x)
    all_y = np.array(all_y)
    all_z = np.array(all_z)

    # run or load rastermap
    cluster_ids = None
    n_actual_clusters = 0

    if rastermap_model is not None:
        # use existing model (handle both dict and Rastermap object)
        if hasattr(rastermap_model, "embedding_clust"):
            embedding_clust = rastermap_model.embedding_clust
        elif isinstance(rastermap_model, dict):
            embedding_clust = rastermap_model.get("embedding_clust", None)
        else:
            embedding_clust = None
        if embedding_clust is not None:
            cluster_ids = embedding_clust.flatten()
            # filter by iscell if model was trained on all ROIs
            if len(cluster_ids) != len(all_x) and len(all_iscell) > 0:
                if len(cluster_ids) == len(all_iscell):
                    cluster_ids = cluster_ids[all_iscell]
                    logger.info(f"Filtered cluster_ids from {len(embedding_clust)} to {len(cluster_ids)} accepted ROIs")
            n_actual_clusters = len(np.unique(cluster_ids[~np.isnan(cluster_ids)]))
            logger.info(f"Using {n_actual_clusters} clusters from saved model")
    elif has_rastermap and len(all_F) > 0:
        # run rastermap
        logger.info(f"Running rastermap with n_clusters={n_clusters}")
        try:
            # zscore the data
            from scipy.stats import zscore
            spks = zscore(all_F.astype("float32"), axis=1)

            # set up rastermap params
            kwargs = {"n_clusters": n_clusters, "n_PCs": min(200, spks.shape[0] - 1), "verbose": False}
            if rastermap_kwargs:
                kwargs.update(rastermap_kwargs)

            model = Rastermap(**kwargs).fit(spks)
            cluster_ids = model.embedding_clust.flatten()
            n_actual_clusters = model.n_clusters

            # save model for future use
            model_save = {
                "embedding": model.embedding,
                "isort": model.isort,
                "embedding_clust": model.embedding_clust,
                "n_clusters": model.n_clusters,
            }
            save_model_path = suite2p_path / "rastermap_model.npy"
            np.save(save_model_path, model_save)
            logger.info(f"Saved rastermap model to {save_model_path}")

        except Exception as e:
            logger.warning(f"Rastermap failed: {e}")
            cluster_ids = None

    # fallback to z-plane coloring if no clusters
    if cluster_ids is None:
        logger.warning("No rastermap clusters available, coloring by z-plane instead")
        cluster_ids = all_z
        n_actual_clusters = len(np.unique(all_z))
        color_label = "Z-depth (μm)"
        cmap = "viridis"
    else:
        color_label = "Cluster"
        cmap = "tab20" if n_actual_clusters <= 20 else "nipy_spectral"

    # create figure
    fig = plt.figure(figsize=figsize, facecolor="black")
    ax = fig.add_subplot(111, projection="3d", facecolor="black")

    # style panes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("white")
    ax.yaxis.pane.set_edgecolor("white")
    ax.zaxis.pane.set_edgecolor("white")

    # plot rejected ROIs first
    if show_rejected and rej_x:
        ax.scatter(rej_x, rej_y, rej_z, c="gray", s=10, alpha=0.3, label="Rejected")

    # plot accepted ROIs colored by cluster
    scatter = ax.scatter(all_x, all_y, all_z, c=cluster_ids, cmap=cmap,
                        s=15, alpha=0.8, edgecolors="none")

    # colorbar
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label(color_label, fontsize=10, color="white")
    cbar.ax.tick_params(colors="white")
    cbar.outline.set_edgecolor("white")

    # style axes
    ax.set_xlabel("X (μm)", fontsize=10, fontweight="bold", color="white", labelpad=10)
    ax.set_ylabel("Y (μm)", fontsize=10, fontweight="bold", color="white", labelpad=10)
    ax.set_zlabel("Z (μm)", fontsize=10, fontweight="bold", color="white", labelpad=10)

    ax.tick_params(colors="white", labelsize=8)
    ax.xaxis._axinfo["grid"]["color"] = (1, 1, 1, 0.2)
    ax.yaxis._axinfo["grid"]["color"] = (1, 1, 1, 0.2)
    ax.zaxis._axinfo["grid"]["color"] = (1, 1, 1, 0.2)

    # title
    n_cells = len(all_x)
    n_planes = len(np.unique(all_z))
    x_range = all_x.max() - all_x.min()
    y_range = all_y.max() - all_y.min()
    z_range = all_z.max() - all_z.min()

    title = f"Rastermap Clusters: {n_cells} cells, {n_actual_clusters} clusters, {n_planes} planes"
    subtitle = f"Volume: {x_range:.0f} × {y_range:.0f} × {z_range:.0f} μm"
    fig.suptitle(f"{title}\n{subtitle}", fontsize=12, fontweight="bold", color="white", y=0.95)

    if show_rejected and rej_x:
        ax.legend(fontsize=9, facecolor="#1a1a1a", edgecolor="white", labelcolor="white")

    ax.view_init(elev=20, azim=45)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="black")
        plt.close(fig)
    else:
        plt.show()

    return fig
