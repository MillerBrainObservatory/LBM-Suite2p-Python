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

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="black")
    ax.set_facecolor("black")

    plt.xlabel("Z-Plane", fontsize=14, fontweight="bold", color="white")
    plt.ylabel("Mean Raw Signal", fontsize=14, fontweight="bold", color="white")
    plt.title(
        "Mean Fluorescence Signal per Z-Plane",
        fontsize=16,
        fontweight="bold",
        color="white",
    )

    plt.errorbar(
        planes,
        mean_signal,
        yerr=std_signal,
        fmt="o-",
        color="cyan",
        ecolor="lightblue",
        elinewidth=2,
        capsize=4,
        markersize=6,
        alpha=0.8,
        label="Mean ± STD",
    )

    plt.xticks(planes, fontsize=12, fontweight="bold", color="white")
    plt.yticks(fontsize=12, fontweight="bold", color="white")

    plt.grid(axis="y", linestyle="--", alpha=0.4, color="white")

    ax.spines["bottom"].set_color("white")
    ax.spines["left"].set_color("white")
    ax.spines["top"].set_color("white")
    ax.spines["right"].set_color("white")

    plt.legend(fontsize=12, facecolor="black", edgecolor="white", labelcolor="white")

    plt.savefig(savepath, bbox_inches="tight", facecolor="black")
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

    plt.figure(figsize=(10, 6), facecolor="black")
    ax = plt.gca()
    ax.set_facecolor("black")

    plt.xlabel("Z-Plane", fontsize=14, fontweight="bold", color="white")
    plt.ylabel("Number of Neurons", fontsize=14, fontweight="bold", color="white")
    plt.title(
        "Accepted vs. Rejected Neurons per Z-Plane",
        fontsize=16,
        fontweight="bold",
        color="white",
    )

    bars1 = plt.bar(
        planes, accepted, label="Accepted Neurons", alpha=0.8, color="#4CAF50"
    )  # Light green
    bars2 = plt.bar(
        planes,
        rejected,
        label="Rejected Neurons",
        alpha=0.8,
        bottom=accepted,
        color="#F57C00",
    )  # Light orange

    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height / 2,
                f"{int(height)}",
                ha="center",
                va="center",
                fontsize=12,
                color="white",
                fontweight="bold",
            )

    for bar1, bar2 in zip(bars1, bars2):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        if height2 > 0:
            plt.text(
                bar2.get_x() + bar2.get_width() / 2,
                height1 + height2 / 2,
                f"{int(height2)}",
                ha="center",
                va="center",
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

    plt.legend(fontsize=12, facecolor="black", edgecolor="white", labelcolor="white")
    plt.savefig(savename, bbox_inches="tight", facecolor="black")


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
            iscell = np.load(iscell_file, allow_pickle=True)[:, 0].astype(bool)
            traces = np.load(traces_file, allow_pickle=True)
        except Exception as e:
            print(f"Skipping plane {zplane_num}: Error loading files - {e}")
            continue

        plane_stats[zplane_num] = {
            "accepted": int(iscell.sum()),
            "rejected": int((~iscell).sum()),
            "mean": float(traces.mean()),
            "std": float(traces.std()),
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
