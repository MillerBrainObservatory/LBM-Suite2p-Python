import glob
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt

from lbm_suite2p_python.zplane import load_ops
from lbm_suite2p_python.utils import get_common_path


def update_ops_paths(ops_files: str | list):
    """
    Update save_path, save_path0, and save_folder in an ops dictionary based on its current location. Use after moving an ops_file or batch of ops_files."""
    if isinstance(ops_files, (str,Path)):
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
    plt.title("Execution Time per Processing Step", fontsize=16, fontweight="bold", color="white")

    plt.bar(planes, reg_time, label="Registration", alpha=0.8, color="#FF5733")
    plt.bar(planes, detect_time, label="Detection", alpha=0.8, bottom=reg_time, color="#33FF57")
    bars3 = plt.bar(planes, extract_time, label="Extraction", alpha=0.8, bottom=reg_time + detect_time, color="#3357FF")

    for bar, total in zip(bars3, total_time):
        height = bar.get_y() + bar.get_height()
        if total > 1:  # Only label if execution time is large enough to be visible
            plt.text(bar.get_x() + bar.get_width() / 2, height + 2, f"{int(total)}",
                     ha="center", va="bottom", fontsize=12, color="white", fontweight="bold")

    plt.xticks(planes, fontsize=12, fontweight="bold", color="white")
    plt.yticks(fontsize=12, fontweight="bold", color="white")

    plt.grid(axis="y", linestyle="--", alpha=0.4, color="white")

    ax.spines["bottom"].set_color("white")
    ax.spines["left"].set_color("white")
    ax.spines["top"].set_color("white")
    ax.spines["right"].set_color("white")

    plt.legend(fontsize=12, facecolor="black", edgecolor="white", labelcolor="white", loc="upper left",
               bbox_to_anchor=(1, 1))

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

    plt.figure(figsize=(10, 6), facecolor="black")
    ax = plt.gca()
    ax.set_facecolor("black")

    plt.xlabel("Z-Plane", fontsize=14, fontweight="bold", color="white")
    plt.ylabel("Mean Raw Signal", fontsize=14, fontweight="bold", color="white")
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
    savename = savepath.joinpath(f"all_neurons_{accepted.sum()}acc_{rejected.sum()}rej.png")

    plt.figure(figsize=(10, 6), facecolor="black")
    ax = plt.gca()
    ax.set_facecolor("black")

    plt.xlabel("Z-Plane", fontsize=14, fontweight="bold", color="white")
    plt.ylabel("Number of Neurons", fontsize=14, fontweight="bold", color="white")
    plt.title("Accepted vs. Rejected Neurons per Z-Plane", fontsize=16, fontweight="bold", color="white")

    bars1 = plt.bar(planes, accepted, label="Accepted Neurons", alpha=0.8, color="#4CAF50")  # Light green
    bars2 = plt.bar(planes, rejected, label="Rejected Neurons", alpha=0.8, bottom=accepted,
                    color="#F57C00")  # Light orange

    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width() / 2, height / 2, f"{int(height)}",
                     ha="center", va="center", fontsize=12, color="white", fontweight="bold")

    for bar1, bar2 in zip(bars1, bars2):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        if height2 > 0:
            plt.text(bar2.get_x() + bar2.get_width() / 2, height1 + height2 / 2, f"{int(height2)}",
                     ha="center", va="center", fontsize=12, color="white", fontweight="bold")

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

    # edge case: the common path will be ops.npy if there's only a single file
    common_path = get_common_path(ops_files)

    plane_save = os.path.join(common_path, "zstats.npy")
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
    # if the file doesn't exist, save it
    if not Path(plane_save).is_file():
        np.save(plane_save, plane_stats_npy)
    # if the file does exist, only save if overwrite is true
    elif Path(plane_save).is_file() and overwrite:
        np.save(plane_save, plane_stats_npy)
    else:
        print(f"File {plane_save} already exists. Skipping.")
    return plane_save


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
        image_files = sorted(glob.glob(str(image_dir / "*.png")) +
                             glob.glob(str(image_dir / "*.jpg")) +
                             glob.glob(str(image_dir / "*.tif")))
    elif isinstance(image_input, list):
        image_files = sorted(map(str, image_input))
    else:
        raise ValueError("image_input must be a directory path or a list of file paths.")

    if not image_files:
        return

    first_image = cv2.imread(image_files[0])
    height, width, _ = first_image.shape
    fps = len(image_files) / duration if duration else 1

    if format == ".mp4":
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_path = savepath
    elif format == ".avi":
        fourcc = cv2.VideoWriter_fourcc(*'HFYU')
        video_path = savepath
    elif format == ".mov":
        fourcc = cv2.VideoWriter_fourcc(*'HFYU')
        video_path = temp_video
    else:
        raise ValueError("Invalid format. Use '.mp4', '.avi', or '.mov'.")

    video_writer = cv2.VideoWriter(str(video_path), fourcc, max(fps, 1), (width, height))

    for image_file in image_files:
        frame = cv2.imread(image_file)
        video_writer.write(frame)

    video_writer.release()

    if format == ".mp4":
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vcodec", "libx264",
            "-acodec", "aac",
            "-preset", "slow",
            "-crf", "18",
            str(savepath)  # Save directly to `savepath`
        ]
        subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"MP4 saved at {savepath}")

    elif format == ".mov":
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-i", str(temp_video),
            "-c:v", "prores_ks",  # Use Apple ProRes codec
            "-profile:v", "3",  # ProRes 422 LT
            "-pix_fmt", "yuv422p10le",
            str(savepath)
        ]
        subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        temp_video.unlink()


def get_fcells_list(ops_list: list):
    if not isinstance(ops_list, list):
        raise ValueError("`ops_list` must be a list")
    f_cells_list = []
    for ops in ops_list:
        ops = load_ops(ops)
        f_cells = np.load(Path(ops['save_path']).joinpath('F.npy'))
        f_cells_list.append(f_cells)
    return f_cells_list


def collect_result_png(ops_list):
    if not isinstance(ops_list, list):
        raise ValueError("`ops_list` must be a list")
    png_list = []
    for ops in ops_list:
        ops = load_ops(ops)
        f_cells = np.load(Path(ops['save_path']).joinpath('segmentation.png'))
        png_list.append(f_cells)
    return png_list
