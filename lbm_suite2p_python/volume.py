import glob
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
import tifffile
from matplotlib import pyplot as plt, patches

from lbm_suite2p_python import load_ops

def get_common_path(ops_files):
    """
    Find the common path of all files in `ops_files`.
    If there is a single file or no common path, return the first non-empty path.
    """
    if len(ops_files) == 1:
        print(f"only 1 op file")
        path = Path(ops_files[0]).parent
        while path.exists() and len(list(path.iterdir())) == 1:  # Traverse up if only one item exists
            print(f"traversing")
            path = path.parent
        print(f"returning {path}")
        return path
    else:
        print(f"multi_file, {os.path.commonpath(ops_files)}")
        return Path(os.path.commonpath(ops_files))


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

    plt.savefig(savepath, bbox_inches="tight", facecolor="black")


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
    print(common_path)

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
    # if the file doesn't exist, save it
    if not Path(plane_save).is_file():
        np.save(plane_save, plane_stats_npy)
    # if the file does exist, only save if overwrite is true
    elif Path(plane_save).is_file() and overwrite:
        np.save(plane_save, plane_stats_npy)
    else:
        print(f"File {plane_save} already exists. Skipping.")
    return plane_save


def plot_volume_projection(ops, savepath, fig_label=None, vmin=None, vmax=None, add_scalebar=False, proj="meanImg"):
    """
    Plot only the max projection image without any segmentation masks.

    Parameters:
    -----------
    ops : dict
        Suite2p ops dictionary containing the 'max_proj' image and optional 'dx' for scalebar.
    savepath : str or Path
        Path to save the output figure.
    fig_label : str, optional
        Label for the figure (Z-plane name or other identifier).
    vmin : float, optional
        Minimum intensity for the grayscale image. Default is 2nd percentile.
    vmax : float, optional
        Maximum intensity for the grayscale image. Default is 98th percentile.
    add_scalebar : bool, optional
        Whether to add a 100 μm scale bar to the image. Requires 'dx' in ops.
    proj : str, optional
        Summary projection to use for the background. Options are "meanImg", "max_proj", "meanImgE". Default is "meanImg".
    """

    if proj == "meanImg":
        txt = "Mean-Image"
    elif proj == "max_proj":
        txt = "Max-Projection"
    elif proj == "meanImgE":
        txt = "Mean-Image (Enhanced)"
    else:
        raise ValueError("Unknown projection type. Options are ['meanImg', 'max_proj', 'meanImgE']")

    savepath = Path(savepath)
    data = ops[proj]
    shape = data.shape

    fig, ax = plt.subplots(figsize=(6, 6), facecolor='black')

    vmin = np.nanpercentile(data, 2) if vmin is None else vmin
    vmax = np.nanpercentile(data, 98) if vmax is None else vmax

    if vmax - vmin < 1e-6:
        vmax = vmin + 1e-6  # Add small offset to prevent NaN issues

    # data = np.clip(data_norm * 255, 0, 255).astype(np.uint8)

    ax.imshow(data, cmap='gray', vmin=vmin, vmax=vmax)

    ax.text(0.5, 1.02, txt, transform=ax.transAxes,
            fontsize=14, fontweight='bold', fontname="Courier New",
            color='white', ha='center', va='bottom')

    if fig_label:
        fig_label = fig_label.replace("_", " ").replace("-", " ").replace(".", " ")
        ax.set_ylabel(fig_label, color='white', fontweight='bold', fontsize=12)

    ax.set_xticks([])
    ax.set_yticks([])

    if add_scalebar and 'dx' in ops:
        pixel_size = ops['dx']
        scale_bar_length = 100 / pixel_size

        scalebar_x = shape[1] * 0.05
        scalebar_y = shape[0] * 0.90

        ax.add_patch(patches.Rectangle(
            (scalebar_x, scalebar_y), scale_bar_length, 5, edgecolor='white', facecolor='white'))
        ax.text(scalebar_x + scale_bar_length / 2, scalebar_y - 10, "100 μm",
                color='white', fontsize=10, ha='center', fontweight='bold')

    plt.tight_layout()
    savepath.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(savepath, dpi=300, facecolor='black')
    plt.show()


def save_images_to_movie(image_dir, savepath, duration=None, format=".mp4"):
    """
    Convert a sequence of saved images into a movie with PowerPoint-compatible encoding.

    Parameters
    ----------
    image_dir : str or Path
        Directory containing saved segmentation images.
    savepath : str or Path
        Path to save the video file.
    duration : int, optional
        Desired total video duration in seconds. If None, defaults to 1 FPS (1 image per second).
    format : str, optional
        Video format: ".mp4" (PowerPoint-compatible), ".avi" (lossless), ".mov" (ProRes). Default is ".mp4".
    """
    image_dir = Path(image_dir)
    savepath = Path(savepath).with_suffix(format)  # Ensure correct file extension
    temp_video = savepath.with_suffix(".avi")  # Temporary AVI file for MOV conversion
    savepath.parent.mkdir(parents=True, exist_ok=True)

    image_files = sorted(glob.glob(str(image_dir / "*.png")) +
                         glob.glob(str(image_dir / "*.jpg")) +
                         glob.glob(str(image_dir / "*.tif")))

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
        fixed_mp4 = savepath.with_suffix(".pptx.mp4")
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vcodec", "libx264",
            "-acodec", "aac",
            "-preset", "slow",
            "-crf", "18",
            str(fixed_mp4)
        ]
        subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        video_path.unlink()
        print(f"✅ PowerPoint-compatible MP4 saved at {fixed_mp4}")

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


def collect_result_png(ops_list):
    if not isinstance(ops_list, list):
        raise ValueError("`ops_list` must be a list")
    png_list = []
    for ops in ops_list:
        ops = load_ops(ops)
        f_cells = np.load(Path(ops['save_path']).joinpath('segmentation.png'))
        png_list.append(f_cells)
    return png_list
