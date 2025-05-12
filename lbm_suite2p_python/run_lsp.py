import os
import re
import traceback
from pathlib import Path
from icecream import ic
import mbo_utilities as mbo
import numpy as np
import tifffile
import suite2p
from scipy.ndimage import uniform_filter1d
from lbm_suite2p_python.utils import dff_percentile

try:
    from suite2p.io.binary import BinaryFile
except ImportError:
    BinaryFile = None

from lbm_suite2p_python.zplane import (
    plot_traces,
    plot_projection,
    plot_noise_distribution,
    load_planar_results,
    load_ops,
)
from . import dff_shot_noise
from .volume import (
    plot_execution_time,
    plot_volume_signal,
    plot_volume_neuron_counts,
    get_volume_stats,
    save_images_to_movie,
)

if mbo.is_running_jupyter():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

try:
    from rastermap import Rastermap

    HAS_RASTERMAP = True
except ImportError:
    Rastermap = None
    utils = None
    HAS_RASTERMAP = False
if HAS_RASTERMAP:
    from lbm_suite2p_python.zplane import plot_rastermap


def _normalize_plane_folder(path):
    name = Path(path).stem
    m = re.search(r"plane[_-](\d+)", name, re.IGNORECASE)
    if not m:
        raise ValueError(f"invalid plane name: {name}")
    return f"plane{int(m.group(1))}"


def _write_raw_binary(tiff_path, out_path):
    data = tifffile.memmap(tiff_path)
    out_path = Path(out_path).with_suffix(".bin")

    if data.ndim != 3:
        raise ValueError("Must be assembled, 3D (T, Y, X)")

    nframes, x, y = data.shape
    bf = BinaryFile(
        Ly=y, Lx=x, filename=str(Path(out_path)), n_frames=nframes, dtype=np.int16
    )

    bf[:] = data
    bf.close()


def _build_ops(metadata: dict, raw_bin: Path) -> dict:
    nt, Ly, Lx = metadata["shape"]
    dx, dy = metadata.get("pixel_resolution", [2, 2])
    return {
        "Ly": Ly,
        "Lx": Lx,
        "fs": round(metadata["frame_rate"], 2),
        "nframes": nt,
        "raw_file": str(raw_bin),
        # "reg_file": str(raw_bin),
        "dx": dx,
        "dy": dy,
        "metadata": metadata,
        "input_format": "binary",
        "do_regmetrics": True,
        "delete_bin": False,
        "move_bin": False,
    }


def run_volume(input_files, save_path=None, user_ops=None, replot=False):
    """
    Processes a full volumetric imaging dataset using Suite2p, handling plane-wise registration,
    segmentation, plotting, and aggregation of volumetric statistics and visualizations.

    Parameters
    ----------
    input_files : list of str or Path
        List of TIFF file paths, each representing a single imaging plane.
    save_path : str or Path, optional
        Base directory to save all outputs.
        If none, will create a "volume" directory in the parent of the first input file.
    user_ops : dict or list, optional
        Dictionary of Suite2p parameters to use for each imaging plane.
    save_path : str, optional
        Subdirectory name within `save_path` for saving results (default: None).
    replot : bool, optional
        If True, regenerate all summary plots even if they already exist (default: False).

    Returns
    -------
    list of str
        List of paths to `ops.npy` files for each plane.

    Raises
    ------
    Exception
        If volumetric summary statistics or any visualization fails to generate.

    Example
    -------
    >> input_files = mbo.get_files(assembled_path, str_contains='tif', max_depth=3)
    >> ops = mbo.params_from_metadata(mbo.get_metadata(input_files[0]), suite2p.default_ops())

    Run volume
    >> output_ops_list = lsp.run_volume(ops, input_files, save_path)

    Notes
    -----
    At the root of `save_path` will be a folder for each z-plane with all suite2p results, as well as
    volumetric outputs at the base of this folder.

    Each z-plane folder contains:
    - Registration, Segmentation and Extraction results (ops, spks, iscell)
    - Summary statistics: execution time, signal strength, acceptance rates
    - Optional rastermap model for visualization of activity across the volume

    Each save_path root contains:
    - Accepted/Rejected histogram, neuron-count x z-plane (acc_rej_bar.png)
    - Execution time for each step in each z-plane (execution_time.png)
    - Mean/Max images, with and without segmentation masks, in GIF/MP4
    - Traces animation over time and neurons
    - Optional rastermap clustering results
    """
    if save_path is None:
        save_path = Path(input_files[0]).parent

    all_ops = []
    for file in tqdm(input_files, desc="Processing Planes"):
        print(f"Processing {file} ---------------")
        output_ops = run_plane(
            input_path=file,
            save_path=str(save_path),
            ops=user_ops,
            replot=replot,
        )
        all_ops.append(output_ops)

    # batch was ran, lets accumulate data
    if isinstance(all_ops[0], dict):
        all_ops = [ops["ops_path"] for ops in all_ops]

    try:
        zstats_file = get_volume_stats(all_ops, overwrite=True)

        all_segs = mbo.get_files(save_path, "segmentation.png", 4)
        all_means = mbo.get_files(save_path, "mean_image.png", 4)
        all_maxs = mbo.get_files(save_path, "max_projection_image.png", 4)
        all_traces = mbo.get_files(save_path, "traces.png", 4)

        save_images_to_movie(
            all_segs, os.path.join(save_path, "segmentation_volume.mp4")
        )
        save_images_to_movie(
            all_means, os.path.join(save_path, "mean_images_volume.mp4")
        )
        save_images_to_movie(all_maxs, os.path.join(save_path, "max_images_volume.mp4"))
        save_images_to_movie(all_traces, os.path.join(save_path, "traces_volume.mp4"))

        plot_volume_neuron_counts(zstats_file, save_path)
        plot_volume_signal(
            zstats_file, os.path.join(save_path, "mean_volume_signal.png")
        )
        plot_execution_time(zstats_file, os.path.join(save_path, "execution_time.png"))

        res_z = [
            load_planar_results(ops_path, z_plane=i)
            for i, ops_path in enumerate(all_ops)
        ]
        all_spks = np.concatenate([res["spks"] for res in res_z], axis=0)
        print(type(all_spks))
        # all_iscell = np.stack([res['iscell'] for res in res_z], axis=-1)
        if HAS_RASTERMAP:
            model = Rastermap(
                n_clusters=100,
                n_PCs=100,
                locality=0.75,
                time_lag_window=15,
            ).fit(all_spks)
            np.save(os.path.join(save_path, "model.npy"), model)
            title_kwargs = {"fontsize": 8, "y": 0.95}
            plot_rastermap(
                all_spks,
                model,
                neuron_bin_size=20,
                xmax=min(2000, all_spks.shape[1]),
                save_path=os.path.join(save_path, "rastermap.png"),
                title_kwargs=title_kwargs,
                title="Rastermap Sorted Activity",
            )
        else:
            print("No rastermap is available.")

    except Exception:
        print("Volume statistics failed.")
        print("Traceback: ", traceback.format_exc())

    print(f"Processing completed for {len(input_files)} files.")
    return all_ops


def run_plane_bin(plane_dir):
    plane_dir = Path(plane_dir)
    ops_path = plane_dir / "ops.npy"
    if ops_path.exists():
        _ = ic(f"Loading ops from existing file: {ops_path}")
        ops = load_ops(str(ops_path))
    else:
        raise ValueError(f"Invalid ops path: {ops_path}")
    ops.update(input_format="binary", delete_bin=False, move_bin=False)

    if "nframes" in ops and "n_frames" not in ops:
        ops["n_frames"] = ops["nframes"]
    if "n_frames" not in ops:
        raise KeyError("run_plane_bin: missing frame count (nframes or n_frames)")
    n_frames = ops["n_frames"]

    Ly, Lx = ops["Ly"], ops["Lx"]
    reg_file = ops["raw_file"]
    with suite2p.io.BinaryFile(
        Ly=Ly, Lx=Lx, filename=reg_file, n_frames=n_frames
    ) as f_reg:
        ops = suite2p.pipeline(f_reg, None, None, None, True, ops, stat=None)
    return ops


def run_plane(
    input_path: str | Path,
    save_path: str | Path | None = None,
    ops: dict | str | Path = None,
    keep_raw: bool = False,
    keep_reg: bool = True,
    force_reg: bool = False,
    force_detect: bool = False,
    **kwargs,
):
    """
    Processes a single imaging plane using suite2p, handling registration, segmentation,
    and plotting of results.

    Parameters
    ----------
    input_path : str or Path
        Full path to the file to process, with the file extension.
    save_path : str or Path, optional
        Directory to save the results.
    ops : dict, str or Path, optional
        Path to or dict of user‐supplied ops.npy. If given, it overrides any existing or generated ops.
    keep_raw : bool, default False
        If True, do not delete the raw binary (`data_raw.bin`) after processing.
    keep_reg : bool, default False
        If True, do not delete the registered binary (`data.bin`) after processing.
    force_reg : bool, default False
        If True, force a new registration even if existing shifts are found in ops.npy.
    force_detect : bool, default False
        If True, force ROI detection even if an existing stat.npy is present.

    Returns
    -------
    dict
        Processed ops dictionary containing results.

    Raises
    ------
    FileNotFoundError
        If `input_tiff` does not exist.
    TypeError
        If `save_folder` is not a string.
    Exception
        If plotting functions fail.

    Notes
    -----
    - ops supplied to the function via `ops_file` will take precendence over previously saved ops.npy files.

    Example
    -------
    >> import mbo_utilities as mbo
    >> import lbm_suite2p_python as lsp

    Get a list of z-planes in Txy format
    >> input_files = mbo.get_files(assembled_path, str_contains='tif', max_depth=3)
    >> metadata = mbo.get_metadata(input_files[0])
    >> ops = suite2p.default_ops()

    Automatically fill in metadata needed for processing (frame rate, pixel resolution, etc..)
    >> mbo_ops = mbo.params_from_metadata(metadata, ops) # handles framerate, Lx/Ly, etc

    Run a single z-plane through suite2p, keeping raw and registered files.
    >> output_ops = lsp.run_plane(input_files[0], save_path="D://data//outputs", keep_raw=True, keep_registered=True)
    """
    if isinstance(input_path, list):
        raise ValueError(
            f"input_path should be a pathlib.Path or string, not: {type(input_path)}"
        )

    p = Path(input_path)
    if p.is_dir():
        raise ValueError(f"Input path must be a file, not a directory: {p}")

    save_root = Path(save_path) if save_path is not None else p.parent
    save_root.mkdir(exist_ok=True)

    s2p_defaults = suite2p.default_ops()
    if p.suffix.lower() in (".tif", ".tiff"):
        metadata = mbo.get_metadata(p)
        folder = _normalize_plane_folder(p)
        plane_dir = save_root / folder
        plane_dir.mkdir(exist_ok=True)
        raw_bin = plane_dir / "data_raw.bin"
        if not raw_bin.exists() or force_reg:
            # if the raw binary does not exist, or we are forcing registration, write it
            print(f"Writing raw binary to {raw_bin}")
            _write_raw_binary(p, raw_bin)
            ops0 = _build_ops(metadata, raw_bin)
    elif p.suffix.lower() in (".bin", "bin"):
        ops0 = {}
        raw_bin = p
        plane_dir = p.parent
        if not p.exists():
            raise ValueError(
                f"Input file {p} is not a valid TIFF file, and no raw binary found at {raw_bin}"
            )
    else:
        raise ValueError(
            f"Unsupported file type: {p.suffix}. Only .tif/.tiff or .bin files are supported."
        )

    ops_path = plane_dir / "ops.npy"
    saved = load_ops(ops_path) if ops_path.is_file() else {}
    user = load_ops(ops) if ops else {}
    print(f"Applying user ops: {user}")
    ops = {**s2p_defaults, **ops0, **saved, **user}

    needs_reg = (
        force_reg
        or (keep_reg and not (plane_dir / "data.bin").exists())
        or "yoff" not in ops
    )
    needs_detect = force_detect or not (plane_dir / "stat.npy").exists()

    ops["zplane"] = int(plane_dir.stem.removeprefix("plane"))
    ops["do_registration"] = int(needs_reg)
    ops["roidetect"] = int(needs_detect)
    ops["move_bin"] = True
    ops["reg_tif"] = True

    if "save_path" not in ops.keys():
        ops["save_path"] = plane_dir

    ops["ops_path"] = ops_path
    np.save(ops_path, ops)

    output_ops = run_plane_bin(plane_dir)

    # cleanup
    if not keep_raw:
        (plane_dir / "data_raw.bin").unlink(missing_ok=True)
    if not keep_reg:
        (plane_dir / "data.bin").unlink(missing_ok=True)

    expected_files = {
        "ops": plane_dir / "ops.npy",
        "stat": plane_dir / "stat.npy",
        "iscell": plane_dir / "iscell.npy",
        "registration": plane_dir / "registration.png",
        "segmentation": plane_dir / "segmentation.png",
        "max_proj": plane_dir / "max_projection_image.png",
        "traces": plane_dir / "traces.png",
        "noise": plane_dir / "shot_noise_distrubution.png",
        "model": plane_dir / "model.npy",
        "rastermap": plane_dir / "rastermap.png",
    }
    try:
        if not all(
            expected_files[key].is_file()
            for key in ["registration", "segmentation", "traces"]
        ):
            print(f"Generating missing plots for {plane_dir.stem}...")

            def safe_delete(file_path):
                if file_path.exists():
                    try:
                        file_path.unlink()
                    except PermissionError:
                        print(
                            f"Error: Cannot delete {file_path}. Ensure it is not open elsewhere."
                        )

            for key in ["registration", "segmentation", "traces"]:
                safe_delete(expected_files[key])

            if expected_files["stat"].is_file():
                res = load_planar_results(output_ops)
                iscell = res["iscell"]
                f = res["F"][iscell]

                dff = dff_percentile(f, percentile=2) * 100
                dff = uniform_filter1d(dff, size=3, axis=1)
                dff_noise = dff_shot_noise(dff, output_ops["fs"])

                ncells = min(30, dff.shape[0])
                if ncells < 10:
                    print(f"Too few cells to plot traces for {plane_dir.stem}.")
                else:
                    print("Plotting traces...")
                    plot_traces(
                        dff, save_path=expected_files["traces"], num_neurons=ncells
                    )
                print("Plotting noise distribution...")
                plot_noise_distribution(dff_noise, save_path=expected_files["noise"])

                if HAS_RASTERMAP:
                    spks = res["spks"][iscell]
                    n_neurons = spks.shape[0]
                    if n_neurons < 200:
                        params = {
                            "n_clusters": None,
                            "n_PCs": min(64, n_neurons - 1),
                            "locality": 0.1,
                            "time_lag_window": 15,
                            "grid_upsample": 0,
                        }
                    else:
                        params = {
                            "n_clusters": 100,
                            "n_PCs": 128,
                            "locality": 0.0,
                            "grid_upsample": 10,
                        }

                    print("Computing rastermap model...")
                    model = Rastermap(**params).fit(spks)
                    np.save(expected_files["model"], model)

                    neuron_bin_size = (
                        1 if n_neurons < 200 else 5 if n_neurons < 500 else 10
                    )
                    xmax = min(spks.shape[1], int(2000 * (200 / n_neurons) ** 0.5))
                    plot_rastermap(
                        spks,
                        model,
                        neuron_bin_size=neuron_bin_size,
                        xmax=xmax,
                        save_path=expected_files["rastermap"],
                        title_kwargs={"fontsize": 8, "y": 0.95},
                        title="Rastermap Sorted Activity",
                    )
                else:
                    print("No rastermap is available.")

            fig_label = kwargs.get("fig_label", plane_dir.stem)
            plot_projection(
                output_ops,
                expected_files["segmentation"],
                fig_label=fig_label,
                display_masks=True,
                add_scalebar=True,
                proj="meanImg",
            )
            plot_projection(
                output_ops,
                expected_files["max_proj"],
                fig_label=fig_label,
                display_masks=False,
                add_scalebar=True,
                proj="max_proj",
            )
            print("Plots generated successfully.")
    except Exception:
        traceback.print_exc()

    return output_ops


def run_grid_search(
    base_ops: dict,
    grid_search_dict: dict,
    input_file: Path | str,
    save_root: Path | str,
):
    """
    Run a grid search over all combinations of the input suite2p parameters.

    Parameters
    ----------
    base_ops : dict
        Dictionary of default Suite2p ops to start from. Each parameter combination will override values in this dictionary.

    grid_search_dict : dict
        Dictionary mapping parameter names (str) to a list of values to grid search.
        Each combination of values across parameters will be run once.

    input_file : str or Path
        Path to the input data file, currently only supports tiff.

    save_root : str or Path
        Root directory where each parameter combination's output will be saved.
        A subdirectory will be created for each run using a short parameter tag.

    Notes
    -----
    - Subfolder names for each parameter are abbreviated to 3-character keys and truncated/rounded values.

    Examples
    --------
    >>> import lbm_suite2p_python as lsp
    >>> import suite2p
    >>> base_ops = suite2p.default_ops()
    >>> base_ops["anatomical_only"] = 3
    >>> base_ops["diameter"] = 6
    >>> lsp.run_grid_search(
    ...     base_ops,
    ...     {"threshold_scaling": [1.0, 1.2], "tau": [0.1, 0.15]},
    ...     input_file="/mnt/data/assembled_plane_03.tiff",
    ...     save_root="/mnt/grid_search/"
    ... )

    This will create the following output directory structure::

        /mnt/data/grid_search/
        ├── thr1.00_tau0.10/
        │   └── suite2p output for threshold_scaling=1.0, tau=0.1
        ├── thr1.00_tau0.15/
        ├── thr1.20_tau0.10/
        └── thr1.20_tau0.15/

    See Also
    --------
    [](http://suite2p.readthedocs.io/en/latest/parameters.html)

    """
    from itertools import product
    from pathlib import Path
    import copy

    save_root = Path(save_root)
    save_root.mkdir(exist_ok=True)

    print(f"Saving grid-search in {save_root}")

    param_names = list(grid_search_dict.keys())
    param_values = list(grid_search_dict.values())
    param_combos = list(product(*param_values))

    for combo in param_combos:
        ops = copy.deepcopy(base_ops)
        combo_dict = dict(zip(param_names, combo))
        ops.update(combo_dict)

        tag_parts = [
            f"{k[:3]}{v:.2f}" if isinstance(v, float) else f"{k[:3]}{v}"
            for k, v in combo_dict.items()
        ]
        tag = "_".join(tag_parts)

        print(f"Running grid search in: {save_root.joinpath(tag)}")
        run_plane(ops, input_file, save_root, save_folder=tag)
