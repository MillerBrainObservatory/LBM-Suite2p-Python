import os
import traceback
from pathlib import Path
import mbo_utilities as mbo

import suite2p

from lbm_suite2p_python import (
    load_ops,
    plot_segmentation,
    plot_registration,
    plot_traces
)
from lbm_suite2p_python.volume import (
    plot_execution_time,
    plot_volume_signal,
    plot_volume_stats,
    get_volume_stats,
)

if mbo.is_running_jupyter():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def run_volume(ops, input_file_list, save_path, save_folder=None, replot=False):
    """
    Processes a volumetric imaging dataset by running Suite2p on multiple planes
    and aggregating volumetric statistics.

    Parameters
    ----------
    ops : dict | list
        Dictionary containing Suite2p parameters.
    input_file_list : list of str or Path
        List of file paths corresponding to imaging planes.
    save_path : str or Path
        Directory to save the results.
    save_folder : str, optional
        Subdirectory for saving results (default: None).
    replot : bool, optional
        If True, regenerates plots even if they exist (default: False).

    Returns
    -------
    list
        List of processed ops dictionaries or paths to ops files.

    Raises
    ------
    Exception
        If volumetric statistics or plots fail.

    Notes
    -----
    - Calls `run_plane` for each plane in `input_file_list`.
    - Computes and saves volumetric statistics.
    - Generates summary plots of segmentation and execution metrics.
    """
    all_ops = []
    for file in tqdm(input_file_list, desc="Processing Planes"):
        print(f"Processing {file} ---------------")
        output_ops = run_plane(
            ops=ops,
            input_file_path=file,
            save_path=str(save_path),
            save_folder=save_folder,
            replot=replot
        )
        all_ops.append(output_ops)

    # batch was ran, lets accumulate data
    if isinstance(all_ops[0], dict):
        all_ops = [ops['ops_path'] for ops in all_ops]

    zstats_file = get_volume_stats(all_ops, overwrite=True)

    try:
        plot_volume_stats(zstats_file, os.path.join(save_path, "acc_rej_bar.png"))
        plot_volume_signal(zstats_file, os.path.join(save_path, "mean_volume_signal.png"))
        plot_execution_time(zstats_file, os.path.join(save_path, "execution_time.png"))
    except Exception:
        print("Volume statistics failed")
        traceback.print_exc()

    print(f"Processing completed for {len(input_file_list)} files.")
    return all_ops

def run_plane(ops, input_file_path, save_path, save_folder=None, replot=False):
    """
    Processes a single imaging plane using suite2p, handling registration, segmentation,
    and plotting of results.

    Parameters
    ----------
    ops : dict
        Dictionary containing suite2p parameters.
    input_file_path : str or Path
        Path to the input TIFF file.
    save_path : str or Path
        Directory to save the results.
    save_folder : str, optional
        Subdirectory for saving results (default: filename of input file).
    replot : bool, optional
        If True, regenerates plots even if they exist (default: False).

    Returns
    -------
    dict
        Processed ops dictionary containing results.

    Raises
    ------
    FileNotFoundError
        If `input_file_path` does not exist.
    TypeError
        If `save_folder` is not a string.
    Exception
        If plotting functions fail.

    Example
    -----
    input_files = mbo.get_files(assembled_path, str_contains='tif', max_depth=3)
    metadata = mbo.get_metadata(input_files[0])
    ops = suite2p.default_ops()
    mbo_ops = mbo.params_from_metadata(metadata, ops) # handles framerate, Lx/Ly, etc
    output_ops = lsp.run_plane(mbo_ops, input_files[0], save_path)
    """
    input_file_path = Path(input_file_path)
    if not input_file_path.is_file():
        raise FileNotFoundError(f"Input data file {input_file_path} does not exist.")

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    if save_folder is None:
        save_folder = input_file_path.stem
    elif not isinstance(save_folder, (str, Path)):
        raise TypeError("save_folder must be a string or path-like object.")
    else:
        save_folder = Path(save_folder)

    if ops is None:
        ops = suite2p.default_ops()
    metadata = mbo.get_metadata(input_file_path)
    ops = mbo.params_from_metadata(metadata, ops)
    ops["tiff_list"] = [input_file_path.name]

    plane_path = save_path / save_folder / "plane0"
    expected_files = {
        "ops": plane_path / "ops.npy",
        "stat": plane_path / "stat.npy",
        "iscell": plane_path / "iscell.npy",
        "registration": plane_path / "registration.png",
        "segmentation": plane_path / "segmentation.png",
        "traces": plane_path / "traces.png",
    }

    if all(expected_files[key].is_file() for key in ["ops", "stat", "iscell"]):
        print(f"{input_file_path} already has segmentation results. Skipping execution.")
        output_ops = load_ops(expected_files["ops"])
    else:
        db = {"data_path": [str(input_file_path.parent)], "save_folder": save_folder, "save_path0": str(save_path)}
        output_ops = suite2p.run_s2p(ops=ops, db=db)

    # remove when we set data.bin path correctly
    # monkey patch to deal with default suite2p/plane0/data.bin save path
    raw_path = save_path / "suite2p" / "plane0" / "data.bin"
    where_raw_should_be_path = plane_path / "data.bin"
    if raw_path.is_file():
        if ops["keep_movie_raw"]:
            print(f"Moving {raw_path} -> {where_raw_should_be_path}")
            if not where_raw_should_be_path.exists():
                raw_path.rename(where_raw_should_be_path)
            else:
                print(f"Warning: {where_raw_should_be_path} already exists. Skipping rename.")
        else:
            try:
                print(f"Deleting {raw_path} due to ops['keep_movie_raw=False'].")
                raw_path.unlink()
                if not any(raw_path.parent.parent.iterdir()):
                    raw_path.parent.parent.rmdir()
            except Exception as e:
                print(f"Failed to delete {raw_path}: {e}")

    try:
        if replot or not all(expected_files[key].is_file() for key in [
            "registration", "segmentation", "traces"]):
            print(f"Generating missing plots for {input_file_path.stem}...")

            def safe_delete(file_path):
                if file_path.exists():
                    try:
                        file_path.unlink()
                    except PermissionError:
                        print(f"Error: Cannot delete {file_path}. Ensure it is not open elsewhere.")

            for key in ["registration", "segmentation", "traces"]:
                safe_delete(expected_files[key])

            plot_registration(output_ops, expected_files["registration"], fig_label=input_file_path.stem)
            plot_segmentation(output_ops, expected_files["segmentation"], fig_label=input_file_path.stem)
            plot_traces(output_ops, expected_files["traces"])
    except Exception as e:
        print(f"Plotting failed: {e}")

    return output_ops
