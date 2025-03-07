import os
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
    get_fcells_list
)


def run_volume(ops, input_file_list, save_path, save_folder=None, replot=False):
    """"""

    all_ops = []
    for file in input_file_list:
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
    print('running volumetric statistics')
    if isinstance(all_ops[0], dict):
        all_ops = [ops['ops_path'] for ops in all_ops]

    # volumetric stats / graphs
    zstats_file = get_volume_stats(all_ops, overwrite=True)

    plot_volume_stats(zstats_file, os.path.join(save_path, "acc_rej_bar.png"))
    plot_volume_signal(zstats_file, os.path.join(save_path, "mean_volume_signal.png"))
    plot_roi_maps(all_ops, os.path.join(save_path, "max_cell_noncell.png"))
    plot_execution_time(zstats_file, os.path.join(save_path, "execution_time.png"))

    fcells_list = get_fcells_list(all_ops)
    flourescence_savepath = os.path.join(save_path, "flourescence.png")
    plot_fluorescence_grid_auto(fcells_list, flourescence_savepath)

    return all_ops


def run_plane(ops, input_file_path, save_path, save_folder=None, replot=False):
    input_file_path = Path(input_file_path)
    if save_folder is None:
        ops["save_folder"] = Path(input_file_path).stem  # path/to/filename.ext becomes "filename"
    else:
        if not isinstance(save_folder, str):
            raise TypeError("save_folder must be a string representing the folder name to save results to.")
    if not input_file_path.is_file():
        raise FileNotFoundError(f"Input data file {input_file_path} does not exist. Must be an existing file.")

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)  # Prevent incorrect root creation

    ops["tiff_list"] = [input_file_path.name]

    # Get metadata and initialize ops
    metadata = mbo.get_metadata(input_file_path)
    ops = ops if ops else mbo.params_from_metadata(metadata, ops)

    # Handle save path
    ops["save_path0"] = str(save_path)

    if save_folder is None:
        save_folder = save_path.name
        # ops["save_folder"] = save_folder

    zplane = input_file_path.stem
    plane_path = save_path / save_folder / "plane0"

    # Expected output files
    expected_files = {
        "ops": plane_path / "ops.npy",
        "stat": plane_path / "stat.npy",
        "iscell": plane_path / "iscell.npy",
        "registration": plane_path / "registration.png",
        "segmentation": plane_path / "segmentation.png",
        "traces": plane_path / "traces.png",
    }

    # If segmentation results exist, skip processing
    # we may want to include optional args for registration / segmentation separately
    if all(expected_files[key].is_file() for key in ["ops", "stat", "iscell"]):
        print(f"{input_file_path} already has segmentation results. Skipping execution.")
        output_ops = load_ops(expected_files["ops"])
    else:
        db = {'data_path': [str(input_file_path.parent)], 'save_folder': [str(save_folder)], 'save_path0': [str(save_path)]}
        output_ops = suite2p.run_s2p(ops=ops, db=db)

    # If replot is False, skip existing plots
    # its computationally cheap to run these plotting functions and its often helpful to access these quickly
    try:
        if replot or not all(expected_files[key].is_file() for key in ["registration", "segmentation_overlay", "segmentation_masks", "traces"]):
            print(f"Generating missing plots for {zplane}...")

            registration_path = Path(expected_files["registration"])
            registration_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

            # Ensure the file does not exist
            if registration_path.exists():
                try:
                    registration_path.unlink()
                except PermissionError:
                    print(f"Error: Cannot delete {registration_path}. Ensure it is not open elsewhere.")

            plot_registration(output_ops, registration_path, fig_label=zplane)

            segmentation_path = Path(expected_files["segmentation"])
            segmentation_path.parent.mkdir(parents=True, exist_ok=True)
            if segmentation_path.exists():
                try:
                    segmentation_path.unlink()
                except PermissionError:
                    print('Error: Cannot delete {segmentation_path}. Ensure it is not open elsewhere.')

            plot_segmentation(output_ops, segmentation_path, fig_label=zplane)

            traces_path = Path(expected_files["traces"])
            traces_path.parent.mkdir(parents=True, exist_ok=True)
            if traces_path.exists():
                try:
                    traces_path.unlink()
                except PermissionError:
                    print('Error: Cannot delete {traces_path}. Ensure it is not open elsewhere.')

                plot_traces(output_ops, traces_path,)

    except Exception as e: # don't lose the raw file if this fails
        print(e)
        print('Returning...')

    return output_ops
