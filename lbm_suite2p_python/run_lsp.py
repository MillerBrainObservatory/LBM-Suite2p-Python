import os
from pathlib import Path
import mbo_utilities as mbo

import suite2p

from lbm_suite2p_python import  get_volume_stats, plot_volume_stats, plot_volume_signal, plot_roi_maps, \
    plot_execution_time, get_fcells_list, plot_fluorescence_grid_auto, load_ops, plot_segmentation, plot_registration, \
    plot_traces


def run_volume(ops, input_file_list, save_path, save_folder=None, replot=False):
    """"""
    if save_folder is None:
        ops["save_folder"] = Path(input_file_list[0]).stem  # path/to/filename.ext becomes "filename"
    else:
        if not isinstance(save_folder, str):
            raise TypeError("save_folder must be a string representing the folder name to save results to.")

    all_ops = []
    for file in input_file_list:
        print(f"Processing {file} ---------------")
        output_ops = run_plane(input_file_path=file, save_path=str(save_path), ops=ops, replot=replot)
        all_ops.append(output_ops)

    # batch was ran, lets accumulate data
    print('running volumetric statistics')
    if isinstance(all_ops[0], dict):
        all_ops = [ops['ops_path'] for ops in all_ops]

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
    if not input_file_path.is_file():
        raise FileNotFoundError(f"Input data file {input_file_path} does not exist. Must be an existing file.")

    save_path = Path(save_path)
    if not save_path.is_dir():
        save_path.mkdir(parents=False, exist_ok=True)  # Prevent incorrect root creation

    ops["tiff_list"] = [input_file_path.name]

    # Get metadata and initialize ops
    metadata = mbo.get_metadata(input_file_path)
    ops = ops if ops else mbo.params_from_metadata(metadata, ops)

    # Handle save path
    ops["save_path0"] = str(save_path)
    save_folder = save_folder if isinstance(save_folder, str) else input_file_path.stem
    zplane = input_file_path.stem
    plane_path = save_path / save_folder / "plane0"

    # Expected output files
    expected_files = {
        "ops": plane_path / "ops.npy",
        "stat": plane_path / "stat.npy",
        "iscell": plane_path / "iscell.npy",
        "registration": plane_path / "registration.png",
        "segmentation_overlay": plane_path / "segmentation_overlay.png",
        "segmentation_masks": plane_path / "segmentation_masks.png",
        "traces": plane_path / "traces.png",
    }

    # If segmentation results exist, skip processing
    # we may want to include optional args for registration / segmentation separately
    if all(expected_files[key].is_file() for key in ["ops", "stat", "iscell"]):
        print(f"{input_file_path} already has segmentation results. Skipping execution.")
        output_ops = load_ops(expected_files["ops"])
    else:
        db = {'data_path': [str(input_file_path.parent)]}  # Suite2p expects List[str]
        output_ops = suite2p.run_s2p(ops=ops, db=db)

    # If replot is False, skip existing plots
    # its computationally cheap to run these plotting functions and its often helpful to access these quickly
    if replot or not all(expected_files[key].is_file() for key in ["registration", "segmentation_overlay", "segmentation_masks", "traces"]):
        print(f"Generating missing plots for {zplane}...")

        plot_registration(output_ops, expected_files["registration"], fig_label=zplane)
        plot_segmentation(output_ops, expected_files["segmentation_overlay"], fig_label=zplane, overlay=True)
        plot_segmentation(output_ops, expected_files["segmentation_masks"], fig_label=zplane, overlay=False)
        plot_traces(output_ops, expected_files["traces"], show_best=True)

    return output_ops
