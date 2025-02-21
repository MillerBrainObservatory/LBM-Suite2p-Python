import os
from pathlib import Path

import suite2p

from lbm_suite2p_python import post_process, get_volume_stats, plot_volume_stats, plot_volume_signal, plot_roi_maps, \
    plot_execution_time, get_fcells_list, plot_fluorescence_grid_auto
from lbm_suite2p_python.__main__ import print


def run_volume(ops, input_file_list, save_path, save_folder=None):
    """"""
    if save_folder is None:
        ops["save_folder"] = Path(input_file_list[0]).stem  # path/to/filename.ext becomes "filename"
    else:
        if not isinstance(save_folder, str):
            raise TypeError("save_folder must be a string representing the folder name to save results to.")

    all_ops = []
    for file in input_file_list:
        print(f"Processing {file} ---------------")
        output_ops = run_plane(input_file_path=file, save_path=str(save_path), ops=ops)
        if isinstance(output_ops, dict):
            # convert to path
            output_ops = output_ops["ops_path"]
        all_ops.append(output_ops)
        post_process(output_ops, overwrite=False)

    # batch was ran, lets accumulate data
    zstats_file = get_volume_stats(all_ops, overwrite=True)

    plot_volume_stats(zstats_file, os.path.join(save_path, "acc_rej_bar.png"))
    plot_volume_signal(zstats_file, os.path.join(save_path, "mean_volume_signal.png"))
    plot_roi_maps(all_ops, os.path.join(save_path, "max_cell_noncell.png"))
    plot_execution_time(zstats_file, os.path.join(save_path, "execution_time.png"))

    fcells_list = get_fcells_list(all_ops)
    flourescence_savepath = os.path.join(save_path, "flourescence.png")
    plot_fluorescence_grid_auto(fcells_list, flourescence_savepath)

    return all_ops


def run_plane(ops, input_file_path, save_path, save_folder=None):

    input_file_path = Path(input_file_path)
    if not input_file_path.is_file():
        raise FileNotFoundError(f"Input data file {input_file_path} does not exist. Must be an existing file.")

    ops["tiff_list"] = [
        str(Path(input_file_path).name),
    ]

    # get metadata
    metadata = mbo.get_metadata(input_file_path)
    ops = mbo.params_from_metadata(metadata, ops)

    # handle save path
    ops["save_path0"] = save_path
    if save_folder is None:
        ops["save_folder"] = input_file_path.stem  # path/to/filename.ext becomes "filename"
    else:
        if not isinstance(save_folder, str):
            raise TypeError("save_folder must be a string representing the folder name to save results to.")

    # TODO: add the plane0 as argument when we figure out how to change it
    ops_file = os.path.join(save_path, input_file_path.stem, "plane0", "ops.npy")
    stat_file = os.path.join(save_path, input_file_path.stem, "plane0", "stat.npy")
    iscell = os.path.join(save_path, input_file_path.stem, "plane0", "iscell.npy")
    if Path(ops_file).is_file() and Path(stat_file).is_file() and Path(iscell).is_file():
        print(f"{input_file_path} already has segmentation results. Skipping execution.")
        return ops_file
    else:
        db = {'data_path': [str(input_file_path.parent)]}  # suite2p expects List[str]
        output_ops = suite2p.run_s2p(ops=ops, db=db)
        return output_ops
