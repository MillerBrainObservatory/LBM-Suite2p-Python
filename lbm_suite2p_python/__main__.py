import os
import pathlib

import numpy as np
import argparse
from pathlib import Path
from functools import partial
import lbm_suite2p_python as lsp
from lbm_suite2p_python import plot_volume_stats, plot_roi_maps
from lbm_suite2p_python.utils import (
    get_volume_stats,
    post_process,
    plot_fluorescence_grid_auto,
)
import mbo_utilities as mbo

import suite2p

current_file = Path(__file__).parent

print = partial(print, flush=True)


def add_args(parser: argparse.ArgumentParser):
    """
    Add command-line arguments to the parser, dynamically adding arguments
    for each key in the `ops` dictionary.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The argument parser to which arguments are added.

    Returns
    -------
    argparse.ArgumentParser
        The parser with added arguments.
    """

    parser.add_argument('--version', type=str, help='Print the version of the package.')
    parser.add_argument('--ops', type=str, help='Path to the ops .npy file.')
    parser.add_argument('--data', type=str, help='Path to the data.')
    parser.add_argument('--max-depth', type=int, help='Number of subdirectories to check for files to process.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files.')
    parser.add_argument('--skip-existing', action='store_true', help='Skip existing files.')

    return parser

def run_plane(ops, input_file_path, save_path, save_folder=None):

    input_file_path = Path(input_file_path)
    if not input_file_path.is_file():
        raise FileNotFoundError(f"Input data file {input_file_path} does not exist. Must be an existing file.")

    # # handle files
    # files = mbo.get_files(input_data_path, 'tif')

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
    if Path(ops_file).is_file() or ops.get("skip_existing", False):
        print("Ops file already exists. Skipping.")
        return ops_file
    db = {'data_path': [str(input_file_path.parent)]}  # suite2p expects List[str]

    output_ops = suite2p.run_s2p(ops=ops, db=db)
    return output_ops

def _get_fcells_from_ops_list(ops_list):
    f_cells_list = []
    for ops in ops_list:
        ops = np.load(ops, allow_pickle=True).item()
        f_cells = np.load(Path(ops['save_path']).joinpath('F.npy'))
        print(f_cells.shape)
        f_cells_list.append(f_cells)
    return f_cells_list

def main():
    """
    The main function that orchestrates the CLI operations.
    """
    print("\n")
    print("-----------LBM-Suite2p-Pipeline -----------")
    print("\n")
    parser = argparse.ArgumentParser(description="LBM-Suite2p-pipeline parameters")
    parser = add_args(parser)
    args = parser.parse_args()

    # Handle version
    if args.version:
        print("lbm_suite2p_python v{}".format(lsp.__version__))
        return

    if args.ops:
        ops = np.load(args.ops, allow_pickle=True).item()
    else:
        ops = suite2p.default_ops()
    if args.data:
        save_path = r"D:\W2_DATA\kbarber\2025-02-10\mk303\results"
        if Path(args.data).is_file():
            output_ops = run_plane(ops, input_file_path=args.data, save_path=save_path,)
            post_process(output_ops, overwrite=True)
            print("Processing complete -----------")

        elif Path(args.data).is_dir():
            files = mbo.get_files(args.data, 'tiff', max_depth=args.max_depth)
            all_ops = []
            for file in files:
                print(f"Processing {file} ---------------")
                output_ops = run_plane(input_file_path=file, save_path=save_path, ops=ops)
                all_ops.append(output_ops)
                post_process(output_ops, overwrite=False)

            # batch was ran, lets accumulate data
            acc_rej_bar_savepath = os.path.join(save_path, "acc_rej_bar.png")
            zstats_file = get_volume_stats(all_ops, overwrite=True)
            plot_volume_stats(zstats_file, acc_rej_bar_savepath)

            max_cell_noncell_savepath = os.path.join(save_path, "max_cell_noncell.png")
            plot_roi_maps(all_ops, max_cell_noncell_savepath)

            fcells_list = _get_fcells_from_ops_list(ops)
            flourescence_savepath = os.path.join(save_path, "flourescence.png")
            plot_fluorescence_grid_auto(fcells_list, flourescence_savepath)

            print("Processing complete -----------")






if __name__ == "__main__":
     main()
