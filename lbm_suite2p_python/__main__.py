import os
import numpy as np
import argparse
from pathlib import Path
from functools import partial
import tifffile
import lbm_suite2p_python as lsp
from lbm_suite2p_python.utils import (
    plot_registration,
    plot_segmentation,
    plot_traces,
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

    return parser


def combine_tiffs(files):
    """
    Combine tiff files into a single tiff file.

    Input Tyx * N_files gives TNyx
    """
    # Load the first file to get the shape
    first_file = files[0]
    first_tiff = tifffile.imread(first_file)
    num_files = len(files)
    num_frames, height, width = first_tiff.shape

    # Create the new tiff file
    new_tiff = np.zeros((num_frames * num_files, height, width), dtype=first_tiff.dtype)

    # Load the tiffs
    for i, f in enumerate(files):
        tiff = tifffile.imread(f)
        new_tiff[i * num_frames:(i + 1) * num_frames] = tiff

    return new_tiff

def run_plane(input_file_path, save_path, ops):

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
    ops["save_folder"] = input_file_path.stem  # path/to/filename.ext becomes "filename"
    db = {'data_path': [str(input_file_path.parent)]}  # suite2p expects List[str]

    output_ops = suite2p.run_s2p(ops=ops, db=db)
    return output_ops


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
        output_ops = run_plane(input_file_path=args.data, save_path=save_path, ops=ops)

        plot_registration(output_ops, os.path.join(save_path, "registration.png"))
        plot_segmentation(output_ops, os.path.join(save_path, "segmentation.png"))
        plot_traces(output_ops, os.path.join(save_path, "traces.png"))

        print("Processing complete -----------")


if __name__ == "__main__":
     main()
