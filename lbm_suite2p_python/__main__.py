import numpy as np
import argparse
from pathlib import Path
from functools import partial
import lbm_suite2p_python as lsp
from lbm_suite2p_python.run_lsp import run_volume, run_plane
from lbm_suite2p_python.utils import (
    post_process,
)
import mbo_utilities as mbo

import suite2p

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
    parser.add_argument('--save', type=str, help='Path to the save folder.')
    parser.add_argument('--max-depth', type=int, help='Number of subdirectories to check for files to process.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files.')
    parser.add_argument('--skip-existing', action='store_true', help='Skip existing files.')

    return parser


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
        if args.save:
            save_path = Path(args.save)
        else:
            save_path = Path(args.data).parent / 'results'
        if Path(args.data).is_file():
            output_ops = run_plane(ops, input_file_path=args.data, save_path=str(save_path))
            post_process(output_ops, overwrite=True)
            print("Processing complete -----------")
        elif Path(args.data).is_dir():
            files = mbo.get_files(args.data, 'tiff', max_depth=args.max_depth)
            output_ops = run_volume(ops, files, save_path=str(save_path), save_folder=str(save_path))
            print("Processing complete -----------")
        else:
            raise FileNotFoundError(f"Input data file {args.data} does not exist. Must be an existing file.")

        return output_ops






if __name__ == "__main__":
     main()
