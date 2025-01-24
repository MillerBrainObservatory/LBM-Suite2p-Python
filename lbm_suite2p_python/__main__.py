import numpy as np
import argparse
from pathlib import Path
from functools import partial
import tifffile
import lbm_suite2p_python as lsp
import suite2p

current_file = Path(__file__).parent

print = partial(print, flush=True)


def _parse_data_path(value):
    """
    Cast the value to an integer if possible, otherwise treat as a file path.
    """
    try:
        return int(value)
    except ValueError:
        return str(Path(value).expanduser().resolve())  # expand ~


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
        new_tiff[i*num_frames:(i+1)*num_frames] = tiff

    return new_tiff

def main():
    """
    The main function that orchestrates the CLI operations.
    """
    print("\n")
    print("-----------LBM-Suite2p-Pipeline -----------")
    print("\n")
    parser = argparse.ArgumentParser(description="LBM-Suite2ppipeline parameters")
    parser = add_args(parser)
    args = parser.parse_args()

    # Handle version
    if args.version:
        print("lbm_suite2p_python v{}".format(lsp.__version__))
        return

    # Load the ops file
    if args.ops:
        ops = np.load(args.ops, allow_pickle=True).item()
    else:
        ops = suite2p.default_ops()
    if args.data:
        data_path = _parse_data_path(args.data)
        files = [x for x in Path(args.data).glob('*.tif*')]
        metadata = lsp.get_metadata(files[0])
        ops = lsp.ops_from_metadata(ops, metadata)

        ops['data_path'] = [data_path]
        save_path = Path(data_path).parent / 'res'

        ops['save_path0'] = str(save_path)
        ops['block_size'] = (64, 64)
        ops['nplanes'] = 2
        ops['tau'] = 1.5
        ops['dx'] = [3.5, 3.5]
        ops['dy'] = [3.5, 3.5]
        ops['combined'] = False

        new_ops = suite2p.run_s2p(ops)

        print("Processing complete -----------")
        return new_ops


if __name__ == "__main__":
    ops_path = main()
