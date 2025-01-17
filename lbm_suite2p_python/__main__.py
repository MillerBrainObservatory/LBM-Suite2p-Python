import numpy as np
import argparse
from pathlib import Path
from functools import partial
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

    ops = suite2p.default_ops()

    # Load the ops file
    if args.ops:
        ops = np.load(args.ops, allow_pickle=True).item()
    if args.data:
        files = [x for x in Path(args.data).glob('*.tif*')]
        metadata = lsp.get_metadata(files[0])
        new_ops = lsp.ops_from_metadata(ops, metadata)
        new_ops['data_path'] = [_parse_data_path(args.data)]
        new_ops['tiff_list'] = [files[0]]
        ops = suite2p.run_s2p(ops)

    print("Processing complete -----------")
    return ops


if __name__ == "__main__":
    ops_path = main()
    print(ops_path)
