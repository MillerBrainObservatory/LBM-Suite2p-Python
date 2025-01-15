import numpy as np
import argparse
from pathlib import Path
from functools import partial
import lbm_suite2p_python as lsp
# import suite2p

current_file = Path(__file__).parent

print = partial(print, flush=True)

def _print_params(params, indent=5):
    for k, v in params.items():
        # if value is a dictionary, recursively call the function
        if isinstance(v, dict):
            print(" " * indent + f"{k}:")
            _print_params(v, indent + 4)
        else:
            print(" " * indent + f"{k}: {v}")


def _parse_data_path(value):
    """
    Cast the value to an integer if possible, otherwise treat as a file path.
    """
    try:
        return int(value)
    except ValueError:
        return str(Path(value).expanduser().resolve())  # expand ~


def _parse_int_float(value):
    """ Cast the value to an integer if possible, otherwise treat as a float. """
    try:
        return int(value)
    except ValueError:
        return float(value)


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

    # non-run flags
    parser.add_argument('--ops', type=str, help='Path to the ops .npy file.')

    return parser


def main():
    """
    The main function that orchestrates the CLI operations.
    """
    print("\n")
    print("-----------LBM-Caiman pipeline -----------")
    print("\n")
    parser = argparse.ArgumentParser(description="LBM-Caiman pipeline parameters")
    parser = add_args(parser)
    args = parser.parse_args()

    # Handle version
    if args.version:
        print("lbm_suite2p_python v{}".format(lsp.__version__))
        return


    print("Processing complete -----------")
    return


if __name__ == "__main__":
    main()
