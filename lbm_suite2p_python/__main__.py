import numpy as np
import argparse
from pathlib import Path
from functools import partial
import lbm_suite2p_python as lsp
import mbo_utilities as mbo

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

    parser.add_argument("--version", type=str, help="Print the version of the package.")
    parser.add_argument("--ops", type=str, help="Path to the ops .npy file.")
    parser.add_argument("--data", type=str, help="Path to the data.")
    parser.add_argument("--save", type=str, help="Path to save the results.")
    parser.add_argument(
        "--subdir", type=str, help="Additional subdirectory add to save-path."
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        help="Number of subdirectories to check for files to process.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files."
    )
    parser.add_argument(
        "--skip-existing", action="store_true", help="Skip existing files."
    )

    return parser


def main():
    """
    The main function that orchestrates the CLI operations.
    """
    print("\n----------- LBM-Suite2p-Pipeline -----------\n")
    from suite2p.default_ops import default_ops

    parser = argparse.ArgumentParser(description="LBM-Suite2p-pipeline parameters")
    parser = add_args(parser)
    args = parser.parse_args()

    if args.version:
        print(f"lbm_suite2p_python v{lsp.__version__}")
        return

    ops = (
        np.load(args.ops, allow_pickle=True).item()
        if args.ops
        else default_ops()
    )

    if not args.data:
        raise ValueError("No input file or directory specified. Use --data")

    input_path = Path(args.data)

    # default to data-path / 'results'
    save_path = Path(args.save) if args.save else input_path.parent / "results"
    save_path.mkdir(parents=True, exist_ok=True)

    # Optional user-defined save folder (e.g., plane_01_runA)
    subdir = Path(args.subdir) if args.subdir else None

    if input_path.is_file():
        output_ops = lsp.run_plane(
            input_path=input_path,
            save_path=save_path,
            ops=ops,
        )
    elif input_path.is_dir():
        files = mbo.get_files(input_path, "tiff", max_depth=args.max_depth)
        output_ops = lsp.run_volume(
            ops=ops, input_files=files, save_path=save_path, save_folder=subdir
        )
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    print("Processing complete -----------")
    return output_ops


if __name__ == "__main__":
    main()
