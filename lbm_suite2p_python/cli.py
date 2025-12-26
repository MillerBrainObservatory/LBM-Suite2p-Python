"""
Command-line interface for lbm_suite2p_python.

Exposes the full pipeline API with all suite2p parameters configurable via CLI.

Usage:
    lsp input_path output_path [options]
    lsp --help
    lsp --list-ops

Examples:
    # basic usage
    lsp /path/to/data.tif /path/to/output

    # process specific z-planes with custom parameters
    lsp /path/to/data --num-zplanes 1 2 3 --diameter 8 --fs 30

    # quick test with limited frames
    lsp /path/to/data --output /tmp/test --frames 500 --num-zplanes 1

    # cellpose-only detection with MBO defaults
    lsp /path/to/data --anatomical-only 4 --diameter 4 --spatial-hp-cp 3
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _snake_to_kebab(name: str) -> str:
    """convert snake_case to kebab-case for CLI args."""
    return name.replace("_", "-")


def _kebab_to_snake(name: str) -> str:
    """convert kebab-case back to snake_case."""
    return name.replace("-", "_")


def _infer_type(value: Any) -> type:
    """infer argparse type from default value."""
    if isinstance(value, bool):
        return None  # handled separately with store_true/store_false
    elif isinstance(value, int):
        return int
    elif isinstance(value, float):
        return float
    elif isinstance(value, str):
        return str
    elif isinstance(value, (list, tuple)):
        if len(value) > 0:
            return type(value[0])
        return str
    return str


def _get_ops_help() -> dict[str, str]:
    """get help text for ops parameters from comments in s2p_ops."""
    # hardcoded help for key parameters (extracted from comments)
    return {
        "fs": "frame rate in Hz (per plane)",
        "tau": "decay time constant for deconvolution",
        "nplanes": "number of planes in the recording",
        "nchannels": "number of channels per plane",
        "diameter": "expected cell diameter for cellpose (pixels)",
        "cellprob_threshold": "cellpose cell probability threshold (lower = more cells)",
        "flow_threshold": "cellpose flow error threshold",
        "anatomical_only": "cellpose detection mode: 0=off, 1=max_proj, 2=mean, 3=enhanced, 4=max",
        "pretrained_model": "cellpose model name (e.g., cpsam, cyto2, nuclei)",
        "do_registration": "whether to run motion correction",
        "nonrigid": "use nonrigid (piecewise) registration",
        "batch_size": "frames per batch for registration",
        "maxregshift": "max registration shift as fraction of image size",
        "smooth_sigma": "gaussian smoothing sigma for registration",
        "threshold_scaling": "scale factor for ROI detection threshold",
        "max_overlap": "max allowed overlap between ROIs (0-1)",
        "sparse_mode": "use sparse mode for cell detection",
        "spatial_scale": "spatial scale for detection: 0=multi, 1=6px, 2=12px, 3=24px",
        "connected": "require ROIs to be spatially connected",
        "roidetect": "run ROI detection",
        "spikedetect": "run spike deconvolution",
        "neuropil_extract": "extract neuropil signals",
        "neucoeff": "neuropil coefficient for signal subtraction",
        "baseline": "baseline mode: maximin or prctile",
        "frames_include": "number of frames to process (-1 = all)",
        "delete_bin": "delete binary files after processing",
        "reg_tif": "save registered tiffs",
    }


def build_parser() -> argparse.ArgumentParser:
    """build the argument parser with all pipeline and ops parameters."""
    from lbm_suite2p_python.default_ops import s2p_ops
    from lbm_suite2p_python import __version__

    parser = argparse.ArgumentParser(
        prog="lsp",
        description="LBM Suite2p Pipeline - process calcium imaging data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  lsp data.tif output/                        # basic processing
  lsp data/ output/ --planes 1 2 3            # specific planes (1-indexed)
  lsp data/ output/ --num-timepoints 500      # quick test with 500 frames
  lsp data/ output/ --diameter 8              # custom cell diameter
  lsp --list-ops                              # show all suite2p parameters
        """,
    )

    # positional arguments
    parser.add_argument(
        "input",
        nargs="?",
        help="input file or directory (tiff, zarr, hdf5, suite2p bin)",
    )
    parser.add_argument(
        "output",
        nargs="?",
        help="output directory for results",
    )

    # info commands
    info = parser.add_argument_group("info")
    info.add_argument(
        "--version", action="store_true", help="print version and exit"
    )
    info.add_argument(
        "--list-ops", action="store_true", help="list all suite2p parameters"
    )

    # pipeline arguments (match mbo_utilities naming)
    pipeline = parser.add_argument_group("pipeline options")
    pipeline.add_argument(
        "--planes", nargs="*", type=int, dest="planes",
        help="z-planes to process (1-indexed, e.g., --planes 1 2 3)"
    )
    pipeline.add_argument(
        "--roi-mode", "--roi", type=int, dest="roi_mode",
        help="ROI mode: None=stitch, 0=split all, N=specific ROI"
    )
    pipeline.add_argument(
        "--num-timepoints", "--frames", type=int, dest="num_timepoints",
        help="number of frames/timepoints to process (for quick testing)"
    )
    pipeline.add_argument(
        "--overwrite", action="store_true",
        help="overwrite existing results"
    )
    pipeline.add_argument(
        "--keep-reg", action="store_true", default=True,
        help="keep registered binary (default: True)"
    )
    pipeline.add_argument(
        "--no-keep-reg", action="store_false", dest="keep_reg",
        help="delete registered binary after processing"
    )
    pipeline.add_argument(
        "--keep-raw", action="store_true", default=False,
        help="keep raw (unregistered) binary"
    )
    pipeline.add_argument(
        "--force-reg", action="store_true",
        help="force re-registration even if binary exists"
    )
    pipeline.add_argument(
        "--force-detect", action="store_true",
        help="force re-detection even if results exist"
    )
    pipeline.add_argument(
        "--save-json", action="store_true",
        help="save ops as JSON (in addition to .npy)"
    )
    pipeline.add_argument(
        "--accept-all-cells", action="store_true",
        help="mark all detected ROIs as accepted cells"
    )

    # dff options
    dff = parser.add_argument_group("dff options")
    dff.add_argument(
        "--dff-window", type=int, dest="dff_window_size",
        help="rolling window size for dF/F calculation"
    )
    dff.add_argument(
        "--dff-percentile", type=int, default=20,
        help="percentile for baseline in dF/F (default: 20)"
    )
    dff.add_argument(
        "--dff-smooth", type=int, dest="dff_smooth_window",
        help="smoothing window for dF/F"
    )

    # cell filter options
    filters = parser.add_argument_group("cell filter options")
    filters.add_argument(
        "--min-diameter-um", type=float,
        help="minimum cell diameter in microns"
    )
    filters.add_argument(
        "--max-diameter-um", type=float,
        help="maximum cell diameter in microns"
    )
    filters.add_argument(
        "--min-diameter-px", type=float,
        help="minimum cell diameter in pixels"
    )
    filters.add_argument(
        "--max-diameter-px", type=float,
        help="maximum cell diameter in pixels"
    )

    # reader options (for raw data)
    reader = parser.add_argument_group("reader options (raw scanimage data)")
    reader.add_argument(
        "--fix-phase", action="store_true",
        help="apply bidirectional phase correction"
    )
    reader.add_argument(
        "--use-fft", action="store_true",
        help="use FFT for subpixel phase correction"
    )

    # dynamically add all ops parameters
    ops_defaults = s2p_ops()
    ops_help = _get_ops_help()
    ops_group = parser.add_argument_group(
        "suite2p parameters",
        description="all suite2p ops can be set via --param-name value"
    )

    # skip params already handled above
    skip_params = {"frames_include"}

    for key, default in ops_defaults.items():
        if key in skip_params:
            continue

        arg_name = f"--{_snake_to_kebab(key)}"
        help_text = ops_help.get(key, "")
        param_type = _infer_type(default)

        if isinstance(default, bool):
            # boolean flags get --flag and --no-flag
            ops_group.add_argument(
                arg_name,
                action="store_true",
                default=None,
                help=help_text or f"enable {key}",
            )
            ops_group.add_argument(
                f"--no-{_snake_to_kebab(key)}",
                action="store_false",
                dest=key,
                help=f"disable {key}",
            )
        elif isinstance(default, (list, tuple)):
            ops_group.add_argument(
                arg_name,
                nargs="*",
                type=param_type,
                default=None,
                help=help_text or f"{key} (default: {default})",
            )
        else:
            ops_group.add_argument(
                arg_name,
                type=param_type,
                default=None,
                help=help_text or f"{key} (default: {default})",
            )

    return parser


def list_ops():
    """print all suite2p parameters with their defaults."""
    from lbm_suite2p_python.default_ops import s2p_ops

    ops = s2p_ops()
    ops_help = _get_ops_help()

    print("\nSuite2p Parameters (use --param-name value to set):\n")
    print(f"{'Parameter':<25} {'Default':<20} Description")
    print("-" * 80)

    # group by category
    categories = {
        "Main Settings": ["nplanes", "nchannels", "fs", "tau", "frames_include"],
        "Registration": ["do_registration", "nonrigid", "batch_size", "maxregshift",
                        "smooth_sigma", "nimg_init", "subpixel"],
        "Cell Detection": ["roidetect", "sparse_mode", "spatial_scale", "threshold_scaling",
                          "max_overlap", "connected", "nbinned", "max_iterations"],
        "Cellpose": ["anatomical_only", "diameter", "cellprob_threshold", "flow_threshold",
                    "pretrained_model", "spatial_hp_cp"],
        "Signal Extraction": ["neuropil_extract", "neucoeff", "spikedetect",
                             "baseline", "win_baseline"],
        "Output": ["delete_bin", "reg_tif", "save_mat", "save_NWB"],
    }

    printed = set()
    for category, keys in categories.items():
        print(f"\n{category}:")
        for key in keys:
            if key in ops:
                default = ops[key]
                help_text = ops_help.get(key, "")
                default_str = str(default)[:18]
                print(f"  --{_snake_to_kebab(key):<22} {default_str:<20} {help_text}")
                printed.add(key)

    # print remaining
    remaining = set(ops.keys()) - printed
    if remaining:
        print("\nOther:")
        for key in sorted(remaining):
            default = ops[key]
            help_text = ops_help.get(key, "")
            default_str = str(default)[:18]
            print(f"  --{_snake_to_kebab(key):<22} {default_str:<20} {help_text}")


def build_cell_filters(args) -> list | None:
    """build cell filters list from CLI args."""
    filters = []

    has_diameter_filter = any([
        args.min_diameter_um,
        args.max_diameter_um,
        args.min_diameter_px,
        args.max_diameter_px,
    ])

    if has_diameter_filter:
        f = {"name": "max_diameter"}
        if args.min_diameter_um:
            f["min_diameter_um"] = args.min_diameter_um
        if args.max_diameter_um:
            f["max_diameter_um"] = args.max_diameter_um
        if args.min_diameter_px:
            f["min_diameter_px"] = args.min_diameter_px
        if args.max_diameter_px:
            f["max_diameter_px"] = args.max_diameter_px
        filters.append(f)

    return filters if filters else None


def build_ops(args, base_ops: dict) -> dict:
    """build ops dict from CLI args, overriding base_ops."""
    from lbm_suite2p_python.default_ops import s2p_ops

    ops = base_ops.copy()
    ops_defaults = s2p_ops()

    # override with any CLI-provided ops values
    for key in ops_defaults.keys():
        # check both snake_case and converted names
        value = getattr(args, key, None)
        if value is not None:
            ops[key] = value

    return ops


def _run_convert(args):
    """Run format conversion subcommand."""
    import argparse
    from lbm_suite2p_python.conversion import convert

    parser = argparse.ArgumentParser(prog="lsp convert", description="Convert between formats")
    parser.add_argument("source", help="Source directory")
    parser.add_argument("--to", "-t", choices=["suite2p", "cellpose"], required=True, help="Target format")
    parser.add_argument("--output", "-o", help="Output directory")

    parsed = parser.parse_args(args)

    result = convert(parsed.source, parsed.to, parsed.output)
    print(f"Conversion complete: {result}")
    return 0


def _run_detect(args):
    """Run format detection subcommand."""
    import argparse
    from lbm_suite2p_python.conversion import detect_format, validate_format

    parser = argparse.ArgumentParser(prog="lsp detect", description="Detect format")
    parser.add_argument("path", help="Path to check")

    parsed = parser.parse_args(args)

    fmt = detect_format(parsed.path)
    validation = validate_format(parsed.path)

    print(f"Format: {fmt}")
    print(f"Valid: {validation['valid']}")
    print(f"ROIs: {validation.get('n_rois', 'N/A')}")
    print(f"Shape: {validation.get('shape', 'N/A')}")
    if validation.get("warnings"):
        print(f"Warnings: {', '.join(validation['warnings'])}")
    return 0


def main():
    """main CLI entrypoint."""
    import lbm_suite2p_python as lsp
    from lbm_suite2p_python import __version__

    # check for subcommands before parsing
    if len(sys.argv) > 1:
        subcommand = sys.argv[1]

        if subcommand == "gui":
            from lbm_suite2p_python.gui import main as gui_main
            sys.argv = sys.argv[1:]  # remove 'lsp' from args
            return gui_main()

        if subcommand == "convert":
            return _run_convert(sys.argv[2:])

        if subcommand == "detect":
            return _run_detect(sys.argv[2:])

    parser = build_parser()
    args = parser.parse_args()

    # handle info commands
    if args.version:
        print(f"lbm_suite2p_python v{__version__}")
        return 0

    if args.list_ops:
        list_ops()
        return 0

    # validate required args
    if not args.input:
        parser.print_help()
        print("\nError: input path is required")
        return 1

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"Error: input path does not exist: {input_path}")
        return 1

    # determine output path
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        output_path = input_path.parent / "suite2p_output"

    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"LBM Suite2p Pipeline v{__version__}")
    print(f"{'='*60}")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")

    # build ops
    base_ops = lsp.default_ops()
    ops = build_ops(args, base_ops)

    # build reader kwargs
    reader_kwargs = {}
    if args.fix_phase:
        reader_kwargs["fix_phase"] = True
    if args.use_fft:
        reader_kwargs["use_fft"] = True

    # build cell filters
    cell_filters = build_cell_filters(args)

    # show key settings
    print(f"\nSettings:")
    if args.planes:
        print(f"  Planes: {args.planes}")
    if args.num_timepoints and args.num_timepoints > 0:
        print(f"  Timepoints: {args.num_timepoints}")
    print(f"  Diameter: {ops.get('diameter', 6)}")
    print(f"  Cellpose model: {ops.get('pretrained_model', 'cpsam')}")
    if cell_filters:
        print(f"  Cell filters: {cell_filters}")

    print(f"\n{'='*60}\n")

    # run pipeline
    try:
        results = lsp.pipeline(
            input_data=input_path,
            save_path=output_path,
            ops=ops,
            planes=args.planes,
            roi_mode=args.roi_mode,
            num_timepoints=args.num_timepoints,
            keep_reg=args.keep_reg,
            keep_raw=args.keep_raw,
            force_reg=args.force_reg,
            force_detect=args.force_detect,
            dff_window_size=args.dff_window_size,
            dff_percentile=args.dff_percentile,
            dff_smooth_window=args.dff_smooth_window,
            cell_filters=cell_filters,
            accept_all_cells=args.accept_all_cells,
            save_json=args.save_json,
            reader_kwargs=reader_kwargs if reader_kwargs else None,
        )

        print(f"\n{'='*60}")
        print(f"Processing complete!")
        print(f"Results saved to: {output_path}")
        print(f"{'='*60}\n")

        return 0

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
