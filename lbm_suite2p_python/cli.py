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
        "algorithm": "detection algorithm: sparsery, sourcery, or cellpose",
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


def build_parser(include_ops: bool = True) -> argparse.ArgumentParser:
    """build the argument parser with all pipeline and ops parameters.

    When ``include_ops`` is False the per-parameter suite2p flags are not
    enumerated (enumerating them imports suite2p). Used for the fast
    ``--help`` / no-arg paths; ``--list-ops`` shows the full list instead.
    """
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
  lsp data/ output/ --tau 1.3 --algorithm cellpose         # decay + detection algorithm
  lsp data/ output/ --fix-phase --phasecorr-method median  # reader phase correction
  lsp data/ output/ --reader-kwargs '{"roi": 2}'           # arbitrary imread kwargs
  lsp data/ output/ --ops-file my_ops.json    # load saved ops (CLI flags override)
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
        "--timepoints", nargs="*", type=int, dest="timepoints",
        help="timepoints to process (1-indexed, e.g., --timepoints 1 50 100)"
    )
    pipeline.add_argument(
        "--roi-mode", "--roi", type=int, dest="roi_mode",
        help="ROI mode: None=stitch, 0=split all, N=specific ROI"
    )
    pipeline.add_argument(
        "--num-timepoints", "--frames", type=int, dest="num_timepoints",
        help="number of timepoints to process (first N, for quick testing)"
    )
    pipeline.add_argument(
        "--num-zplanes", type=int, dest="num_zplanes",
        help="number of z-planes to process (first N)"
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
        "--stream", action="store_true",
        help="stream frames from the lazy array through suite2p instead of "
             "writing data_raw.bin / data.bin (only reg_outputs.npy persists; "
             "registered frames reconstituted on the fly). Functional channel only."
    )
    pipeline.add_argument(
        "--save-json", action="store_true",
        help="save ops as JSON (in addition to .npy)"
    )
    pipeline.add_argument(
        "--accept-all-cells", action="store_true",
        help="mark all detected ROIs as accepted cells"
    )
    pipeline.add_argument(
        "--workers", type=int, default=4,
        help="number of zplane worker processes (default: 4). "
             "Use 1 for sequential, or 0 / negative for auto = "
             "min(num_planes, cpu_count//2, 8). "
             "Cellpose on GPU may OOM with multiple workers."
    )
    pipeline.add_argument(
        "--skip-volumetric", action="store_true", dest="skip_volumetric",
        help="skip merge_mrois, volume_stats, and volumetric plots after per-plane processing"
    )
    pipeline.add_argument(
        "--threads-per-worker", type=int, dest="threads_per_worker", default=2,
        help="cap BLAS / OMP / numba / torch threads per worker process "
             "(default: 2). Total CPU load ~ workers * threads_per_worker. "
             "Set to 0 or a negative value to use library defaults "
             "(typically 1 thread per core, which oversubscribes when workers > 1)."
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
    dff.add_argument(
        "--norm-method", choices=["dff", "zscore"], default="zscore",
        dest="norm_method",
        help="normalization for norm_traces.npy (default: zscore)"
    )
    dff.add_argument(
        "--correct-neuropil", dest="correct_neuropil",
        action=argparse.BooleanOptionalAction, default=True,
        help="subtract 0.7*Fneu before dF/F (default: on; use --no-correct-neuropil to disable)"
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

    # reader options (forwarded to mbo_utilities.imread)
    reader = parser.add_argument_group(
        "reader options",
        description="Forwarded to mbo_utilities.imread(). --reader-kwargs passes "
                    "any imread argument as JSON and works for every lazy array "
                    "type (tiff, zarr, hdf5, bin, ...); the named flags below are "
                    "conveniences for raw ScanImage TIFFs."
    )
    reader.add_argument(
        "--fix-phase", action=argparse.BooleanOptionalAction, default=None,
        help="bidirectional phase correction (ScanImage TIFF; reader default: on)"
    )
    reader.add_argument(
        "--use-fft", action=argparse.BooleanOptionalAction, default=None,
        help="FFT-based subpixel phase correction (ScanImage TIFF)"
    )
    reader.add_argument(
        "--phasecorr-method", choices=["mean", "median", "max"], default=None,
        help="phase-correction reduction method (ScanImage TIFF; default: mean)"
    )
    reader.add_argument(
        "--channel", type=int, default=None,
        help="zero-based color channel to read (multi-channel sources)"
    )
    reader.add_argument(
        "--reader-kwargs", type=str, default=None, metavar="JSON",
        help='any mbo_utilities.imread() kwargs as JSON, e.g. \'{"roi": 2}\''
    )

    # writer options (forwarded to mbo_utilities.imwrite)
    writer = parser.add_argument_group(
        "writer options",
        description="Forwarded to mbo_utilities.imwrite() (a different argument "
                    "set than the reader). --writer-kwargs passes any imwrite "
                    "argument as JSON."
    )
    writer.add_argument(
        "--writer-kwargs", type=str, default=None, metavar="JSON",
        help='any mbo_utilities.imwrite() kwargs as JSON, e.g. \'{"target_chunk_mb": 200}\''
    )

    # rastermap options
    rmap = parser.add_argument_group("rastermap options")
    rmap.add_argument(
        "--rastermap", action="store_true",
        help="enable rastermap (planar + volumetric) with default settings"
    )
    rmap.add_argument(
        "--rastermap-kwargs", type=str, default=None, metavar="JSON",
        help='rastermap config JSON with "planar"/"volumetric" keys, '
             'e.g. \'{"planar": {"n_clusters": 50}}\''
    )

    # advanced
    advanced = parser.add_argument_group("advanced")
    advanced.add_argument(
        "--ops-file", type=str, default=None,
        help="base ops from a .npy/.json file or suite2p dir; CLI flags override"
    )
    advanced.add_argument(
        "--replot", action=argparse.BooleanOptionalAction, default=True,
        help="regenerate per-plane figures (default: on)"
    )

    # dynamically add all ops parameters
    ops_group = parser.add_argument_group(
        "suite2p parameters",
        description="all suite2p ops can be set via --param-name value; "
                    "run 'lsp --list-ops' to list them"
    )

    # enumerating every suite2p op requires importing suite2p; skip it for the
    # fast --help / no-arg paths (see build_parser docstring)
    if not include_ops:
        return parser

    from lbm_suite2p_python.default_ops import s2p_ops
    ops_defaults = s2p_ops()
    ops_help = _get_ops_help()

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
        "Cell Detection": ["roidetect", "algorithm", "sparse_mode", "spatial_scale",
                          "threshold_scaling", "max_overlap", "connected", "nbinned",
                          "max_iterations"],
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


class _Tee:
    """Duplicate writes to multiple text streams (e.g., stdout + file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
            except Exception:
                pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self):
        return False


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


def _parse_json_arg(flag: str, value: str) -> dict:
    """parse a JSON object from a CLI flag value; exit cleanly on error."""
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as e:
        raise SystemExit(f"Error: {flag} is not valid JSON: {e}")
    if not isinstance(parsed, dict):
        raise SystemExit(f"Error: {flag} must be a JSON object")
    return parsed


def _load_ops_file(path: str) -> dict:
    """load a base ops dict from a .json/.npy file or a suite2p directory."""
    from lbm_suite2p_python import load_ops

    p = Path(path).expanduser()
    if p.suffix.lower() == ".json":
        if not p.exists():
            raise SystemExit(f"Error: --ops-file not found: {p}")
        with open(p, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise SystemExit("Error: --ops-file JSON must be an object")
        return data
    return load_ops(p)


def build_reader_kwargs(args) -> dict | None:
    """build imread kwargs from CLI reader args (None if empty)."""
    kw = {}
    if args.fix_phase is not None:
        kw["fix_phase"] = args.fix_phase
    if args.use_fft is not None:
        kw["use_fft"] = args.use_fft
    if args.phasecorr_method is not None:
        kw["phasecorr_method"] = args.phasecorr_method
    if args.channel is not None:
        kw["channel"] = args.channel
    if args.reader_kwargs:
        kw.update(_parse_json_arg("--reader-kwargs", args.reader_kwargs))
    return kw or None


def build_writer_kwargs(args) -> dict | None:
    """build writer kwargs from CLI writer args (None if empty)."""
    if args.writer_kwargs:
        return _parse_json_arg("--writer-kwargs", args.writer_kwargs)
    return None


def build_rastermap_kwargs(args) -> dict | None:
    """build rastermap_kwargs from CLI args (None if disabled)."""
    if args.rastermap_kwargs:
        return _parse_json_arg("--rastermap-kwargs", args.rastermap_kwargs)
    if args.rastermap:
        return {"planar": {}, "volumetric": {}}
    return None


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

    argv = sys.argv[1:]

    # fast info paths: don't import suite2p / the pipeline stack just to print
    if "--version" in argv:
        print(f"lbm_suite2p_python v{__version__}")
        return 0

    if "--list-ops" in argv:
        list_ops()
        return 0

    # --help and the no-arg usage dump only need the static parser; building the
    # full parser enumerates every suite2p op, which imports suite2p.
    include_ops = bool(argv) and "-h" not in argv and "--help" not in argv
    parser = build_parser(include_ops=include_ops)
    args = parser.parse_args()

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

    # parse config/advanced args up front so invalid JSON or a missing ops
    # file fails before any logging or processing work begins
    base_extra_ops = _load_ops_file(args.ops_file) if args.ops_file else None
    reader_kwargs = build_reader_kwargs(args)
    writer_kwargs = build_writer_kwargs(args)
    rastermap_kwargs = build_rastermap_kwargs(args)

    log_path = output_path / "log.txt"
    log_file = open(log_path, "w", encoding="utf-8", buffering=1)
    _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(_orig_stdout, log_file)
    sys.stderr = _Tee(_orig_stderr, log_file)
    print(f"Logging to: {log_path}")

    from importlib.metadata import PackageNotFoundError, version as _pkg_version
    try:
        _mbo_version = _pkg_version("mbo_utilities")
    except PackageNotFoundError:
        _mbo_version = "unknown"
    try:
        _s2p_version = _pkg_version("suite2p")
    except PackageNotFoundError:
        _s2p_version = "not installed"

    print(f"\n{'='*60}")
    print(f"LBM Suite2p Pipeline v{__version__}")
    print(f"  mbo_utilities       v{_mbo_version}")
    print(f"  lbm_suite2p_python  v{__version__}")
    print(f"  suite2p             v{_s2p_version}")
    print(f"{'='*60}")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")

    # build ops (optionally starting from a user-supplied ops file).
    # import default_ops directly: build_parser(include_ops=True) already did
    # `from ...default_ops import s2p_ops`, which binds the default_ops submodule
    # onto the package, shadowing the lazily-exported function — so
    # `lsp.default_ops` would resolve to the module here, not the callable.
    import lbm_suite2p_python as lsp
    from lbm_suite2p_python.default_ops import default_ops
    base_ops = default_ops(ops=base_extra_ops) if base_extra_ops else default_ops()
    ops = build_ops(args, base_ops)

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
    if reader_kwargs:
        print(f"  Reader: {reader_kwargs}")
    if rastermap_kwargs:
        print(f"  Rastermap: {sorted(rastermap_kwargs)}")

    print(f"\n{'='*60}\n")

    # run pipeline
    import time
    _pipeline_start = time.time()
    try:
        lsp.pipeline(
            input_data=input_path,
            save_path=output_path,
            ops=ops,
            planes=args.planes,
            timepoints=args.timepoints,
            roi_mode=args.roi_mode,
            num_timepoints=args.num_timepoints,
            num_zplanes=args.num_zplanes,
            keep_reg=args.keep_reg,
            keep_raw=args.keep_raw,
            force_reg=args.force_reg or args.overwrite,
            force_detect=args.force_detect or args.overwrite,
            stream=args.stream,
            dff_window_size=args.dff_window_size,
            dff_percentile=args.dff_percentile,
            dff_smooth_window=args.dff_smooth_window,
            norm_method=args.norm_method,
            correct_neuropil=args.correct_neuropil,
            cell_filters=cell_filters,
            accept_all_cells=args.accept_all_cells,
            save_json=args.save_json,
            reader_kwargs=reader_kwargs,
            writer_kwargs=writer_kwargs,
            rastermap_kwargs=rastermap_kwargs,
            replot=args.replot,
            workers=args.workers,
            skip_volumetric=args.skip_volumetric,
            threads_per_worker=args.threads_per_worker,
        )

        _elapsed = time.time() - _pipeline_start
        _h, _rem = divmod(int(_elapsed), 3600)
        _m, _s = divmod(_rem, 60)
        print(f"\n{'='*60}")
        print(f"Processing complete!")
        print(f"Results saved to: {output_path}")
        print(f"Total elapsed: {_h:02d}:{_m:02d}:{_s:02d} ({_elapsed:.1f}s)")
        print(f"{'='*60}\n")

        return 0

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        sys.stdout, sys.stderr = _orig_stdout, _orig_stderr
        try:
            log_file.close()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
