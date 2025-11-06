import logging
import time
from pathlib import Path
import os
import traceback
from contextlib import nullcontext
from itertools import product
import copy
import gc

import numpy as np

from lbm_suite2p_python import default_ops
from lbm_suite2p_python.postprocessing import (
    ops_to_json,
    load_planar_results,
    load_ops,
)
from mbo_utilities.log import get as get_logger

from lbm_suite2p_python.zplane import save_pc_panels_and_metrics, plot_zplane_figures

logger = get_logger("run_lsp")

from lbm_suite2p_python._benchmarking import get_cpu_percent, get_ram_used
from lbm_suite2p_python.volume import (
    plot_volume_signal,
    plot_volume_neuron_counts,
    get_volume_stats,
)
from mbo_utilities.file_io import (
    get_plane_from_filename,
)  # derive_tag_from_filename, PIPELINE_TAGS

PIPELINE_TAGS = ("plane", "roi", "z", "plane_", "roi_", "z_")


def derive_tag_from_filename(path):
    """
    Derive a folder tag from a filename based on “planeN”, “roiN”, or "tagN" patterns.

    Parameters
    ----------
    path : str or pathlib.Path
        File path or name whose stem will be parsed.

    Returns
    -------
    str
        If the stem starts with “plane”, “roi”, or “res” followed by an integer,
        returns that tag plus the integer (e.g. “plane3”, “roi7”, “res2”).
        Otherwise returns the original stem unchanged.

    Examples
    --------
    >>> derive_tag_from_filename("plane_01.tif")
    'plane1'
    >>> derive_tag_from_filename("plane2.bin")
    'plane2'
    >>> derive_tag_from_filename("roi5.raw")
    'roi5'
    >>> derive_tag_from_filename("ROI_10.dat")
    'roi10'
    >>> derive_tag_from_filename("res-3.h5")
    'res3'
    >>> derive_tag_from_filename("assembled_data_1.tiff")
    'assembled_data_1'
    >>> derive_tag_from_filename("file_12.tif")
    'file_12'
    """
    name = Path(path).stem
    for tag in PIPELINE_TAGS:
        low = name.lower()
        if low.startswith(tag):
            suffix = name[len(tag) :]
            if suffix and (suffix[0] in ("_", "-")):
                suffix = suffix[1:]
            if suffix.isdigit():
                return f"{tag}{int(suffix)}"
    return name


def run_volume(
    input_files: list,
    save_path: str | Path = None,
    ops: dict | str | Path = None,
    keep_reg: bool = True,
    keep_raw: bool = False,
    force_reg: bool = False,
    force_detect: bool = False,
    dff_window_size: int = 500,
    dff_percentile: int = 20,
    save_json: bool = False,
    **kwargs,
):
    """
    Processes a full volumetric imaging dataset using Suite2p, handling plane-wise registration,
    segmentation, plotting, and aggregation of volumetric statistics and visualizations.

    Supports planar, contiguous .zarr, tiff, suite2p .bin and automatically merges multi-ROI datasets
    acquired with ScanImage's multi-ROI mode.

    Parameters
    ----------
    input_files : list of str or Path
        List of file paths, each representing a single imaging plane. Supported formats:
        - .tif files (e.g., "plane01.tif", "plane02.tif")
        - .bin files from mbo.imwrite (e.g., "plane01_stitched/data_raw.bin")
        - .zarr files (e.g., "plane01_roi01.zarr", "plane01_roi02.zarr")
        For binary inputs, must have accompanying ops.npy in parent directory.
    save_path : str or Path, optional
        Base directory to save all outputs.
        If None, creates a "volume" directory in the parent of the first input file.
        For binary inputs with ops.npy, processing occurs in-place at the parent directory.
    ops : dict or str or Path, optional
        Suite2p parameters to use for each imaging plane. Can be:
        - Dictionary of parameters
        - Path to ops.npy file
        - None (uses defaults from default_ops())
    keep_raw : bool, default False
        If True, do not delete the raw binary (data_raw.bin) after processing.
    keep_reg : bool, default True
        If True, keep the registered binary (data.bin) after processing.
    force_reg : bool, default False
        If True, force re-registration even if refImg/meanImg/xoff exist in ops.npy.
    force_detect : bool, default False
        If True, force ROI detection even if stat.npy exists and is non-empty.
    dff_window_size : int, default 500
        Number of frames to use for rolling percentile baseline in ΔF/F₀ calculations.
    dff_percentile : int, default 20
        Percentile to use for baseline F₀ estimation (e.g., 20 = 20th percentile).
    save_json : bool, default False
        If True, saves ops as JSON in addition to .npy format.
    **kwargs
        Additional keyword arguments passed to run_plane().

    Returns
    -------
    list of Path
        List of paths to ops.npy files for each plane (or merged plane if mROI).

    Notes
    -----
    **Directory Structure:**

    For standard single-ROI data::

        save_path/
        ├── plane01/
        │   ├── ops.npy, stat.npy, F.npy, Fneu.npy, spks.npy, iscell.npy
        │   ├── data.bin (registered binary, if keep_reg=True)
        │   └── [visualization PNGs]
        ├── plane02/
        │   └── ...
        ├── volume_stats.npy          # Per-plane statistics
        ├── mean_volume_signal.png    # Signal strength across planes
        └── rastermap.png             # Clustered activity (if rastermap installed)

    **Multi-ROI Merging:**

    When input filenames contain "roi" (case-insensitive), e.g., "plane01_roi01.tif",
    "plane01_roi02.tif", the pipeline automatically detects multi-ROI acquisition and
    performs horizontal stitching after planar processing::

        save_path/
        ├── plane01_roi01/           # Individual ROI results
        │   └── [Suite2p outputs]
        ├── plane01_roi02/
        │   └── [Suite2p outputs]
        ├── merged_mrois/            # Merged results (used for volumetric stats)
        │   ├── plane01/
        │   │   ├── ops.npy          # Merged ops with Lx = sum of ROI widths
        │   │   ├── stat.npy         # Concatenated ROIs with xpix offsets applied
        │   │   ├── F.npy, spks.npy  # Concatenated traces
        │   │   ├── data.bin         # Horizontally stitched binary
        │   │   └── [merged visualizations]
        │   └── plane02/
        │       └── ...
        └── [volumetric outputs as above]

    The merging process:
    - Groups directories by plane number (e.g., "plane01_roi01", "plane01_roi02" → "plane01")
    - Horizontally concatenates images (refImg, meanImg, meanImgE, max_proj)
    - Adjusts stat["xpix"] and stat["med"] coordinates to account for horizontal offset
    - Concatenates fluorescence traces (F, Fneu, spks) and cell classifications (iscell)
    - Creates stitched binary files by horizontally stacking frames

    **Supported Input Scenarios:**

    1. TIFF files (standard workflow)::

        input_files = ["plane01.tif", "plane02.tif", "plane03.tif"]
        lsp.run_volume(input_files, save_path="outputs")

    2. Binary files from interrupted processing::

        input_files = [
            "plane01_stitched/data_raw.bin",
            "plane02_stitched/data_raw.bin",
        ]
        lsp.run_volume(input_files)  # Processes in-place

    3. Multi-ROI TIFF files (automatic merging)::

        input_files = [
            "plane01_roi01.tif", "plane01_roi02.tif",
            "plane02_roi01.tif", "plane02_roi02.tif",
        ]
        lsp.run_volume(input_files, save_path="outputs")

    4. Mixed input types::

        input_files = [
            "plane01.tif",                      # New TIFF
            "plane02_stitched/data_raw.bin",    # Existing binary
        ]
        lsp.run_volume(input_files, save_path="outputs")

    See Also
    --------
    run_plane : Process a single imaging plane
    run_plane_bin : Process an existing binary file through Suite2p pipeline
    merge_mrois : Manual multi-ROI merging function
    """
    from mbo_utilities.file_io import get_files, get_plane_from_filename

    if not input_files:
        raise Exception("No input files given.")

    start = time.time()
    if save_path is None:
        save_path = Path(input_files[0]).parent

    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True)

    all_ops = []
    for z, file in enumerate(input_files):
        file_path = Path(file)
        start_file = time.time()

        # Create a fresh kwargs dict for each iteration to avoid cross-contamination
        call_kwargs = dict(kwargs)

        # Determine save_path based on input type
        # For binary inputs with ops.npy, use the parent directory (in-place processing)
        # For TIFF/ZARR/other formats, create subdirectory based on filename tag
        if file_path.suffix == ".bin" and file_path.parent.joinpath("ops.npy").exists():
            # Input is a binary from mbo.imwrite or previous processing
            # Process in-place (parent directory contains ops.npy and data_raw.bin)
            print(f"Detected existing binary with ops.npy: {file_path}")
            tag = file_path.parent.name
            plane_save_path = file_path.parent

            # Prefer data_raw.bin if it exists, otherwise use whatever binary was provided
            if (file_path.parent / "data_raw.bin").exists():
                input_to_process = file_path.parent / "data_raw.bin"
            else:
                input_to_process = file_path
        else:
            # Input is TIFF, ZARR, or other format - derive tag from filename
            tag = derive_tag_from_filename(file_path.name)
            plane_num = get_plane_from_filename(tag, fallback=len(all_ops))
            plane_save_path = Path(save_path).joinpath(tag)
            plane_save_path.mkdir(exist_ok=True)
            input_to_process = file_path
            # Set plane number for non-binary inputs
            call_kwargs["plane"] = plane_num

        # Always call run_plane - it has all the logic to determine what needs processing
        try:
            ops_file = run_plane(
                input_path=input_to_process,
                save_path=plane_save_path,
                ops=ops,
                keep_reg=keep_reg,
                keep_raw=keep_raw,
                force_reg=force_reg,
                force_detect=force_detect,
                dff_window_size=dff_window_size,
                dff_percentile=dff_percentile,
                save_json=save_json,
                **call_kwargs,
            )
            all_ops.append(ops_file)
            print(f"Completed {file_path.name} -> {ops_file}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            import traceback
            traceback.print_exc()
            # Continue with next file rather than failing entire volume
            continue
        finally:
            end_file = time.time()
            print(f"Time for {file}: {(end_file - start_file) / 60:0.1f} min")
            print(f"CPU {get_cpu_percent():4.1f}% | RAM {get_ram_used() / 1024:5.2f} GB")
            gc.collect()

    end = time.time()
    print(f"Total time for volume: {(end - start) / 60:0.1f} min")

    if "roi" in Path(input_files[0]).stem.lower():
        print("Detected mROI data, merging ROIs for each z-plane...")
        from .merging import merge_mrois

        merged_savepath = save_path.joinpath("merged_mrois")
        merge_mrois(save_path, merged_savepath)
        save_path = merged_savepath

        all_ops = sorted(get_files(merged_savepath, "ops.npy", 2))
        print(f"Planes found after merge: {len(all_ops)}")
    else:
        all_ops = sorted(get_files(save_path, "ops.npy", 2))
        print(f"No mROI data detected, planes found: {len(all_ops)}")

    try:
        zstats_file = get_volume_stats(all_ops, overwrite=True)

        plot_volume_neuron_counts(zstats_file, save_path)
        plot_volume_signal(
            zstats_file, os.path.join(save_path, "mean_volume_signal.png")
        )
        # todo: why is suite2p not saving timings to ops.npy?
        # plot_execution_time(zstats_file, os.path.join(save_path, "execution_time.png"))

        res_z = [
            load_planar_results(ops_path, z_plane=i)
            for i, ops_path in enumerate(all_ops)
        ]
        all_spks = np.concatenate([res["spks"] for res in res_z], axis=0)
        try:
            print("Importing rastermap...")
            from rastermap import Rastermap
            from lbm_suite2p_python.zplane import plot_rastermap
            print("Rastermap import complete...")

            HAS_RASTERMAP = True
        except ImportError:
            Rastermap = None
            HAS_RASTERMAP = False
            plot_rastermap = None
        if HAS_RASTERMAP:
            model = Rastermap(
                n_clusters=100,
                n_PCs=100,
                locality=0.75,
                time_lag_window=15,
            ).fit(all_spks)
            np.save(os.path.join(save_path, "model.npy"), model)
            title_kwargs = {"fontsize": 8, "y": 0.95}
            if plot_rastermap is not None:
                plot_rastermap(
                    all_spks,
                    model,
                    neuron_bin_size=20,
                    xmax=min(2000, all_spks.shape[1]),
                    save_path=os.path.join(save_path, "rastermap.png"),
                    title_kwargs=title_kwargs,
                    title="Rastermap Sorted Activity",
                )
        else:
            print("No rastermap is available.")

    except Exception:
        print("Volume statistics failed.")
        print("Traceback: ", traceback.format_exc())

    print(f"Processing completed for {len(input_files)} files.")
    return all_ops


def _should_write_bin(ops_path: Path, force: bool = False, *, validate_chan2: bool | None = None, expected_dtype: np.dtype = np.int16) -> bool:
    if force:
        return True
    ops_path = Path(ops_path)
    if not ops_path.is_file():
        return True
    raw_path = ops_path.parent / "data_raw.bin"
    reg_path = ops_path.parent / "data.bin"
    chan2_path = ops_path.parent / "data_chan2.bin"

    # If neither raw nor registered binary exists, need to write
    if not raw_path.is_file() and not reg_path.is_file():
        return True

    # Use whichever binary exists for validation (prefer raw)
    binary_to_validate = raw_path if raw_path.is_file() else reg_path
    try:
        ops = np.load(ops_path, allow_pickle=True).item()
        if validate_chan2 is None:
            validate_chan2 = (ops.get("align_by_chan", 1) == 2)
        Ly = ops.get("Ly")
        Lx = ops.get("Lx")
        nframes_raw = ops.get("nframes_chan1") or ops.get("nframes") or ops.get("num_frames")
        if (None in (nframes_raw, Ly, Lx)) or (nframes_raw <= 0 or Ly <= 0 or Lx <= 0):
            return True
        expected_size_raw = int(nframes_raw) * int(Ly) * int(Lx) * np.dtype(expected_dtype).itemsize
        actual_size_raw = binary_to_validate.stat().st_size
        if actual_size_raw != expected_size_raw or actual_size_raw == 0:
            return True
        try:
            arr = np.memmap(binary_to_validate, dtype=expected_dtype, mode="r", shape=(int(nframes_raw), int(Ly), int(Lx)))
            _ = arr[0, 0, 0]
            del arr
        except Exception:
            return True
        if validate_chan2:
            nframes_chan2 = ops.get("nframes_chan2")
            if (not chan2_path.is_file()) or (nframes_chan2 is None) or (nframes_chan2 <= 0):
                return True
            expected_size_chan2 = int(nframes_chan2) * int(Ly) * int(Lx) * np.dtype(expected_dtype).itemsize
            actual_size_chan2 = chan2_path.stat().st_size
            if actual_size_chan2 != expected_size_chan2 or actual_size_chan2 == 0:
                return True
            try:
                arr2 = np.memmap(chan2_path, dtype=expected_dtype, mode="r", shape=(int(nframes_chan2), int(Ly), int(Lx)))
                _ = arr2[0, 0, 0]
                del arr2
            except Exception:
                return True
        return False
    except Exception as e:
        print(f"Bin validation failed for {ops_path.parent}: {e}")
        return True


def _should_register(ops_path: str | Path) -> bool:
    """
    Determine whether Suite2p registration still needs to be performed.

    Registration is considered complete if any of the following hold:
      - A reference image (refImg) exists and is a valid ndarray
      - meanImg exists (Suite2p always produces it post-registration)
      - Valid registration offsets (xoff/yoff) are present

    Returns True if registration *should* be run, False otherwise.
    """
    ops = load_ops(ops_path)

    has_ref = isinstance(ops.get("refImg"), np.ndarray)
    has_mean = isinstance(ops.get("meanImg"), np.ndarray)
    has_offsets = ("xoff" in ops and np.any(np.isfinite(ops["xoff"]))) or (
        "yoff" in ops and np.any(np.isfinite(ops["yoff"]))
    )
    has_metrics = any(k in ops for k in ("regDX", "regPC", "regPC1", "regDX1"))

    # registration done if any of these are true
    registration_done = has_ref or has_mean or has_offsets or has_metrics
    return not registration_done


def run_plane_bin(ops) -> bool:
    from contextlib import nullcontext
    from suite2p.io.binary import BinaryFile
    from suite2p.run_s2p import pipeline

    ops = load_ops(ops)
    Ly, Lx = int(ops["Ly"]), int(ops["Lx"])

    raw_file = ops.get("raw_file")
    n_func = ops.get("nframes_chan1") or ops.get("nframes") or ops.get("n_frames")
    if raw_file is None or n_func is None:
        raise KeyError("Missing raw_file or nframes_chan1")
    n_func = int(n_func)

    ops_parent = Path(ops["ops_path"]).parent
    ops["save_path"] = ops_parent

    reg_file = ops_parent / "data.bin"
    ops["reg_file"] = str(reg_file)

    chan2_file = ops.get("chan2_file", "")
    use_chan2 = bool(chan2_file) and Path(chan2_file).exists()
    n_chan2 = int(ops.get("nframes_chan2", 0)) if use_chan2 else 0

    n_align = n_func if not use_chan2 else min(n_func, n_chan2)
    if n_align <= 0:
        raise ValueError("Non-positive frame count after alignment selection.")
    if use_chan2 and (n_func != n_chan2):
        print(f"[run_plane_bin] Trimming to {n_align} frames (func={n_func}, chan2={n_chan2}).")

    ops["functional_chan"] = 1
    ops["align_by_chan"] = 2 if use_chan2 else 1
    ops["nchannels"] = 2 if use_chan2 else 1
    ops["nframes"] = n_align
    ops["nframes_chan1"] = n_align
    if use_chan2:
        ops["nframes_chan2"] = n_align

    if "diameter" in ops:
        if ops["diameter"] is not None and np.isnan(ops["diameter"]):
            ops["diameter"] = 8
        if (ops["diameter"] in (None, 0)) and ops.get("anatomical_only", 0) > 0:
            ops["diameter"] = 8
            print("Warning: diameter was not set, defaulting to 8.")

    reg_file_chan2 = ops_parent / "data_chan2_reg.bin" if use_chan2 else None

    ops["anatomical_red"] = False
    ops["chan2_thres"] = 0.1

    with (
        BinaryFile(Ly=Ly, Lx=Lx, filename=str(reg_file), n_frames=n_align) as f_reg,
        BinaryFile(Ly=Ly, Lx=Lx, filename=str(raw_file), n_frames=n_align) as f_raw,
        (BinaryFile(Ly=Ly, Lx=Lx, filename=str(reg_file_chan2), n_frames=n_align) if use_chan2 else nullcontext()) as f_reg_chan2,
        (BinaryFile(Ly=Ly, Lx=Lx, filename=str(chan2_file), n_frames=n_align) if use_chan2 else nullcontext()) as f_raw_chan2,
    ):
        ops = pipeline(
            f_reg=f_reg,
            f_raw=f_raw,
            f_reg_chan2=f_reg_chan2 if use_chan2 else None,
            f_raw_chan2=f_raw_chan2 if use_chan2 else None,
            run_registration=bool(ops.get("do_registration", True)),
            ops=ops,
            stat=None,
        )

    if use_chan2:
        ops["reg_file_chan2"] = str(reg_file_chan2)
    np.save(ops["ops_path"], ops)
    return True


def run_plane(
    input_path: str | Path,
    save_path: str | Path | None = None,
    ops: dict | str | Path = None,
    chan2_file: str | Path | None = None,
    keep_raw: bool = False,
    keep_reg: bool = True,
    force_reg: bool = False,
    force_detect: bool = False,
    dff_window_size: int = 300,
    dff_percentile: int = 20,
    save_json: bool = False,
    **kwargs,
) -> Path:
    """
    Processes a single imaging plane using suite2p, handling registration, segmentation,
    and plotting of results.

    Parameters
    ----------
    input_path : str or Path
        Full path to the file to process, with the file extension.
    save_path : str or Path, optional
        Directory to save the results.
    ops : dict, str or Path, optional
        Path to or dict of user‐supplied ops.npy. If given, it overrides any existing or generated ops.
    chan2_file : str, optional
        Path to structural / anatomical data used for registration.
    keep_raw : bool, default false
        if true, do not delete the raw binary (`data_raw.bin`) after processing.
    keep_reg : bool, default false
        if true, do not delete the registered binary (`data.bin`) after processing.
    force_reg : bool, default false
        if true, force a new registration even if existing shifts are found in ops.npy.
    force_detect : bool, default false
        if true, force roi detection even if an existing stat.npy is present.
    dff_window_size : int, default 10
        Size of the window for calculating dF/F traces.
    dff_percentile : int, default 8
        Percentile to use for baseline F₀ estimation in dF/F calculation.
    save_json : bool, default True
        If true, saves ops as a JSON file in addition to npy.
    **kwargs : dict, optional

    Returns
    -------
    dict
        Processed ops dictionary containing results.

    Raises
    ------
    FileNotFoundError
        If `input_tiff` does not exist.
    TypeError
        If `save_folder` is not a string.
    Exception
        If plotting functions fail.

    Notes
    -----
    - ops supplied to the function via `ops_file` will take precendence over previously saved ops.npy files.

    Example
    -------
    >> import mbo_utilities as mbo
    >> import lbm_suite2p_python as lsp

    Get a list of z-planes in Txy format
    >> input_files = mbo.get_files(assembled_path, str_contains='tif', max_depth=3)
    >> metadata = mbo.get_metadata(input_files[0])
    >> ops = suite2p.default_ops()

    Automatically fill in metadata needed for processing (frame rate, pixel resolution, etc..)
    >> mbo_ops = mbo.params_from_metadata(metadata, ops) # handles framerate, Lx/Ly, etc

    Run a single z-plane through suite2p, keeping raw and registered files.
    >> output_ops = lsp.run_plane(input_files[0], save_path="D://data//outputs", keep_raw=True, keep_registered=True, force_reg=True, force_detect=True)
    """
    from mbo_utilities.array_types import MboRawArray
    from mbo_utilities.lazy_array import imread, imwrite
    from mbo_utilities.metadata import get_metadata

    if "debug" in kwargs:
        logger.setLevel(logging.DEBUG)
        logger.info("Debug mode enabled.")

    assert isinstance(
        input_path, (Path, str)
    ), f"input_path should be a pathlib.Path or string, not: {type(input_path)}"
    input_path = Path(input_path)
    input_parent = input_path.parent

    assert isinstance(
        save_path, (Path, str, type(None))
    ), f"save_path should be a pathlib.Path or string, not: {type(save_path)}"
    if save_path is None:
        logger.debug(f"save_path is None, using parent of input file: {input_parent}")
        save_path = input_parent
    else:
        save_path = Path(save_path)
        if not save_path.parent.is_dir():
            raise ValueError(
                f"save_path does not have a valid parent directory: {save_path}"
            )
        save_path.mkdir(exist_ok=True)

    # Check if input is already a binary at the target location
    is_binary_input = input_path.suffix == ".bin"
    binary_at_target = is_binary_input and input_path.parent == save_path
    skip_imwrite = False

    if binary_at_target:
        print(f"Input is already a binary at target location: {input_path}")
        # Check if ops.npy exists
        if not (save_path / "ops.npy").exists():
            raise FileNotFoundError(
                f"ops.npy not found at {save_path}. Cannot process binary without ops file."
            )
        skip_imwrite = True

    ops_default = default_ops()
    ops_user = load_ops(ops) if ops else {}
    ops = {**ops_default, **ops_user, "data_path": str(input_path.resolve())}

    # suite2p diameter handling
    if (
        isinstance(ops["diameter"], list)
        and len(ops["diameter"]) > 1
        and ops["aspect"] == 1.0
    ):
        ops["aspect"] = ops["diameter"][0] / ops["diameter"][1]  # noqa

    # Skip imread if we're using existing binary
    if skip_imwrite:
        file = None
        # Load metadata from existing ops.npy
        existing_ops = np.load(save_path / "ops.npy", allow_pickle=True).item()
        metadata = {k: v for k, v in existing_ops.items() if k in ("plane", "fs", "dx", "dy", "Ly", "Lx", "nframes")}
    else:
        file = imread(input_path)
        if isinstance(file, MboRawArray):
            raise TypeError(
                "Input file appears to be a raw array. Please provide a planar input file."
            )
        if hasattr(file, "metadata"):
            metadata = file.metadata  # noqa
        else:
            metadata = get_metadata(input_path)

    if "plane" in ops:
        plane = ops["plane"]
        metadata["plane"] = plane
    elif "plane" in metadata:
        plane = metadata["plane"]
        ops["plane"] = plane
    else:
        # get the plane from the filename
        plane = get_plane_from_filename(input_path, ops.get("plane", None))
        ops["plane"] = plane
        metadata["plane"] = plane

    plane_dir = save_path
    ops["save_path"] = str(plane_dir.resolve())

    needs_detect = False
    if force_detect:
        print(f"Roi detection forced for plane {plane}.")
        needs_detect = True
    elif ops["roidetect"]:
        if (plane_dir / "stat.npy").is_file():
            # make sure this is a valid stat.npy file
            stat = np.load(plane_dir / "stat.npy", allow_pickle=True)
            if stat is None or len(stat) == 0:
                print(
                    f"Detected empty stat.npy, forcing roi detection for plane {plane}."
                )
                needs_detect = True
            else:
                print(
                    f"Roi detection skipped, stat.npy already exists for plane {plane}."
                )
                needs_detect = False
        else:
            print(
                f"ops['roidetect'] is True with no stat.npy file present, "
                f"proceeding with segmentation/detection for plane {plane}."
            )
            needs_detect = True
    elif (plane_dir / "stat.npy").is_file():
        # check contents of stat.npy
        stat = np.load(plane_dir / "stat.npy", allow_pickle=True)
        if stat is None or len(stat) == 0:
            print(f"Detected empty stat.npy, forcing roi detection for plane {plane}.")
            needs_detect = True
        else:
            print(f"Roi detection skipped, stat.npy already exists for plane {plane}.")
            needs_detect = True

    ops_file = plane_dir / "ops.npy"

    if skip_imwrite:
        print(f"Skipping binary write, using existing binary at {input_path}")
    elif _should_write_bin(ops_file, force=force_reg):
        md_combined = {**metadata, **ops}
        imwrite(
            file,
            plane_dir,
            ext=".bin",
            metadata=md_combined,
            register_z=False,
            output_name="data_raw.bin",
            overwrite=True
        )
    else:
        print(
            f"Skipping data_raw.bin write, already exists and passes data validation checks."
        )

    ops_outpath = (
        np.load(ops_file, allow_pickle=True).item()
        if (plane_dir / "ops.npy").exists()
        else {}
    )

    if force_reg:
        needs_reg = True
    else:
        if not ops_file.exists():
            needs_reg = True
        else:
            needs_reg = _should_register(ops_file)

    # Build ops dict - user settings should not be overwritten
    ops = {
        **ops_default,
        **ops_outpath,
        **ops_user,
        "ops_path": str(ops_file),
        "save_path": str(plane_dir),
        "raw_file": str((plane_dir / "data_raw.bin").resolve()),
        "reg_file": str((plane_dir / "data.bin").resolve()),
    }

    # Only set do_registration/roidetect if user hasn't explicitly provided them
    if "do_registration" not in ops_user:
        ops["do_registration"] = int(needs_reg)
    if "roidetect" not in ops_user:
        ops["roidetect"] = int(needs_detect)

    # optional structural (channel 2) input
    if chan2_file is not None:
        chan2_file = Path(chan2_file)
        if not chan2_file.exists():
            raise FileNotFoundError(f"chan2_path not found: {chan2_file}")

        chan2_data = imread(chan2_file)
        chan2_md = chan2_data.metadata if hasattr(chan2_data, "metadata") else {}
        chan2_frames = chan2_md.get("num_frames") or chan2_md.get("nframes") or chan2_data.shape[0]

        # write channel 2 binary automatically
        imwrite(chan2_data, plane_dir, ext=".bin", metadata=chan2_md, register_z=False, structural=True)
        ops["chan2_file"] = str((plane_dir / "data_chan2.bin").resolve())
        ops["nframes_chan2"] = int(chan2_frames)
        ops["nchannels"] = 2
        ops["align_by_chan"] = 2

    if "nframes" not in ops:
        if "metadata" in ops and "shape" in ops["metadata"]:
            ops["nframes"] = ops["metadata"]["shape"][0]
        elif "num_frames" in metadata:
            ops["nframes"] = metadata["num_frames"]
        elif "nframes" in metadata:
            ops["nframes"] = metadata["nframes"]
        elif file is not None and hasattr(file, "shape") and len(file.shape) >= 1:
            ops["nframes"] = file.shape[0]
        elif "shape" in metadata:
            ops["nframes"] = metadata["shape"][0]
        else:
            raise KeyError(
                "missing frame count (nframes) in ops or metadata, and cannot infer from data"
            )

    try:
        processed = run_plane_bin(ops)
    except Exception as e:
        print(e)
        processed = False

    if not processed:
        print(f"Skipping {ops_file.name}, processing was not completed.")
        return ops_file

    if save_json:
        # convert ops dict to JSON serializable and save as ops.json
        ops_to_json(ops_file)

    raw_file = Path(ops.get("raw_file", plane_dir / "data_raw.bin"))
    reg_file = Path(ops.get("reg_file", plane_dir / "data.bin"))

    try:
        if not keep_raw and raw_file.exists():
            raw_file.unlink(missing_ok=True)
        if not keep_reg and reg_file.exists():
            reg_file.unlink(missing_ok=True)
    except Exception as e:
        print(e)

    save_pc_panels_and_metrics(ops_file, plane_dir / "pc_metrics")

    try:
        plot_zplane_figures(
            plane_dir,
            dff_percentile=dff_percentile,
            dff_window_size=dff_window_size,
            run_rastermap=kwargs.get("run_rastermap", False),
        )
    except Exception:
        traceback.print_exc()
    return ops_file


def run_grid_search(
    base_ops: dict,
    grid_search_dict: dict,
    input_file: Path | str,
    save_root: Path | str,
    force_reg: bool,
    force_detect: bool,
):
    """
    Run a grid search over all combinations of the input suite2p parameters.

    Parameters
    ----------
    base_ops : dict
        Dictionary of default Suite2p ops to start from. Each parameter combination will override values in this dictionary.

    grid_search_dict : dict
        Dictionary mapping parameter names (str) to a list of values to grid search.
        Each combination of values across parameters will be run once.

    input_file : str or Path
        Path to the input data file, currently only supports tiff.

    save_root : str or Path
        Root directory where each parameter combination's output will be saved.
        A subdirectory will be created for each run using a short parameter tag.

    force_reg : bool
        Whether to force suite2p registration.

    force_detect : bool
        Whether to force suite2p detection.

    Notes
    -----
    - Subfolder names for each parameter are abbreviated to 3-character keys and truncated/rounded values.
    - For available Suite2p parameters, see: http://suite2p.readthedocs.io/en/latest/parameters.html

    Examples
    --------
    >>> import lbm_suite2p_python as lsp
    >>> import suite2p
    >>> base_ops = suite2p.default_ops()
    >>> base_ops["anatomical_only"] = 3
    >>> base_ops["diameter"] = 6
    >>> lsp.run_grid_search(
    ...     base_ops,
    ...     {"threshold_scaling": [1.0, 1.2], "tau": [0.1, 0.15]},
    ...     input_file="/mnt/data/assembled_plane_03.tiff",
    ...     save_root="/mnt/grid_search/"
    ... )

    This will create the following output directory structure::

        /mnt/data/grid_search/
        ├── thr1.00_tau0.10/
        │   └── suite2p output for threshold_scaling=1.0, tau=0.1
        ├── thr1.00_tau0.15/
        ├── thr1.20_tau0.10/
        └── thr1.20_tau0.15/

    """

    save_root = Path(save_root)
    save_root.mkdir(exist_ok=True)

    print(f"Saving grid-search in {save_root}")

    param_names = list(grid_search_dict.keys())
    param_values = list(grid_search_dict.values())
    param_combos = list(product(*param_values))

    for combo in param_combos:
        ops = copy.deepcopy(base_ops)
        combo_dict = dict(zip(param_names, combo))
        ops.update(combo_dict)

        tag_parts = [
            f"{k[:3]}{v:.2f}" if isinstance(v, float) else f"{k[:3]}{v}"
            for k, v in combo_dict.items()
        ]
        tag = "_".join(tag_parts)
        save_path = save_root / tag
        print(f"\nRunning grid search combination: {tag}")

        ops_file = save_path / "ops.npy"

        # Skip runs that are already registered
        if ops_file.exists() and not force_reg and not _should_register(ops_file):
            print(f"Skipping {tag}: registration already complete.")
            continue

        run_plane(
            input_path=input_file,
            save_path=save_path,
            ops=ops,
            keep_reg=True,
            keep_raw=True,
            force_reg=force_reg,
            force_detect=force_detect,
        )
