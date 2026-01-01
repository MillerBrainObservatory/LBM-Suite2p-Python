"""
grid search module for parameter optimization.

provides functions to:
- run grid search over suite2p/cellpose parameters
- collect and compare results across parameter combinations
- visualize quality metrics and detection results
"""

import copy
import shutil
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import skew

from lbm_suite2p_python.postprocessing import (
    load_ops,
    load_planar_results,
    dff_shot_noise,
)
from lbm_suite2p_python.zplane import (
    plot_zplane_figures,
)

# registration parameters that require re-registration when changed
REGISTRATION_PARAMS = {
    "do_registration", "two_step_registration", "keep_movie_raw",
    "nimg_init", "batch_size", "maxregshift", "align_by_chan",
    "reg_tif", "reg_tif_chan2", "subpixel",
    "smooth_sigma_time", "smooth_sigma",
    "th_badframes", "norm_frames", "force_refImg", "pad_fft",
    "nonrigid", "block_size", "snr_thresh", "maxregshiftNR",
    "1Preg", "spatial_hp_reg", "pre_smooth", "spatial_taper",
}


def grid_search(
    input_data,
    save_path: Path | str,
    grid_params: dict,
    ops: dict = None,
    planes: list | int = None,
    roi_mode: int = None,
    force_reg: bool = False,
    force_detect: bool = True,
    reader_kwargs: dict = None,
    writer_kwargs: dict = None,
    # deprecated parameters
    roi: int = None,
    **kwargs,
):
    """
    Run a grid search over all combinations of Suite2p parameters.

    Tests all combinations of parameters in `grid_params`, running detection
    for each combination. When only searching detection parameters, the binary
    is written and registered once, then reused for all combinations.

    Parameters
    ----------
    input_data : str, Path, list, or lazy array
        Input data source. Can be:
        - Path to a file (TIFF, Zarr, HDF5, .bin)
        - Path to a directory containing supported files
        - An mbo_utilities lazy array (MboRawArray, Suite2pArray, etc.)
    save_path : str or Path
        Root directory where results will be saved.
    grid_params : dict
        Dictionary mapping parameter names to lists of values to test.
        All combinations will be tested (Cartesian product).
    ops : dict, optional
        Base ops dictionary. If None, uses `default_ops()`.
    planes : int or list, optional
        Which z-planes to process (1-indexed, 0 is not valid).
    roi_mode : int, optional
        ROI handling for multi-ROI ScanImage data.
    force_reg : bool, default False
        If True, force registration even if already done.
    force_detect : bool, default True
        If True, force ROI detection for each combination.
    reader_kwargs : dict, optional
        Keyword arguments passed to mbo_utilities.imread().
    writer_kwargs : dict, optional
        Keyword arguments passed when writing binary files.
    **kwargs
        Additional arguments passed to Suite2p.

    Returns
    -------
    Path
        Path to save_path directory containing all results.

    Examples
    --------
    >>> import lbm_suite2p_python as lsp
    >>>
    >>> # search detection params on plane 7
    >>> lsp.grid_search(
    ...     input_data="D:/data/raw.tif",
    ...     save_path="D:/results/grid_search",
    ...     planes=7,
    ...     grid_params={
    ...         "threshold_scaling": [0.8, 1.0, 1.2],
    ...         "diameter": [6, 8],
    ...     },
    ... )
    >>>
    >>> # collect and analyze results
    >>> df = lsp.collect_grid_results("D:/results/grid_search", grid_params)
    >>> best = lsp.get_best_parameters(df)
    """
    import warnings

    # validate planes parameter (1-indexed, 0 is not valid)
    if planes is not None:
        planes_list = [planes] if isinstance(planes, int) else list(planes)
        if 0 in planes_list:
            raise ValueError(
                "planes parameter uses 1-based indexing. "
                "Plane 0 is not valid - use 1 for the first plane."
            )
        if any(p < 0 for p in planes_list):
            raise ValueError(
                "planes parameter cannot contain negative values. "
                "Use 1-based indexing (1 for first plane, 2 for second, etc.)."
            )

    # handle deprecated parameter names
    if roi is not None:
        warnings.warn(
            "The 'roi' parameter is deprecated and will be removed in a future version. "
            "Use 'roi_mode' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if roi_mode is None:
            roi_mode = roi
        else:
            raise ValueError("Cannot specify both 'roi' (deprecated) and 'roi_mode'.")
    from mbo_utilities import imread
    from mbo_utilities._writers import _write_plane
    from mbo_utilities.metadata import get_param, get_voxel_size, detect_stack_type

    from lbm_suite2p_python.default_ops import default_ops
    from lbm_suite2p_python.run_lsp import (
        run_plane,
        run_plane_bin,
    )
    from lbm_suite2p_python.utils import _is_lazy_array, _get_num_planes
    from mbo_utilities.arrays import _normalize_planes, supports_roi

    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True, parents=True)

    reader_kwargs = reader_kwargs or {}
    writer_kwargs = writer_kwargs or {}

    # always start with full default ops, then overlay user ops
    base_default_ops = default_ops()
    if ops is not None:
        base_default_ops.update(ops)
    ops = base_default_ops

    # check if any registration parameters are being searched
    reg_params_in_grid = set(grid_params.keys()) & REGISTRATION_PARAMS
    searching_reg_params = len(reg_params_in_grid) > 0

    if searching_reg_params:
        print(f"Registration parameters in grid: {reg_params_in_grid}")
        print("Each combination will require separate registration.")
    else:
        print("Only detection parameters in grid - will reuse registered binary.")

    print(f"Grid search: {save_path}")
    print(f"Parameters: {list(grid_params.keys())}")

    # load input data
    print(f"\nLoading input data...")
    if _is_lazy_array(input_data):
        arr = input_data
        filenames = getattr(arr, "filenames", [])
        print(f"  Input: {type(arr).__name__} (pre-loaded array)")
    elif isinstance(input_data, (str, Path)):
        input_path = Path(input_data)
        print(f"  Input: {input_path}")
        arr = imread(input_path, **reader_kwargs)
        print(f"  Loaded as: {type(arr).__name__}")
        filenames = getattr(arr, "filenames", [input_path])
    else:
        raise TypeError(
            f"input_data must be a path or lazy array. Got: {type(input_data)}"
        )

    # configure ROI on the lazy array
    if roi_mode is not None and supports_roi(arr):
        arr.roi = roi_mode

    # get dimensions
    num_planes = _get_num_planes(arr)
    num_frames = arr.shape[0]
    Ly, Lx = arr.shape[-2], arr.shape[-1]

    print(f"\nDataset info:")
    print(f"  Shape: {arr.shape}")
    print(f"  Frames: {num_frames}")
    print(f"  Planes: {num_planes}")
    print(f"  Dimensions: {Ly} x {Lx}")

    # normalize planes to process
    planes_to_process = _normalize_planes(planes, num_planes)
    print(f"  Processing planes: {[p+1 for p in planes_to_process]}")

    # extract metadata and configure base ops
    metadata = dict(getattr(arr, "metadata", {}) or {})
    fs = get_param(metadata, "fs")
    if fs:
        ops["fs"] = fs

    voxel = get_voxel_size(metadata)
    if voxel.dx != 1.0 or voxel.dy != 1.0:
        ops["dx"] = voxel.dx
        ops["dy"] = voxel.dy
        ops["pixel_resolution"] = list(voxel.pixel_resolution)
    if voxel.dz is not None:
        ops["dz"] = voxel.dz

    # detect stack type for metadata
    stack_type = detect_stack_type(metadata)
    ops["stack_type"] = stack_type

    param_names = list(grid_params.keys())
    param_values = list(grid_params.values())
    param_combos = list(product(*param_values))
    n_total = len(param_combos)

    print(f"\nTotal combinations: {n_total}")

    # process each plane
    for plane_idx in planes_to_process:
        plane_num = plane_idx + 1
        print(f"\n{'='*60}")
        print(f"Processing plane {plane_num}")
        print(f"{'='*60}")

        # get plane dimensions
        if num_planes > 1:
            sample_frame = arr[0, plane_idx]
        else:
            sample_frame = arr[0] if arr.ndim == 3 else arr[0, 0]
        plane_Ly, plane_Lx = sample_frame.shape[-2], sample_frame.shape[-1]

        # create plane subdirectory
        plane_tag = f"plane{plane_num:02d}" if len(planes_to_process) > 1 else ""
        plane_save_path = save_path / plane_tag if plane_tag else save_path
        plane_save_path.mkdir(exist_ok=True, parents=True)

        # if not searching registration params, write binary once to _base directory
        base_dir = None
        base_ops_file = None

        if not searching_reg_params:
            base_dir = plane_save_path / "_base"
            base_dir.mkdir(exist_ok=True, parents=True)
            base_bin_file = base_dir / "data_raw.bin"
            base_ops_file = base_dir / "ops.npy"

            # build base ops for this plane (start with defaults, then user ops)
            base_ops = copy.deepcopy(default_ops())
            base_ops.update(ops)
            base_ops.update({
                "Ly": plane_Ly,
                "Lx": plane_Lx,
                "nframes": num_frames,
                "nframes_chan1": num_frames,
                "plane": plane_num,
                "data_path": str(plane_save_path),
                "save_path": str(base_dir),
                "ops_path": str(base_ops_file),
                "raw_file": str(base_bin_file),
                "reg_file": str(base_dir / "data.bin"),
                "shape": (num_frames, plane_Ly, plane_Lx),
            })

            # write binary if needed
            if not base_bin_file.exists() or force_reg:
                print(f"\nWriting base binary ({num_frames} frames, {plane_Ly}x{plane_Lx})...")
                _write_plane(
                    arr,
                    base_bin_file,
                    overwrite=True,
                    metadata=base_ops,
                    plane_index=plane_idx if num_planes > 1 else None,
                    **writer_kwargs,
                )
                np.save(base_ops_file, base_ops)
            else:
                print(f"\nUsing existing base binary: {base_bin_file}")

            # run registration once on base
            base_reg_file = base_dir / "data.bin"
            if not base_reg_file.exists() or force_reg:
                print(f"Running registration on base...")
                reg_ops = copy.deepcopy(default_ops())
                if base_ops_file.exists():
                    reg_ops.update(load_ops(base_ops_file))
                reg_ops["do_registration"] = 1
                reg_ops["roidetect"] = 0
                np.save(base_ops_file, reg_ops)
                run_plane_bin(base_ops_file)
                print(f"Registration complete.")
            else:
                print(f"Using existing registered binary: {base_reg_file}")

        # run each parameter combination
        for i, combo in enumerate(param_combos, 1):
            combo_dict = dict(zip(param_names, combo))

            # create readable folder name
            tag = _combo_to_tag(combo_dict)
            combo_save_path = plane_save_path / tag

            print(f"\n[{i}/{n_total}] {tag}")

            combo_ops_file = combo_save_path / "ops.npy"
            stat_file = combo_save_path / "stat.npy"

            # skip if already processed
            if combo_ops_file.exists() and stat_file.exists() and not force_detect:
                print(f"  Skipping: already complete")
                continue

            combo_save_path.mkdir(exist_ok=True, parents=True)

            if not searching_reg_params:
                base_reg_file = base_dir / "data.bin"
                combo_reg_file = combo_save_path / "data.bin"

                if base_reg_file.exists() and not combo_reg_file.exists():
                    shutil.copy2(base_reg_file, combo_reg_file)

                base_loaded_ops = load_ops(base_ops_file)
                combo_ops = copy.deepcopy(default_ops())
                combo_ops.update(base_loaded_ops)
                combo_ops.update(combo_dict)
                combo_ops.update({
                    "save_path": str(combo_save_path),
                    "ops_path": str(combo_ops_file),
                    "reg_file": str(combo_reg_file),
                    "do_registration": 0,
                    "roidetect": 1,
                })
                np.save(combo_ops_file, combo_ops)

                print(f"  Running detection...")
                run_plane_bin(combo_ops_file)

            else:
                combo_ops = copy.deepcopy(default_ops())
                combo_ops.update(ops)
                combo_ops.update(combo_dict)

                run_plane(
                    input_path=input_data if isinstance(input_data, (str, Path)) else filenames[0] if filenames else None,
                    save_path=plane_save_path,
                    ops=combo_ops,
                    keep_reg=True,
                    keep_raw=False,
                    force_reg=force_reg,
                    force_detect=force_detect,
                    plane_name=tag,
                    reader_kwargs=reader_kwargs,
                    writer_kwargs=writer_kwargs,
                    **kwargs,
                )

            # generate plots
            try:
                plot_zplane_figures(
                    combo_save_path,
                    dff_percentile=kwargs.get("dff_percentile", 20),
                    dff_window_size=kwargs.get("dff_window_size"),
                    dff_smooth_window=kwargs.get("dff_smooth_window"),
                )
            except Exception as e:
                print(f"  Warning: Plot generation failed: {e}")

    print(f"\n{'='*60}")
    print(f"Grid search complete: {n_total} combinations")
    print(f"Results in: {save_path}")
    print(f"{'='*60}")

    return save_path


def _combo_to_tag(combo_dict: dict) -> str:
    """Convert parameter combo dict to folder name tag."""
    tag_parts = [
        f"{k[:3]}{v:.2f}" if isinstance(v, float) else f"{k[:3]}{v}"
        for k, v in combo_dict.items()
    ]
    return "_".join(tag_parts)


def collect_grid_results(
    save_path: Path | str,
    grid_params: dict = None,
) -> pd.DataFrame:
    """
    Collect quality metrics from all grid search combinations.

    Parameters
    ----------
    save_path : Path or str
        Root directory containing grid search results.
    grid_params : dict, optional
        Grid parameters dict to extract parameter values from ops.
        If None, only combo name is included.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per combination, containing:
        - combo: folder name
        - n_accepted, n_rejected: cell counts
        - snr_median, snr_iqr: signal-to-noise ratio
        - skew_median, skew_iqr: trace skewness
        - noise_median, noise_iqr: shot noise
        - parameter columns from grid_params
    """
    save_path = Path(save_path)
    results = []

    for combo_dir in sorted(save_path.iterdir()):
        if not combo_dir.is_dir():
            continue
        if combo_dir.name in ("_base", "__pycache__"):
            continue

        ops_file = combo_dir / "ops.npy"
        if not ops_file.exists():
            # check subdirectories
            for subdir in combo_dir.iterdir():
                if subdir.is_dir() and (subdir / "ops.npy").exists():
                    ops_file = subdir / "ops.npy"
                    break
            else:
                continue

        try:
            metrics = compute_combo_metrics(ops_file)
            result = {"combo": combo_dir.name, "path": str(ops_file), **metrics}

            # add grid parameters
            if grid_params:
                loaded_ops = load_ops(ops_file)
                for param in grid_params.keys():
                    result[param] = loaded_ops.get(param)

            results.append(result)

        except Exception as e:
            print(f"Skipping {combo_dir.name}: {e}")

    if not results:
        print("No results found!")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df.sort_values("snr_median", ascending=False)
    return df


def compute_combo_metrics(ops_path: Path | str, neuropil_coef: float = 0.7) -> dict:
    """
    Compute quality metrics for a single parameter combination.

    Parameters
    ----------
    ops_path : Path or str
        Path to ops.npy file or directory containing it.
    neuropil_coef : float, default 0.7
        Neuropil correction coefficient.

    Returns
    -------
    dict
        Dictionary containing:
        - n_accepted, n_rejected: cell counts
        - snr_median, snr_iqr: signal-to-noise ratio statistics
        - skew_median, skew_iqr: skewness statistics
        - noise_median, noise_iqr: shot noise statistics
    """
    res = load_planar_results(ops_path)
    ops = load_ops(ops_path)
    fs = ops.get("fs", 30.0)

    iscell = res["iscell"]
    mask = iscell[:, 0].astype(bool) if iscell.ndim == 2 else iscell.astype(bool)
    n_accepted = mask.sum()
    n_rejected = len(mask) - n_accepted

    if n_accepted == 0:
        return {
            "n_accepted": 0,
            "n_rejected": n_rejected,
            "snr_median": np.nan,
            "snr_iqr": np.nan,
            "skew_median": np.nan,
            "skew_iqr": np.nan,
            "noise_median": np.nan,
            "noise_iqr": np.nan,
        }

    F = res["F"][mask]
    Fneu = res["Fneu"][mask]
    stat = res["stat"][mask] if isinstance(res["stat"], np.ndarray) else [
        s for s, m in zip(res["stat"], mask) if m
    ]

    # neuropil-corrected fluorescence and dF/F
    F_corr = F - neuropil_coef * Fneu
    baseline = np.percentile(F_corr, 20, axis=1, keepdims=True)
    baseline = np.maximum(baseline, 1e-6)
    dff = (F_corr - baseline) / baseline

    # snr: signal std / noise (MAD estimator)
    signal = np.std(dff, axis=1)
    noise_est = np.median(np.abs(np.diff(dff, axis=1)), axis=1) / 0.6745
    snr = signal / (noise_est + 1e-6)

    # shot noise using existing function
    shot_noise = dff_shot_noise(dff, fs)

    # skewness from stat if available, else compute
    skewness = []
    for i, s in enumerate(stat):
        if isinstance(s, dict) and "skew" in s:
            skewness.append(s["skew"])
        else:
            skewness.append(skew(dff[i]))
    skewness = np.array(skewness)

    return {
        "n_accepted": n_accepted,
        "n_rejected": n_rejected,
        "snr_median": np.median(snr),
        "snr_iqr": np.percentile(snr, 75) - np.percentile(snr, 25),
        "skew_median": np.median(skewness),
        "skew_iqr": np.percentile(skewness, 75) - np.percentile(skewness, 25),
        "noise_median": np.median(shot_noise),
        "noise_iqr": np.percentile(shot_noise, 75) - np.percentile(shot_noise, 25),
    }


def get_best_parameters(df: pd.DataFrame, grid_params: dict = None) -> dict:
    """
    Find best parameter combinations by different criteria.

    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame from collect_grid_results().
    grid_params : dict, optional
        Grid parameters to include in output.

    Returns
    -------
    dict
        Dictionary with keys 'best_snr', 'best_skew', 'best_noise',
        each containing the best row as a dict.
    """
    if len(df) == 0:
        return {}

    result = {}

    # best by SNR (highest)
    best_snr_idx = df["snr_median"].idxmax()
    result["best_snr"] = df.loc[best_snr_idx].to_dict()

    # best by skewness (highest = more events)
    best_skew_idx = df["skew_median"].idxmax()
    result["best_skew"] = df.loc[best_skew_idx].to_dict()

    # best by noise (lowest)
    best_noise_idx = df["noise_median"].idxmin()
    result["best_noise"] = df.loc[best_noise_idx].to_dict()

    # best by cell count (highest)
    best_count_idx = df["n_accepted"].idxmax()
    result["best_count"] = df.loc[best_count_idx].to_dict()

    return result


def print_best_parameters(df: pd.DataFrame, grid_params: dict = None):
    """Print summary of best parameters by different criteria."""
    best = get_best_parameters(df, grid_params)

    if not best:
        print("No results to analyze.")
        return

    print("Best Parameters by Different Criteria:")
    print("=" * 60)

    for criterion, row in best.items():
        name = criterion.replace("best_", "").upper()
        print(f"\n{name}: {row['combo']}")
        print(f"  Cells: {row['n_accepted']}")
        print(f"  SNR: {row['snr_median']:.3f}")
        print(f"  Skewness: {row['skew_median']:.3f}")
        print(f"  Shot Noise: {row['noise_median']:.4f}")

        if grid_params:
            for p in grid_params:
                if p in row:
                    print(f"  {p}: {row[p]}")


def plot_grid_metrics(
    df: pd.DataFrame,
    grid_params: dict = None,
    save_path: Path | str = None,
    figsize: tuple = (15, 10),
):
    """
    Plot quality metrics comparison across grid search combinations.

    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame from collect_grid_results().
    grid_params : dict, optional
        Grid parameters for parameter effect plots.
    save_path : Path or str, optional
        Path to save figure. If None, displays with plt.show().
    figsize : tuple, default (15, 10)
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if len(df) == 0:
        print("No results to plot.")
        return None

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    metrics = [
        ("snr_median", "SNR (median)", "higher is better", "#2ecc71"),
        ("skew_median", "Skewness (median)", "higher = more events", "#9b59b6"),
        ("noise_median", "Shot Noise (median)", "lower is better", "#e74c3c"),
    ]

    # row 1: metrics by combination
    for col, (metric, label, note, color) in enumerate(metrics):
        ax = axes[0, col]
        x = range(len(df))
        ax.bar(x, df[metric], color=color, alpha=0.8, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(df["combo"], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(label)
        ax.set_title(f"{label}\n({note})")
        ax.axhline(df[metric].median(), color="white", linestyle="--", alpha=0.5)

    # row 2: parameter effects on SNR
    params = list(grid_params.keys()) if grid_params else []
    for col, param in enumerate(params[:3]):
        ax = axes[1, col]
        if param not in df.columns:
            ax.axis("off")
            continue

        grouped = df.groupby(param)["snr_median"].agg(["mean", "std"]).reset_index()
        x = range(len(grouped))
        ax.bar(
            x, grouped["mean"], yerr=grouped["std"].fillna(0),
            color="#3498db", alpha=0.8, edgecolor="white", capsize=5
        )
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in grouped[param]])
        ax.set_xlabel(param, fontweight="bold")
        ax.set_ylabel("SNR (mean ± std)")
        ax.set_title(f"Effect of {param} on SNR")

    # hide unused subplots
    for col in range(len(params), 3):
        axes[1, col].axis("off")

    plt.suptitle("Grid Search Quality Metrics", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="black")
        plt.close(fig)
        print(f"Saved: {save_path}")
    else:
        plt.show()

    return fig


def plot_grid_distributions(
    df: pd.DataFrame,
    results_dir: Path | str,
    n_top: int = 4,
    save_path: Path | str = None,
    figsize: tuple = (15, 5),
):
    """
    Plot distributions of quality metrics for top combinations.

    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame from collect_grid_results().
    results_dir : Path or str
        Root directory containing grid search results.
    n_top : int, default 4
        Number of top combinations to include.
    save_path : Path or str, optional
        Path to save figure.
    figsize : tuple, default (15, 5)
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if len(df) == 0:
        print("No results to plot.")
        return None

    results_dir = Path(results_dir)
    top_combos = df.head(n_top)["combo"].tolist()

    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    colors = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12", "#9b59b6", "#1abc9c"]

    for combo, color in zip(top_combos, colors):
        combo_dir = results_dir / combo
        ops_file = combo_dir / "ops.npy"
        if not ops_file.exists():
            continue

        try:
            res = load_planar_results(ops_file)
            ops = load_ops(ops_file)
            fs = ops.get("fs", 30.0)

            mask = res["iscell"][:, 0].astype(bool)
            if mask.sum() == 0:
                continue

            F = res["F"][mask]
            Fneu = res["Fneu"][mask]
            stat = [s for s, m in zip(res["stat"], mask) if m]

            F_corr = F - 0.7 * Fneu
            baseline = np.percentile(F_corr, 20, axis=1, keepdims=True)
            baseline = np.maximum(baseline, 1e-6)
            dff = (F_corr - baseline) / baseline

            # compute metrics
            signal = np.std(dff, axis=1)
            noise = np.median(np.abs(np.diff(dff, axis=1)), axis=1) / 0.6745
            snr = signal / (noise + 1e-6)
            shot_noise = dff_shot_noise(dff, fs)
            skewness = np.array([
                s.get("skew", skew(dff[i])) for i, s in enumerate(stat)
            ])

            # plot distributions
            axes[0].hist(snr, bins=30, alpha=0.5, color=color, label=combo, density=True)
            axes[1].hist(skewness, bins=30, alpha=0.5, color=color, label=combo, density=True)
            axes[2].hist(shot_noise, bins=30, alpha=0.5, color=color, label=combo, density=True)

        except Exception as e:
            print(f"Error loading {combo}: {e}")

    axes[0].set_xlabel("SNR")
    axes[0].set_ylabel("Density")
    axes[0].set_title("SNR Distribution")
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("Skewness")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Skewness Distribution")

    axes[2].set_xlabel("Shot Noise")
    axes[2].set_ylabel("Density")
    axes[2].set_title("Shot Noise Distribution")

    plt.suptitle(f"Quality Metric Distributions (Top {n_top} by SNR)", fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="black")
        plt.close(fig)
        print(f"Saved: {save_path}")
    else:
        plt.show()

    return fig


def plot_grid_masks(
    df: pd.DataFrame,
    results_dir: Path | str,
    n_top: int = 4,
    img_key: str = "meanImg",
    save_path: Path | str = None,
    figsize: tuple = (12, 12),
):
    """
    Plot detection masks for top combinations side-by-side.

    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame from collect_grid_results().
    results_dir : Path or str
        Root directory containing grid search results.
    n_top : int, default 4
        Number of top combinations to show.
    img_key : str, default "meanImg"
        Background image key from ops.
    save_path : Path or str, optional
        Path to save figure.
    figsize : tuple, default (12, 12)
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if len(df) == 0:
        print("No results to plot.")
        return None

    results_dir = Path(results_dir)
    top_combos = df.head(n_top)

    # determine grid layout
    n_cols = min(n_top, 2)
    n_rows = (n_top + n_cols - 1) // n_cols

    plt.style.use("dark_background")
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_top == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    axes = axes.flatten()

    for ax, (_, row) in zip(axes, top_combos.iterrows()):
        combo_dir = results_dir / row["combo"]
        ops_file = combo_dir / "ops.npy"

        if not ops_file.exists():
            ax.axis("off")
            continue

        try:
            ops = load_ops(ops_file)
            img = ops.get(img_key, ops.get("refImg", np.zeros((512, 512))))

            ax.imshow(
                img, cmap="gray",
                vmin=np.percentile(img, 1),
                vmax=np.percentile(img, 99)
            )

            # draw ROIs
            stat = np.load(combo_dir / "stat.npy", allow_pickle=True)
            iscell = np.load(combo_dir / "iscell.npy")[:, 0].astype(bool)

            for i, s in enumerate(stat):
                if iscell[i]:
                    ax.scatter(s["xpix"], s["ypix"], s=0.1, c="lime", alpha=0.3)

            title = f"{row['combo']}\n{row['n_accepted']} cells, SNR={row['snr_median']:.2f}"
            ax.set_title(title, fontsize=10)

        except Exception as e:
            ax.set_title(f"{row['combo']}\nError: {e}", fontsize=8)

        ax.axis("off")

    # hide unused axes
    for ax in axes[len(top_combos):]:
        ax.axis("off")

    plt.suptitle(f"Top {n_top} by SNR", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="black")
        plt.close(fig)
        print(f"Saved: {save_path}")
    else:
        plt.show()

    return fig


def save_grid_results(df: pd.DataFrame, save_path: Path | str):
    """Save grid search results to CSV."""
    save_path = Path(save_path)
    csv_path = save_path / "grid_search_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    return csv_path
