import json
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.ndimage import percentile_filter
from scipy.stats import norm

from mbo_utilities.metadata import get_voxel_size
from mbo_utilities.util import load_npy


def _normalize_iscell(iscell):
    """Ensure iscell is 1D boolean array."""
    if iscell.ndim == 2:
        iscell = iscell[:, 0]
    return iscell.astype(bool)


def _load_plane_data(plane_dir, iscell=None, stat=None, ops=None):
    """
    Load iscell, stat, and ops from plane_dir if not provided.

    Parameters
    ----------
    plane_dir : str or Path, optional
        Path to Suite2p plane directory.
    iscell : np.ndarray, optional
        Pre-loaded iscell array.
    stat : np.ndarray, optional
        Pre-loaded stat array.
    ops : dict, optional
        Pre-loaded ops dictionary.

    Returns
    -------
    tuple
        (iscell, stat, ops, plane_dir) with loaded data.
    """
    if plane_dir is not None:
        plane_dir = Path(plane_dir)
        if plane_dir.suffix == ".npy":
            plane_dir = plane_dir.parent

        if iscell is None:
            iscell_path = plane_dir / "iscell.npy"
            if iscell_path.exists():
                iscell = np.load(iscell_path, allow_pickle=True)

        if stat is None:
            stat_path = plane_dir / "stat.npy"
            if stat_path.exists():
                stat = np.load(stat_path, allow_pickle=True)

        if ops is None:
            ops_path = plane_dir / "ops.npy"
            if ops_path.exists():
                ops = np.load(ops_path, allow_pickle=True).item()

    return iscell, stat, ops, plane_dir


def _save_filtered_iscell(plane_dir, iscell_filtered, iscell_original=None):
    """
    Save filtered iscell to plane_dir, preserving probability column.

    Parameters
    ----------
    plane_dir : Path
        Path to Suite2p plane directory.
    iscell_filtered : np.ndarray
        1D boolean array of filtered cell classifications.
    iscell_original : np.ndarray, optional
        Original 2D iscell array to preserve probabilities from.
    """
    plane_dir = Path(plane_dir)
    iscell_path = plane_dir / "iscell.npy"

    # Load original to preserve probabilities
    if iscell_original is None and iscell_path.exists():
        iscell_original = np.load(iscell_path, allow_pickle=True)

    if iscell_original is not None and iscell_original.ndim == 2:
        # Preserve probability column
        iscell_2d = iscell_original.copy()
        iscell_2d[:, 0] = iscell_filtered.astype(float)
    else:
        # Create 2D array with probability=1 for all cells
        iscell_2d = np.column_stack([
            iscell_filtered.astype(float),
            iscell_filtered.astype(float)  # prob=1 for cells, 0 for non-cells
        ])

    np.save(iscell_path, iscell_2d)
    print(f"Saved filtered iscell to {iscell_path}")


def filter_by_diameter(
    plane_dir=None,
    iscell=None,
    stat=None,
    ops=None,
    min_mult: float = 0.3,
    max_mult: float = 3.0,
    save: bool = False,
):
    """
    Filter cells by diameter relative to median diameter.

    Sets iscell=False for ROIs whose radius is outside [min_mult, max_mult]
    times the median diameter (from ops['diameter'] or computed from radii).

    Parameters
    ----------
    plane_dir : str or Path, optional
        Path to Suite2p plane directory. If provided, loads iscell, stat, ops.
    iscell : np.ndarray, optional
        Cell classification array (n_rois,) or (n_rois, 2).
    stat : np.ndarray or list, optional
        Suite2p stat array with ROI statistics.
    ops : dict, optional
        Suite2p ops dictionary.
    min_mult : float, default 0.3
        Minimum diameter as multiple of median.
    max_mult : float, default 3.0
        Maximum diameter as multiple of median.
    save : bool, default False
        If True, save updated iscell.npy to plane_dir.

    Returns
    -------
    iscell_filtered : np.ndarray
        Updated iscell array (1D boolean).
    removed_mask : np.ndarray
        Boolean mask of ROIs that were removed.
    info : dict
        Dictionary with 'diameters_px', 'lower_px', 'upper_px', 'n_removed'.
    """
    iscell, stat, ops, plane_dir = _load_plane_data(plane_dir, iscell, stat, ops)
    iscell_orig = _normalize_iscell(iscell)
    iscell = iscell_orig.copy()

    if "radius" not in stat[0]:
        from suite2p.detection.stats import roi_stats
        stat = roi_stats(
            stat,
            ops["Ly"],
            ops["Lx"],
            aspect=ops.get("aspect", None),
            diameter=ops.get("diameter", None),
            max_overlap=ops.get("max_overlap", None),
            do_crop=ops.get("soma_crop", 1),
        )

    radii = np.array([s["radius"] for s in stat])
    diameters_px = 2 * radii
    median_diam = ops.get("diameter", np.median(diameters_px))
    lower, upper = min_mult * median_diam, max_mult * median_diam

    valid = (diameters_px >= lower) & (diameters_px <= upper)
    removed_mask = ~valid & iscell_orig
    iscell = iscell_orig & valid

    n_removed = removed_mask.sum()
    if n_removed > 0:
        print(f"filter_by_diameter: removed {n_removed} ROIs (diameter not in [{lower:.1f}, {upper:.1f}] px)")

    if save and plane_dir is not None:
        _save_filtered_iscell(plane_dir, iscell, iscell_orig)

    return iscell, removed_mask, {
        "diameters_px": diameters_px,
        "lower_px": lower,
        "upper_px": upper,
        "n_removed": n_removed,
    }


def filter_by_max_diameter(
    plane_dir=None,
    iscell=None,
    stat=None,
    ops=None,
    max_diameter_um: float = None,
    max_diameter_px: float = None,
    min_diameter_um: float = None,
    min_diameter_px: float = None,
    pixel_size_um: float = None,
    save: bool = False,
):
    """
    Filter cells by diameter bounds in microns or pixels.

    Sets iscell=False for ROIs whose diameter is outside the specified bounds.
    Diameter is computed as 2 * radius from the ellipse fit.

    Parameters
    ----------
    plane_dir : str or Path, optional
        Path to Suite2p plane directory. If provided, loads iscell, stat, ops.
    iscell : np.ndarray, optional
        Cell classification array (n_rois,) or (n_rois, 2).
    stat : np.ndarray or list, optional
        Suite2p stat array with ROI statistics.
    ops : dict, optional
        Suite2p ops dictionary.
    max_diameter_um : float, optional
        Maximum allowed diameter in microns.
    max_diameter_px : float, optional
        Maximum allowed diameter in pixels.
    min_diameter_um : float, optional
        Minimum allowed diameter in microns.
    min_diameter_px : float, optional
        Minimum allowed diameter in pixels.
    pixel_size_um : float, optional
        Pixel size in microns. If None, attempts to read from ops.
    save : bool, default False
        If True, save updated iscell.npy to plane_dir.

    Returns
    -------
    iscell_filtered : np.ndarray
        Updated iscell array (1D boolean).
    removed_mask : np.ndarray
        Boolean mask of ROIs that were removed.
    info : dict
        Dictionary with 'diameters_px', 'min_px', 'max_px', 'n_removed'.

    Examples
    --------
    >>> # Filter by max 22 microns diameter
    >>> iscell_filtered, removed, info = filter_by_max_diameter(
    ...     plane_dir="path/to/plane01", max_diameter_um=22
    ... )

    >>> # Filter by diameter range in pixels
    >>> iscell_filtered, removed, info = filter_by_max_diameter(
    ...     plane_dir, min_diameter_px=5, max_diameter_px=30
    ... )

    >>> # Filter and save to iscell.npy
    >>> iscell_filtered, removed, info = filter_by_max_diameter(
    ...     plane_dir, max_diameter_um=22, save=True
    ... )
    """
    iscell, stat, ops, plane_dir = _load_plane_data(plane_dir, iscell, stat, ops)
    iscell_orig = _normalize_iscell(iscell)

    # Need at least one bound
    has_max = max_diameter_um is not None or max_diameter_px is not None
    has_min = min_diameter_um is not None or min_diameter_px is not None
    if not has_max and not has_min:
        raise ValueError("Must specify at least one of: max_diameter_um, max_diameter_px, "
                         "min_diameter_um, min_diameter_px")

    # get pixel size for unit conversion
    if pixel_size_um is None and ops is not None:
        voxel = get_voxel_size(ops)
        pixel_size_um = (voxel.dx + voxel.dy) / 2 if voxel.dx != 1.0 or voxel.dy != 1.0 else None

    # Get radii from stat
    if "radius" not in stat[0]:
        radii = np.array([np.sqrt(len(s["xpix"]) / np.pi) for s in stat])
    else:
        radii = np.array([s["radius"] for s in stat])

    diameters_px = 2 * radii

    # Convert bounds to pixels
    max_px = None
    min_px = None

    if max_diameter_um is not None:
        if pixel_size_um is None:
            raise ValueError("Cannot convert max_diameter_um to pixels: pixel_size_um not found")
        max_px = max_diameter_um / pixel_size_um
    elif max_diameter_px is not None:
        max_px = max_diameter_px

    if min_diameter_um is not None:
        if pixel_size_um is None:
            raise ValueError("Cannot convert min_diameter_um to pixels: pixel_size_um not found")
        min_px = min_diameter_um / pixel_size_um
    elif min_diameter_px is not None:
        min_px = min_diameter_px

    # Apply filter
    valid = np.ones(len(diameters_px), dtype=bool)
    if max_px is not None:
        valid &= diameters_px <= max_px
    if min_px is not None:
        valid &= diameters_px >= min_px

    removed_mask = ~valid & iscell_orig
    iscell_filtered = iscell_orig & valid
    n_removed = removed_mask.sum()

    if n_removed > 0:
        bounds_str = []
        if min_px is not None:
            bounds_str.append(f"min={min_px:.1f}px")
        if max_px is not None:
            bounds_str.append(f"max={max_px:.1f}px")
        print(f"filter_by_max_diameter: removed {n_removed} ROIs ({', '.join(bounds_str)})")

    if save and plane_dir is not None:
        _save_filtered_iscell(plane_dir, iscell_filtered)

    return iscell_filtered, removed_mask, {
        "diameters_px": diameters_px,
        "min_px": min_px,
        "max_px": max_px,
        "n_removed": n_removed,
    }


def filter_by_area(
    plane_dir=None,
    iscell=None,
    stat=None,
    ops=None,
    min_area_px: float = None,
    max_area_px: float = None,
    min_mult: float = None,
    max_mult: float = None,
    save: bool = False,
):
    """
    Filter cells by total area (in pixels).

    Can specify absolute bounds (min_area_px, max_area_px) or relative bounds
    as multiples of the median area (min_mult, max_mult).

    Parameters
    ----------
    plane_dir : str or Path, optional
        Path to Suite2p plane directory. If provided, loads iscell, stat, ops.
    iscell : np.ndarray, optional
        Cell classification array.
    stat : np.ndarray or list, optional
        Suite2p stat array.
    ops : dict, optional
        Suite2p ops dictionary (not used, for consistent interface).
    min_area_px : float, optional
        Minimum allowed area in pixels.
    max_area_px : float, optional
        Maximum allowed area in pixels.
    min_mult : float, optional
        Minimum area as multiple of median (e.g., 0.25 = 25% of median).
    max_mult : float, optional
        Maximum area as multiple of median (e.g., 4.0 = 400% of median).
    save : bool, default False
        If True, save updated iscell.npy to plane_dir.

    Returns
    -------
    iscell_filtered : np.ndarray
        Updated iscell array (1D boolean).
    removed_mask : np.ndarray
        Boolean mask of ROIs that were removed.
    info : dict
        Dictionary with 'areas_px', 'min_px', 'max_px', 'n_removed'.
    """
    iscell, stat, ops, plane_dir = _load_plane_data(plane_dir, iscell, stat, ops)
    iscell_orig = _normalize_iscell(iscell)

    areas = np.array([len(s["xpix"]) for s in stat])
    median_area = np.median(areas[iscell_orig]) if iscell_orig.any() else np.median(areas)

    # Determine bounds
    min_px = min_area_px
    max_px = max_area_px

    if min_mult is not None:
        min_px = min_mult * median_area if min_px is None else min(min_px, min_mult * median_area)
    if max_mult is not None:
        max_px = max_mult * median_area if max_px is None else max(max_px, max_mult * median_area)

    # Default bounds if nothing specified
    if min_px is None and max_px is None:
        min_px = 0.25 * median_area
        max_px = 4.0 * median_area

    # Apply filter
    valid = np.ones(len(areas), dtype=bool)
    if min_px is not None:
        valid &= areas >= min_px
    if max_px is not None:
        valid &= areas <= max_px

    removed_mask = ~valid & iscell_orig
    iscell_filtered = iscell_orig & valid
    n_removed = removed_mask.sum()

    if n_removed > 0:
        bounds_str = []
        if min_px is not None:
            bounds_str.append(f"min={min_px:.0f}px")
        if max_px is not None:
            bounds_str.append(f"max={max_px:.0f}px")
        print(f"filter_by_area: removed {n_removed} ROIs ({', '.join(bounds_str)})")

    if save and plane_dir is not None:
        _save_filtered_iscell(plane_dir, iscell_filtered)

    return iscell_filtered, removed_mask, {
        "areas_px": areas,
        "min_px": min_px,
        "max_px": max_px,
        "median_area": median_area,
        "n_removed": n_removed,
    }


def filter_by_eccentricity(
    plane_dir=None,
    iscell=None,
    stat=None,
    ops=None,
    max_ratio: float = 5.0,
    min_ratio: float = None,
    save: bool = False,
):
    """
    Filter ROIs by aspect ratio (elongation).

    Uses bounding box dimensions to compute aspect ratio. High ratios
    indicate elongated shapes (likely not cell bodies).

    Parameters
    ----------
    plane_dir : str or Path, optional
        Path to Suite2p plane directory. If provided, loads iscell, stat, ops.
    iscell : np.ndarray, optional
        Cell classification array.
    stat : np.ndarray or list, optional
        Suite2p stat array.
    ops : dict, optional
        Suite2p ops dictionary (not used, for consistent interface).
    max_ratio : float, default 5.0
        Maximum allowed aspect ratio (width/height or height/width).
    min_ratio : float, optional
        Minimum allowed aspect ratio.
    save : bool, default False
        If True, save updated iscell.npy to plane_dir.

    Returns
    -------
    iscell_filtered : np.ndarray
        Updated iscell array (1D boolean).
    removed_mask : np.ndarray
        Boolean mask of ROIs that were removed.
    info : dict
        Dictionary with 'ratios', 'min_ratio', 'max_ratio', 'n_removed'.
    """
    iscell, stat, ops, plane_dir = _load_plane_data(plane_dir, iscell, stat, ops)
    iscell_orig = _normalize_iscell(iscell)

    ratios = []
    for s in stat:
        h = s["ypix"].max() - s["ypix"].min() + 1
        w = s["xpix"].max() - s["xpix"].min() + 1
        ratio = max(h, w) / max(1, min(h, w))
        ratios.append(ratio)
    ratios = np.array(ratios)

    # Apply filter
    valid = np.ones(len(ratios), dtype=bool)
    if max_ratio is not None:
        valid &= ratios <= max_ratio
    if min_ratio is not None:
        valid &= ratios >= min_ratio

    removed_mask = ~valid & iscell_orig
    iscell_filtered = iscell_orig & valid
    n_removed = removed_mask.sum()

    if n_removed > 0:
        bounds_str = []
        if min_ratio is not None:
            bounds_str.append(f"min={min_ratio:.1f}")
        if max_ratio is not None:
            bounds_str.append(f"max={max_ratio:.1f}")
        print(f"filter_by_eccentricity: removed {n_removed} ROIs (ratio {', '.join(bounds_str)})")

    if save and plane_dir is not None:
        _save_filtered_iscell(plane_dir, iscell_filtered)

    return iscell_filtered, removed_mask, {
        "ratios": ratios,
        "min_ratio": min_ratio,
        "max_ratio": max_ratio,
        "n_removed": n_removed,
    }


def apply_filters(
    plane_dir=None,
    iscell=None,
    stat=None,
    ops=None,
    filters: list = None,
    save: bool = False,
):
    """
    Apply multiple cell filters in sequence.

    Chains filter functions together, passing the filtered iscell from each
    step to the next. All filters use a consistent interface.

    Parameters
    ----------
    plane_dir : str or Path, optional
        Path to Suite2p plane directory. If provided, loads iscell, stat, ops.
    iscell : np.ndarray, optional
        Cell classification array.
    stat : np.ndarray or list, optional
        Suite2p stat array.
    ops : dict, optional
        Suite2p ops dictionary.
    filters : list of dict
        List of filter configurations. Each dict must have:
        - 'name': str - filter function name (e.g., 'max_diameter', 'area', 'eccentricity')
        - Additional keys are passed as kwargs to the filter function.

        Available filters:
        - 'diameter': filter_by_diameter (min_mult, max_mult)
        - 'max_diameter': filter_by_max_diameter (max_diameter_um, max_diameter_px, min_diameter_um, min_diameter_px)
        - 'area': filter_by_area (min_area_px, max_area_px, min_mult, max_mult)
        - 'eccentricity': filter_by_eccentricity (max_ratio, min_ratio)

    save : bool, default False
        If True, save final filtered iscell.npy to plane_dir.

    Returns
    -------
    iscell_filtered : np.ndarray
        Final filtered iscell array (1D boolean).
    total_removed : np.ndarray
        Boolean mask of all ROIs removed by any filter.
    filter_results : list of dict
        Results from each filter, including removed_mask and info.

    Examples
    --------
    >>> # Apply multiple filters
    >>> iscell, removed, results = apply_filters(
    ...     plane_dir="path/to/plane01",
    ...     filters=[
    ...         {"name": "max_diameter", "max_diameter_um": 22},
    ...         {"name": "area", "min_mult": 0.25, "max_mult": 4.0},
    ...         {"name": "eccentricity", "max_ratio": 5.0},
    ...     ],
    ...     save=True
    ... )

    >>> # Use with pre-loaded data
    >>> iscell, removed, results = apply_filters(
    ...     iscell=iscell, stat=stat, ops=ops,
    ...     filters=[{"name": "max_diameter", "max_diameter_px": 30}]
    ... )
    """
    # Map filter names to functions
    FILTER_MAP = {
        "diameter": filter_by_diameter,
        "max_diameter": filter_by_max_diameter,
        "area": filter_by_area,
        "eccentricity": filter_by_eccentricity,
    }

    iscell, stat, ops, plane_dir = _load_plane_data(plane_dir, iscell, stat, ops)
    iscell_orig = _normalize_iscell(iscell)
    iscell_current = iscell_orig.copy()

    if filters is None:
        filters = []

    filter_results = []
    total_removed = np.zeros(len(iscell_current), dtype=bool)

    for filter_config in filters:
        config = filter_config.copy()
        name = config.pop("name", None)

        if name is None:
            raise ValueError("Each filter must have a 'name' key")

        if name not in FILTER_MAP:
            raise ValueError(f"Unknown filter: {name}. Available: {list(FILTER_MAP.keys())}")

        filter_fn = FILTER_MAP[name]

        # Don't save intermediate results, only final
        config["save"] = False

        # Apply filter
        iscell_current, removed, info = filter_fn(
            iscell=iscell_current,
            stat=stat,
            ops=ops,
            **config
        )

        total_removed |= removed
        filter_results.append({
            "name": name,
            "config": config,  # original user params (without 'name' and 'save')
            "removed_mask": removed,
            "info": info,
        })

    # Save final result if requested
    if save and plane_dir is not None:
        _save_filtered_iscell(plane_dir, iscell_current)

    n_total = total_removed.sum()
    n_orig = iscell_orig.sum()
    print(f"apply_filters: {n_total} total ROIs removed ({n_orig - n_total}/{n_orig} cells remaining)")

    return iscell_current, total_removed, filter_results


def mode_robust(x):
    """Half-sample mode robust estimator."""
    x = np.sort(x)
    n = len(x)
    if n == 1:
        return x[0]
    if n == 2:
        return np.mean(x)
    if n == 3:
        d1, d2 = x[1]-x[0], x[2]-x[1]
        if d1 < d2:
            return np.mean(x[:2])
        elif d2 < d1:
            return np.mean(x[1:])
        else:
            return x[1]
    # recursive half-sample mode
    N = n//2 + n%2 - 1
    wmin = np.inf
    j = 0
    for i in range(N):
        w = x[i+N-1] - x[i]
        if w < wmin:
            wmin = w
            j = i
    return mode_robust(x[j:j+N+1])


def compute_event_exceptionality(traces, N=5, robust_std=False):
    """
    traces: ndarray (n_cells x T)
    N: number of consecutive samples
    robust_std: use robust std estimate instead of simple RMS
    """
    md = np.maximum(np.apply_along_axis(mode_robust, 1, traces), 0)

    ff1 = (traces.T - md).T
    ff1 = -ff1 * (ff1 < 0)

    if robust_std:
        sd_r = []
        for row in ff1:
            vals = row[row > 0]
            if len(vals) == 0:
                sd_r.append(1.0)
                continue
            iqr = np.percentile(vals, 75) - np.percentile(vals, 25)
            sd_r.append(iqr / 1.349)
        sd_r = np.array(sd_r) * 2
    else:
        Ns = (ff1 > 0).sum(axis=1)
        sd_r = np.sqrt((ff1**2).sum(axis=1) / np.maximum(Ns, 1))

    # compute z-scores relative to noise
    z = (traces.T - md) / (3 * sd_r)
    z = z.T

    # tail probability of seeing value >= z under N(0,1)
    p = 1 - norm.cdf(z)
    p[p <= 0] = 1e-12
    logp = np.log(p)

    # moving sum over N consecutive samples
    kernel = np.ones(N)
    erfc = np.array([np.convolve(row, kernel, mode="same") for row in logp])

    # fitness score = min(erfc) (lower = more exceptional)
    fitness = erfc.min(axis=1)

    return fitness, erfc, sd_r, md


def ops_to_json(ops: dict | str | Path, outpath=None, indent=2):
    """
    Convert a Suite2p ops.npy file (or dict) to JSON.

    Parameters
    ----------
    ops : dict or str or Path
        Either a loaded ops dictionary or the path to an ops.npy file.
    outpath : str or Path, optional
        Output path for ops.json. If None, saves as 'ops.json' in the same
        directory as the input ops.npy (or current directory if ops is a dict).
    indent : int
        Indentation for JSON formatting.

    Returns
    -------
    Path
        Path to the written ops.json file.
    """
    # Load dict if given a path
    if isinstance(ops, (str, Path)):
        ops_path = Path(ops)
        if ops_path.is_dir():
            ops_path = ops_path / "ops.npy"
        if ops_path.suffix != ".npy":
            raise ValueError(f"Expected .npy file, got {ops_path}")
        ops_dict = np.load(ops_path, allow_pickle=True).item()
        base_dir = ops_path.parent
    elif isinstance(ops, dict):
        ops_dict = ops
        base_dir = Path.cwd()
    else:
        raise TypeError(f"`ops` must be dict, str, or Path, not {type(ops)}")

    # Convert numpy types to JSON serializable
    def _serialize(obj):
        if isinstance(obj, (np.generic, np.bool_)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Unserializable type {type(obj)}")

    # Decide output path
    if outpath is None:
        outpath = base_dir / "ops.json"
    else:
        outpath = Path(outpath)

    with open(outpath, "w") as f:
        json.dump(ops_dict, f, indent=indent, default=_serialize)

    print(f"Saved {outpath}")
    return outpath


def normalize_traces(F, mode="percentile"):
    """
    Normalize fluorescence traces F to [0, 1] range.
    Parameters
    ----------
    F : ndarray
        2d array of fluorescence traces (n_neurons x n_timepoints).
    mode : str
        Normalization mode, either "per_neuron" or "percentile".

    Returns
    -------
    F_norm : ndarray
        Normalized fluorescence traces in [0, 1] range.

    Notes
    -----
    - "per_neuron": scales each neuron's trace based on its own min and max.
    - "percentile": scales each neuron's trace based on its 1st and 99th percentiles.
    - If min == max for each cell, the trace is set to all zeros to avoid division by zero.
    """
    F_norm = np.zeros_like(F, dtype=float)

    if mode == "per_neuron":
        for i in range(F.shape[0]):
            f = F[i]
            fmax = np.max(f)
            fmin = np.min(f)
            if fmax > fmin:
                F_norm[i] = (f - fmin) / (fmax - fmin)
            else:
                F_norm[i] = f * 0
    elif mode == "percentile":
        for i in range(F.shape[0]):
            f = F[i]
            fmin = np.percentile(f, 1)
            fmax = np.percentile(f, 99)
            if fmax > fmin:
                F_norm[i] = (f - fmin) / (fmax - fmin)  # noqa
            else:
                F_norm[i] = f * 0
    return F_norm


def dff_rolling_percentile(
    f_trace,
    window_size: int = None,
    percentile: int = 20,
    use_median_floor: bool = False,
    smooth_window: int = None,
    fs: float = None,
    tau: float = None,
):
    """
    Compute ΔF/F₀ using a rolling percentile baseline.

    Parameters
    ----------
    f_trace : np.ndarray
        (N_neurons, N_frames) fluorescence traces.
    window_size : int, optional
        Size of the rolling window for baseline estimation (in frames).
        If None, auto-calculated as ~10 × tau × fs (default: 300 frames).
    percentile : int, default 20
        Percentile to use for baseline F₀ estimation.
    use_median_floor : bool, default False
        Set a minimum F₀ floor at 1% of the median fluorescence.
    smooth_window : int, optional
        Size of temporal smoothing window (in frames) applied after dF/F.
        If None, auto-calculated as ~0.5 × tau × fs to emphasize transients
        while reducing noise. Set to 0 or 1 to disable smoothing.
    fs : float, optional
        Frame rate in Hz. Used to auto-calculate window sizes if tau is provided.
    tau : float, optional
        Calcium indicator decay time constant in seconds (e.g., 1.0 for GCaMP6s).
        Used to auto-calculate window sizes if fs is provided.

    Returns
    -------
    dff : np.ndarray
        (N_neurons, N_frames) ΔF/F₀ traces.

    Notes
    -----
    Window size recommendations:

    * Baseline window (~10 × tau × fs): Should span multiple transients so the
      percentile filter can find baseline between events.
    * Smooth window (~0.5 × tau × fs): Should be shorter than typical transients
      to preserve them while averaging out noise.

    For GCaMP6s (tau ~ 1.0s) at 30 Hz: window_size ~ 300 frames (10 seconds),
    smooth_window ~ 15 frames (0.5 seconds).

    For GCaMP6f (tau ~ 0.4s) at 30 Hz: window_size ~ 120 frames (4 seconds),
    smooth_window ~ 6 frames (0.2 seconds).
    """
    from scipy.ndimage import uniform_filter1d

    if not isinstance(f_trace, np.ndarray):
        raise TypeError("f_trace must be a numpy array")
    if f_trace.ndim != 2:
        raise ValueError("f_trace must be a 2D array with shape (N_neurons, N_frames)")
    if f_trace.shape[0] == 0 or f_trace.shape[1] == 0:
        raise ValueError("f_trace must not be empty")

    # Auto-calculate window sizes based on tau and fs
    if window_size is None:
        if tau is not None and fs is not None:
            # ~10 × tau × fs for baseline window
            window_size = int(10 * tau * fs)
        else:
            # Default fallback
            window_size = 300

    if smooth_window is None:
        if tau is not None and fs is not None:
            # ~0.5 × tau × fs for smoothing (preserve transients, reduce noise)
            smooth_window = max(1, int(0.5 * tau * fs))
        else:
            # Default: no smoothing if parameters not provided
            smooth_window = 1

    # Ensure odd window size for symmetric filtering
    window_size = max(3, window_size)

    # Compute baseline using rolling percentile
    f0 = np.array(
        [
            percentile_filter(f, percentile, size=window_size, mode="nearest")
            for f in f_trace
        ]
    )
    if use_median_floor:
        floor = np.median(f_trace, axis=1, keepdims=True) * 0.01
        f0 = np.maximum(f0, floor)

    # Compute dF/F
    dff = (f_trace - f0) / (f0 + 1e-6)  # 1e-6 to avoid division by zero

    # Apply temporal smoothing if requested
    if smooth_window is not None and smooth_window > 1:
        dff = uniform_filter1d(dff, size=smooth_window, axis=1, mode="nearest")

    return dff


def dff_median_filter(f_trace):
    """
    Compute ΔF/F₀ using a rolling median filter baseline.

    Parameters:
    -----------
    f_trace : np.ndarray
        (N_neurons, N_frames) fluorescence traces.

    Returns:
    --------
    dff : np.ndarray
        (N_neurons, N_frames) ΔF/F₀ traces.
    """
    if not isinstance(f_trace, np.ndarray):
        raise TypeError("f_trace must be a numpy array")
    if f_trace.ndim != 2:
        raise ValueError("f_trace must be a 2D array with shape (N_neurons, N_frames)")
    if f_trace.shape[0] == 0 or f_trace.shape[1] == 0:
        raise ValueError("f_trace must not be empty")

    f0 = np.median(f_trace, axis=1, keepdims=True) * 0.01
    return (f_trace - f0) / (f0 + 1e-6)  # 1e-6 to avoid division by zero


def dff_shot_noise(dff, fr):
    """
    Estimate the shot noise level of calcium imaging traces.

    This metric quantifies the noise level based on frame-to-frame differences,
    assuming slow calcium dynamics compared to the imaging frame rate. It was
    introduced by Rupprecht et al. (2021) [1] as a standardized method for comparing
    noise levels across datasets with different acquisition parameters.

    The noise level :math:`\\nu` is computed as:

    .. math::

        \\nu = \\frac{\\mathrm{median}_t\\left( \\left| \\Delta F/F_{t+1} - \\Delta F/F_t \\right| \\right)}{\\sqrt{f_r}}

    where
      - :math:`\\Delta F/F_t` is the fluorescence trace at time :math:`t`
      - :math:`f_r` is the imaging frame rate (in Hz).

    Parameters
    ----------
    dff : np.ndarray
        Array of shape (n_neurons, n_frames), containing raw :math:`\\Delta F/F` traces
        (percent units, **without neuropil subtraction**).
    fr : float
        Frame rate of the recording in Hz.

    Returns
    -------
    np.ndarray
        Noise level :math:`\\nu` for each neuron, expressed in %/√Hz units.

    Notes
    -----
    - The metric relies on the slow dynamics of calcium signals compared to frame rate.
    - Higher values of :math:`\\nu` indicate higher shot noise.
    - Units are % divided by √Hz, and while unconventional, they enable comparison across frame rates.

    References
    ----------
    [1] Rupprecht et al., "Large-scale calcium imaging & noise levels",
        A Neuroscientific Blog (2021).
        https://gcamp6f.com/2021/10/04/large-scale-calcium-imaging-noise-levels/
    """
    return np.median(np.abs(np.diff(dff, axis=1)), axis=1) / np.sqrt(fr)


def compute_trace_quality_score(
    F,
    Fneu=None,
    stat=None,
    fs=30.0,
    weights=None,
):
    """
    Compute a weighted quality score for sorting neurons by signal quality.

    Combines SNR, skewness, and shot noise into a single score for ranking
    neurons from best to worst signal quality. Higher scores indicate better
    quality traces.

    Parameters
    ----------
    F : np.ndarray
        Fluorescence traces, shape (n_neurons, n_frames).
    Fneu : np.ndarray, optional
        Neuropil fluorescence traces, shape (n_neurons, n_frames).
        If None, no neuropil correction is applied.
    stat : np.ndarray or list, optional
        Suite2p stat array containing ROI statistics. If provided, uses
        pre-computed skewness from stat['skew']. Otherwise computes from traces.
    fs : float, default 30.0
        Frame rate in Hz, used for shot noise calculation.
    weights : dict, optional
        Weights for each metric. Keys: 'snr', 'skewness', 'shot_noise'.
        Default: {'snr': 1.0, 'skewness': 0.8, 'shot_noise': 0.5}
        Note: shot_noise is inverted (lower noise = higher score).

    Returns
    -------
    dict
        Dictionary containing:
        - 'score': Combined quality score (n_neurons,)
        - 'sort_idx': Indices that sort neurons by score (descending)
        - 'snr': SNR values (n_neurons,)
        - 'skewness': Skewness values (n_neurons,)
        - 'shot_noise': Shot noise values (n_neurons,)
        - 'weights': Weights used for scoring

    Notes
    -----
    Each metric is z-scored before weighting to ensure comparable scales:
    - SNR: signal std / noise estimate (higher = better)
    - Skewness: positive skew indicates calcium transients (higher = better)
    - Shot noise: frame-to-frame variability (lower = better, so inverted)

    Examples
    --------
    >>> import numpy as np
    >>> from lbm_suite2p_python.postprocessing import compute_trace_quality_score
    >>> F = np.load("F.npy")
    >>> Fneu = np.load("Fneu.npy")
    >>> result = compute_trace_quality_score(F, Fneu, fs=30.0)
    >>> sorted_F = F[result['sort_idx']]  # Traces sorted by quality
    """
    from scipy.stats import skew

    if weights is None:
        weights = {'snr': 1.0, 'skewness': 0.8, 'shot_noise': 0.5}

    n_neurons = F.shape[0]

    # Neuropil correction
    if Fneu is not None:
        F_corr = F - 0.7 * Fneu
    else:
        F_corr = F

    # Compute baseline and dF/F
    baseline = np.percentile(F_corr, 20, axis=1, keepdims=True)
    baseline = np.maximum(baseline, 1e-6)
    dff = (F_corr - baseline) / baseline

    # SNR
    signal = np.std(dff, axis=1)
    noise = np.median(np.abs(np.diff(dff, axis=1)), axis=1) / 0.6745
    snr = signal / (noise + 1e-6)

    # Skewness
    if stat is not None:
        # Use pre-computed skewness from Suite2p stat
        skewness = np.array([s.get('skew', np.nan) for s in stat])
        # Fill NaN with computed values
        nan_mask = np.isnan(skewness)
        if nan_mask.any():
            skewness[nan_mask] = skew(dff[nan_mask], axis=1)
    else:
        # Compute from traces
        skewness = skew(dff, axis=1)

    # Shot noise
    shot_noise = dff_shot_noise(dff, fs)

    # Normalize metrics to z-scores
    def safe_zscore(x):
        """Z-score with handling for constant arrays."""
        std = np.nanstd(x)
        if std < 1e-10:
            return np.zeros_like(x)
        return (x - np.nanmean(x)) / std

    snr_z = safe_zscore(snr)
    skewness_z = safe_zscore(skewness)
    # Invert shot noise (lower noise = higher score)
    shot_noise_z = -safe_zscore(shot_noise)

    # Compute weighted score
    score = (
        weights['snr'] * snr_z +
        weights['skewness'] * skewness_z +
        weights['shot_noise'] * shot_noise_z
    )

    # Handle any NaN values
    score = np.nan_to_num(score, nan=-np.inf)

    # Sort indices (descending - best first)
    sort_idx = np.argsort(score)[::-1]

    return {
        'score': score,
        'sort_idx': sort_idx,
        'snr': snr,
        'skewness': skewness,
        'shot_noise': shot_noise,
        'weights': weights,
    }


def sort_traces_by_quality(
    F,
    Fneu=None,
    stat=None,
    fs=30.0,
    weights=None,
):
    """
    Sort fluorescence traces by quality score (best to worst).

    Convenience function that computes quality scores and returns sorted traces.

    Parameters
    ----------
    F : np.ndarray
        Fluorescence traces, shape (n_neurons, n_frames).
    Fneu : np.ndarray, optional
        Neuropil fluorescence traces.
    stat : np.ndarray or list, optional
        Suite2p stat array for pre-computed skewness.
    fs : float, default 30.0
        Frame rate in Hz.
    weights : dict, optional
        Weights for each metric. Default: {'snr': 1.0, 'skewness': 0.8, 'shot_noise': 0.5}

    Returns
    -------
    F_sorted : np.ndarray
        Traces sorted by quality (best first).
    sort_idx : np.ndarray
        Indices used to sort (can be used to sort other arrays).
    quality : dict
        Full quality metrics from compute_trace_quality_score().

    Examples
    --------
    >>> F_sorted, sort_idx, quality = sort_traces_by_quality(F, Fneu)
    >>> # Also sort stat and iscell arrays
    >>> stat_sorted = stat[sort_idx]
    >>> iscell_sorted = iscell[sort_idx]
    """
    quality = compute_trace_quality_score(F, Fneu, stat, fs, weights)
    sort_idx = quality['sort_idx']
    F_sorted = F[sort_idx]

    return F_sorted, sort_idx, quality


def load_planar_results(ops: dict | str | Path, z_plane: list | int = None) -> dict:
    """
    Load stat, iscell, spks files and return as a dict. Does NOT filter by valid cells, arrays contain both
    accepted and rejected neurons. Filter for accepted-only via ``iscell_mask = iscell[:, 0].astype(bool)``.

    Parameters
    ----------
    ops : dict, str or Path
        Dict of or path to the ops.npy file. Can be a fully qualified path or a directory containing ops.npy.
    z_plane : int or None, optional
        the z-plane index for this file. If provided, it is stored in the output.

    Returns
    -------
    dict
        Dictionary with keys: 'F' (fluorescence traces, n_rois x n_frames),
        'Fneu' (neuropil fluorescence), 'spks' (deconvolved spikes),
        'stat' (ROI statistics array), 'iscell' (classification array where
        column 0 is 0/1 rejected/accepted and column 1 is probability),
        and 'z_plane' (z-plane index array).

    See Also
    --------
    lbm_suite2p_python.load_ops
    lbm_suite2p_python.load_traces
    """
    if isinstance(ops, list):
        raise ValueError(f"Input should not be a list!")
    if isinstance(ops, (str, Path)):
        if Path(ops).is_dir():
            ops = Path(ops).joinpath("ops.npy")
            if not ops.exists():
                raise FileNotFoundError(f"ops.npy not found in given directory: {ops}")
    output_ops = load_ops(ops)

    save_path = Path(output_ops["save_path"])

    # Check all required files exist
    required_files = {
        "F.npy": save_path / "F.npy",
        "Fneu.npy": save_path / "Fneu.npy",
        "spks.npy": save_path / "spks.npy",
        "stat.npy": save_path / "stat.npy",
        "iscell.npy": save_path / "iscell.npy",
    }

    missing_files = [name for name, path in required_files.items() if not path.exists()]
    if missing_files:
        raise FileNotFoundError(
            f"Missing required files in {save_path}: {', '.join(missing_files)}"
        )

    F = np.load(required_files["F.npy"])
    Fneu = np.load(required_files["Fneu.npy"])
    spks = np.load(required_files["spks.npy"])
    stat = np.load(required_files["stat.npy"], allow_pickle=True)

    # iscell is (n_rois, 2): column 0 is is_cell (0/1), column 1 is probability
    iscell = np.load(required_files["iscell.npy"], allow_pickle=True)

    n_neurons = spks.shape[0]
    if z_plane is None:
        z_plane_arr = output_ops.get("plane", np.zeros(n_neurons, dtype=int))
    else:
        z_plane_arr = np.full(n_neurons, z_plane, dtype=int)
    return {
        "F": F,
        "Fneu": Fneu,
        "spks": spks,
        "stat": stat,
        "iscell": iscell,  # Full (n_rois, 2) array: [:, 0] is bool, [:, 1] is probability
        "z_plane": z_plane_arr,
    }


def load_ops(ops_input: str | Path | list[str | Path]) -> dict:
    """
    Load a Suite2p ops.npy file.

    Uses cross-platform loader to handle ops.npy files created on different
    operating systems (Windows vs Linux/Mac Path objects in pickled data).

    Parameters
    ----------
    ops_input : str, Path, or dict
        Path to ops.npy file, directory containing ops.npy, or an
        already-loaded ops dict.

    Returns
    -------
    dict
        Suite2p operations dictionary containing pipeline parameters
        and results metadata.
    """
    if isinstance(ops_input, (str, Path)):
        ops_path = Path(ops_input)
        if ops_path.is_dir():
            ops_path = ops_path / "ops.npy"
        if not ops_path.exists():
            raise FileNotFoundError(f"ops.npy not found: {ops_path}")
        ops = load_npy(ops_path)
        return ops.item() if ops.ndim == 0 else ops
    elif isinstance(ops_input, dict):
        return ops_input
    print("Warning: No valid ops file provided, returning empty dict.")
    return {}


def load_traces(ops):
    """
    Load fluorescence traces and related data from an ops file directory and return valid cells.

    This function loads the raw fluorescence traces, neuropil traces, and spike data from the directory
    specified in the ops dictionary. It also loads the 'iscell' file and returns only the traces corresponding
    to valid cells (i.e. where iscell is True).

    Parameters
    ----------
    ops : dict
        Dictionary containing at least the key 'save_path', which specifies the directory where the following
        files are stored: 'F.npy', 'Fneu.npy', 'spks.npy', and 'iscell.npy'.

    Returns
    -------
    F_valid : ndarray
        Array of fluorescence traces for valid cells (n_valid x n_timepoints).
    Fneu_valid : ndarray
        Array of neuropil fluorescence traces for valid cells (n_valid x n_timepoints).
    spks_valid : ndarray
        Array of spike data for valid cells (n_valid x n_timepoints).

    Notes
    -----
    The 'iscell.npy' file is expected to be an array where the first column (iscell[:, 0]) contains
    boolean values indicating valid cells.
    """
    save_path = Path(ops['save_path'])
    F = np.load(save_path.joinpath('F.npy'))
    Fneu = np.load(save_path.joinpath('Fneu.npy'))
    spks = np.load(save_path.joinpath('spks.npy'))
    iscell = np.load(save_path.joinpath('iscell.npy'), allow_pickle=True)[:, 0].astype(bool)

    F_valid = F[iscell]
    Fneu_valid = Fneu[iscell]
    spks_valid = spks[iscell]

    return F_valid, Fneu_valid, spks_valid

