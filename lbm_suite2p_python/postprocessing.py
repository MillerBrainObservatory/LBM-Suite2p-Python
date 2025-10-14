import json
from pathlib import Path

import numpy as np
from scipy.ndimage import percentile_filter
from scipy.stats import norm



def _normalize_iscell(iscell):
    """Ensure iscell is 1D boolean array."""
    if iscell.ndim == 2:
        iscell = iscell[:, 0]
    return iscell.astype(bool)

def filter_by_diameter(iscell, stat, ops, min_mult=0.3, max_mult=3.0):
    """
    Set iscell=False for ROIs whose radius is out of range relative to ops['diameter'].
    """
    iscell = _normalize_iscell(iscell)

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
    median_diam = ops.get("diameter", np.median(radii))
    lower, upper = min_mult * median_diam, max_mult * median_diam
    iscell &= (radii >= lower) & (radii <= upper)
    return iscell


def filter_by_area(iscell, stat, min_mult=0.25, max_mult=4.0):
    """
    Filter cells by total area (in pixels).

    Cells with area outside [min_mult*median_area, max_mult*median_area] are rejected.
    """
    iscell = _normalize_iscell(iscell)

    areas = np.array([len(s["xpix"]) for s in stat])
    median_area = np.median(areas)
    lower, upper = min_mult * median_area, max_mult * median_area
    iscell &= (areas >= lower) & (areas <= upper)
    return iscell


def filter_by_eccentricity(iscell, stat, max_ratio=5.0):
    """
    Set iscell=False for ROIs that are extremely elongated (bounding box disproportion).
    """
    iscell = _normalize_iscell(iscell)

    ecc = []
    for s in stat:
        h = s["ypix"].max() - s["ypix"].min() + 1
        w = s["xpix"].max() - s["xpix"].min() + 1
        ratio = max(h, w) / max(1, min(h, w))
        ecc.append(ratio <= max_ratio)

    iscell &= np.array(ecc, dtype=bool)
    return iscell


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


def dff_rolling_percentile(f_trace, window_size=300, percentile=20, use_median_floor: bool=False):
    """
    Compute ΔF/F₀ using a rolling percentile baseline.

    Parameters:
    -----------
    f_trace : np.ndarray
        (N_neurons, N_frames) fluorescence traces.
    window_size : int
        Size of the rolling window (in frames).
    percentile : int
        Percentile to use for baseline F₀ estimation.
    use_median_floor : bool, optional
        Set a minimum F₀ floor at 1% of the median fluorescence.

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


    f0 = np.array(
        [
            percentile_filter(f, percentile, size=window_size, mode="nearest")
            for f in f_trace
        ]
    )
    if use_median_floor:
        floor = np.median(f_trace, axis=1, keepdims=True) * 0.01
        f0 = np.maximum(f0, floor)

    return (f_trace - f0) / (f0 + 1e-6)  # 1e-6 to avoid division by zero


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


def load_planar_results(ops: dict | str | Path, z_plane: list | int = None) -> dict:
    """
    Load stat, iscell, spks files and return as a dict. Does NOT filter by valid cells, array contain both
    accepted and rejected neurons. Filter for accepted-only via f[iscell] or fneue[iscell] if needed.

    Parameters
    ----------
    ops : dict, str or Path
        Dict of or path to the ops.npy file. Can be a fully qualified path or a directory containing ops.npy.
    z_plane : int or None, optional
        the z-plane index for this file. If provided, it is stored in the output.

    Returns
    -------
    dict
        dictionary with keys:
        - 'F': fluorescence traces loaded from F.npy,
        - 'Fneu': neuropil fluorescence traces loaded from Fneu.npy,
        - 'spks': spike traces loaded from spks.npy,
        - 'stat': stats loaded from stat.npy,
        - 'iscell': boolean array from iscell.npy,
        - 'cellprob': cell probability from classifier.
        - 'z_plane': an array (of shape [n_neurons,]) with the provided z_plane index.

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

    F = np.load(save_path.joinpath("F.npy"))
    Fneu = np.load(save_path.joinpath("Fneu.npy"))
    spks = np.load(save_path.joinpath("spks.npy"))
    stat = np.load(save_path.joinpath("stat.npy"), allow_pickle=True)
    iscell = np.load(save_path.joinpath("iscell.npy"), allow_pickle=True)[:, 0].astype(
        bool
    )
    cellprob = np.load(save_path.joinpath("iscell.npy"), allow_pickle=True)[:, 1]
    # model = np.load(save_path.joinpath("model.npy"), allow_pickle=True).item()

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
        "iscell": iscell,
        "cellprob": cellprob,
        "z_plane": z_plane_arr,
        # "rastermap_model": model,
    }


def load_ops(ops_input: str | Path | list[str | Path]) -> dict:
    """Simple utility load a suite2p npy file"""
    if isinstance(ops_input, (str, Path)):
        return np.load(ops_input, allow_pickle=True).item()
    elif isinstance(ops_input, dict):
        return ops_input
    print("Warning: No valid ops file provided, returning empty dict.")
    return {}


