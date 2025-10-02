from pathlib import Path
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from mbo_utilities import get_files
from skimage.filters.rank import threshold_percentile

from lbm_suite2p_python.postprocessing import (
    load_ops,
    load_planar_results,
    dff_rolling_percentile,
    dff_shot_noise,
    filter_by_area,
    filter_by_diameter,
    filter_by_eccentricity, compute_event_exceptionality,
)
from lbm_suite2p_python.zplane import plot_masks, plot_traces

if __name__ == "__main__":
    inpath = Path(r"D:\W2_DATA\kbarber\2025-03-01\green.extracted.processed.results\merged_mrois")
    ops_ind = 8

    # Load ops + results
    ops_files = get_files(inpath, str_contains="ops", max_depth=3)
    ops = load_ops(ops_files[ops_ind])
    planar_results = load_planar_results(ops)

    F = planar_results["F"]
    stat = planar_results["stat"]
    iscell = planar_results["iscell"]

    print(f"Num accepted: {np.sum(iscell)}, rejected: {np.sum(iscell == 0)}")

    # Functional metric: ΔF/F and shot noise
    dff = dff_rolling_percentile(F, window_size=300, percentile=8, use_median_floor=False)
    shot_noise = dff_shot_noise(dff, ops["fs"])

    # Filters applied to iscell
    diam_min_mult, diam_max_mult = 0.3, 3.0
    iscell_diam = filter_by_diameter(iscell, stat, ops, min_mult=diam_min_mult, max_mult=diam_max_mult)

    area_min_mult, area_max_mult = 0.25, 4.0
    iscell_area = filter_by_area(iscell, stat, min_mult=area_min_mult, max_mult=area_max_mult)

    max_eccentric_ratio = 5.0
    iscell_ecc  = filter_by_eccentricity(iscell, stat, max_ratio=max_eccentric_ratio)

    # Event exceptionality
    n=5
    thr_prct = 80
    fitness, erfc, sd_r, md = compute_event_exceptionality(dff, N=n, robust_std=True)
    thresh = np.percentile(fitness, thr_prct)
    iscell_event = iscell & (fitness < thresh)

    # Background image
    meanImg = ops.get("meanImgE", ops.get("meanImg", np.zeros((ops["Ly"], ops["Lx"]))))

    # Plot each filter condition
    ops_dir = ops_files[ops_ind].parent
    outdir = ops_dir.joinpath("filter_comparison")
    outdir.mkdir(exist_ok=True)

    # eliminated = original iscell minus the filtered result
    eliminated_diam = iscell & ~iscell_diam
    eliminated_area = iscell & ~iscell_area
    eliminated_ecc = iscell & ~iscell_ecc
    eliminated_event = iscell & ~iscell_event

    diam_path = outdir.joinpath(f"diameter_filter_{diam_min_mult:.2f}_{diam_max_mult:.2f}.png")
    diam_path_traces = outdir.joinpath(f"diameter_filter_traces_{diam_min_mult:.2f}_{diam_max_mult:.2f}.png")
    plot_masks(
        meanImg, stat, eliminated_diam, savepath=diam_path, title="Eliminated by Diameter filter"
    )
    plot_traces(
        F,
        save_path=diam_path_traces,
        cell_indices=eliminated_diam,
        title="Traces eliminated by Diameter filter",
        fps=ops["fs"],
    )

    area_path = outdir.joinpath(f"area_filter_{area_min_mult:.2f}_{area_max_mult:.2f}.png")
    area_path_traces = outdir.joinpath(f"area_filter_traces_{area_min_mult:.2f}_{area_max_mult:.2f}.png")
    plot_masks(meanImg, stat, eliminated_area, savepath=area_path,
               title="Eliminated by Area filter")
    plot_traces(
        F,
        save_path=area_path_traces,
        cell_indices=eliminated_area,
        title="Traces eliminated by Area filter",
        fps=ops["fs"],
    )

    elong_path = outdir.joinpath(f"eccentricity_filter_{max_eccentric_ratio:.2f}.png")
    elong_path_traces = outdir.joinpath(f"eccentricity_filter_traces_{max_eccentric_ratio:.2f}.png")
    plot_masks(meanImg, stat, eliminated_ecc, savepath=elong_path,
               title="Eliminated by Eccentricity filter")
    plot_traces(
        F,
        save_path=elong_path_traces,
        cell_indices=eliminated_ecc,
        title="Traces eliminated by Eccentricity filter",
        fps=ops["fs"],
    )

    event_path = outdir.joinpath(f"event_filter_{thresh:.4f}.png")
    event_path_traces = outdir.joinpath(f"event_filter_traces_{thresh:.4f}.png")
    plot_masks(meanImg, stat, eliminated_event, savepath=event_path,
               title=f"Eliminated by Event filter (thr={thresh:.4f})")
    plot_traces(
        F,
        save_path=event_path_traces,
        cell_indices=eliminated_event,
        title="Traces eliminated by Event filter",
        fps=ops["fs"],
    )

    print(f"Saved all filtered overlays to {outdir}")
