# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "numpy",
#     "h5py",
#     "scipy",
#     "mat73",
# ]
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full", app_title="CaImAn ROI Diagnostics")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import h5py
    from scipy.spatial.distance import cdist
    from scipy.ndimage import zoom
    from scipy.stats import skew as scipy_skew

    # dark mode
    plt.style.use("dark_background")
    plt.rcParams.update({
        "figure.facecolor": "#181818",
        "axes.facecolor": "#181818",
        "savefig.facecolor": "#181818",
        "figure.dpi": 100,
    })
    return Path, cdist, h5py, mo, np, plt, scipy_skew, zoom


@app.cell
def _(mo):
    data_path = mo.ui.text(
        value="//rbo-s1/S1_DATA/lbm/jdemas/bi_hemisphere/output",
        label="Path",
        full_width=True,
    )
    plane_num = mo.ui.number(value=1, start=1, stop=30, label="Plane")
    return data_path, plane_num


@app.cell
def _(data_path, mo, plane_num):
    mo.hstack([data_path, plane_num], widths=[0.9, 0.1])
    return


@app.cell
def _(Path, data_path, h5py, np, plane_num, zoom):
    import scipy.io as sio
    import mat73

    def _load(fp):
        d = {}
        # try h5py first (matlab v7.3)
        try:
            with h5py.File(fp, "r") as f:
                for k in f.keys():
                    try:
                        a = f[k][:]
                        if a.ndim == 2 and a.shape[0] > 1 and a.shape[1] > 1:
                            if 0.1 < a.shape[0] / a.shape[1] < 10:
                                a = a.T
                        d[k] = a
                    except:
                        pass
            return d
        except OSError:
            pass

        # try mat73 (another v7.3 loader)
        try:
            d = mat73.loadmat(str(fp))
            return d
        except:
            pass

        # try scipy (matlab v5/v7)
        try:
            raw = sio.loadmat(str(fp), squeeze_me=True)
            for k, v in raw.items():
                if not k.startswith("_"):
                    d[k] = v
            return d
        except:
            pass

        return d

    _p = Path(data_path.value) / f"caiman_output_plane_{plane_num.value}.mat"
    if _p.exists():
        D = _load(_p)
        Cn = D.get("Cn", D.get("Ym"))
        if Cn is None:
            Cn = np.zeros((512, 512))
        fov = Cn.shape

        # handle different key names
        _ac = D.get("Ac_keep", D.get("A", None))
        _t = D.get("T_keep", D.get("C", None))  # raw traces
        _c = D.get("C_keep", D.get("S", _t))    # deconvolved or same as raw

        if _ac is not None and _t is not None:
            Ac = np.array(_ac)
            T = np.array(_t)
            C = np.array(_c) if _c is not None else T

            # handle shape variations
            if Ac.ndim == 2:
                # sparse format (n_pixels, n_rois) - need footprints
                pass  # keep as is for now
            n_rois = Ac.shape[0] if Ac.ndim == 3 else Ac.shape[1]
            n_frames = T.shape[0] if T.shape[0] < T.shape[1] else T.shape[1]
            if T.shape[0] > T.shape[1]:
                T = T.T
            if C.shape[0] > C.shape[1]:
                C = C.T

            _acx = D.get("acx", D.get("center", None))
            _acy = D.get("acy", None)
            if _acx is not None:
                if _acy is not None:
                    acx = np.array(_acx).flatten().astype(int)
                    acy = np.array(_acy).flatten().astype(int)
                elif isinstance(_acx, np.ndarray) and _acx.ndim == 2:
                    acx = _acx[:, 0].astype(int)
                    acy = _acx[:, 1].astype(int)
                else:
                    acx = np.array(_acx).flatten().astype(int)
                    acy = np.zeros_like(acx)
            else:
                acx = np.zeros(n_rois, dtype=int)
                acy = np.zeros(n_rois, dtype=int)

            _max_dim = max(fov)
            if _max_dim > 400:
                _scale = 400 / _max_dim
                Cn_small = zoom(Cn, _scale, order=1)
                scale_factor = _scale
            else:
                Cn_small = Cn
                scale_factor = 1.0

            loaded = True
        else:
            D, Cn, Cn_small = {}, np.zeros((512, 512)), np.zeros((512, 512))
            fov, n_rois, n_frames = (512, 512), 0, 0
            acx, acy = np.array([]), np.array([])
            Ac, T, C = np.zeros((0, 0, 0)), np.zeros((0, 0)), np.zeros((0, 0))
            scale_factor = 1.0
            loaded = False
    else:
        D, Cn, Cn_small = {}, np.zeros((512, 512)), np.zeros((512, 512))
        fov, n_rois, n_frames = (512, 512), 0, 0
        acx, acy = np.array([]), np.array([])
        Ac, T, C = np.zeros((0, 0, 0)), np.zeros((0, 0)), np.zeros((0, 0))
        scale_factor = 1.0
        loaded = False
    return Ac, C, Cn, Cn_small, D, T, acx, acy, fov, loaded, n_frames, n_rois, scale_factor


@app.cell
def _(loaded, mo, n_frames, n_rois, fov):
    status = mo.md(f"**{n_rois:,} ROIs** | {n_frames:,} frames | {fov[0]}x{fov[1]}") if loaded else mo.md("no data")
    return (status,)


@app.cell
def _(Ac, T, acx, acy, cdist, loaded, np, scipy_skew):
    if loaded and len(acx) > 0:
        _n = len(acx)

        mask_sizes = np.zeros(_n)
        compactness = np.zeros(_n)
        for _i in range(_n):
            _m = Ac[_i]
            _thresh = _m.max() * 0.1
            _npix = (_m > _thresh).sum()
            mask_sizes[_i] = _npix
            _ys, _xs = np.where(_m > _thresh)
            if len(_ys) > 0:
                _bbox = ((_ys.max() - _ys.min() + 1) * (_xs.max() - _xs.min() + 1))
                compactness[_i] = _npix / max(1, _bbox)

        trace_snr = np.zeros(_n)
        trace_skew = np.zeros(_n)
        for _i in range(_n):
            _tr = T[:, _i]
            _std = _tr.std()
            trace_snr[_i] = _tr.mean() / (_std + 1e-9)
            trace_skew[_i] = scipy_skew(_tr)

        _coords = np.column_stack([acx, acy])
        if _n > 5000:
            nn_dists = np.full(_n, np.nan)
            for _i in range(_n):
                _dists = np.sqrt((acx - acx[_i])**2 + (acy - acy[_i])**2)
                _dists[_i] = np.inf
                nn_dists[_i] = _dists.min()
        else:
            _d = cdist(_coords, _coords)
            np.fill_diagonal(_d, np.inf)
            nn_dists = _d.min(axis=1)

        snr_order = np.argsort(trace_snr)[::-1]
        stats_computed = True
    else:
        mask_sizes = np.array([])
        compactness = np.array([])
        trace_snr = np.array([])
        trace_skew = np.array([])
        nn_dists = np.array([])
        snr_order = np.array([])
        stats_computed = False
    return compactness, mask_sizes, nn_dists, snr_order, stats_computed, trace_skew, trace_snr


@app.cell
def _(loaded, mo, n_rois):
    roi_slider = mo.ui.slider(0, max(1, n_rois - 1), value=0, label="ROI index", show_value=True) if loaded and n_rois > 0 else None
    return (roi_slider,)


@app.cell
def _(Ac, Cn, T, acx, acy, compactness, fov, loaded, mask_sizes, mo, nn_dists, np, plt, roi_slider, trace_skew, trace_snr):
    if loaded and roi_slider is not None:
        _idx = roi_slider.value
        _cx, _cy = int(acx[_idx]), int(acy[_idx])
        _fp = Ac[_idx]
        _fp_size = _fp.shape[0]
        _half = _fp_size // 2

        _dists = np.sqrt((acx - _cx)**2 + (acy - _cy)**2)
        _dists[_idx] = np.inf
        _neighbor_idx = np.argsort(_dists)[:4]

        _fig = plt.figure(figsize=(14, 8))
        _gs = _fig.add_gridspec(3, 4, height_ratios=[1.2, 1, 1], hspace=0.3, wspace=0.3)

        _ax1 = _fig.add_subplot(_gs[0, 0])
        _crop_half = 50
        _y0, _y1 = max(0, _cy - _crop_half), min(fov[0], _cy + _crop_half)
        _x0, _x1 = max(0, _cx - _crop_half), min(fov[1], _cx + _crop_half)
        _crop = Cn[_y0:_y1, _x0:_x1]
        if _crop.size > 0:
            _p1, _p99 = np.percentile(_crop, [1, 99])
            _ax1.imshow(_crop, cmap="gray", vmin=_p1, vmax=_p99, extent=[_x0, _x1, _y1, _y0])

            # draw main ROI contour in cyan
            _fp_y0 = _cy - _half
            _fp_x0 = _cx - _half
            _ax1.contour(
                np.arange(_fp_x0, _fp_x0 + _fp_size),
                np.arange(_fp_y0, _fp_y0 + _fp_size),
                _fp, levels=[_fp.max() * 0.3], colors=["cyan"], linewidths=2
            )

            # draw neighbor contours in magenta
            _neighbor_colors = ["magenta", "yellow", "orange", "lime"]
            for _j, _ni in enumerate(_neighbor_idx):
                _ncx, _ncy = int(acx[_ni]), int(acy[_ni])
                _nfp = Ac[_ni]
                _nfp_y0 = _ncy - _half
                _nfp_x0 = _ncx - _half
                if _nfp.max() > 0:
                    _ax1.contour(
                        np.arange(_nfp_x0, _nfp_x0 + _fp_size),
                        np.arange(_nfp_y0, _nfp_y0 + _fp_size),
                        _nfp, levels=[_nfp.max() * 0.3], colors=[_neighbor_colors[_j]], linewidths=1
                    )
        _ax1.set_title(f"ROI #{_idx} (cyan) + 4 neighbors", fontsize=10)
        _ax1.axis("off")

        _ax2 = _fig.add_subplot(_gs[0, 1])
        _ax2.imshow(_fp, cmap="hot")
        _ax2.set_title(f"footprint ({_fp_size}x{_fp_size})", fontsize=10)
        _ax2.axis("off")

        _ax3 = _fig.add_subplot(_gs[0, 2:])
        _ax3.axis("off")
        _stats_text = f"ROI #{_idx}\n\nposition: ({_cx}, {_cy})\nsize: {mask_sizes[_idx]:.0f} px\ncompactness: {compactness[_idx]:.2f}\nnearest neighbor: {nn_dists[_idx]:.1f} px\nSNR: {trace_snr[_idx]:.2f}\nskew: {trace_skew[_idx]:.2f}"
        _ax3.text(0.1, 0.9, _stats_text, transform=_ax3.transAxes, fontsize=11,
                  verticalalignment="top", family="monospace")

        _ax4 = _fig.add_subplot(_gs[1, :2])
        _trace = T[:, _idx]
        _step = max(1, len(_trace) // 800)
        _ax4.plot(_trace[::_step], lw=0.6, color="cyan")
        _ax4.set_title("raw trace", fontsize=10)
        _ax4.set_xlabel("frame")

        _ax5 = _fig.add_subplot(_gs[1, 2:])
        _baseline = np.percentile(_trace, 10)
        _dff = (_trace - _baseline) / (_baseline + 1e-9)
        _ax5.plot(_dff[::_step], lw=0.6, color="lime")
        _ax5.axhline(0, color="gray", ls="--", lw=0.5)
        _ax5.set_title("dF/F (10th percentile baseline)", fontsize=10)
        _ax5.set_xlabel("frame")

        _ax6 = _fig.add_subplot(_gs[2, :])
        _neighbor_colors = ["magenta", "yellow", "orange", "lime"]
        for _j, _ni in enumerate(_neighbor_idx):
            _nt = T[:, _ni]
            _nt_norm = (_nt - _nt.min()) / (_nt.max() - _nt.min() + 1e-9)
            _ax6.plot(_nt_norm[::_step] + _j * 1.2, lw=0.5, color=_neighbor_colors[_j],
                      label=f"#{_ni} ({_dists[_ni]:.0f}px)")
        _ax6.set_title("4 nearest neighbors (stacked)", fontsize=10)
        _ax6.set_xlabel("frame")
        _ax6.legend(loc="upper right", fontsize=8)

        plt.tight_layout()
        inspect_fig = mo.as_html(_fig)
        plt.close(_fig)
    else:
        inspect_fig = mo.md("no data")
    return (inspect_fig,)


@app.cell
def _(inspect_fig, loaded, mo, roi_slider):
    inspect_content = mo.vstack([roi_slider, inspect_fig]) if loaded and roi_slider else mo.md("no data")
    return (inspect_content,)


@app.cell
def _(Ac, Cn, T, acx, acy, fov, loaded, mo, np, plt, snr_order, trace_snr):
    if loaded and len(snr_order) > 0:
        _n_show = 6
        _good_idx = snr_order[:_n_show]
        _bad_idx = snr_order[-_n_show:][::-1]
        _fp_size = Ac.shape[1]
        _half = _fp_size // 2

        _fig, _axes = plt.subplots(4, _n_show, figsize=(14, 10))

        def _draw_roi_with_neighbors(ax, idx, main_color):
            _cx, _cy = int(acx[idx]), int(acy[idx])
            _fp = Ac[idx]

            _crop_half = 40
            _y0, _y1 = max(0, _cy - _crop_half), min(fov[0], _cy + _crop_half)
            _x0, _x1 = max(0, _cx - _crop_half), min(fov[1], _cx + _crop_half)
            _crop = Cn[_y0:_y1, _x0:_x1]

            if _crop.size > 0:
                _p1, _p99 = np.percentile(_crop, [1, 99])
                ax.imshow(_crop, cmap="gray", vmin=_p1, vmax=_p99, extent=[_x0, _x1, _y1, _y0])

                # main ROI contour
                _fp_y0 = _cy - _half
                _fp_x0 = _cx - _half
                if _fp.max() > 0:
                    ax.contour(
                        np.arange(_fp_x0, _fp_x0 + _fp_size),
                        np.arange(_fp_y0, _fp_y0 + _fp_size),
                        _fp, levels=[_fp.max() * 0.3], colors=[main_color], linewidths=1.5
                    )

                # find and draw neighbors
                _dists = np.sqrt((acx - _cx)**2 + (acy - _cy)**2)
                _dists[idx] = np.inf
                _neighbors = np.argsort(_dists)[:3]

                for _ni in _neighbors:
                    _ncx, _ncy = int(acx[_ni]), int(acy[_ni])
                    _nfp = Ac[_ni]
                    _nfp_y0 = _ncy - _half
                    _nfp_x0 = _ncx - _half
                    if _nfp.max() > 0:
                        ax.contour(
                            np.arange(_nfp_x0, _nfp_x0 + _fp_size),
                            np.arange(_nfp_y0, _nfp_y0 + _fp_size),
                            _nfp, levels=[_nfp.max() * 0.3], colors=["gray"], linewidths=0.5, alpha=0.7
                        )
            ax.axis("off")

        for _i, _idx in enumerate(_good_idx):
            _draw_roi_with_neighbors(_axes[0, _i], _idx, "cyan")
            _axes[0, _i].set_title(f"#{_idx} snr={trace_snr[_idx]:.1f}", fontsize=8, color="lime")

            _trace = T[:, _idx]
            _step = max(1, len(_trace) // 200)
            _axes[1, _i].plot(_trace[::_step], lw=0.4, color="cyan")
            _axes[1, _i].axis("off")

        _axes[0, 0].set_ylabel("HIGH SNR", fontsize=10, color="lime", rotation=0, labelpad=50)

        for _i, _idx in enumerate(_bad_idx):
            _draw_roi_with_neighbors(_axes[2, _i], _idx, "red")
            _axes[2, _i].set_title(f"#{_idx} snr={trace_snr[_idx]:.1f}", fontsize=8, color="red")

            _trace = T[:, _idx]
            _step = max(1, len(_trace) // 200)
            _axes[3, _i].plot(_trace[::_step], lw=0.4, color="red")
            _axes[3, _i].axis("off")

        _axes[2, 0].set_ylabel("LOW SNR", fontsize=10, color="red", rotation=0, labelpad=50)

        plt.suptitle(f"Top {_n_show} highest vs lowest SNR (gray = neighbors)", fontsize=12)
        plt.tight_layout()
        compare_content = mo.as_html(_fig)
        plt.close(_fig)
    else:
        compare_content = mo.md("no data")
    return (compare_content,)


@app.cell
def _(Ac, Cn, acx, acy, fov, loaded, mask_sizes, mo, nn_dists, np, plt):
    if loaded and len(acx) > 0:
        _n = len(acx)
        _fp_size = Ac.shape[1]
        _thresh_frac = 0.1

        # size categories
        _tiny = (mask_sizes >= 1) & (mask_sizes <= 4)
        _small = (mask_sizes >= 5) & (mask_sizes <= 10)
        _medium = (mask_sizes >= 11) & (mask_sizes <= 20)
        _large = mask_sizes > 20

        _n_tiny = _tiny.sum()
        _n_small = _small.sum()
        _n_medium = _medium.sum()
        _n_large = _large.sum()

        # find overlapping pairs
        _crowded_idx = np.where(nn_dists < _fp_size)[0]
        overlap_pairs = []
        for _i in _crowded_idx[:1000]:
            _fp1 = Ac[_i]
            _mask1 = _fp1 > _fp1.max() * _thresh_frac

            _dists = np.sqrt((acx - acx[_i])**2 + (acy - acy[_i])**2)
            _dists[_i] = np.inf
            _close = np.where(_dists < _fp_size)[0]

            for _j in _close:
                if _j <= _i:
                    continue
                _fp2 = Ac[_j]
                _mask2 = _fp2 > _fp2.max() * _thresh_frac
                _overlap = (_mask1 & _mask2).sum()
                _min_size = min(_mask1.sum(), _mask2.sum())
                if _min_size > 0 and _overlap / _min_size > 0.1:
                    overlap_pairs.append((_i, _j, _overlap / _min_size, _dists[_j]))

        overlap_pairs = sorted(overlap_pairs, key=lambda x: -x[2])
        _n_overlapping = len(overlap_pairs)

        _fig = plt.figure(figsize=(14, 9))
        _gs = _fig.add_gridspec(2, 4, height_ratios=[1, 1.2], hspace=0.3, wspace=0.3)

        # row 0: size distribution bar chart + stats
        _ax0 = _fig.add_subplot(_gs[0, 0])
        _categories = ["1-4px", "5-10px", "11-20px", ">20px"]
        _counts = [_n_tiny, _n_small, _n_medium, _n_large]
        _colors = ["red", "orange", "yellow", "lime"]
        _bars = _ax0.bar(_categories, _counts, color=_colors, edgecolor="white", linewidth=0.5)
        _ax0.set_ylabel("count")
        _ax0.set_title("ROI size distribution", fontsize=10)
        for _bar, _cnt in zip(_bars, _counts):
            _ax0.text(_bar.get_x() + _bar.get_width()/2, _bar.get_height() + _n*0.01,
                     f"{_cnt:,}", ha="center", va="bottom", fontsize=9)

        # stats summary
        _ax1 = _fig.add_subplot(_gs[0, 1])
        _ax1.axis("off")
        _pct_tiny = 100 * _n_tiny / _n
        _pct_small = 100 * _n_small / _n
        _pct_suspect = 100 * (_n_tiny + _n_small) / _n
        _stats = f"""SIZE BREAKDOWN
Total ROIs: {_n:,}

1-4 px:   {_n_tiny:,} ({_pct_tiny:.1f}%)
5-10 px:  {_n_small:,} ({_pct_small:.1f}%)
11-20 px: {_n_medium:,} ({100*_n_medium/_n:.1f}%)
>20 px:   {_n_large:,} ({100*_n_large/_n:.1f}%)

<=10 px total: {_n_tiny + _n_small:,} ({_pct_suspect:.1f}%)
"""
        _ax1.text(0.05, 0.95, _stats, transform=_ax1.transAxes, fontsize=10,
                 verticalalignment="top", family="monospace")

        # overlap stats
        _ax2 = _fig.add_subplot(_gs[0, 2])
        _ax2.axis("off")
        _overlap_stats = f"""OVERLAP ANALYSIS
Pairs with >10% overlap:
  {_n_overlapping:,}

Top overlapping pairs:
"""
        for _p in overlap_pairs[:6]:
            _overlap_stats += f"  #{_p[0]} & #{_p[1]}: {_p[2]*100:.0f}%\n"
        _ax2.text(0.05, 0.95, _overlap_stats, transform=_ax2.transAxes, fontsize=10,
                 verticalalignment="top", family="monospace")

        # size histogram
        _ax3 = _fig.add_subplot(_gs[0, 3])
        _ax3.hist(mask_sizes, bins=50, color="steelblue", edgecolor="none", range=(0, 100))
        _ax3.axvline(4, color="red", ls="--", lw=1.5, label="4px")
        _ax3.axvline(10, color="orange", ls="--", lw=1.5, label="10px")
        _ax3.set_xlabel("mask size (px)")
        _ax3.set_ylabel("count")
        _ax3.set_title("size histogram (0-100px)", fontsize=10)
        _ax3.legend(fontsize=8)

        # row 1: zoomed region with actual masks rendered
        # pick a dense region
        _cx_mid, _cy_mid = fov[1] // 2, fov[0] // 2
        _region_size = 150  # px

        # find ROIs in this region
        _in_region = np.where(
            (acx >= _cx_mid - _region_size//2) & (acx < _cx_mid + _region_size//2) &
            (acy >= _cy_mid - _region_size//2) & (acy < _cy_mid + _region_size//2)
        )[0]

        _y0, _y1 = _cy_mid - _region_size//2, _cy_mid + _region_size//2
        _x0, _x1 = _cx_mid - _region_size//2, _cx_mid + _region_size//2
        _y0, _y1 = max(0, _y0), min(fov[0], _y1)
        _x0, _x1 = max(0, _x0), min(fov[1], _x1)

        _crop = Cn[_y0:_y1, _x0:_x1]
        _crop_h, _crop_w = _crop.shape

        # correlation image
        _ax4 = _fig.add_subplot(_gs[1, 0])
        if _crop.size > 0:
            _p1, _p99 = np.percentile(_crop, [1, 99])
            _ax4.imshow(_crop, cmap="gray", vmin=_p1, vmax=_p99)
        _ax4.set_title(f"correlation image\n({_crop_h}x{_crop_w} region, {len(_in_region)} ROIs)", fontsize=10)
        _ax4.axis("off")

        # mask overlay - all ROIs
        _ax5 = _fig.add_subplot(_gs[1, 1])
        _mask_img = np.zeros((_crop_h, _crop_w, 3), dtype=np.float32)
        if _crop.size > 0:
            _cn_norm = (_crop - _crop.min()) / (_crop.max() - _crop.min() + 1e-9)
            _mask_img[:, :, 0] = _cn_norm * 0.3
            _mask_img[:, :, 1] = _cn_norm * 0.3
            _mask_img[:, :, 2] = _cn_norm * 0.3

        _fp_half = _fp_size // 2
        for _i in _in_region:
            _rx, _ry = acx[_i] - _x0, acy[_i] - _y0
            _fp = Ac[_i]
            _mask = _fp > _fp.max() * _thresh_frac

            # color by size
            if mask_sizes[_i] <= 4:
                _col = np.array([1, 0, 0])  # red
            elif mask_sizes[_i] <= 10:
                _col = np.array([1, 0.5, 0])  # orange
            elif mask_sizes[_i] <= 20:
                _col = np.array([1, 1, 0])  # yellow
            else:
                _col = np.array([0, 1, 0])  # green

            # place footprint
            for _dy in range(_fp_size):
                for _dx in range(_fp_size):
                    _py = _ry - _fp_half + _dy
                    _px = _rx - _fp_half + _dx
                    if 0 <= _py < _crop_h and 0 <= _px < _crop_w and _mask[_dy, _dx]:
                        _mask_img[_py, _px] = _col * 0.7 + _mask_img[_py, _px] * 0.3

        _ax5.imshow(np.clip(_mask_img, 0, 1))
        _ax5.set_title("all masks (color = size)\nred=1-4, orange=5-10, yellow=11-20, green=>20", fontsize=9)
        _ax5.axis("off")

        # only tiny masks
        _ax6 = _fig.add_subplot(_gs[1, 2])
        _tiny_img = np.zeros((_crop_h, _crop_w, 3), dtype=np.float32)
        if _crop.size > 0:
            _tiny_img[:, :, 0] = _cn_norm * 0.3
            _tiny_img[:, :, 1] = _cn_norm * 0.3
            _tiny_img[:, :, 2] = _cn_norm * 0.3

        _tiny_in_region = [_i for _i in _in_region if mask_sizes[_i] <= 10]
        for _i in _tiny_in_region:
            _rx, _ry = acx[_i] - _x0, acy[_i] - _y0
            _fp = Ac[_i]
            _mask = _fp > _fp.max() * _thresh_frac
            _col = np.array([1, 0, 0]) if mask_sizes[_i] <= 4 else np.array([1, 0.5, 0])

            for _dy in range(_fp_size):
                for _dx in range(_fp_size):
                    _py = _ry - _fp_half + _dy
                    _px = _rx - _fp_half + _dx
                    if 0 <= _py < _crop_h and 0 <= _px < _crop_w and _mask[_dy, _dx]:
                        _tiny_img[_py, _px] = _col

        _ax6.imshow(np.clip(_tiny_img, 0, 1))
        _ax6.set_title(f"only tiny masks (<=10px)\n{len(_tiny_in_region)} in this region", fontsize=10)
        _ax6.axis("off")

        # overlapping pairs visualization
        _ax7 = _fig.add_subplot(_gs[1, 3])
        if len(overlap_pairs) > 0:
            _i, _j, _ofrac, _ = overlap_pairs[0]
            _fp1, _fp2 = Ac[_i], Ac[_j]
            _combined = np.zeros((*_fp1.shape, 3))
            _combined[:, :, 0] = _fp1 / (_fp1.max() + 1e-9)
            _combined[:, :, 2] = _fp2 / (_fp2.max() + 1e-9)
            _combined[:, :, 1] = np.minimum(_fp1 / (_fp1.max() + 1e-9), _fp2 / (_fp2.max() + 1e-9))
            _ax7.imshow(np.clip(_combined, 0, 1))
            _ax7.set_title(f"top overlap: #{_i} & #{_j}\n{_ofrac*100:.0f}% overlap", fontsize=10)
        else:
            _ax7.text(0.5, 0.5, "no overlaps", ha="center", va="center")
        _ax7.axis("off")

        plt.suptitle("Size & Overlap Analysis", fontsize=12)
        plt.tight_layout()
        overlap_content = mo.as_html(_fig)
        plt.close(_fig)
    else:
        overlap_content = mo.md("no data")
    return (overlap_content,)


@app.cell
def _(Ac, Cn, T, acx, acy, fov, loaded, mask_sizes, mo, np, plt, trace_snr):
    if loaded and len(acx) > 0:
        _n = len(acx)
        _fp_size = Ac.shape[1]
        _thresh_frac = 0.1

        # categorize ROIs
        _tiny_idx = np.where((mask_sizes >= 1) & (mask_sizes <= 4))[0]
        _small_idx = np.where((mask_sizes >= 5) & (mask_sizes <= 10))[0]
        _medium_idx = np.where((mask_sizes >= 11) & (mask_sizes <= 20))[0]
        _large_idx = np.where(mask_sizes > 20)[0]

        # pick random examples from each category
        _rng = np.random.default_rng(42)
        _n_examples = 4

        def _get_examples(idx_arr, n):
            if len(idx_arr) == 0:
                return []
            return _rng.choice(idx_arr, min(n, len(idx_arr)), replace=False).tolist()

        _tiny_ex = _get_examples(_tiny_idx, _n_examples)
        _small_ex = _get_examples(_small_idx, _n_examples)
        _medium_ex = _get_examples(_medium_idx, _n_examples)
        _large_ex = _get_examples(_large_idx, _n_examples)

        _fig, _axes = plt.subplots(4, 8, figsize=(16, 10))

        _categories = [
            ("1-4px", _tiny_ex, "red"),
            ("5-10px", _small_ex, "orange"),
            ("11-20px", _medium_ex, "yellow"),
            (">20px", _large_ex, "lime"),
        ]

        for _row, (_label, _examples, _color) in enumerate(_categories):
            _axes[_row, 0].set_ylabel(_label, fontsize=10, color=_color, rotation=0, labelpad=60, ha="right")

            for _col, _idx in enumerate(_examples):
                # mask in context
                _cx, _cy = int(acx[_idx]), int(acy[_idx])
                _fp = Ac[_idx]
                _fp_half = _fp_size // 2

                _crop_half = 30
                _y0, _y1 = max(0, _cy - _crop_half), min(fov[0], _cy + _crop_half)
                _x0, _x1 = max(0, _cx - _crop_half), min(fov[1], _cx + _crop_half)
                _crop = Cn[_y0:_y1, _x0:_x1]

                _ax_img = _axes[_row, _col * 2]
                if _crop.size > 0:
                    _p1, _p99 = np.percentile(_crop, [1, 99])
                    _ax_img.imshow(_crop, cmap="gray", vmin=_p1, vmax=_p99)

                    # overlay mask contour
                    _fp_y_start = (_cy - _fp_half) - _y0
                    _fp_x_start = (_cx - _fp_half) - _x0
                    _fp_crop = np.zeros_like(_crop)
                    _fy0 = max(0, _fp_y_start)
                    _fx0 = max(0, _fp_x_start)
                    _fy1 = min(_crop.shape[0], _fp_y_start + _fp_size)
                    _fx1 = min(_crop.shape[1], _fp_x_start + _fp_size)
                    _src_y0 = max(0, -_fp_y_start)
                    _src_x0 = max(0, -_fp_x_start)
                    _src_y1 = _src_y0 + (_fy1 - _fy0)
                    _src_x1 = _src_x0 + (_fx1 - _fx0)
                    if _fy1 > _fy0 and _fx1 > _fx0:
                        _fp_crop[_fy0:_fy1, _fx0:_fx1] = _fp[_src_y0:_src_y1, _src_x0:_src_x1]
                        _ax_img.contour(_fp_crop, levels=[_fp.max() * 0.3], colors=[_color], linewidths=1.5)

                _ax_img.set_title(f"#{_idx}\n{mask_sizes[_idx]:.0f}px, snr={trace_snr[_idx]:.1f}", fontsize=8)
                _ax_img.axis("off")

                # trace
                _ax_tr = _axes[_row, _col * 2 + 1]
                _trace = T[:, _idx]
                _step = max(1, len(_trace) // 150)
                _ax_tr.plot(_trace[::_step], lw=0.5, color=_color)
                _ax_tr.axis("off")

            # fill empty slots
            for _col in range(len(_examples), _n_examples):
                _axes[_row, _col * 2].axis("off")
                _axes[_row, _col * 2 + 1].axis("off")

        plt.suptitle("Size Category Examples: mask (left) + trace (right)", fontsize=12)
        plt.tight_layout()
        examples_content = mo.as_html(_fig)
        plt.close(_fig)
    else:
        examples_content = mo.md("no data")
    return (examples_content,)


@app.cell
def _(Ac, Cn, T, acx, acy, fov, loaded, mo, nn_dists, np, plt):
    if loaded and len(acx) > 0:
        _n = len(acx)
        _fp_size = Ac.shape[1]
        _half = _fp_size // 2
        _thresh_frac = 0.1

        # compute all overlapping pairs
        _crowded_idx = np.where(nn_dists < _fp_size)[0]
        all_overlaps = []
        for _i in _crowded_idx:
            _fp1 = Ac[_i]
            _mask1 = _fp1 > _fp1.max() * _thresh_frac
            _npix1 = _mask1.sum()

            _dists = np.sqrt((acx - acx[_i])**2 + (acy - acy[_i])**2)
            _dists[_i] = np.inf
            _close = np.where(_dists < _fp_size)[0]

            for _j in _close:
                if _j <= _i:
                    continue
                _fp2 = Ac[_j]
                _mask2 = _fp2 > _fp2.max() * _thresh_frac
                _npix2 = _mask2.sum()
                _overlap = (_mask1 & _mask2).sum()
                _min_size = min(_npix1, _npix2)
                if _min_size > 0:
                    _ofrac = _overlap / _min_size
                    if _ofrac > 0.05:
                        all_overlaps.append((_i, _j, _ofrac, _overlap, _dists[_j], _npix1, _npix2))

        all_overlaps = sorted(all_overlaps, key=lambda x: -x[2])

        _fig = plt.figure(figsize=(16, 12))
        _gs = _fig.add_gridspec(3, 4, height_ratios=[1, 1.2, 1.2], hspace=0.35, wspace=0.3)

        # row 0: stats and histograms
        _ax0 = _fig.add_subplot(_gs[0, 0])
        _ax0.axis("off")
        _n_pairs = len(all_overlaps)
        _n_high = sum(1 for p in all_overlaps if p[2] > 0.5)
        _n_medium = sum(1 for p in all_overlaps if 0.2 < p[2] <= 0.5)
        _n_low = sum(1 for p in all_overlaps if p[2] <= 0.2)
        _stats = f"""OVERLAP STATISTICS

Total pairs (>5%): {_n_pairs:,}

>50% overlap: {_n_high:,}
20-50% overlap: {_n_medium:,}
5-20% overlap: {_n_low:,}

Unique ROIs involved:
  {len(set(p[0] for p in all_overlaps) | set(p[1] for p in all_overlaps)):,}
"""
        _ax0.text(0.05, 0.95, _stats, transform=_ax0.transAxes, fontsize=10,
                 verticalalignment="top", family="monospace")

        # overlap fraction histogram
        _ax1 = _fig.add_subplot(_gs[0, 1])
        if len(all_overlaps) > 0:
            _ofracs = [p[2] * 100 for p in all_overlaps]
            _ax1.hist(_ofracs, bins=30, color="steelblue", edgecolor="none")
            _ax1.axvline(50, color="red", ls="--", lw=1.5)
            _ax1.axvline(20, color="orange", ls="--", lw=1.5)
            _ax1.set_xlabel("overlap %")
            _ax1.set_ylabel("count")
            _ax1.set_title("overlap fraction distribution", fontsize=10)

        # overlap vs distance scatter
        _ax2 = _fig.add_subplot(_gs[0, 2])
        if len(all_overlaps) > 0:
            _dists_plot = [p[4] for p in all_overlaps]
            _ofracs_plot = [p[2] * 100 for p in all_overlaps]
            _ax2.scatter(_dists_plot, _ofracs_plot, s=5, alpha=0.5, c="cyan")
            _ax2.set_xlabel("distance (px)")
            _ax2.set_ylabel("overlap %")
            _ax2.set_title("overlap vs distance", fontsize=10)

        # overlap pixels histogram
        _ax3 = _fig.add_subplot(_gs[0, 3])
        if len(all_overlaps) > 0:
            _opix = [p[3] for p in all_overlaps]
            _ax3.hist(_opix, bins=30, color="orange", edgecolor="none")
            _ax3.set_xlabel("overlap pixels")
            _ax3.set_ylabel("count")
            _ax3.set_title("overlap size distribution", fontsize=10)

        # rows 1-2: example overlapping pairs with masks and traces
        _top_pairs = all_overlaps[:8]
        for _pi, _pair in enumerate(_top_pairs):
            _row = 1 + _pi // 4
            _col = _pi % 4

            _i, _j, _ofrac, _opix, _dist, _sz1, _sz2 = _pair
            _ax = _fig.add_subplot(_gs[_row, _col])

            # get region around the pair
            _cx = (acx[_i] + acx[_j]) // 2
            _cy = (acy[_i] + acy[_j]) // 2
            _crop_half = 45
            _y0, _y1 = max(0, _cy - _crop_half), min(fov[0], _cy + _crop_half)
            _x0, _x1 = max(0, _cx - _crop_half), min(fov[1], _cx + _crop_half)
            _crop = Cn[_y0:_y1, _x0:_x1]

            if _crop.size > 0:
                _p1, _p99 = np.percentile(_crop, [1, 99])
                _ax.imshow(_crop, cmap="gray", vmin=_p1, vmax=_p99, extent=[_x0, _x1, _y1, _y0])

                # draw both ROI contours
                _fp1 = Ac[_i]
                _fp2 = Ac[_j]
                _cx1, _cy1 = int(acx[_i]), int(acy[_i])
                _cx2, _cy2 = int(acx[_j]), int(acy[_j])

                if _fp1.max() > 0:
                    _ax.contour(
                        np.arange(_cx1 - _half, _cx1 - _half + _fp_size),
                        np.arange(_cy1 - _half, _cy1 - _half + _fp_size),
                        _fp1, levels=[_fp1.max() * 0.3], colors=["red"], linewidths=1.5
                    )
                if _fp2.max() > 0:
                    _ax.contour(
                        np.arange(_cx2 - _half, _cx2 - _half + _fp_size),
                        np.arange(_cy2 - _half, _cy2 - _half + _fp_size),
                        _fp2, levels=[_fp2.max() * 0.3], colors=["blue"], linewidths=1.5
                    )

                # also show any other neighbors in gray
                _in_view = np.where(
                    (acx >= _x0) & (acx < _x1) & (acy >= _y0) & (acy < _y1)
                )[0]
                for _other in _in_view:
                    if _other != _i and _other != _j:
                        _ofp = Ac[_other]
                        _ocx, _ocy = int(acx[_other]), int(acy[_other])
                        if _ofp.max() > 0:
                            _ax.contour(
                                np.arange(_ocx - _half, _ocx - _half + _fp_size),
                                np.arange(_ocy - _half, _ocy - _half + _fp_size),
                                _ofp, levels=[_ofp.max() * 0.3], colors=["gray"], linewidths=0.5, alpha=0.5
                            )

            _ax.set_title(f"#{_i} (red) & #{_j} (blue)\n{_ofrac*100:.0f}% overlap, {_dist:.0f}px apart", fontsize=8)
            _ax.axis("off")

        # fill empty slots
        for _pi in range(len(_top_pairs), 8):
            _row = 1 + _pi // 4
            _col = _pi % 4
            _fig.add_subplot(_gs[_row, _col]).axis("off")

        plt.suptitle("Overlap Analysis - Top Overlapping Pairs", fontsize=12)
        plt.tight_layout()
        overlaps_content = mo.as_html(_fig)
        plt.close(_fig)
    else:
        overlaps_content = mo.md("no data")
    return (overlaps_content,)


@app.cell
def _(loaded, mo, n_frames):
    movie_frame = mo.ui.slider(0, max(1, n_frames - 1), value=0, label="Frame", show_value=True) if loaded else None
    movie_alpha = mo.ui.slider(0, 1, value=0.5, step=0.05, label="Mask alpha", show_value=True) if loaded else None
    return movie_alpha, movie_frame


@app.cell
def _(Ac, C, Cn, D, acx, acy, fov, loaded, mask_sizes, mo, movie_alpha, movie_frame, np, plt):
    if loaded and movie_frame is not None:
        _frame_idx = movie_frame.value
        _alpha = movie_alpha.value
        _fp_size = Ac.shape[1]
        _fp_half = _fp_size // 2
        _thresh_frac = 0.1

        # pick 3 regions: center, top-left quadrant, bottom-right quadrant
        _regions = [
            ("center", fov[1] // 2, fov[0] // 2),
            ("top-left", fov[1] // 4, fov[0] // 4),
            ("bottom-right", 3 * fov[1] // 4, 3 * fov[0] // 4),
        ]
        _region_size = 120

        # get temporal data for this frame
        _c_frame = C[_frame_idx, :]  # deconvolved activity

        # check if we have raw movie data (Y) or background (b, f)
        _has_raw = "Y" in D or "Yr" in D
        _has_bg = "b" in D and "f" in D

        _fig, _axes = plt.subplots(3, 4, figsize=(14, 10))

        for _ri, (_name, _cx, _cy) in enumerate(_regions):
            _y0, _y1 = max(0, _cy - _region_size//2), min(fov[0], _cy + _region_size//2)
            _x0, _x1 = max(0, _cx - _region_size//2), min(fov[1], _cx + _region_size//2)
            _crop_h = _y1 - _y0
            _crop_w = _x1 - _x0

            # find ROIs in this region
            _in_region = np.where(
                (acx >= _x0) & (acx < _x1) & (acy >= _y0) & (acy < _y1)
            )[0]

            # correlation image as base
            _cn_crop = Cn[_y0:_y1, _x0:_x1]
            _p1, _p99 = np.percentile(_cn_crop, [1, 99]) if _cn_crop.size > 0 else (0, 1)

            # col 0: correlation image
            _axes[_ri, 0].imshow(_cn_crop, cmap="gray", vmin=_p1, vmax=_p99)
            _axes[_ri, 0].set_title(f"{_name}\ncorrelation", fontsize=9)
            _axes[_ri, 0].axis("off")

            # col 1: reconstructed neural activity for this frame
            _neural_img = np.zeros((_crop_h, _crop_w), dtype=np.float32)
            for _i in _in_region:
                _rx, _ry = acx[_i] - _x0, acy[_i] - _y0
                _fp = Ac[_i]
                _mask = _fp > _fp.max() * _thresh_frac
                _activity = _c_frame[_i]

                for _dy in range(_fp_size):
                    for _dx in range(_fp_size):
                        _py = _ry - _fp_half + _dy
                        _px = _rx - _fp_half + _dx
                        if 0 <= _py < _crop_h and 0 <= _px < _crop_w and _mask[_dy, _dx]:
                            _neural_img[_py, _px] += _activity * _fp[_dy, _dx]

            _vmax = np.percentile(_neural_img[_neural_img > 0], 98) if np.any(_neural_img > 0) else 1
            _axes[_ri, 1].imshow(_neural_img, cmap="hot", vmin=0, vmax=max(0.01, _vmax))
            _axes[_ri, 1].set_title(f"activity frame {_frame_idx}", fontsize=9)
            _axes[_ri, 1].axis("off")

            # col 2: correlation with mask overlay (using alpha)
            _cn_norm = (_cn_crop - _p1) / (_p99 - _p1 + 1e-9)
            _cn_norm = np.clip(_cn_norm, 0, 1)
            _overlay = np.stack([_cn_norm, _cn_norm, _cn_norm], axis=-1)

            for _i in _in_region:
                _rx, _ry = acx[_i] - _x0, acy[_i] - _y0
                _fp = Ac[_i]
                _mask = _fp > _fp.max() * _thresh_frac

                if mask_sizes[_i] <= 4:
                    _col = np.array([1, 0, 0])
                elif mask_sizes[_i] <= 10:
                    _col = np.array([1, 0.5, 0])
                elif mask_sizes[_i] <= 20:
                    _col = np.array([1, 1, 0])
                else:
                    _col = np.array([0, 1, 0])

                for _dy in range(_fp_size):
                    for _dx in range(_fp_size):
                        _py = _ry - _fp_half + _dy
                        _px = _rx - _fp_half + _dx
                        if 0 <= _py < _crop_h and 0 <= _px < _crop_w and _mask[_dy, _dx]:
                            _overlay[_py, _px] = (1 - _alpha) * _overlay[_py, _px] + _alpha * _col

            _axes[_ri, 2].imshow(np.clip(_overlay, 0, 1))
            _axes[_ri, 2].set_title(f"masks (alpha={_alpha:.2f})", fontsize=9)
            _axes[_ri, 2].axis("off")

            # col 3: activity + mask contours
            _activity_rgb = np.zeros((_crop_h, _crop_w, 3), dtype=np.float32)
            _neural_norm = _neural_img / (max(0.01, _vmax))
            _activity_rgb[:, :, 0] = np.clip(_neural_norm, 0, 1)

            # add mask contours
            for _i in _in_region:
                _rx, _ry = acx[_i] - _x0, acy[_i] - _y0
                _fp = Ac[_i]
                _mask = _fp > _fp.max() * 0.3

                # draw edge pixels
                for _dy in range(_fp_size):
                    for _dx in range(_fp_size):
                        _py = _ry - _fp_half + _dy
                        _px = _rx - _fp_half + _dx
                        if 0 <= _py < _crop_h and 0 <= _px < _crop_w and _mask[_dy, _dx]:
                            # check if edge
                            _is_edge = False
                            for _ddy, _ddx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                _ny, _nx = _dy + _ddy, _dx + _ddx
                                if 0 <= _ny < _fp_size and 0 <= _nx < _fp_size:
                                    if not _mask[_ny, _nx]:
                                        _is_edge = True
                                        break
                                else:
                                    _is_edge = True
                                    break
                            if _is_edge:
                                _activity_rgb[_py, _px] = [0, 1, 1]  # cyan edge

            _axes[_ri, 3].imshow(np.clip(_activity_rgb, 0, 1))
            _axes[_ri, 3].set_title(f"activity + contours\n{len(_in_region)} ROIs", fontsize=9)
            _axes[_ri, 3].axis("off")

        plt.suptitle(f"Regional Views - Frame {_frame_idx}", fontsize=12)
        plt.tight_layout()
        movie_fig = mo.as_html(_fig)
        plt.close(_fig)
    else:
        movie_fig = mo.md("no data")
    return (movie_fig,)


@app.cell
def _(loaded, mo, movie_alpha, movie_fig, movie_frame):
    if loaded and movie_frame and movie_alpha:
        movie_content = mo.vstack([
            mo.hstack([movie_frame, movie_alpha], widths=[0.7, 0.3]),
            movie_fig
        ])
    else:
        movie_content = mo.md("no data")
    return (movie_content,)


@app.cell
def _(compare_content, examples_content, inspect_content, loaded, mo, movie_content, overlap_content, overlaps_content, status):
    if loaded:
        main_ui = mo.vstack([
            status,
            mo.ui.tabs({
                "Inspect": inspect_content,
                "Compare": compare_content,
                "Sizes": overlap_content,
                "Overlaps": overlaps_content,
                "Examples": examples_content,
                "Movie": movie_content,
            })
        ])
    else:
        main_ui = mo.md("**No data loaded.** Enter path to CaImAn output directory above.")
    return (main_ui,)


@app.cell
def _(main_ui):
    main_ui
    return


if __name__ == "__main__":
    app.run()
