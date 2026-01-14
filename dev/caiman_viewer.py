# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "numpy",
#     "h5py",
#     "scipy",
# ]
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full", app_title="CaImAn Viewer")


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

    # dark mode
    plt.style.use("dark_background")
    plt.rcParams.update({
        "figure.facecolor": "#181818",
        "axes.facecolor": "#181818",
        "savefig.facecolor": "#181818",
        "figure.dpi": 100,
    })
    return Path, cdist, h5py, mo, np, plt, zoom


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
    def _load(fp):
        d = {}
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

    _p = Path(data_path.value) / f"caiman_output_plane_{plane_num.value}.mat"
    if _p.exists():
        D = _load(_p)
        Cn = D.get("Cn", D.get("Ym"))
        fov = Cn.shape
        n_rois = D["Ac_keep"].shape[0]
        n_frames = D["T_keep"].shape[0]
        acx = D["acx"].flatten().astype(int)
        acy = D["acy"].flatten().astype(int)
        Ac = D["Ac_keep"]
        T = D["T_keep"]
        C = D["C_keep"]

        # precompute downsampled Cn for display (max 400px)
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
        D = {}
        Cn = np.zeros((512, 512))
        Cn_small = np.zeros((512, 512))
        fov = (512, 512)
        n_rois = 0
        n_frames = 0
        acx = np.array([])
        acy = np.array([])
        Ac = np.zeros((0, 0, 0))
        T = np.zeros((0, 0))
        C = np.zeros((0, 0))
        scale_factor = 1.0
        loaded = False
    return Ac, C, Cn, Cn_small, D, T, acx, acy, fov, loaded, n_frames, n_rois, scale_factor


@app.cell
def _(loaded, mo, n_frames, n_rois, fov):
    status = mo.md(f"**{n_rois:,} ROIs** · {n_frames:,} frames · {fov[0]}×{fov[1]}") if loaded else mo.md("no data")
    return (status,)


# precompute stats on subsample
@app.cell
def _(Ac, T, acx, acy, cdist, loaded, np):
    if loaded and len(acx) > 0:
        _n = min(3000, len(acx))
        _rng = np.random.default_rng(42)
        sampled_idx = _rng.choice(len(acx), _n, replace=False)

        # mask sizes
        mask_sizes = np.array([(a > a.max() * 0.1).sum() for a in Ac[sampled_idx]])

        # nearest neighbor (use cdist on subset for memory)
        _coords = np.column_stack([acx[sampled_idx], acy[sampled_idx]])
        _d = cdist(_coords, _coords)
        np.fill_diagonal(_d, np.inf)
        nn_dists = _d.min(axis=1)

        # trace snr
        _T = T[:, sampled_idx]
        trace_snr = _T.mean(axis=0) / (_T.std(axis=0) + 1e-6)
    else:
        sampled_idx = np.array([])
        mask_sizes = np.array([])
        nn_dists = np.array([])
        trace_snr = np.array([])
    return mask_sizes, nn_dists, sampled_idx, trace_snr


# explore tab
@app.cell
def _(loaded, mo, sampled_idx):
    neuron_slider = mo.ui.slider(0, max(1, len(sampled_idx) - 1), value=0, label="Neuron", show_value=True) if loaded and len(sampled_idx) > 0 else None
    return (neuron_slider,)


@app.cell
def _(Ac, Cn, T, acx, acy, fov, loaded, mo, neuron_slider, np, plt, sampled_idx):
    if loaded and neuron_slider is not None:
        _nidx = sampled_idx[neuron_slider.value]
        _cx, _cy = int(acx[_nidx]), int(acy[_nidx])

        _fig, _axes = plt.subplots(1, 3, figsize=(11, 3))

        # mask
        _axes[0].imshow(Ac[_nidx], cmap="hot")
        _axes[0].set_title(f"#{_nidx}")
        _axes[0].axis("off")

        # zoomed region
        _half = 35
        _y0, _y1 = max(0, _cy - _half), min(fov[0], _cy + _half)
        _x0, _x1 = max(0, _cx - _half), min(fov[1], _cx + _half)
        _crop = Cn[_y0:_y1, _x0:_x1]
        if _crop.size > 0:
            _p1, _p99 = np.percentile(_crop, [1, 99])
            _axes[1].imshow(_crop, cmap="gray", vmin=_p1, vmax=_p99)
            _axes[1].plot(_cx - _x0, _cy - _y0, "r+", ms=10, mew=2)
        _axes[1].set_title(f"({_cx}, {_cy})")
        _axes[1].axis("off")

        # trace
        _trace = T[:, _nidx]
        _step = max(1, len(_trace) // 400)
        _axes[2].plot(_trace[::_step], lw=0.5, color="cyan")
        _axes[2].set_title(f"snr={_trace.mean()/_trace.std():.2f}")
        _axes[2].set_xlabel("frame")

        plt.tight_layout()
        explore_fig = mo.as_html(_fig)
        plt.close(_fig)
    else:
        explore_fig = mo.md("")
    return (explore_fig,)


@app.cell
def _(explore_fig, loaded, mo, neuron_slider):
    explore_content = mo.vstack([neuron_slider, explore_fig]) if loaded and neuron_slider else mo.md("no data")
    return (explore_content,)


# traces tab
@app.cell
def _(T, loaded, mo, np, plt, sampled_idx):
    if loaded and len(sampled_idx) > 0:
        _n = min(15, len(sampled_idx))
        _step = max(1, T.shape[0] // 400)

        _fig, _ax = plt.subplots(figsize=(10, 5))
        for _i in range(_n):
            _trace = T[::_step, sampled_idx[_i]]
            _tn = (_trace - _trace.min()) / (_trace.max() - _trace.min() + 1e-6)
            _ax.plot(_tn + _i * 1.1, lw=0.4)
        _ax.set_xlabel("frame")
        _ax.set_ylabel("neuron")
        _ax.set_title(f"{_n} traces (normalized, stacked)")
        plt.tight_layout()
        traces_content = mo.as_html(_fig)
        plt.close(_fig)
    else:
        traces_content = mo.md("no data")
    return (traces_content,)


# coverage tab - this is the key one
@app.cell
def _(Ac, Cn_small, acx, acy, fov, loaded, mo, np, plt, scale_factor):
    if loaded and len(acx) > 0:
        _fig, _axes = plt.subplots(1, 3, figsize=(12, 4))

        # 1. downsampled correlation image
        _p1, _p99 = np.percentile(Cn_small, [1, 99])
        _axes[0].imshow(Cn_small, cmap="gray", vmin=_p1, vmax=_p99)
        _axes[0].set_title(f"correlation ({Cn_small.shape[0]}×{Cn_small.shape[1]})")
        _axes[0].axis("off")

        # 2. ROI centers on downsampled image
        _axes[1].imshow(Cn_small, cmap="gray", vmin=_p1, vmax=_p99)
        _acx_s = (acx * scale_factor).astype(int)
        _acy_s = (acy * scale_factor).astype(int)
        _idx = np.random.default_rng(0).choice(len(acx), min(2000, len(acx)), replace=False)
        _axes[1].scatter(_acx_s[_idx], _acy_s[_idx], c="red", s=0.3, alpha=0.5)
        _axes[1].set_title(f"{len(acx):,} ROI centers")
        _axes[1].axis("off")

        # 3. coverage heatmap - how many masks per region
        _small_shape = Cn_small.shape
        _coverage = np.zeros(_small_shape, dtype=np.float32)
        _fp_size = Ac.shape[1]
        _fp_half = _fp_size // 2

        # bin ROIs into coverage map
        for _i in range(min(10000, len(acx))):
            _y = int(acy[_i] * scale_factor)
            _x = int(acx[_i] * scale_factor)
            # add gaussian-ish blob
            _r = max(2, int(_fp_half * scale_factor))
            _y0, _y1 = max(0, _y - _r), min(_small_shape[0], _y + _r)
            _x0, _x1 = max(0, _x - _r), min(_small_shape[1], _x + _r)
            _coverage[_y0:_y1, _x0:_x1] += 1

        _im = _axes[2].imshow(_coverage, cmap="hot")
        _axes[2].set_title(f"coverage (max {int(_coverage.max())} ROIs/region)")
        _axes[2].axis("off")
        plt.colorbar(_im, ax=_axes[2], shrink=0.8)

        plt.tight_layout()
        coverage_content = mo.as_html(_fig)
        plt.close(_fig)
    else:
        coverage_content = mo.md("no data")
    return (coverage_content,)


# crowding tab
@app.cell
def _(Ac, Cn, acx, acy, fov, loaded, mask_sizes, mo, nn_dists, np, plt, sampled_idx):
    if loaded and len(nn_dists) > 0:
        _fig, _axes = plt.subplots(2, 3, figsize=(12, 7))

        # histograms
        _axes[0, 0].hist(nn_dists, bins=40, color="steelblue", edgecolor="none")
        _axes[0, 0].axvline(5, color="red", ls="--", lw=1.5)
        _n_crowded = (nn_dists < 5).sum()
        _axes[0, 0].set_title(f"NN dist: {_n_crowded} < 5px ({100*_n_crowded/len(nn_dists):.0f}%)")
        _axes[0, 0].set_xlabel("pixels")

        _axes[0, 1].hist(mask_sizes, bins=40, color="steelblue", edgecolor="none")
        _axes[0, 1].axvline(10, color="red", ls="--", lw=1.5)
        _n_tiny = (mask_sizes < 10).sum()
        _axes[0, 1].set_title(f"size: {_n_tiny} < 10px ({100*_n_tiny/len(mask_sizes):.0f}%)")
        _axes[0, 1].set_xlabel("pixels")

        # scatter
        _axes[0, 2].scatter(nn_dists, mask_sizes, s=2, alpha=0.3, c="cyan")
        _axes[0, 2].axvline(5, color="red", ls="--", lw=1, alpha=0.7)
        _axes[0, 2].axhline(10, color="orange", ls="--", lw=1, alpha=0.7)
        _axes[0, 2].set_xlabel("NN dist")
        _axes[0, 2].set_ylabel("size")
        _axes[0, 2].set_title("crowded + small = bad")

        # example crowded pairs (small crops)
        _crowded = np.where(nn_dists < 5)[0]
        _fp_size = Ac.shape[1]
        for _pi in range(3):
            if _pi < len(_crowded):
                _ci = _crowded[_pi]
                _nidx = sampled_idx[_ci]
                _cx, _cy = int(acx[_nidx]), int(acy[_nidx])
                _half = 30
                _y0, _y1 = max(0, _cy - _half), min(fov[0], _cy + _half)
                _x0, _x1 = max(0, _cx - _half), min(fov[1], _cx + _half)
                _crop = Cn[_y0:_y1, _x0:_x1]
                if _crop.size > 0:
                    _p1, _p99 = np.percentile(_crop, [1, 99])
                    _axes[1, _pi].imshow(_crop, cmap="gray", vmin=_p1, vmax=_p99)
                    _axes[1, _pi].plot(_cx - _x0, _cy - _y0, "r+", ms=8, mew=2)
                    _axes[1, _pi].set_title(f"#{_nidx} nn={nn_dists[_ci]:.1f}px")
            _axes[1, _pi].axis("off")

        plt.tight_layout()
        crowding_content = mo.as_html(_fig)
        plt.close(_fig)
    else:
        crowding_content = mo.md("no data")
    return (crowding_content,)


# quality tab
@app.cell
def _(Ac, loaded, mask_sizes, mo, np, plt, sampled_idx, trace_snr):
    if loaded and len(trace_snr) > 0:
        _fig, _axes = plt.subplots(2, 3, figsize=(12, 7))

        # snr histogram
        _axes[0, 0].hist(trace_snr, bins=40, color="steelblue", edgecolor="none")
        _axes[0, 0].axvline(1, color="red", ls="--", lw=1.5)
        _n_noisy = (trace_snr < 1).sum()
        _axes[0, 0].set_title(f"SNR: {_n_noisy} < 1 ({100*_n_noisy/len(trace_snr):.0f}%)")
        _axes[0, 0].set_xlabel("SNR")

        # size vs snr
        _axes[0, 1].scatter(mask_sizes, trace_snr, s=2, alpha=0.3, c="cyan")
        _axes[0, 1].axvline(10, color="red", ls="--", lw=1, alpha=0.7)
        _axes[0, 1].axhline(1, color="red", ls="--", lw=1, alpha=0.7)
        _axes[0, 1].set_xlabel("size")
        _axes[0, 1].set_ylabel("SNR")
        _axes[0, 1].set_title("quality quadrants")

        # summary
        _axes[0, 2].axis("off")
        _bad = ((trace_snr < 1) | (mask_sizes < 10)).sum()
        _axes[0, 2].text(0.1, 0.7, f"Total sampled: {len(sampled_idx):,}", fontsize=12, transform=_axes[0, 2].transAxes)
        _axes[0, 2].text(0.1, 0.5, f"Low SNR (<1): {_n_noisy:,}", fontsize=12, color="orange", transform=_axes[0, 2].transAxes)
        _axes[0, 2].text(0.1, 0.3, f"Tiny (<10px): {(mask_sizes < 10).sum():,}", fontsize=12, color="orange", transform=_axes[0, 2].transAxes)
        _axes[0, 2].text(0.1, 0.1, f"Either: {_bad:,} ({100*_bad/len(sampled_idx):.0f}%)", fontsize=12, color="red", transform=_axes[0, 2].transAxes)

        # example tiny/noisy masks
        _bad_idx = np.where((trace_snr < 1) | (mask_sizes < 10))[0]
        for _pi in range(3):
            if _pi < len(_bad_idx):
                _bi = _bad_idx[_pi]
                _nidx = sampled_idx[_bi]
                _axes[1, _pi].imshow(Ac[_nidx], cmap="hot")
                _axes[1, _pi].set_title(f"#{_nidx} sz={mask_sizes[_bi]:.0f} snr={trace_snr[_bi]:.1f}")
            _axes[1, _pi].axis("off")

        plt.tight_layout()
        quality_content = mo.as_html(_fig)
        plt.close(_fig)
    else:
        quality_content = mo.md("no data")
    return (quality_content,)


# masks gallery
@app.cell
def _(Ac, loaded, mo, np, plt, sampled_idx):
    if loaded and len(sampled_idx) > 0:
        _n = min(25, len(sampled_idx))
        _idx = np.linspace(0, len(sampled_idx) - 1, _n, dtype=int)

        _fig, _axes = plt.subplots(5, 5, figsize=(8, 8))
        for _i, _ax in enumerate(_axes.flat):
            if _i < _n:
                _nidx = sampled_idx[_idx[_i]]
                _ax.imshow(Ac[_nidx], cmap="hot")
                _ax.set_title(f"#{_nidx}", fontsize=7)
            _ax.axis("off")

        plt.suptitle(f"{_n} of {len(sampled_idx):,} sampled ({Ac.shape[0]:,} total)", fontsize=10)
        plt.tight_layout()
        masks_content = mo.as_html(_fig)
        plt.close(_fig)
    else:
        masks_content = mo.md("no data")
    return (masks_content,)


# movie tab
@app.cell
def _(loaded, mo, n_frames):
    frame_slider = mo.ui.slider(0, max(1, n_frames - 1), value=0, label="Frame", show_value=True) if loaded else None
    return (frame_slider,)


@app.cell
def _(Ac, C, Cn_small, acx, acy, fov, frame_slider, loaded, mo, np, plt, scale_factor):
    if loaded and frame_slider is not None:
        _fidx = frame_slider.value
        _c_t = C[_fidx, :]

        # build reconstruction at small scale
        _small_shape = Cn_small.shape
        _neural = np.zeros(_small_shape, dtype=np.float32)
        _fp_size = Ac.shape[1]

        _active = np.where(_c_t > 0)[0]
        for _i in _active[:2000]:
            _y = int(acy[_i] * scale_factor)
            _x = int(acx[_i] * scale_factor)
            _r = max(1, int((_fp_size // 2) * scale_factor))
            _y0, _y1 = max(0, _y - _r), min(_small_shape[0], _y + _r + 1)
            _x0, _x1 = max(0, _x - _r), min(_small_shape[1], _x + _r + 1)
            _neural[_y0:_y1, _x0:_x1] += _c_t[_i]

        _fig, _axes = plt.subplots(1, 2, figsize=(10, 4))

        _p1, _p99 = np.percentile(Cn_small, [1, 99])
        _axes[0].imshow(Cn_small, cmap="gray", vmin=_p1, vmax=_p99)
        _axes[0].set_title("correlation")
        _axes[0].axis("off")

        _vmax = np.percentile(_neural[_neural > 0], 98) if np.any(_neural > 0) else 1
        _axes[1].imshow(_neural, cmap="hot", vmin=0, vmax=max(0.01, _vmax))
        _axes[1].set_title(f"frame {_fidx} activity")
        _axes[1].axis("off")

        plt.tight_layout()
        movie_fig = mo.as_html(_fig)
        plt.close(_fig)
    else:
        movie_fig = mo.md("")
    return (movie_fig,)


@app.cell
def _(frame_slider, loaded, mo, movie_fig):
    movie_content = mo.vstack([frame_slider, movie_fig]) if loaded and frame_slider else mo.md("no data")
    return (movie_content,)


# tabs
@app.cell
def _(coverage_content, crowding_content, explore_content, loaded, masks_content, mo, movie_content, quality_content, status, traces_content):
    if loaded:
        main_ui = mo.vstack([
            status,
            mo.ui.tabs({
                "Explore": explore_content,
                "Traces": traces_content,
                "Coverage": coverage_content,
                "Crowding": crowding_content,
                "Quality": quality_content,
                "Masks": masks_content,
                "Movie": movie_content,
            })
        ])
    else:
        main_ui = mo.md("no data loaded")
    return (main_ui,)


@app.cell
def _(main_ui):
    main_ui
    return


if __name__ == "__main__":
    app.run()
