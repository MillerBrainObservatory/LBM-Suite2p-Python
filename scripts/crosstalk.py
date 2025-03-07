from pathlib import Path
import numpy as np
import os
import tifffile

from multiprocessing import Pool
from scipy.signal import find_peaks
from scipy.stats import gamma

import matplotlib.pyplot as plt
import fastplotlib as fpl
import seaborn as sns

import lbm_suite2p_python as lsp
import mbo_utilities as mbo

import lbm_suite2p_python.crosstalk


def calculate_crosstalk_coeff(im3d, exclude_below=1, sigma=0.01, peak_width=1,
                              verbose=True, estimate_gamma=True, estimate_from_last_n_planes=None,
                              n_proc=1, show_plots=True, save_plots=None, force_positive=True,
                              m_penalty=0, bounds=None, fit_above_percentile=0, fig_scale=3,
                              n_per_cavity=None):
    import numpy as n
    m_opts = []
    m_firsts = []
    all_liks = []
    m_opt_liks = []
    m_first_liks = []
    im3d = im3d.copy()
    if n_per_cavity is None:
        n_per_cavity = im3d.shape[0] // 2
    if force_positive:
        im3d = im3d - im3d.min(axis=(1, 2), keepdims=True)

    ms = n.linspace(0, 1, 101)
    nz, ny, nx = im3d.shape

    if estimate_from_last_n_planes is None:
        estimate_from_last_n_planes = n_per_cavity

    if save_plots is not None:
        plot_dir = os.path.join(save_plots, 'crosstalk_plots')
        os.makedirs(plot_dir, exist_ok=True)

    fs = []
    n_plots = estimate_from_last_n_planes
    n_cols = 5
    n_rows = n.ceil(n_plots / n_cols).astype(int)

    for i in range(estimate_from_last_n_planes):
        # print("Plot for plane %d" % i)
        Y = im3d[nz - i - 1].flatten()
        X = im3d[nz - i - 1 - n_per_cavity].flatten()
        fit_thresh = n.percentile(X, fit_above_percentile)
        # print(fit_thresh)
        idxs = X > n.percentile(X, fit_above_percentile)
        # print(len(idxs), X.shape)

        # For each potential scaling factor, 0 0.01 0.02 ... 1
        if n_proc == 1:
            liks = n.array([lbm_suite2p_python.crosstalk.sum_log_lik_one_line(m, X[idxs], Y[idxs], sigma_0=sigma, m_penalty=m_penalty) for m in ms])
        else:
            p = Pool(n_proc)
            liks = p.starmap(lbm_suite2p_python.crosstalk.sum_log_lik_one_line, [(m, X[idxs], Y[idxs], 0, sigma, 1e-10, m_penalty) for m in ms])
            liks = n.array(liks)

        m_opt = ms[n.argmin(liks)]
        pks = find_peaks(-liks, width=peak_width)[0]
        m_first = ms[pks[0]]

        m_opts.append(m_opt)
        m_firsts.append(m_first)
        all_liks.append(liks)
        m_opt_liks.append(liks.min())
        m_first_liks.append(liks[pks[0]])

        if verbose:
            print("Plane %d and %d, m_opt: %.2f and m_first: %.2f" % (i, i + n_per_cavity, m_opt, m_first))

        if bounds is None:
            bounds = (0, n.percentile(X, 99.95))
    m_opts = n.array(m_opts)
    m_firsts = n.array(m_firsts)

    best_ms = m_opts[m_opts == m_firsts]
    best_m = best_ms.mean()

    if estimate_gamma:
        gx = gamma.fit(m_opts)
        x = n.linspace(0, 1, 1001)
        gs = gamma.pdf(x, *gx)
        f = plt.figure(figsize=(3, 3))
        plt.hist(m_opts, density=True, log=False, bins=n.arange(0, 1.01, 0.01))
        plt.plot(x, gs)
        plt.yticks([])
        plt.scatter([x[n.argmax(gs)]], [n.max(gs)], label='Best coeff: %.3f' % x[n.argmax(gs)])
        plt.legend()
        plt.xlabel("Coeff value")
        plt.ylabel("")
        plt.xlim(0, 0.4)
        plt.title("Histogram of est. coefficients per plane")
        if save_plots is not None:
            plt.savefig(os.path.join(plot_dir, 'gamma_fit.png'), dpi=200)
        if show_plots:
            plt.show()
        plt.close()
        fs.append(f)
        best_m = x[n.argmax(gs)]

    return m_opts, m_firsts, best_m

if __name__ == '__main__':
    data_dir = Path("D:/W2_DATA/kbarber/2025-02-17/mk303/assembled")
    save_path = data_dir.parent
    input_files = mbo.get_files(data_dir, 'tif', 1)
    fpath_6 = input_files[5]
    fpath_7 = input_files[6]
    ops6 = np.load(r"D:\W2_DATA\kbarber\2025-02-17\mk303\results\plane_06\plane0\ops.npy", allow_pickle=True).item()
    ops7 = np.load(r"D:\W2_DATA\kbarber\2025-02-17\mk303\results\plane_07\plane0\ops.npy", allow_pickle=True).item()
    ops6_ref = ops6['meanImg']
    ops7_ref = ops7['meanImg']
    im3d = np.stack([ops6_ref, ops7_ref])
    m_o, m_f, best_m = calculate_crosstalk_coeff(im3d, estimate_gamma=False, save_plots=save_path)
    print('done')
