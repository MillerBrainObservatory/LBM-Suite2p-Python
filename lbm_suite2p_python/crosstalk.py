import os
from multiprocessing import Pool

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import gamma

from lbm_suite2p_python import gaussian


def calculate_crosstalk_coeff(im3d, exclude_below=1, sigma=0.01, peak_width=1,
                              verbose=True, estimate_gamma=True, estimate_from_last_n_planes=None,
                              n_proc=1, show_plots=True, save_plots=None, force_positive=True,
                              m_penalty=0, bounds=None, fit_above_percentile=0, fig_scale=3,
                              n_per_cavity=None):
    ## from
    plt.style.use('seaborn')
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

    ms = np.linspace(0, 1, 101)
    nz, ny, nx = im3d.shape

    if estimate_from_last_n_planes is None:
        estimate_from_last_n_planes = n_per_cavity

    if save_plots is not None:
        plot_dir = os.path.join(save_plots, 'crosstalk_plots')
        os.makedirs(plot_dir, exist_ok=True)

    fs = []
    n_plots = estimate_from_last_n_planes
    n_cols = 5
    n_rows = np.ceil(n_plots / n_cols).astype(int)

    for i in range(estimate_from_last_n_planes):
        # print("Plot for plane %d" % i)
        Y = im3d[nz - i - 1].flatten()
        X = im3d[nz - i - 1 - n_per_cavity].flatten()
        fit_thresh = np.percentile(X, fit_above_percentile)
        # print(fit_thresh)
        idxs = X > np.percentile(X, fit_above_percentile)
        # print(len(idxs), X.shape)

        if n_proc == 1:
            liks = np.array([sum_log_lik_one_line(m, X[idxs], Y[idxs], sigma_0=sigma, m_penalty=m_penalty) for m in ms])
        else:
            p = Pool(n_proc)
            liks = p.starmap(sum_log_lik_one_line, [(m, X[idxs], Y[idxs], 0, sigma, 1e-10, m_penalty) for m in ms])
            liks = np.array(liks)

        m_opt = ms[np.argmin(liks)]
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
            bounds = (0, np.percentile(X, 99.95))
    m_opts = np.array(m_opts)
    m_firsts = np.array(m_firsts)

    best_ms = m_opts[m_opts == m_firsts]
    best_m = best_ms.mean()

    if estimate_gamma:
        gx = gamma.fit(m_opts)
        x = np.linspace(0, 1, 1001)
        gs = gamma.pdf(x, *gx)
        f = plt.figure(figsize=(3, 3))
        plt.hist(m_opts, density=True, log=False, bins=np.arange(0, 1.01, 0.01))
        plt.plot(x, gs)
        plt.yticks([])
        plt.scatter([x[np.argmax(gs)]], [np.max(gs)], label='Best coeff: %.3f' % x[np.argmax(gs)])
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
        best_m = x[np.argmax(gs)]

    return m_opts, m_firsts, best_m


def sum_log_lik_one_line(m, x, y, b=0, sigma_0=10, c=1e-10, m_penalty=0):
    mu = m * x + b
    lik_line = gaussian(y, mu, sigma_0)
    lik = lik_line

    log_lik = np.log(lik + c - m * m_penalty).sum()

    return -log_lik
