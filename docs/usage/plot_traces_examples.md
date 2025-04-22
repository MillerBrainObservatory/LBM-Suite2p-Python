---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.0
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Function Demo: `plot_traces`

```{code-cell} ipython3
from pathlib import Path
import mbo_utilities as mbo
import lbm_suite2p_python as lsp
```

## Load traces with `lbm_suite2p_python.load_traces`

```{code-cell} ipython3
# lazy way to find all ops.npy files in this directory
ops_files = mbo.get_files(r"D:\W2_DATA", 'ops.npy', max_depth=10)
ops_files[:3]
```

```{code-cell} ipython3
ops = lsp.load_ops(ops_files[6])
f, _, _ = lsp.load_traces(ops)
```

```{code-cell} ipython3
plot_traces(f, './plot_traces_raw.png', signal_units='raw', fps=metadata["frame_rate"])
```

```{code-cell} ipython3
dff = lsp.dff_percentile(f) * 100
```

```{code-cell} ipython3
# lsp.plot_traces(dff, './plot_traces_single.png', signal_units='dff', num_neurons=1)  # dff is default
lsp.plot_traces(dff, './plot_traces_single_offset_50.png', signal_units='dff', num_neurons=2, offset=250)  # dff is default
```

```{code-cell} ipython3
lsp.plot_traces(dff, './plot_traces_dff.png', signal_units='dff')  # dff is default
```

```{code-cell} ipython3
lsp.plot_traces(dff[::2], './plot_traces_dff_100n_nipy.png', signal_units='dff', num_neurons=100, cmap="nipy_spectral")  # dff is default
```

```{code-cell} ipython3
bin_factor = 2
dff_binned = lsp.bin1d(dff, bin_factor)
fps_binned = metadata["frame_rate"] / bin_factor

plot_traces(dff_binned, './plot_traces_dff_binned_2x.png', signal_units='dff', fps=fps_binned)
```

```{code-cell} ipython3

```
