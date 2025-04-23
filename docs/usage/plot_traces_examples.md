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

# Function Demo: plot_traces

{func}`lbm_suite2p_python.plot_traces`

`plot_traces` is a convenience function for plotting suite2p traces.

Data input is in the same format as F.npy: `[n_neurons x n_timepoints]`.

The main features of this function:
- automatic time and signal scalebars that adjust to the input data
- automatic offset between traces calculated from the magnitude of the baseline signal
- lower traces mask into upper traces to highlight the high-magnitude transients

```{code-cell} ipython3
from pathlib import Path
import mbo_utilities as mbo
import lbm_suite2p_python as lsp
```

## Preparation

{func}`mbo_utilities.get_files`
{func}`lbm_suite2p_python.load_ops`
{func}`lbm_suite2p_python.load_traces`

This is a lazy way to get any ops.npy files in this directory.

```{code-cell} ipython3
ops_files = mbo.get_files(r"D:\W2_DATA", 'ops.npy', max_depth=10)
ops_files[:3]
```

We then use the ops.npy file to retrieve 

```{code-cell} ipython3
ops = lsp.load_ops(ops_files[6])
f, fneu, spks = lsp.load_traces(ops)  # only need f here
```

## Raw trace plot

```{code-cell} ipython3
lsp.plot_traces(f, './plot_traces_raw.png', signal_units='raw', fps=metadata["frame_rate"])
```

```{figure} ../_images/plot_traces_raw.png
:alt: Raw signal traces
:width: 100%

Raw fluorescence traces (no normalization).
```

## Percentile-normalized DF/F

```{code-cell} ipython3
dff = lsp.dff_percentile(f) * 100
```

## Single neuron

```{code-cell} ipython3
lsp.plot_traces(dff, './plot_traces_single.png', signal_units='dff', num_neurons=1)
```

```{figure} ../_images/plot_traces_single.png
:alt: Single neuron DF/F
:width: 100%

Single neuron trace (percentile-based DF/F).
```

## Two neurons with manual offset

```{code-cell} ipython3
lsp.plot_traces(dff, './plot_traces_single_gap50.png', signal_units='dff', num_neurons=2, offset=250)
```

```{figure} ../_images/plot_traces_single_gap50.png
:alt: Two neurons with offset
:width: 100%

Two traces with manually specified offset.
```

## Default dff

```{code-cell} ipython3
lsp.plot_traces(dff, './plot_traces_dff.png', signal_units='dff')
```

```{figure} ../_images/plot_traces_dff.png
:alt: Default DF/F traces
:width: 100%

Default appearance with vertical stack and scalebars.
```

## More neurons

```{code-cell} ipython3
lsp.plot_traces(dff[:50], './plot_traces_dff_50n.png', signal_units='dff', num_neurons=50)
```

```{figure} ../_images/plot_traces_dff_50n.png
:alt: 50 neuron traces
:width: 100%

First 50 neurons, default colormap.
```

### 100 neurons, custom colormap

```{code-cell} ipython3
lsp.plot_traces(dff[::2], './plot_traces_dff_100n_nipy.png', signal_units='dff', num_neurons=100, cmap="nipy_spectral")
```

```{figure} ../_images/plot_traces_dff_100n_nipy.png
:alt: 100 neurons, nipy spectral colormap
:width: 100%

100 traces using the `nipy_spectral` colormap.
```

## Temporal binning

### 2× binning

```{code-cell} ipython3
bin_factor = 2
dff_binned = lsp.bin1d(dff, bin_factor)
fps_binned = metadata["frame_rate"] / bin_factor
lsp.plot_traces(dff_binned, './plot_traces_dff_binned_2x.png', signal_units='dff', fps=fps_binned)
```

```{figure} ../_images/plot_traces_dff_binned_2x.png
:alt: Binned 2x
:width: 100%

Traces downsampled by factor 2. Scale bar adjusts accordingly.
```

### 4× binning

```{code-cell} ipython3
bin_factor = 4
dff_binned_4x = lsp.bin1d(dff, bin_factor)
fps_binned_4x = metadata["frame_rate"] / bin_factor
lsp.plot_traces(dff_binned_4x, './plot_traces_dff_binned_4x.png', signal_units='dff', fps=fps_binned_4x)
```

```{figure} ../_images/plot_traces_dff_binned_4x.png
:alt: Binned 4x
:width: 100%

Traces downsampled by factor 4. Horizontal scale bar still valid.
```
