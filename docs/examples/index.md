(notebooks)=

# Example Notebooks

Interactive Jupyter notebooks demonstrating LBM-Suite2p-Python workflows.

```{note}
The canonical examples are maintained in the [`demos/notebooks/`](https://github.com/MillerBrainObservatory/LBM-Suite2p-Python/tree/master/demos/notebooks) directory.
These are kept up-to-date with the latest API and recommended workflows.
```

## Quick Links

| Notebook | Description |
|----------|-------------|
| [Quickstart](https://github.com/MillerBrainObservatory/LBM-Suite2p-Python/blob/master/demos/notebooks/quickstart.ipynb) | Basic workflow: `run_plane()` and `run_volume()` |
| [Anatomical Grid Search](https://github.com/MillerBrainObservatory/LBM-Suite2p-Python/blob/master/demos/notebooks/anatomical_grid_search.ipynb) | Parameter optimization with Cellpose |
| [Tau & Spike Inference](https://github.com/MillerBrainObservatory/LBM-Suite2p-Python/blob/master/demos/notebooks/tau_spike_inference_analysis.ipynb) | Analyzing deconvolution parameters |

## Demo Scripts

For command-line workflows, see [`demos/scripts/`](https://github.com/MillerBrainObservatory/LBM-Suite2p-Python/tree/master/demos/scripts):

- **`run_plane.py`** - Single plane processing
- **`run_volume.py`** - Full volume processing with Cellpose
- **`reg_chan2.py`** - Two-channel registration

## Running Locally

Clone the repository and navigate to the demos folder:

```bash
git clone https://github.com/MillerBrainObservatory/LBM-Suite2p-Python.git
cd LBM-Suite2p-Python/demos/notebooks
jupyter lab quickstart.ipynb
```

## API Reference

See the {doc}`User Guide <../user_guide>` for detailed parameter explanations and the {doc}`API Reference <../api>` for function signatures.
