# Data preparation

Suite2p accepts multiple input formats, but `.tiff` is the only format that has been tested thoroughly.

See suite2p's {doc}`inputs documentation <suite2p:inputs>` for more details on the use of other input formats such as `h5` and `.zarr`.

## mbo_utilities

To prepare your raw ScanImage TIFFs for Suite2p, use the functions from [`mbo_utilities`](https://millerbrainobservatory.github.io/mbo_utilities/). These utilities read, deinterleave, and assemble the raw data into per-plane TIFFs.

Refer to the full guide: {doc}`mbo_utilities:index`

```python
import mbo_utilities as mbo
scan = mbo.read_scan("/path/to/raw/*.tiff")
mbo.save_as(scan, "/path/to/assembled", ext=".tiff")
```

- Each Z-plane is saved as a separate TIFF.
- TIFFs are compatible with Suite2p.
- For details, see {doc}`mbo_utilities:assembly`

## Create `ops` from metadata

You can process a single imaging plane using the {func}`run_plane` function, which wraps `suite2p` registration, segmentation, and result plotting in a single call.

```{eval-rst}
.. autosummary:: lbm_suite2p_python.run_lsp
```

## Usage

```python
import mbo_utilities as mbo
import lbm_suite2p_python as lsp
import suite2p

# Find input TIFF files
input_files = mbo.get_files(assembled_path, str_contains='tif', max_depth=3)
metadata = mbo.get_metadata(input_files[0])

# Get and configure Suite2p ops
ops = suite2p.default_ops()
mbo_ops = mbo.params_from_metadata(metadata, ops)

# Run a single z-plane through Suite2p
output_ops = lsp.run_plane(
    ops=mbo_ops,
    input_file_path=input_files[0],
    save_path="output_directory"
)
```

---

(installation)=
# Installation Guide

This package has been tested on Linux and macOS using **Miniforge3** with **Python 3.11+**.

## Quick Install

### With pip
```bash
pip install lbm_suite2p_python
```

### From source (recommended for development)
```bash
git clone https://github.com/millerbrainobservatory/lbm_suite2p_python.git
cd lbm_suite2p_python
pip install -e .
```

### With Conda (using Miniforge3)
```bash
conda create -n lbm python=3.11 -c conda-forge
conda activate lbm
git clone https://github.com/millerbrainobservatory/lbm_suite2p_python.git
cd lbm_suite2p_python
pip install -e .
```

## GUI Dependencies

Some pipelines require GUI support (e.g. for interactive widgets or napari).

:::{sphinx-tabs}

=== "Linux / macOS"
```bash
sudo apt install libxcursor-dev libgl1-mesa-dev libglu1-mesa-dev freeglut3-dev
```

=== "Windows"
Install system-level dependencies using your package manager (e.g. Chocolatey or MSYS2) if needed. GUI-based features (e.g., napari) may require proper OpenGL drivers and a working Python Qt backend.

:::

## Troubleshooting

### Git LFS Error: `smudge filter lfs failed`
If you see an error like:
```
error: external filter 'git-lfs filter-process' failed
fatal: docs/source/_static/guide_hello_world.png: smudge filter lfs failed
```

Use the following command to skip LFS smudge during clone or sync:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync --all-extras --active
```

To debug:
```bash
git lfs logs last
```

This disables downloading large binary files such as images or model checkpoints managed by Git LFS.

