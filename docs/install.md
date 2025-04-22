(installation)=
# Installation Guide

This package has been tested on Linux and Windows 10 with support for Python 3.10, 3.11 and 3.12.

**Note**: Install times will differ depending on the chosen Python version. For example, a `gui` install on python 3.10 will build `imgui-bundle` from source, increasing the install time by several minutes.We recommend **Python 3.11** for the greatest compatibility and install speed.

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

