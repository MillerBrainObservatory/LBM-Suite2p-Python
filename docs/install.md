(installation)=
# Installation Guide

This package has been tested on Linux and Windows 10 with support for Python 3.10, 3.11 and 3.12.

**Note**: Install times will differ depending on the chosen Python version. For example, a `gui` install on Python 3.10 will build `imgui-bundle` from source, increasing the install time by several minutes. We recommend **Python 3.11** for the greatest compatibility and install speed.

## Quick Install

::::{tab-set}

:::{tab-item} With pip
```bash
pip install lbm_suite2p_python
```
:::

:::{tab-item} From source (recommended for development)
```bash
git clone https://github.com/millerbrainobservatory/lbm_suite2p_python.git
cd lbm_suite2p_python
pip install -e .
```
:::

:::{tab-item} With Conda (Miniforge3)
```bash
conda create -n lbm python=3.11 -c conda-forge
conda activate lbm
git clone https://github.com/millerbrainobservatory/lbm_suite2p_python.git
cd lbm_suite2p_python
pip install -e .
```
:::

::::

## GUI Dependencies

::::{tab-set}

:::{tab-item} Linux / macOS
```bash
sudo apt install libxcursor-dev libgl1-mesa-dev libglu1-mesa-dev freeglut3-dev
```
:::

:::{tab-item} Windows
Install system-level dependencies using your package manager (e.g. Chocolatey or MSYS2) if needed. GUI-based features (e.g., napari) may require proper OpenGL drivers and a working Python Qt backend.
:::

::::

## Troubleshooting

### Git LFS Error: `smudge filter lfs failed`

If you see:
```
error: external filter 'git-lfs filter-process' failed
fatal: docs/source/_static/guide_hello_world.png: smudge filter lfs failed
```

Disable smudge during sync:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync --all-extras --active
```

To debug:
```bash
git lfs logs last
```

This avoids downloading large binary files (e.g. images, model checkpoints) managed by Git LFS.
