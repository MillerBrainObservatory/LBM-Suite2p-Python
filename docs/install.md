(installation)=
# Installation Guide

LBM-Suite2p-Python has been developed to be a pure `pip` install.

This makes the choice of virtual-environment less relevant, you can use `venv`, `uv (recommended)`, `conda`, it does not matter.

`````` {tip}
:class: dropdown

While this pipeline is early in development, we recommend keeping a version of the codebase locally using `git`. 
This will allow you to quickly pull changes and incorperate them into your environment without waiting for a pypi release.

``` {code} bash
git clone https://github.com/MillerBrainObservatory/LBM-Suite2p-Python.git
cd LBM-Suite2p-Python
pip install -e "."
```
``````

::::{tab-set}

:::{tab-item} With uv

```bash

# create a new project folder
mkdir my_project
cd my_project

# create our environment
uv venv --python 3.12.9
uv pip install lbm_suite2p_python
```
:::

:::{tab-item} With Conda (Miniforge3)
```bash
conda create -n lsp -c conda-forge python=3.12.9
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
You will need [msvcc redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170#visual-studio-2015-2017-2019-and-2022)
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
