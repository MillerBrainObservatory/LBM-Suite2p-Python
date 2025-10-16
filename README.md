# Light Beads Microscopy (LBM) Pipeline: Suite2p

[![PyPI - Version](https://img.shields.io/pypi/v/lbm-suite2p-python)](https://pypi.org/project/lbm-suite2p-python/)

[![Documentation](https://img.shields.io/badge/Documentation-blue?style=for-the-badge&logo=readthedocs&logoColor=white)](https://millerbrainobservatory.github.io/LBM-Suite2p-Python/index.html)

This package is still in a *late-beta* stage of development.

A pipeline for processing volumetric 2-photon Light Beads Microscopy (LBM) datasets.

This pipeline uses the following open-source software:

- [suite2p](https://github.com/MouseLand/suite2p)
- [cellpose](https://github.com/MouseLand/cellpose)
- [rastermap](https://github.com/MouseLand/rastermap)
- [mbo_utilities](https://github.com/MillerBrainObservatory/mbo_utilities)
- [scanreader](https://github.com/atlab/scanreader)


[![LBM](https://zenodo.org/badge/DOI/10.1007/978-3-319-76207-4_15.svg)](https://doi.org/10.1038/s41592-021-01239-8)

---

## Installation

This pipeline is installable with `pip`:

```bash
pip install lbm_suite2p_python
# with uv: uv pip install lbm_suite2p_python
```

We highly encourage the use of a virtual environment. If you are unfamiliar with virtual environments, see our documentation [here](https://millerbrainobservatory.github.io/mbo_utilities/venvs.html).

You may also use git to clone and install locally for updates not yet released to pypi:

```bash
git clone https://github.com/MillerBrainObservatory/LBM-Suite2p-Python.git
cd LBM-Suite2p-Python

# make sure your virtual environment is active
pip install "." 
```

## Features

### 2.0.0

- Process ScanImage multi-ROI as separate datasets
- Post-processing cell filters for area, exceptional events and eccentricity

### 1.0.0

- Suite2p planar segmentation
- DF/F, baseline calculation and documentation
- Aggregate planar outputs into volumetric dataset

## Issues

Widgets may throw "Invalid Rect" errors. This can be safely ignored until it is [resolved](https://github.com/pygfx/wgpu-py/issues/716#issuecomment-2880853089).

---

## Acknowledgements

This pipeline is mostly a volumetric wrapper around [suite2p](https://github.com/MouseLand/suite2p), [cellpose](https://github.com/MouseLand/cellpose) and [Suite3D](https://github.com/alihaydaroglu/suite3d). We thank the contributors to those projects.

Thank you to the developers of [scanreader](https://github.com/atlab/scanreader), which provides a clean interface to ScanImage metadata using only tifffile and numpy.
