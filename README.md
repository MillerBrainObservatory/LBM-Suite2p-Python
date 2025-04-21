## Light Beads Microscopy (LBM) Pipeline: Suite2p

!Note: LBM-Suite2p-Python is still in a *beta* stage of development.

[![Documentation](https://img.shields.io/badge/Documentation-black?style=for-the-badge&logo=readthedocs&logoColor=white)](https://millerbrainobservatory.github.io/LBM-Suite-2p-Python/)

A pipeline for processing 2-photon Light Beads Microscopy (LBM) datasets. 

This pipeline uses the following software:

- [suite2p](https://github.com/MouseLand/suite2p)
- [cellpose](https://github.com/MouseLand/cellpose)
- [rastermap](https://github.com/MouseLand/rastermap)
- [mbo_utilities](https://github.com/MillerBrainObservatory/mbo_utilities)

The LBM technlology described in the Nature Methods publication:

[![DOI](https://zenodo.org/badge/DOI/10.1007/978-3-319-76207-4_15.svg)](https://doi.org/10.1038/s41592-021-01239-8)

# LBM-Suite2p-Python

Light Beads Microscopy Pipeline based on the suite2p pipeline with Cellpose and Suite3D.

## Installation

1. `pip`

``` bash
conda create -n lsp python=3.10
conda activate lsp
pip install https://github.com/MillerBrainObservatory/LBM_Suite2p_python.git
```

## Usage

``` bash
lsp --path/to/file.tiff # run a single z-plane
lsp --path/do/dir --max-depth 2 # run all z-planes up to this depth
```
