# LBM-Suite2p-Python

Light Beads Microscopy Pipeline based on the suite2p pipeline with Cellpose and Suite3D.

Very early in development.

## Installation

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
