# LBM-Suite2p-Python

[![PyPI - Version](https://img.shields.io/pypi/v/lbm-suite2p-python)](https://pypi.org/project/lbm-suite2p-python/)
[![Documentation](https://img.shields.io/badge/Documentation-blue?style=for-the-badge&logo=readthedocs&logoColor=white)](https://millerbrainobservatory.github.io/LBM-Suite2p-Python/index.html)
[![DOI](https://zenodo.org/badge/DOI/10.1007/978-3-319-76207-4_15.svg)](https://doi.org/10.1038/s41592-021-01239-8)

A volumetric 2-photon calcium imaging processing pipeline for Light Beads Microscopy (LBM) datasets, built on Suite2p.

A GUI is available via [mbo_utilities]() Functionality is available via GUI
around [mbo_utilities]() which handles .tiff, .zarr, .bin, and .h5 file I/O

> **Status:** Late-beta stage of development

## Overview

LBM-Suite2p-Python processes multi-plane calcium imaging data by:
1. Converting raw TIFF files to binary format for Suite2p
2. Running registration, segmentation, and extraction on individual z-planes
3. Aggregating planar results into volumetric outputs
4. Providing visualization and analysis tools

**Key Features:**
- Planar-by-planar Suite2p processing optimized for volumetric data
- ScanImage multi-ROI support with automatic detection and merging
- Robust binary validation to avoid redundant reprocessing
- Post-processing filters (area, eccentricity, exceptional events)
- ΔF/F calculation with rolling percentile baseline
- Volumetric visualization and Rastermap clustering

## Quick Start

### Installation

```bash
# Recommended: using uv
uv pip install lbm_suite2p_python

# Or with pip
pip install lbm_suite2p_python

# Development install from source
git clone https://github.com/MillerBrainObservatory/LBM-Suite2p-Python.git
cd LBM-Suite2p-Python
pip install -e "."
```

**Python requirement:** 3.12.7 to <3.12.10

See [Installation Guide](https://millerbrainobservatory.github.io/LBM-Suite2p-Python/install.html) for detailed instructions.

### Basic Usage

```python
import mbo_utilities as mbo
import lbm_suite2p_python as lsp

# Get list of z-plane TIFFs (planar timeseries format: T, Y, X)
files = mbo.get_files(data_dir, "tiff", max_depth=3)

# Get metadata and create ops
metadata = mbo.get_metadata(files[0])
ops = mbo.params_from_metadata(metadata)

# Process entire volume
output_ops = lsp.run_volume(
    input_files=files,
    save_path=save_dir,
    ops=ops,
    keep_reg=True,      # Keep registered binaries
    force_reg=False,    # Skip if already registered
    force_detect=False  # Skip if stat.npy exists
)
```

**Process a single plane:**
```python
ops_file = lsp.run_plane(
    input_path=plane_file,
    save_path=plane_dir,
    ops=ops,
    keep_raw=False,  # Delete data_raw.bin after processing
    keep_reg=True    # Keep data.bin (registered binary)
)
```

See [User Guide](https://millerbrainobservatory.github.io/LBM-Suite2p-Python/user_guide.html) for complete examples and parameter tuning.

## Output Structure

**Per-plane outputs:**
```
{save_path}/{plane_tag}/
├── ops.npy              # Suite2p operations and results
├── stat.npy             # ROI statistics (pixel coordinates, weights)
├── F.npy                # Fluorescence traces (n_rois, n_frames)
├── Fneu.npy             # Neuropil fluorescence
├── spks.npy             # Deconvolved spike traces
├── iscell.npy           # Classification (is_cell, probability)
├── data.bin             # Registered binary (if keep_reg=True)
├── pc_metrics/          # Principal component analysis
└── *.png                # Visualization plots
```

**Volumetric outputs (at save_path root):**
```
{save_path}/
├── volume_stats.npy     # Per-plane statistics
├── rastermap.png        # Clustered activity (if rastermap installed)
└── mean_volume_signal.png  # Signal strength across planes
```

## Key Parameters

Critical Suite2p parameters (see [Parameter Guide](https://millerbrainobservatory.github.io/LBM-Suite2p-Python/user_guide.html#parameters)):

| Parameter | Effect | Default | Recommended |
|-----------|--------|---------|-------------|
| `diameter` | Expected cell size (pixels) | 0 (auto) | 6-15 for neurons |
| `threshold_scaling` | Detection sensitivity | 1.0 | 0.8-1.5 (higher = fewer ROIs) |
| `tau` | Ca²⁺ decay constant (s) | 1.0 | 0.7-2.0 (GCaMP6f: ~1.0) |
| `fs` | Frame rate (Hz) | 10.0 | Must match acquisition |
| `nonrigid` | Block-wise registration | True | True for drifting FOV |
| `do_registration` | Registration mode | 1 (run if needed) | 0=skip, 1=auto, 2=force |

**LBM-specific defaults:**
- `nplanes=1` - Process each z-plane independently
- `do_bidiphase=0` - Assumes pre-corrected data
- `do_regmetrics=True` - Always compute registration quality

## Documentation

- **[Full Documentation](https://millerbrainobservatory.github.io/LBM-Suite2p-Python/)**
- **[Installation Guide](https://millerbrainobservatory.github.io/LBM-Suite2p-Python/install.html)**
- **[User Guide](https://millerbrainobservatory.github.io/LBM-Suite2p-Python/user_guide.html)** - Complete usage examples
- **[API Reference](https://millerbrainobservatory.github.io/LBM-Suite2p-Python/api.html)**
- **[Glossary](https://millerbrainobservatory.github.io/LBM-Suite2p-Python/glossary.html)**
- **[CLAUDE.md](./CLAUDE.md)** - Technical deep-dive for AI assistants

## Built With

This pipeline integrates several open-source tools:

- **[Suite2p](https://github.com/MouseLand/suite2p)** - Core registration and segmentation
- **[Cellpose](https://github.com/MouseLand/cellpose)** - Anatomical segmentation (optional)
- **[Rastermap](https://github.com/MouseLand/rastermap)** - Activity clustering (optional)
- **[mbo_utilities](https://github.com/MillerBrainObservatory/mbo_utilities)** - ScanImage I/O and metadata
- **[scanreader](https://github.com/atlab/scanreader)** - ScanImage metadata parsing

## Citation

If you use this pipeline, please cite:

```bibtex
@article{pachitariu2017suite2p,
  title={Suite2p: beyond 10,000 neurons with standard two-photon microscopy},
  author={Pachitariu, Marius and Stringer, Carsen and Dipoppa, Mario and Schr{\"o}der, Sylvia and Rossi, L Federico and Dalgleish, Henry and Carandini, Matteo and Harris, Kenneth D},
  journal={bioRxiv},
  pages={061507},
  year={2017},
  publisher={Cold Spring Harbor Laboratory}
}

@article{duemani2021light,
  title={Light-beads microscopy for volumetric imaging},
  author={Duemani Reddy, Gaddum and Kelleher, Kripa and Fink, Reid and Saggau, Peter},
  journal={Nature methods},
  volume={18},
  number={9},
  pages={1082--1090},
  year={2021},
  publisher={Nature Publishing Group}
}
```

## Issues & Support

- **Bug reports:** [GitHub Issues](https://github.com/MillerBrainObservatory/LBM-Suite2p-Python/issues)
- **Questions:** See [Suite2p documentation](https://suite2p.readthedocs.io/) for Suite2p-specific questions
- **Known issues:** Widgets may throw "Invalid Rect" errors ([upstream issue](https://github.com/pygfx/wgpu-py/issues/716#issuecomment-2880853089))

## Contributing

Contributions are welcome! This project follows Suite2p's conventions and uses:
- **Ruff** for linting and formatting (line length: 88, numpy docstring style)
- **pytest** for testing
- **Sphinx** for documentation

## Acknowledgements

This pipeline is primarily a volumetric wrapper around the excellent work by:
- **Suite2p team** (Marius Pachitariu, Carsen Stringer, et al.)
- **Cellpose team** (Carsen Stringer, Marius Pachitariu, et al.)
- **Suite3D** (Ali Haydaroglu)
- **scanreader team** (atlab)

Special thanks to the Miller Brain Observatory team for testing and feedback.

## License

See LICENSE file for details.
