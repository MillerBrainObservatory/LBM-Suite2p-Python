# LBM-Suite2p-Python

> **Status:** Late-beta stage of development

[![Documentation](https://img.shields.io/badge/Documentation-blue?style=for-the-badge&logo=readthedocs&logoColor=white)](https://millerbrainobservatory.github.io/LBM-Suite2p-Python/index.html)

[![PyPI - Version](https://img.shields.io/pypi/v/lbm-suite2p-python)](https://pypi.org/project/lbm-suite2p-python/)
[![DOI](https://zenodo.org/badge/DOI/10.1007/978-3-319-76207-4_15.svg)](https://doi.org/10.1038/s41592-021-01239-8)

A volumetric 2-photon calcium imaging processing pipeline for Light Beads Microscopy (LBM) datasets, built on Suite2p.

A GUI is available via [mbo_utilities](https://millerbrainobservatory.github.io/mbo_utilities/index.html#gui) (GUI functionality will lag behind this pipeline).

## Installation

LBM-Suite2p-Python is a pure `pip` install. You can use `venv`, `uv` (recommended), or `conda`. Just remove the `uv` prefix.

```bash
# create a new project folder
mkdir my_project
cd my_project

# (uv only) create environment and install
uv venv --python 3.12.9
uv pip install lbm_suite2p_python
```

### Optional Dependencies

```bash
# With rastermap for activity clustering visualization
uv pip install "lbm_suite2p_python[rastermap]"

# With cellpose for anatomical cell detection (includes PyTorch)
uv pip install "lbm_suite2p_python[cellpose]"

# All optional dependencies
uv pip install "lbm_suite2p_python[all]"
```

### Development Installation

While this pipeline is in active development, you can keep a local copy to quickly pull changes:

```bash
git clone https://github.com/MillerBrainObservatory/LBM-Suite2p-Python.git
cd LBM-Suite2p-Python
uv pip install .
```

### GUI Dependencies

**Linux / macOS:**

```bash
sudo apt install libxcursor-dev libgl1-mesa-dev libglu1-mesa-dev freeglut3-dev
```

**Windows:**
Install [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)

### Troubleshooting

When installing from github, you may get:

**Git LFS Error:** If you see `smudge filter lfs failed`:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync --all-extras --active
```

## Quick Start

```python
import lbm_suite2p_python as lsp

results = lsp.pipeline(
    input_data="D:/data/raw",   # path to file, directory, or list of files
    save_path=None,             # default: save next to input
    ops=None,                   # default: use MBO-optimized parameters
    planes=None,                # default: process all planes
    roi=None,                   # default: stitch multi-ROI data
    keep_reg=True,              # default: keep data.bin (registered binary)
    keep_raw=False,             # default: delete data_raw.bin after processing
    force_reg=False,            # default: skip if already registered
    force_detect=False,         # default: skip if stat.npy exists
    dff_window_size=None,       # default: auto-calculate from tau and framerate
    dff_percentile=20,          # default: 20th percentile for baseline
    dff_smooth_window=None,     # default: auto-calculate from tau and framerate
)
```

## Output Gallery

### Planar Results

Each z-plane produces diagnostic images automatically saved during processing.

<table>
<tr>
<td align="center">
<img src="docs/_images/outputs/01_correlation_segmentation.png" alt="Correlation Segmentation" width="280"/>
<br><sub>correlation image with ROI overlay</sub>
</td>
<td align="center">
<img src="docs/_images/outputs/03_mean_segmentation.png" alt="Mean Segmentation" width="280"/>
<br><sub>mean image with ROI overlay</sub>
</td>
</tr>
<tr>
<td align="center">
<img src="docs/_images/outputs/05_quality_diagnostics.png" alt="Quality Diagnostics" width="280"/>
<br><sub>ROI quality metrics</sub>
</td>
<td align="center">
<img src="docs/_images/outputs/08_traces_dff.png" alt="ΔF/F Traces" width="280"/>
<br><sub>ΔF/F traces sorted by quality</sub>
</td>
</tr>
</table>

### Volumetric Results

Volume-level visualizations combine data across all z-planes.

<table>
<tr>
<td align="center">
<img src="docs/_images/volume/all_planes_masks.png" alt="All Planes Masks" width="400"/>
<br><sub>ROI masks across all z-planes</sub>
</td>
</tr>
<tr>
<td align="center">
<img src="docs/_images/volume/roi_map_3d.png" alt="3D ROI Map" width="300"/>
<br><sub>3D ROI centroids colored by SNR</sub>
</td>
<td align="center">
<img src="docs/_images/volume/rastermap.png" alt="Rastermap" width="400"/>
<br><sub>activity sorted by similarity (rastermap)</sub>
</td>
</tr>
</table>

## Built With

This pipeline integrates several open-source tools:

- **[Suite2p](https://github.com/MouseLand/suite2p)** - Core registration and segmentation
- **[Cellpose](https://github.com/MouseLand/cellpose)** - Anatomical segmentation (optional)
- **[Rastermap](https://github.com/MouseLand/rastermap)** - Activity clustering (optional)
- **[mbo_utilities](https://github.com/MillerBrainObservatory/mbo_utilities)** - ScanImage I/O and metadata
- **[scanreader](https://github.com/atlab/scanreader)** - ScanImage metadata parsing

## Issues & Support

- **Bug reports:** [GitHub Issues](https://github.com/MillerBrainObservatory/LBM-Suite2p-Python/issues)
- **Questions:** See [Suite2p documentation](https://suite2p.readthedocs.io/) for Suite2p-specific questions
- **Known issues:** Widgets may throw "Invalid Rect" errors ([upstream issue](https://github.com/pygfx/wgpu-py/issues/716#issuecomment-2880853089))

## Contributing

Contributions are welcome! This project follows Suite2p's conventions and uses:

- **Ruff** for linting and formatting (line length: 88, numpy docstring style)
- **pytest** for testing
- **Sphinx** for documentation
