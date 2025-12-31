<p align="center">
<img src="docs/_static/logo_suit2p.png" height="180" alt="LBM-Suite2p-Python logo">
</p>

<p align="center">
<a href="https://github.com/MillerBrainObservatory/LBM-Suite2p-Python/actions/workflows/test_python.yml"><img src="https://github.com/MillerBrainObservatory/LBM-Suite2p-Python/actions/workflows/test_python.yml/badge.svg" alt="Tests"></a>
<a href="https://badge.fury.io/py/lbm-suite2p-python"><img src="https://badge.fury.io/py/lbm-suite2p-python.svg" alt="PyPI version"></a>
<a href="https://millerbrainobservatory.github.io/LBM-Suite2p-Python/"><img src="https://img.shields.io/badge/docs-online-green" alt="Documentation"></a>
<a href="https://doi.org/10.1038/s41592-021-01239-8"><img src="https://zenodo.org/badge/DOI/10.1007/978-3-319-76207-4_15.svg" alt="DOI"></a>
</p>

<p align="center">
<a href="#installation"><b>Installation</b></a> ·
<a href="https://millerbrainobservatory.github.io/LBM-Suite2p-Python/"><b>Documentation</b></a> ·
<a href="https://millerbrainobservatory.github.io/LBM-Suite2p-Python/user_guide.html"><b>User Guide</b></a> ·
<a href="https://github.com/MillerBrainObservatory/LBM-Suite2p-Python/issues"><b>Issues</b></a>
</p>

A volumetric 2-photon calcium imaging processing pipeline for [Light Beads Microscopy](https://github.com/MillerBrainObservatory) (LBM) datasets, built on Suite2p.

- **Process volumetric calcium imaging data** - motion correction, cell detection, and signal extraction across z-planes
- **Automated quality diagnostics** - ROI quality metrics, ΔF/F traces, and correlation maps
- **Scalable architecture** - process single planes or entire volumes with consistent parameters

<p align="center">
<img src="docs/_images/volume/all_planes_masks.png" height="180" alt="All Planes Masks" /><img src="docs/_images/outputs/08_traces_dff.png" height="180" alt="ΔF/F Traces" /><img src="docs/_images/volume/roi_map_3d.png" height="180" alt="3D ROI Map" />
<br/>
<em>Planar Suite2p outputs combined into a 3D representation of neural activity</em>
</p>

> **Note:**
> `lbm_suite2p_python` is in **late-beta** stage of active development. File an [issue](https://github.com/MillerBrainObservatory/LBM-Suite2p-Python/issues) for bugs or feature requests.

## Installation

`lbm_suite2p_python` is available on [PyPI](https://pypi.org/project/lbm-suite2p-python/):

> We recommend using a virtual environment. For help setting up a virtual environment, see [the MBO guide on virtual environments](https://millerbrainobservatory.github.io/guides/venvs.html).

```bash
# create a new project folder
mkdir my_project && cd my_project

# create environment and install (uv recommended)
uv venv --python 3.12.9
uv pip install lbm_suite2p_python

# or with pip
pip install lbm_suite2p_python
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

```bash
git clone https://github.com/MillerBrainObservatory/LBM-Suite2p-Python.git
cd LBM-Suite2p-Python
uv pip install -e ".[dev]"
```

## Quick Start

```python
import lbm_suite2p_python as lsp

results = lsp.pipeline(
    input_data="D:/data/raw",   # path to file, directory, or list of files
    save_path=None,             # default: save next to input
    ops=None,                   # default: use MBO-optimized parameters
    planes=None,                # default: process all planes (1-indexed)
    roi_mode=None,              # default: stitch multi-ROI data
    keep_reg=True,              # default: keep data.bin (registered binary)
    keep_raw=False,             # default: delete data_raw.bin after processing
    force_reg=False,            # default: skip if already registered
    force_detect=False,         # default: skip if stat.npy exists
    dff_window_size=None,       # default: auto-calculate from tau and framerate
    dff_percentile=20,          # default: 20th percentile for baseline
    dff_smooth_window=None,     # default: auto-calculate from tau and framerate
)
```

> [User Guide](https://millerbrainobservatory.github.io/LBM-Suite2p-Python/user_guide.html) for full API reference and examples

## Output Gallery

### Planar Results

Each z-plane produces diagnostic images automatically saved during processing.

<p align="center">
<table>
<tr>
<td align="center">
<img src="docs/_images/outputs/01_correlation_segmentation.png" alt="Correlation Segmentation" width="280"/>
<br/><em>correlation image with ROI overlay</em>
</td>
<td align="center">
<img src="docs/_images/outputs/03_mean_segmentation.png" alt="Mean Segmentation" width="280"/>
<br/><em>mean image with ROI overlay</em>
</td>
</tr>
<tr>
<td align="center">
<img src="docs/_images/outputs/05_quality_diagnostics.png" alt="Quality Diagnostics" width="280"/>
<br/><em>ROI quality metrics</em>
</td>
<td align="center">
<img src="docs/_images/outputs/08_traces_dff.png" alt="ΔF/F Traces" width="280"/>
<br/><em>ΔF/F traces sorted by quality</em>
</td>
</tr>
</table>
</p>

### Volumetric Results

Volume-level visualizations combine data across all z-planes.

<p align="center">
<table>
<tr>
<td align="center">
<img src="docs/_images/volume/orthoslices.png" alt="Orthoslices" width="300"/>
<br/><em>XZ/YZ orthogonal projections</em>
</td>
<td align="center">
<img src="docs/_images/volume/rastermap.png" alt="Rastermap" width="400"/>
<br/><em>activity sorted by similarity (rastermap)</em>
</td>
</tr>
</table>
</p>

## GUI

A graphical interface is available via [mbo_utilities](https://millerbrainobservatory.github.io/mbo_utilities/index.html#gui):

```bash
pip install mbo_utilities
mbo                    # launch GUI
mbo /path/to/data      # open file directly
```

> **Note:** GUI functionality may lag behind the latest pipeline features.

## Troubleshooting

<details>
<summary><b>Git LFS Download Errors</b></summary>

If you see `smudge filter lfs failed` when installing from GitHub:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv pip install git+https://github.com/MillerBrainObservatory/LBM-Suite2p-Python.git
```

Or set it permanently:

```powershell
# Windows
[System.Environment]::SetEnvironmentVariable('GIT_LFS_SKIP_SMUDGE', '1', 'User')
```

```bash
# Linux/macOS
echo 'export GIT_LFS_SKIP_SMUDGE=1' >> ~/.bashrc
source ~/.bashrc
```

</details>

<details>
<summary><b>GUI Dependencies</b></summary>

**Linux / macOS:**

```bash
sudo apt install libxcursor-dev libgl1-mesa-dev libglu1-mesa-dev freeglut3-dev
```

**Windows:**
Install [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)

</details>

## Built With

- **[Suite2p](https://github.com/MouseLand/suite2p)** - Core registration and segmentation
- **[Cellpose](https://github.com/MouseLand/cellpose)** - Anatomical segmentation (optional)
- **[Rastermap](https://github.com/MouseLand/rastermap)** - Activity clustering (optional)
- **[mbo_utilities](https://github.com/MillerBrainObservatory/mbo_utilities)** - ScanImage I/O and metadata

## Issues & Support

- **Bug reports:** [GitHub Issues](https://github.com/MillerBrainObservatory/LBM-Suite2p-Python/issues)
- **Questions:** See [documentation](https://millerbrainobservatory.github.io/LBM-Suite2p-Python/) or [Suite2p docs](https://suite2p.readthedocs.io/)

## Contributing

Contributions are welcome! This project uses:

- **Ruff** for linting and formatting (line length: 88, numpy docstring style)
- **pytest** for testing
- **Sphinx** for documentation
