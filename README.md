# LBM-Suite2p-Python

> **Status:** Late-beta stage of development

[![Documentation](https://img.shields.io/badge/Documentation-blue?style=for-the-badge&logo=readthedocs&logoColor=white)](https://millerbrainobservatory.github.io/LBM-Suite2p-Python/index.html)

[![PyPI - Version](https://img.shields.io/pypi/v/lbm-suite2p-python)](https://pypi.org/project/lbm-suite2p-python/)
[![DOI](https://zenodo.org/badge/DOI/10.1007/978-3-319-76207-4_15.svg)](https://doi.org/10.1038/s41592-021-01239-8)

A volumetric 2-photon calcium imaging processing pipeline for Light Beads Microscopy (LBM) datasets, built on Suite2p.

A GUI is available via [mbo_utilities](https://millerbrainobservatory.github.io/mbo_utilities/index.html#gui).

GUI functionality will lag behind this pipeline.

## Quick Start

See the [installation documentation](https://millerbrainobservatory.github.io/LBM-Suite2p-Python/install.html) for GUI dependencies and troubleshooting.

```bash
uv pip install lbm_suite2p_python
```

### Basic Usage

```python
import lbm_suite2p_python as lsp

ops = {"two_step_registration": 1}
files = [
    r"D://demo//plane05_stitched.zarr",
    r"D://demo//plane06_stitched.zarr",
]

# Process entire volume
output_ops = lsp.run_volume(
    input_files=files,
    save_path=None,     # save next to tiffs
    ops=ops,
    keep_reg=True,      # Keep registered binaries
    force_reg=False,    # Skip if already registered
    force_detect=False  # Skip if stat.npy exists
)
```

**Process a single plane:**
```python
ops_file = lsp.run_plane(
    input_path=files[0],
    save_path=None,
    ops=ops,
    keep_raw=False,  # Delete data_raw.bin after processing
    keep_reg=True    # Keep data.bin (registered binary)
)
```

## Documentation

- **[Installation Guide](https://millerbrainobservatory.github.io/LBM-Suite2p-Python/install.html)**
- **[User Guide](https://millerbrainobservatory.github.io/LBM-Suite2p-Python/user_guide.html)** - Complete usage examples
- **[API Reference](https://millerbrainobservatory.github.io/LBM-Suite2p-Python/api.html)**

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
