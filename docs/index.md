---
bibliography: refs.bib
---

# LBM-Suite2p-Python Documentation

Volumetric 2-photon calcium imaging processing pipeline using [Suite2p](https://github.com/MouseLand/suite2p) and [Cellpose](https://cellpose.readthedocs.io/en/latest/).

Primarily intended for Light Beads Microscopy (LBM) datasets, but viable with any planar or volumetric data.

**Key capabilities:**

- Volumetric and/or Planar registration, detection, segmentation and deconvolution
- Automatic detection and merging of ScanImage multi-ROI acquisitions
- Post-processing filters for cell quality (area, eccentricity, event exceptionality)
- ΔF/F calculation with multiple baseline methods
- Volumetric statistics and Rastermap clustering

## Navigation

See the {doc}`user guide <user_guide>` for complete examples and parameter tuning.

```{toctree}
---
maxdepth: 1
---
user_guide
postprocessing
processing_flow
API Reference <api>
glossary
```

```{toctree}
---
caption: Additional Resources
hidden: true
maxdepth: 1
---
image_gallery
pipeline_comparison
```

## Quick Example

```python
import lbm_suite2p_python as lsp

# Process entire volume with the unified pipeline
results = lsp.pipeline(
    input_data="D:/data/raw",   # path to file, directory, or list of files
    save_path=None,             # default: save next to input
    ops=None,                   # default: use MBO-optimized parameters
    planes=None,                # default: process all planes
)
```

---

## Helpful Suite2p Resources

| Topic | Resource | Description |
|-------|----------|-------------|
| **Parameters** | [Suite2p Settings](https://suite2p.readthedocs.io/en/latest/settings.html) | Complete parameter reference |
| **Detection** | [ROI Detection](https://youtu.be/NcC0YxQ9o3A) | Video overview of detection algorithm |
| **Registration** | [Issue #921](https://github.com/MouseLand/suite2p/issues/921) | Troubleshooting registration artifacts |
| **Critical Parameters** | [Issue #129](https://github.com/MouseLand/suite2p/issues/129) | Discussion of tau, threshold_scaling, diameter |
| **ROI Overlap** | [Issue #851](https://github.com/MouseLand/suite2p/issues/851) | Understanding max_overlap and allow_overlap |
| **Fluorescence Signals** | [Issue #627](https://github.com/MouseLand/suite2p/issues/627) | Explanation of F and Fneu outputs |

---

## External Links

- [Suite2p Documentation](https://suite2p.readthedocs.io/) - Core pipeline documentation
- [Cellpose Documentation](https://cellpose.readthedocs.io/) - Anatomical segmentation
- [mbo_utilities Documentation](https://millerbrainobservatory.github.io/mbo_utilities/) - ScanImage I/O and assembly
- [GitHub Repository](https://github.com/MillerBrainObservatory/LBM-Suite2p-Python) - Source code and issue tracker

---

## Acknowledgements

This pipeline is built on excellent open-source tools:

- **[Suite2p](https://github.com/MouseLand/suite2p)** - Registration and segmentation (Pachitariu, Stringer, et al.)
- **[Cellpose](https://github.com/MouseLand/cellpose)** - Anatomical segmentation (Stringer, Pachitariu, et al.)
- **[Rastermap](https://github.com/MouseLand/rastermap)** - Activity clustering (Stringer, Pachitariu)
- **[Suite3D](https://github.com/alihaydaroglu/suite3d)** - Volumetric processing inspiration (Haydaroglu)

Special thanks to the Miller Brain Observatory team and all contributors for testing and feedback.
