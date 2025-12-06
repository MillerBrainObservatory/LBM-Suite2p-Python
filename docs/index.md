---
bibliography: refs.bib
---

# LBM-Suite2p-Python Documentation

A volumetric 2-photon calcium imaging processing pipeline for Light Beads Microscopy (LBM) datasets.

This package processes multi-plane calcium imaging data through a three-step workflow:

1. **Convert** raw TIFF files to binary format for Suite2p
2. **Process** each z-plane independently (registration → segmentation → extraction)
3. **Aggregate** planar results into volumetric outputs with visualization

**Key capabilities:**
- Planar-by-planar Suite2p processing optimized for volumetric datasets
- Automatic detection and merging of ScanImage multi-ROI acquisitions
- Robust binary validation to avoid redundant reprocessing
- Post-processing filters for cell quality (area, eccentricity, event exceptionality)
- ΔF/F calculation with multiple baseline methods
- Volumetric statistics and Rastermap clustering

## Navigation

See the {doc}`user guide <user_guide>` for complete examples and parameter tuning.

```{toctree}
---
maxdepth: 2
---
user_guide
postprocessing
API Reference <api>
glossary
```

```{toctree}
---
caption: Additional Resources
hidden: true
maxdepth: 1
---
discussions/index
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

## Citation

If you use this pipeline, please cite Suite2p and LBM:

```{bibliography}
:filter: docname in docnames
:style: plain
```


---

## Acknowledgements

This pipeline is built on excellent open-source tools:

- **[Suite2p](https://github.com/MouseLand/suite2p)** - Registration and segmentation (Pachitariu, Stringer, et al.)
- **[Cellpose](https://github.com/MouseLand/cellpose)** - Anatomical segmentation (Stringer, Pachitariu, et al.)
- **[Rastermap](https://github.com/MouseLand/rastermap)** - Activity clustering (Stringer, Pachitariu)
- **[scanreader](https://github.com/atlab/scanreader)** - ScanImage metadata parsing (atlab)
- **[Suite3D](https://github.com/alihaydaroglu/suite3d)** - Volumetric processing inspiration (Haydaroglu)

Special thanks to the Miller Brain Observatory team and all contributors for testing and feedback.
