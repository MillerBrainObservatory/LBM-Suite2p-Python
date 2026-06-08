---
bibliography: refs.bib
---

# LBM-Suite2p-Python Documentation

Volumetric 2-photon calcium imaging processing pipeline using [Suite2p](https://github.com/MouseLand/suite2p) and [Cellpose](https://cellpose.readthedocs.io/en/latest/).

A high-level wrapper around suite2p primarily intended for Light Beads Microscopy (LBM) datasets, but compatible with any data supported by mbo_utilities [imread](https://millerbrainobservatory.github.io/mbo_utilities/array_types.html)

**Features:**

- Registration, cell-detection, segmentation and deconvolution
- Uniform cell curation filters (area, eccentricity, event exceptionality)
- ΔF/F calculation with multiple baseline methods
- Volumetric statistics, visualizations and clustering
- Extensive documentation

See the {doc}`user guide <user_guide>` for complete examples and parameter tuning.

```{toctree}
---
maxdepth: 1
---
quickstart
user_guide
projections
postprocessing
processing_flow
classifiers
metadata_reference
pipeline_comparison
API <api>
glossary
image_gallery
```

## Additional Resources

### Helpful Suite2p Discussions

| Topic | Resource | Description |
|-------|----------|-------------|
| **Parameters** | [Suite2p Settings](https://suite2p.readthedocs.io/en/latest/settings.html) | Complete parameter reference |
| **Detection** | [ROI Detection](https://youtu.be/NcC0YxQ9o3A) | Video overview of detection algorithm |
| **Registration** | [Issue #921](https://github.com/MouseLand/suite2p/issues/921) | Troubleshooting registration artifacts |
| **Critical Parameters** | [Issue #129](https://github.com/MouseLand/suite2p/issues/129) | Discussion of tau, threshold_scaling, diameter |
| **ROI Overlap** | [Issue #851](https://github.com/MouseLand/suite2p/issues/851) | Understanding max_overlap and allow_overlap |
| **Fluorescence Signals** | [Issue #627](https://github.com/MouseLand/suite2p/issues/627) | Explanation of F and Fneu outputs |

### Referenced Documentation

- [Suite2p Documentation](https://suite2p.readthedocs.io/) - Core pipeline documentation
- [Cellpose Documentation](https://cellpose.readthedocs.io/) - Anatomical segmentation

## Acknowledgements

This pipeline is built on excellent open-source tools:

- **[Suite2p](https://github.com/MouseLand/suite2p)** - Registration and segmentation (Pachitariu, Stringer, et al.)
- **[Cellpose](https://github.com/MouseLand/cellpose)** - Anatomical segmentation (Stringer, Pachitariu, et al.)
- **[Rastermap](https://github.com/MouseLand/rastermap)** - Activity clustering (Stringer, Pachitariu)
- **[Suite3D](https://github.com/alihaydaroglu/suite3d)** - Volumetric processing inspiration (Haydaroglu)
