# LBM-Suite2p-Python Documentation 

## Documentation Contents

```{toctree}
---
maxdepth: 2
---
Notebooks <examples/index>
Functions <api>
Suite2p Publication <publication>
Manual Curation <manual_curation>
Glossary <glossary>

```

## Helpful Suite2p Issues

| Issue | Topic | Summary |
|-------|-------|---------|
| [#921](https://github.com/MouseLand/suite2p/issues/921) | Registration Artifacts | Try **smaller `block_size`** and **larger `spatial_taper`** to reduce wobbliness caused by black background inclusion in registration. |
| [#880](https://github.com/MouseLand/suite2p/issues/880) | Running on a Cluster | Discussion on **running Suite2p on a cluster**. |
| [#851](https://github.com/MouseLand/suite2p/issues/851) | ROI Overlap | ROIs will overlap if signals overlap, but if an ROI overlaps more than `ops['max_overlap']`, it gets discarded. Set `ops['max_overlap'] = 1` to keep all ROIs. `ops['allow_overlap'] = 0` (default) ignores overlapping pixels when computing signals. |
| [#837](https://github.com/MouseLand/suite2p/issues/837) | ROI Binning | `tau` determines data binning before ROI detection: **Bin size** = `ops['tau'] * ops['fs']`. More binning helps with noisy data. [ROI Detection Overview](https://youtu.be/NcC0YxQ9o3A). |
| [#787](https://github.com/MouseLand/suite2p/issues/787) | Thresholding | `ops['threshold_scaling']` affects dim regions. |
| [#690](https://github.com/MouseLand/suite2p/issues/690) | Channel Registration | Discussion on **registering one channel to another**. |
| [#627](https://github.com/MouseLand/suite2p/issues/627) | Fluorescence & Neuropil Signals | Explanation of `F` (Fluorescence) and `Fneu` (Neuropil Fluorescence) in Suite2p output. |
| [#758](https://github.com/MouseLand/suite2p/issues/758#issuecomment-956588935) | Running on a Cluster | More discussion on **looping over databases (`dbs`)** when running Suite2p on a cluster. |

## Links / Resources

[suite2p github](https://github.com/MouseLand/suite2p/tree/main)
[suite2p docs](https://suite2p.readthedocs.io/en/latest/index.html)
[suite2p paper](https://www.biorxiv.org/content/10.1101/061507v2)
[marius deconv paper](https://www.jneurosci.org/content/38/37/7976)

----------------

## Pipeline Overview

WIP
