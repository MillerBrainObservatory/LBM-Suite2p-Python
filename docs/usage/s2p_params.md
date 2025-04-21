# Cell Detection Parameters

It would be very convenient to be able to say "increase parameter `p` to detect more cells".

The {doc}`suite2p documentation <suite2p:celldetection>` only make that declaration for `threshold_scaling`,
stating that decreasing this parameter will yield more cells.

However, we have found this not always to be the case. 

``` {admonition} Example dataset
:name: dataset_overview
:class: dropdown

*This dataset was collected at the Miller Brain Observatory by Will Snyder of the Charles Gilbert Lab.*

| Attribute              | Value                      | Description                                                |
|------------------------|----------------------------|------------------------------------------------------------|
| `fov`                 | [448, 896] μm              | Total field of view in microns (X × Y).                    |
| `fov_px`              | [448, 896] px              | Field of view in pixels (X × Y).                           |
| `pixel_resolution`    | [2.0, 2.0] μm/px           | Microns per pixel in each spatial dimension.              |
| `frame_rate`          | 17.07 Hz                   | Imaging frame rate.                                        |
| `num_frames`          | 9                          | Number of timepoints.                                      |
| `num_planes`          | 14                         | Number of Z-planes.                                        |
| `num_rois`            | 2                          | Number of simultaneously acquired ROIs.                    |
| `ndim`                | 4                          | Dataset has 4 dimensions: (Z, T, Y, X).                    |
| `dtype`               | `uint16`                   | Pixel data type.                                           |
| `sample_format`       | `int16`                    | Raw TIFF sample format.                                    |
| `raw_width`           | 224 px                     | Width of each ROI in the raw TIFF (slow axis).            |
| `raw_height`          | 912 px                     | Total TIFF height (stacked ROIs vertically).              |
| `roi_width_px`        | 224 px                     | Width of individual ROI.                                   |
| `roi_height_px`       | 448 px                     | Height of individual ROI.                                  |
| `tiff_pages`          | 126                        | Total number of frames stored in TIFF file.                |
| `objective_resolution`| 61 μm/deg                  | Optical resolution of the objective.                       |
| `z_step_pollen`       | *None*                     | Z-step metadata not available.                             |
| `size`                | 17.16 GB                   | Approximate total dataset size in bytes.                   |

> This scan is suitable for validating deinterleaving, z-plane reconstruction, and basic visualization. The short duration and small number of timepoints make it ideal for debugging or prototyping processing pipelines.

```

## Parameters

The parameters covered here are only some of the more influential parameters.

- `threshold_scaling`: This is a scale factor which determines **how bright a signal needs to be to {term}`seed` an ROI**.

- `sparse_mode`: Speeds up ROI detection by iteratively seeding and growing regions distinct regions on a binned movie, helpful for separating dendritic segmentation (set to False) and somatic segmentation (set to True).

- `anatomical_only`: Enables anatomical segmentation using [Cellpose](https://www.cellpose.org/), bypassing Suite2p's functional ROI detection. The value determines which image is used:

  | Value | Image Used         | Description                                                                 |
  |-------|--------------------|-----------------------------------------------------------------------------|
  | `1`   | `max_proj / mean_img` | Ratio of the max projection to the mean image; highlights active areas relative to baseline. |
  | `2`   | `mean_img`         | The average image over all frames; provides baseline structural contrast. |
  | `3`   | `meanImgE`         | An enhanced version of the mean image using Suite2p’s sharpening/filtering; highlights edges and features. |
  | `0` or False | Disabled   | Anatomical detection is off; functional detection (correlation-based) is used instead. |

- `diameter`: Approximate diameter of ROIs, in pixels. Used to set the scale for filters and Cellpose segmentation.

- `flow_threshold`: Minimum Cellpose flow error to consider a region valid. Lower values include more ROIs.

- `cellprob_threshold`: Probability threshold from Cellpose’s output to determine whether a pixel belongs to a cell. More negative values include more pixels.

- `spatial_hp_cp`: Amount of high-pass filtering applied to the image before Cellpose segmentation. A float between `0` and `1`.

- `tau`: Fluorescence decay time constant (in seconds). Used to determine binning length for the movie and influences temporal filtering and deconvolution.

- `max_overlap`: Maximum allowed spatial overlap (0–1) between ROIs. 0 will delete every ROI, 1 will keep all ROI's.
