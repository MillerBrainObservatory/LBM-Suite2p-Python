(discuss_reg)=
# Registration

```{note}
The terms motion-correction and registration are often used interchangeably.
Similarly, non-rigid and piecewise-rigid are often used interchangeably.
Here, piecewise-rigid registration is the **method** used to correct for non-rigid motion.
```

## Overview

We use {ref}`suite2p:Registration` to ensure spatial alignment across all frames in a movie. This means a neuron that appears in one location in frame 0 remains in the same spot in frame N.

Registration uses either a single-channel movie (default: channel 1), or an alternate channel if `ops['align_by_chan'] = 2`.

The result of registration is a binary file (`data.bin`) aligned to a reference frame.
Suite2p creates this reference frame by selecting the top correlated subset of frames and refining their alignment iteratively.
You can customize this via `ops['nimg_init']`.

```python
from suite2p.registration import register
refImg = register.pick_initial_reference(ops)
```

Use `ops['refImg']` to inspect the reference if needed.

## Rigid and Non-Rigid Registration

Suite2p first runs rigid registration (frame-wide shifts using phase correlation), followed by optional non-rigid registration (local shifts in blocks).

Set `ops['nonrigid'] = True` to enable block-based correction.

### Key Parameters

| Parameter              | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| `align_by_chan`       | Channel to align by (1 or 2).                                                |
| `nonrigid`            | Whether to apply non-rigid registration.                                    |
| `block_size`          | Size of blocks in [Y, X] for nonrigid. Default `[128, 128]`.                |
| `maxregshift`         | Maximum fraction of FOV allowed to shift (rigid).                           |
| `maxregshiftNR`       | Max shift in pixels for non-rigid registration.                             |
| `smooth_sigma`        | Gaussian smoothing sigma. Improves template clarity.                        |
| `batch_size`          | Number of frames per registration batch.                                    |

```{admonition} Recommended tuning
:class: tip
- Increase `nimg_init` if your template looks noisy or blurry
- If there is remaining motion after rigid, try `nonrigid = True`
- If registration looks unstable, try decreasing `maxregshift` or `maxregshiftNR`
```

## Output Files

- `data.bin`: motion-corrected movie (channel 1)
- `data_chan2.bin`: registered channel 2 (if `nchannels = 2`)
- `ops['xoff']`, `ops['yoff']`: rigid shifts for each frame
- `ops['corrXY']`: phase-correlation scores frame vs reference

## Metrics

Suite2p computes PC-based registration metrics via {ref}`registration`:

```python
from suite2p.registration import metrics
ops = metrics.get_pc_metrics(ops, use_red=False)
```

| Metric         | Description                                                                                         |
|----------------|-----------------------------------------------------------------------------------------------------|
| `regPC`        | Spatial principal components used to compute metrics.                                               |
| `tPC`          | Time courses of principal components used for alignment QC.                                         |
| `regDX`        | Shift distance between split PCs. Lower values = better registration.                              |

You can visualize these with Suite2p’s GUI or custom plotting tools. Even with good metrics, always visually inspect registered frames.

```{important}
Visual inspection trumps all. Even if `regDX` is low, double check the video. And if it looks good but metrics are high, trust your eyes.
```

## CLI for Registration Metrics

You can also use the command-line:

```bash
reg_metrics path/to/data --nplanes 1 --smooth_sigma 1.2
```

To see available CLI options:

```bash
reg_metrics --help
```

Example output:

```text
Avg_Rigid: 0.000000     Avg_Average NR: 0.028889        Avg_Max NR: 0.120000
Max_Rigid: 0.000000     Max_Average NR: 0.044444        Max_Max NR: 0.200000
```

These values are from `ops['regDX']`, and should be close to zero in well-registered data.

