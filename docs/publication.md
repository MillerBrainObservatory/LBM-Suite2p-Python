# Suite2p: beyond 10,000 neurons with standard two-photon microscopy

## Steps

1) image registration
2) region-of-interest (ROI) detection;
3) ROI labelling and quality control;
4) activity extraction with neuropil correction and spike deconvolution.

## Claims:

- Suite2p does not assume that cell activity is indepenedent of neuropil
- Finds 2x as many neurons as CNMF
- Preferrale to algorithms based on greedy segmentation of neighboring pixels

## Registration

- Many techniques find the maximum correlative peak between reference and target.
- Disadvantage that it relies on low-frequencys: neuropil, small graduents of pixel fluctuation
   - High frequency is essential for registration, i.e. edges, activity

[Whitened Image]("./_images/whitened.png")

## ROI Detection



## Deconvolution

- Core is a model that assumes *the signal of an individual pixel is the sum of signals originating in distinct active compartments* (soma, dendrite, spine) and more diffuse contamination from neuropil
- CNMF extraction significantly comprimised by subsampling (why?)

### Neuropil

- produces contamination from average activity of out-of-focus dentrides/axons
- 2p scopes have PSF one magnitude wider in Z vs X/Y
- So, single plane is really a weighted average of a volume extenting 10's of micron

[Neuropil]("./_images/neuropil.png")

(a) Example cell (blue) and neuropil areas (red-shaded annuli) superimposed on the variance image (white), for an
imaging session in superficial V1. The neuropil signal is defined as the signal inside the red-shaded annuli but not in
any detected ROIs.

- The neuropil signal is highly correlated (related) to the signal recorded from neurons (somata).
- The neuropil signal varies slowly across space (it changes gradually over distance).
- Sometimes, when the neuropil signal gets very large, it reflects periods of high network activity (when neurons are firing a lot).

This is modeled through **spatial basis functions**

Spatial Basis Functions:

- Imagine dividing the whole image into small overlapping "tiles" (like tiles on a floor).
- Each tile represents a smooth, predictable pattern of how the neuropil signal changes across space.
- These tiles are modeled using "raised cosines" (smooth, circular patterns) that are sized based on the average size of a neuron.

ROI Signals:

- Each neuron has its own unique activity pattern over time.
- Each pixel in an ROI contributes to the same timecourse (signal over time) for that neuron, scaled by how strongly that pixel is part of the neuron.

Fitting the model:

The recorded signal at each pixel is the sum of three things:

- The neuron activity (signals from the ROIs).
- The neuropil signal (from the overlapping tiles).
- Measurement noise (random errors in the recording).

Super res mean image:

- brain constantly in motion, generating continuous x/y offsets
- same as registration
- instead of shifting frame, bin frames by their x/y offset @subpixel resolution
- get mean-image for each XY bin, combine into super image

### Suite2p classifier

- labels cells as cell/noncell based on statistics
- activity dependent: skewness, variance, corr with surrounding px
- anatomy dependent: area, aspect ratio
- quality metric: posterior probability tht each ROI == soma

## Extraction and Deconvolution

Get final flourescent signal.

Skipping lots of math. 

- Remove neuropil contamination
- (sometimes useful) estimate spike train underlying the calcium trace
- Recommendation: unconstrained non-negative deconvolution using exponential kernels

## Benchmark vs CNMF

**Dataset: `[5000x512x512]`** with 64GB RAM

- Single CNMF optimization: 8 minutes
- Single Suite2p optimization: 7 seconds
- Large difference due to difference extraction algorithm / sparse matrix
   - SVD Low rank approximation
   - CNMF can't use this approx, invalidates the positivity constraints CNMF imposes on calcium dynamics

CNMF uses a *single global neuropil signal* model
- Suite2p with this "single neuropil" model == lots of false ROI's
- Estimate that CaImAn does the same

Essentially, a global neuropil means that all pixels recieve the same contribution of contamination.
