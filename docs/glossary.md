# Glossary

```{glossary}

segmentation
  The general process of dividing an image based on the contents of that image, in our case, based on neuron location.

source-extraction
  Umbrella term for all of the individual processes that produce a segmented image.

deconvolution
  The process performed after segmentation to the resulting traces to infer spike times from flourescence values.

CNMF
  The name for a set of algorithms within the flatironinstitute's [CaImAn Pipeline](https://github.com/flatironinstitute/CaImAn) that initialize parameters and run source extraction.

Rigid-registration
  The object retains shape and size.

Non-rigid-registration
  The object is moved and transforms shape or size.

pixel-resolution
  The length of each pixel, in micron (px/um).

batch-path
  Location where results are saved in the form `path/to/batch.pickle`.
  All results (registration/segmentation outputs) will be saved in the same directory as this file.
  The `.pickle` file is a table with the paths to the files and information about theh algorithm which was run.

```
