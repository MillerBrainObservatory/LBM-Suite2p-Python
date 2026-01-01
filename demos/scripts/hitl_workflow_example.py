"""
Example: Full Human-in-the-Loop (HITL) Cellpose Workflow

This script demonstrates the complete workflow for processing calcium imaging
data with iterative cellpose training:

1. Run initial pipeline on z-plane 7
2. Evaluate results and decide if retraining is needed
3. Enhance summary image for annotation
4. Annotate cells in the Cellpose GUI
5. Train a custom cellpose model
6. Re-detect with the trained model

Dataset: mk311_11_04_2025_180mw_right_ppc_go_to_2x-mROI-896x896um_224x448px_2um-px_17p07Hz_green_00001.tif
Z-plane: 7

Author: Generated with Claude Code
Date: 2025-12-31
"""

from pathlib import Path

import lbm_suite2p_python as lsp

# =============================================================================
# Configuration
# =============================================================================

# Input data
DATA_PATH = Path(
    r"D:\SERVER_DATA\raw_scanimage_tiffs\mk311_11_04_2025_180mw_right_ppc_go_to_2x-mROI-896x896um_224x448px_2um-px_17p07Hz_green_00001.tif"
)

# Output directory (will create subdirectories for each plane)
OUTPUT_DIR = Path(r"D:\lsp_output\mk311_11_04_2025_hitl_workflow")

# Processing parameters
PLANE = 7  # 1-indexed z-plane to process
ROI_MODE = None  # None = stitch mROIs together (default for LBM data)

# =============================================================================
# Step 1: Initial Pipeline Run
# =============================================================================

print("=" * 60)
print("STEP 1: Running initial pipeline on z-plane 7")
print("=" * 60)

# Configure ops for initial detection
# Using cellpose with cyto3 model for first pass
ops = lsp.default_ops()
ops.update({
    # Registration
    "do_registration": 1,
    "nonrigid": True,

    # Detection - use anatomical mode with cellpose
    "anatomical_only": 3,  # Use meanImgE for detection
    "cellpose_model_path": "cyto3",
    "diameter": 0,  # Auto-estimate diameter
    "flow_threshold": 0.4,
    "cellprob_threshold": 0.0,

    # Spatial high-pass for detection (helps with uneven illumination)
    "spatial_hp_cp": 25,

    # Spike detection
    "spikedetect": True,
    "tau": 1.0,  # GCaMP decay time
})

# Check for existing results to allow re-running the script
existing_ops = list(OUTPUT_DIR.glob("**/ops.npy"))
if existing_ops:
    # Use existing results
    plane_dir = existing_ops[0].parent
    print(f"Found existing results at: {plane_dir}")
    print("Skipping pipeline run. Delete the output directory to re-run.")
    results = [existing_ops[0]]
else:
    # Run the pipeline
    results = lsp.pipeline(
        DATA_PATH,
        save_path=OUTPUT_DIR,
        ops=ops,
        planes=[PLANE],
        roi_mode=ROI_MODE,
        cell_filters=[{"name": "max_diameter", "min_diameter_um": 4, "max_diameter_um": 35}],
        accept_all_cells=False,  # Use classifier
    )
    # Get path to the plane results
    plane_dir = results[0].parent if results else OUTPUT_DIR / f"plane{PLANE:02d}"

print(f"\nInitial results saved to: {plane_dir}")

# =============================================================================
# Step 2: Evaluate Results
# =============================================================================

print("\n" + "=" * 60)
print("STEP 2: Evaluate initial detection results")
print("=" * 60)

# Generate diagnostic plots
lsp.plot_zplane_figures(plane_dir)
lsp.plot_traces(plane_dir)
lsp.plot_plane_diagnostics(plane_dir)

print(f"""
Review the generated figures in: {plane_dir}

Key things to check:
1. Are cells being detected correctly?
2. Are there false positives (non-cells marked as cells)?
3. Are there false negatives (cells missed)?
4. Check the diameter histogram - are detected diameters reasonable?

If detection needs improvement, proceed to Step 3.
Otherwise, skip to the end.
""")

# =============================================================================
# Step 3: Enhance Summary Image for Annotation
# =============================================================================

print("\n" + "=" * 60)
print("STEP 3: Enhance summary image for annotation")
print("=" * 60)

# Create enhanced images using different methods to see which works best
enhanced_paths = []

for method in ["meanImgE", "log_ratio", "max_proj"]:
    enhanced_path = lsp.enhance_summary_image(
        plane_dir,
        method=method,
        denoise=True,
        spatial_hp=0,  # Already applied in pipeline
        percentile_range=(1, 99),
    )
    enhanced_paths.append(enhanced_path)

print(f"""
Enhanced images saved to {plane_dir}:
- enhanced_meanImgE.tif  (spatial high-pass of mean)
- enhanced_log_ratio.tif (log ratio emphasizes active regions)
- enhanced_max_proj.tif  (maximum projection)

Review these and choose the one with best cell visibility.
""")

# =============================================================================
# Step 4: Annotate Cells in Cellpose GUI
# =============================================================================

print("\n" + "=" * 60)
print("STEP 4: Annotate cells in Cellpose GUI")
print("=" * 60)

# Prepare annotation data (creates image + initial masks for GUI)
annotation_dir = plane_dir / "annotation"
lsp.annotate(
    plane_dir,
    output_path=annotation_dir,
    use_existing_masks=True,  # Start with current detections as a baseline
)

print(f"""
Annotation data prepared in: {annotation_dir}

To annotate:
1. Open the Cellpose GUI:
   >>> lsp.open_in_gui("{annotation_dir}")

2. In the GUI:
   - Load the image file
   - Correct any mis-detected cells (delete false positives)
   - Draw any missed cells (add false negatives)
   - Save segmentation (Ctrl+S)

3. Export annotations when done

The annotations will be used to train a custom model in Step 5.
""")

# Uncomment to automatically open the GUI:
# lsp.open_in_gui(annotation_dir)

# =============================================================================
# Step 5: Train Custom Cellpose Model
# =============================================================================

print("\n" + "=" * 60)
print("STEP 5: Train custom Cellpose model")
print("=" * 60)

# After annotation, train a model on the corrected data
# NOTE: This assumes you've completed annotations from Step 4

training_dir = plane_dir / "training"

print(f"""
To train a custom model after annotation:

>>> model_path = lsp.train_cellpose(
...     "{annotation_dir}",
...     model_name="mk311_plane7_model",
...     base_model="cyto3",
...     n_epochs=500,
...     learning_rate=0.2,
... )

This will:
1. Load your annotated images and masks
2. Train a cellpose model starting from cyto3
3. Save the model to ~/.cellpose/models/

Training typically takes 5-30 minutes depending on:
- Number of annotated images
- Number of epochs
- GPU availability
""")

# Uncomment when you've completed annotations:
# model_path = lsp.train_cellpose(
#     annotation_dir,
#     model_name="mk311_plane7_model",
#     base_model="cyto3",
#     n_epochs=500,
#     learning_rate=0.2,
# )

# =============================================================================
# Step 6: Re-detect with Trained Model
# =============================================================================

print("\n" + "=" * 60)
print("STEP 6: Re-detect with trained model")
print("=" * 60)

print(f"""
After training, re-detect cells with your custom model:

>>> lsp.redetect(
...     "{plane_dir}",
...     model_path=model_path,  # or path to your trained model
...     diameter=None,  # Auto-estimate
...     flow_threshold=0.4,
...     cellprob_threshold=0.0,
...     run_extraction=True,
...     run_classification=True,
...     run_spikes=True,
... )

This will:
1. Run cellpose detection with your trained model
2. Extract fluorescence traces (F, Fneu)
3. Run the ROI classifier (iscell)
4. Compute deconvolved spikes

The original data.bin is preserved, so you can re-run detection
multiple times without re-registering.
""")

# Uncomment when you have a trained model:
# lsp.redetect(
#     plane_dir,
#     model_path=model_path,
#     diameter=None,
#     flow_threshold=0.4,
#     cellprob_threshold=0.0,
# )

# =============================================================================
# Step 7: Final Evaluation
# =============================================================================

print("\n" + "=" * 60)
print("STEP 7: Final evaluation")
print("=" * 60)

print(f"""
After re-detection, evaluate the final results:

>>> lsp.plot_zplane_figures("{plane_dir}")
>>> lsp.plot_traces("{plane_dir}")
>>> lsp.plot_plane_diagnostics("{plane_dir}")

Compare with the initial detection:
- Did the number of detected cells change?
- Are the traces cleaner?
- Are there fewer artifacts?

If still not satisfied, iterate:
1. Add more annotations
2. Retrain the model
3. Re-detect again
""")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 60)
print("WORKFLOW COMPLETE")
print("=" * 60)

print(f"""
Files created in {OUTPUT_DIR}:

plane07/
├── ops.npy          # Suite2p options and summary images
├── stat.npy         # ROI statistics
├── F.npy            # Raw fluorescence traces
├── Fneu.npy         # Neuropil traces
├── spks.npy         # Deconvolved spikes
├── iscell.npy       # Cell classifier output
├── data.bin         # Registered movie (can re-run detection)
├── dff.npy          # dF/F traces
├── enhanced_*.tif   # Enhanced images for annotation
├── masks.npy        # Detection masks
├── annotation/      # Annotation data for GUI
└── figures/         # Diagnostic plots

To use the trained model on other planes:
>>> lsp.redetect("path/to/other/plane", model_path=model_path)

To use on future datasets:
>>> ops = lsp.default_ops()
>>> ops["cellpose_model_path"] = str(model_path)
>>> lsp.pipeline(new_data, ops=ops)
""")
