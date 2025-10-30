"""
Anatomical Detection Grid Search

This script runs a grid search over Cellpose anatomical detection parameters on multiple z-planes.

Grid Parameters:
- anatomical_only: [1, 2, 3] - Which image to use for Cellpose
- spatial_hp_cp: [0, 0.5, 1] - High-pass filtering before Cellpose
- diameter: [4, 8] - Expected cell diameter in pixels

Planes tested: 2, 7, 13
Total combinations: 3 × 3 × 2 = 18 per plane
"""

import numpy as np
from pathlib import Path
import mbo_utilities as mbo
import lbm_suite2p_python as lsp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# Configuration
# =============================================================================

# Paths
data_dir = Path(r"D:/demo/ome_v2/sharded")
save_root = Path(r"D:/demo/ome_v2/anatomical_gridsearch")
save_root.mkdir(exist_ok=True, parents=True)

# Planes to process
planes_to_process = [2, 7, 13]

print(f"Data directory: {data_dir}")
print(f"Save directory: {save_root}")
print(f"Planes: {planes_to_process}")

# =============================================================================
# Grid Search Parameters
# =============================================================================

# Grid search dictionary
search_dict = {
    "anatomical_only": [1, 2, 3],      # 1: max_proj/mean_img, 2: mean_img, 3: meanImgE
    "spatial_hp_cp": [0, 0.5, 1.0],    # High-pass filter amount (0-1)
    "diameter": [4, 8]                  # Expected cell diameter in pixels
}

# Total combinations
n_combinations = np.prod([len(v) for v in search_dict.values()])
print(f"\nGrid search will test {n_combinations} parameter combinations per plane")
print(f"Total runs: {n_combinations * len(planes_to_process)}")

# Print all combinations
print("\nParameter combinations:")
for anat in search_dict["anatomical_only"]:
    for hp in search_dict["spatial_hp_cp"]:
        for diam in search_dict["diameter"]:
            print(f"  anatomical_only={anat}, spatial_hp_cp={hp}, diameter={diam}")

# =============================================================================
# Base Operations Setup
# =============================================================================

# Get a sample zarr file to extract metadata
sample_zarr = data_dir / f"plane{planes_to_process[0]:02d}_stitched.zarr"
print(f"\nReading metadata from: {sample_zarr}")

# Get metadata from zarr
metadata = mbo.get_metadata(str(sample_zarr))
print(f"\nMetadata extracted:")
print(f"  Frame rate: {metadata.get('frame_rate', 'N/A')} Hz")
print(f"  Dimensions: {metadata.get('Ly', 'N/A')} x {metadata.get('Lx', 'N/A')}")
print(f"  Frames: {metadata.get('num_frames', 'N/A')}")

# Create base ops from metadata
from lbm_suite2p_python.default_ops import default_ops
base_ops = default_ops()

# Run grid search on each plane
all_results = {}

for plane_idx in planes_to_process:
    print(f"\n{'='*80}")
    print(f"Processing Plane {plane_idx}")
    print(f"{'='*80}")

    # Get input zarr file
    input_zarr = data_dir / f"plane{plane_idx:02d}_stitched.zarr"

    if not input_zarr.exists():
        print(f"WARNING: {input_zarr} not found, skipping...")
        continue

    print(f"Input: {input_zarr}")

    # Create save path for this plane
    plane_save_dir = save_root / f"plane{plane_idx:02d}"
    plane_save_dir.mkdir(exist_ok=True, parents=True)
    print(f"Output: {plane_save_dir}")

    try:
        # Run grid search
        lsp.run_grid_search(
            base_ops=base_ops,
            grid_search_dict=search_dict,
            input_file=str(input_zarr),
            save_root=str(plane_save_dir),
            force_reg=False,        # Don't force re-registration
            force_detect=True       # Force re-detection for each combination
        )

        all_results[plane_idx] = plane_save_dir
        print(f"✓ Completed plane {plane_idx}")

    except Exception as e:
        print(f"✗ Error processing plane {plane_idx}: {e}")
        import traceback
        traceback.print_exc()
        continue

print(f"\n{'='*80}")
print(f"Grid search complete!")
print(f"Processed {len(all_results)} planes")
print(f"{'='*80}")

# =============================================================================
# Analyze Results
# =============================================================================

print("\n" + "="*80)
print("Analyzing Results")
print("="*80)

# Collect results from all runs
results_data = []

for plane_idx, plane_dir in all_results.items():
    print(f"\nAnalyzing plane {plane_idx}...")

    # Find all grid search subdirectories
    for combo_dir in plane_dir.iterdir():
        if not combo_dir.is_dir():
            continue

        # Look for plane0 subdirectory (Suite2p output)
        plane0_dir = combo_dir / "plane0"
        if not plane0_dir.exists():
            continue

        ops_file = plane0_dir / "ops.npy"
        stat_file = plane0_dir / "stat.npy"
        iscell_file = plane0_dir / "iscell.npy"

        if not ops_file.exists():
            print(f"  Skipping {combo_dir.name}: no ops.npy")
            continue

        # Load ops
        ops = np.load(ops_file, allow_pickle=True).item()

        # Count ROIs
        n_total = 0
        n_accepted = 0
        n_rejected = 0

        if iscell_file.exists():
            iscell = np.load(iscell_file)
            n_total = len(iscell)
            n_accepted = int(iscell[:, 0].sum())
            n_rejected = n_total - n_accepted
        elif stat_file.exists():
            stat = np.load(stat_file, allow_pickle=True)
            n_total = len(stat)

        # Extract parameters from ops or directory name
        anatomical_only = ops.get('anatomical_only', None)
        spatial_hp_cp = ops.get('spatial_hp_cp', None)
        diameter = ops.get('diameter', None)

        # Store results
        results_data.append({
            'plane': plane_idx,
            'combo_name': combo_dir.name,
            'anatomical_only': anatomical_only,
            'spatial_hp_cp': spatial_hp_cp,
            'diameter': diameter,
            'n_total': n_total,
            'n_accepted': n_accepted,
            'n_rejected': n_rejected,
            'detection_time': ops.get('timing', {}).get('detection', np.nan)
        })

# Create DataFrame
df = pd.DataFrame(results_data)

if len(df) > 0:
    # Sort by plane and accepted cells
    df = df.sort_values(['plane', 'n_accepted'], ascending=[True, False])

    print(f"\n{'='*80}")
    print("Grid Search Results Summary")
    print(f"{'='*80}")
    print(df.to_string(index=False))

    # Save results
    results_csv = save_root / "grid_search_summary.csv"
    df.to_csv(results_csv, index=False)
    print(f"\n✓ Results saved to: {results_csv}")

    # Print best combination per plane
    print(f"\n{'='*80}")
    print("Best Parameter Combination per Plane (by accepted cells)")
    print(f"{'='*80}")
    for plane_idx in planes_to_process:
        plane_data = df[df['plane'] == plane_idx]
        if len(plane_data) > 0:
            best = plane_data.iloc[0]
            print(f"\nPlane {plane_idx}:")
            print(f"  anatomical_only: {best['anatomical_only']}")
            print(f"  spatial_hp_cp: {best['spatial_hp_cp']}")
            print(f"  diameter: {best['diameter']}")
            print(f"  → {best['n_accepted']} accepted cells (out of {best['n_total']} total)")
else:
    print("\n⚠ No results found!")

# =============================================================================
# Visualize Parameter Effects
# =============================================================================

if len(df) > 0:
    print("\n" + "="*80)
    print("Generating Visualizations")
    print("="*80)

    # Set style
    sns.set_style('whitegrid')

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Anatomical Detection Grid Search Results', fontsize=16, fontweight='bold')

    # Plot 1: Accepted cells by anatomical_only
    ax = axes[0, 0]
    sns.boxplot(data=df, x='anatomical_only', y='n_accepted', ax=ax)
    ax.set_title('Effect of anatomical_only on Cell Detection')
    ax.set_xlabel('anatomical_only (1=max/mean, 2=mean, 3=meanImgE)')
    ax.set_ylabel('Accepted Cells')

    # Plot 2: Accepted cells by spatial_hp_cp
    ax = axes[0, 1]
    sns.boxplot(data=df, x='spatial_hp_cp', y='n_accepted', ax=ax)
    ax.set_title('Effect of spatial_hp_cp on Cell Detection')
    ax.set_xlabel('spatial_hp_cp (High-pass filter 0-1)')
    ax.set_ylabel('Accepted Cells')

    # Plot 3: Accepted cells by diameter
    ax = axes[1, 0]
    sns.boxplot(data=df, x='diameter', y='n_accepted', ax=ax)
    ax.set_title('Effect of diameter on Cell Detection')
    ax.set_xlabel('Diameter (pixels)')
    ax.set_ylabel('Accepted Cells')

    # Plot 4: Accepted cells by plane
    ax = axes[1, 1]
    sns.boxplot(data=df, x='plane', y='n_accepted', ax=ax)
    ax.set_title('Cell Detection by Plane')
    ax.set_xlabel('Plane Number')
    ax.set_ylabel('Accepted Cells')

    plt.tight_layout()

    # Save figure
    fig_path = save_root / "parameter_effects.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Figure saved to: {fig_path}")

    # Heatmap of parameter interactions
    fig, axes = plt.subplots(1, len(planes_to_process), figsize=(15, 5))
    if len(planes_to_process) == 1:
        axes = [axes]
    fig.suptitle('Parameter Interaction Heatmaps (n_accepted)', fontsize=14, fontweight='bold')

    for idx, plane_idx in enumerate(planes_to_process):
        plane_data = df[df['plane'] == plane_idx]
        if len(plane_data) > 0:
            # Create pivot table
            pivot = plane_data.pivot_table(
                values='n_accepted',
                index='anatomical_only',
                columns='diameter',
                aggfunc='mean'
            )

            ax = axes[idx]
            sns.heatmap(pivot, annot=True, fmt='.0f', cmap='viridis', ax=ax, cbar_kws={'label': 'Accepted Cells'})
            ax.set_title(f'Plane {plane_idx}')
            ax.set_xlabel('Diameter')
            ax.set_ylabel('anatomical_only')

    plt.tight_layout()

    # Save heatmap
    heatmap_path = save_root / "parameter_heatmaps.png"
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    print(f"✓ Heatmap saved to: {heatmap_path}")

    print("\n" + "="*80)
    print("Complete!")
    print("="*80)
    print(f"\nOutput directory: {save_root}")
    print(f"  - grid_search_summary.csv")
    print(f"  - parameter_effects.png")
    print(f"  - parameter_heatmaps.png")
    print(f"  - plane02/ (18 combinations)")
    print(f"  - plane07/ (18 combinations)")
    print(f"  - plane13/ (18 combinations)")
else:
    print("\n⚠ No data to plot!")

print("\n" + "="*80)
print("Script finished successfully!")
print("="*80)
