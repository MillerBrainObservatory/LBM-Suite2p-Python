"""
Analyze existing anatomical grid search results without re-running the grid search.

This script parses all grid search combination directories and extracts:
- Parameter combinations from directory names
- Cell detection statistics from ops.npy and iscell.npy
- Timing information
- Generates summary CSV and visualization plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re


def parse_combo_name(combo_name):
    """
    Parse parameter values from combination directory name.

    Example: "ana1_spa0.50_dia4" -> {anatomical_only: 1, spatial_hp_cp: 0.5, diameter: 4}
    """
    # Pattern: ana{int}_spa{float}_dia{int}
    pattern = r'ana(\d+)_spa([\d.]+)_dia(\d+)'
    match = re.match(pattern, combo_name)

    if match:
        return {
            'anatomical_only': int(match.group(1)),
            'spatial_hp_cp': float(match.group(2)),
            'diameter': int(match.group(3))
        }
    return None


def analyze_grid_search_results(results_root, planes_to_analyze=None):
    """
    Analyze all grid search results in the given directory.

    Parameters
    ----------
    results_root : str or Path
        Root directory containing plane{XX}/ subdirectories
    planes_to_analyze : list of int, optional
        Specific plane numbers to analyze. If None, analyzes all found.

    Returns
    -------
    pd.DataFrame
        Summary dataframe with all results
    """
    results_root = Path(results_root)
    results_data = []

    # Find all plane directories
    plane_dirs = sorted([d for d in results_root.iterdir() if d.is_dir() and d.name.startswith('plane')])

    if not plane_dirs:
        raise ValueError(f"No plane directories found in {results_root}")

    print(f"Found {len(plane_dirs)} plane directories")

    for plane_dir in plane_dirs:
        # Extract plane number from directory name
        plane_match = re.search(r'plane(\d+)', plane_dir.name)
        if not plane_match:
            continue

        plane_idx = int(plane_match.group(1))

        # Skip if not in requested planes
        if planes_to_analyze is not None and plane_idx not in planes_to_analyze:
            continue

        print(f"\nAnalyzing {plane_dir.name}...")

        # Find all combination directories
        combo_dirs = [d for d in plane_dir.iterdir() if d.is_dir()]
        print(f"  Found {len(combo_dirs)} parameter combinations")

        for combo_dir in combo_dirs:
            # Parse parameters from directory name
            params = parse_combo_name(combo_dir.name)
            if params is None:
                print(f"  Skipping {combo_dir.name}: cannot parse name")
                continue

            # Check for required files
            ops_file = combo_dir / "ops.npy"
            stat_file = combo_dir / "stat.npy"
            iscell_file = combo_dir / "iscell.npy"

            if not ops_file.exists():
                print(f"  Skipping {combo_dir.name}: no ops.npy")
                continue

            # Load ops
            try:
                ops = np.load(ops_file, allow_pickle=True).item()
            except Exception as e:
                print(f"  Error loading {combo_dir.name}/ops.npy: {e}")
                continue

            # Count ROIs
            n_total = 0
            n_accepted = 0
            n_rejected = 0

            if iscell_file.exists():
                try:
                    iscell = np.load(iscell_file)
                    n_total = len(iscell)
                    n_accepted = int(iscell[:, 0].sum())
                    n_rejected = n_total - n_accepted
                except Exception as e:
                    print(f"  Error loading {combo_dir.name}/iscell.npy: {e}")
            elif stat_file.exists():
                try:
                    stat = np.load(stat_file, allow_pickle=True)
                    n_total = len(stat)
                except Exception as e:
                    print(f"  Error loading {combo_dir.name}/stat.npy: {e}")

            # Extract timing if available
            detection_time = np.nan
            if 'timing' in ops and isinstance(ops['timing'], dict):
                detection_time = ops['timing'].get('detection', np.nan)

            # Store results
            result = {
                'plane': plane_idx,
                'combo_name': combo_dir.name,
                'combo_path': str(combo_dir),
                'anatomical_only': params['anatomical_only'],
                'spatial_hp_cp': params['spatial_hp_cp'],
                'diameter': params['diameter'],
                'n_total': n_total,
                'n_accepted': n_accepted,
                'n_rejected': n_rejected,
                'acceptance_rate': n_accepted / n_total if n_total > 0 else 0,
                'detection_time_sec': detection_time,
                'Ly': ops.get('Ly', np.nan),
                'Lx': ops.get('Lx', np.nan),
                'nframes': ops.get('nframes', np.nan)
            }

            results_data.append(result)
            print(f"  ✓ {combo_dir.name}: {n_accepted}/{n_total} cells accepted")

    # Create DataFrame
    df = pd.DataFrame(results_data)

    if len(df) == 0:
        raise ValueError("No valid results found!")

    # Sort by plane and accepted cells
    df = df.sort_values(['plane', 'n_accepted'], ascending=[True, False])

    return df


def print_summary(df, save_path=None):
    """Print summary statistics and optionally save to CSV."""
    print(f"\n{'='*80}")
    print("Grid Search Results Summary")
    print(f"{'='*80}")

    # Print full table
    print(df[['plane', 'combo_name', 'anatomical_only', 'spatial_hp_cp',
              'diameter', 'n_accepted', 'n_rejected', 'acceptance_rate']].to_string(index=False))

    # Save CSV
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"\n✓ Full results saved to: {save_path}")

    # Print best combination per plane
    print(f"\n{'='*80}")
    print("Best Parameter Combination per Plane (by accepted cells)")
    print(f"{'='*80}")

    for plane_idx in sorted(df['plane'].unique()):
        plane_data = df[df['plane'] == plane_idx]
        if len(plane_data) > 0:
            best = plane_data.iloc[0]
            print(f"\nPlane {plane_idx}:")
            print(f"  Directory: {best['combo_name']}")
            print(f"  anatomical_only: {best['anatomical_only']}")
            print(f"  spatial_hp_cp: {best['spatial_hp_cp']}")
            print(f"  diameter: {best['diameter']}")
            print(f"  → {best['n_accepted']} accepted / {best['n_total']} total")
            print(f"  → Acceptance rate: {best['acceptance_rate']:.1%}")


def plot_results(df, save_dir=None):
    """Generate visualization plots for grid search results."""
    if len(df) == 0:
        print("\n⚠ No data to plot!")
        return

    sns.set_style('whitegrid')

    # Plot 1: Parameter effects
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Anatomical Detection Grid Search Results', fontsize=16, fontweight='bold')

    # Accepted cells by anatomical_only
    ax = axes[0, 0]
    sns.boxplot(data=df, x='anatomical_only', y='n_accepted', ax=ax)
    ax.set_title('Effect of anatomical_only on Cell Detection')
    ax.set_xlabel('anatomical_only (1=max/mean, 2=mean, 3=meanImgE)')
    ax.set_ylabel('Accepted Cells')

    # Accepted cells by spatial_hp_cp
    ax = axes[0, 1]
    sns.boxplot(data=df, x='spatial_hp_cp', y='n_accepted', ax=ax)
    ax.set_title('Effect of spatial_hp_cp on Cell Detection')
    ax.set_xlabel('spatial_hp_cp (High-pass filter 0-1)')
    ax.set_ylabel('Accepted Cells')

    # Accepted cells by diameter
    ax = axes[1, 0]
    sns.boxplot(data=df, x='diameter', y='n_accepted', ax=ax)
    ax.set_title('Effect of diameter on Cell Detection')
    ax.set_xlabel('Diameter (pixels)')
    ax.set_ylabel('Accepted Cells')

    # Accepted cells by plane
    ax = axes[1, 1]
    sns.boxplot(data=df, x='plane', y='n_accepted', ax=ax)
    ax.set_title('Cell Detection by Plane')
    ax.set_xlabel('Plane Number')
    ax.set_ylabel('Accepted Cells')

    plt.tight_layout()

    if save_dir:
        fig_path = Path(save_dir) / "parameter_effects.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Parameter effects plot saved to: {fig_path}")

    plt.show()

    # Plot 2: Heatmaps
    planes = sorted(df['plane'].unique())
    n_planes = len(planes)

    fig, axes = plt.subplots(1, n_planes, figsize=(5 * n_planes, 5))
    if n_planes == 1:
        axes = [axes]

    fig.suptitle('Parameter Interaction Heatmaps (n_accepted)', fontsize=14, fontweight='bold')

    for idx, plane_idx in enumerate(planes):
        plane_data = df[df['plane'] == plane_idx]

        # Create pivot table
        pivot = plane_data.pivot_table(
            values='n_accepted',
            index='anatomical_only',
            columns='diameter',
            aggfunc='mean'
        )

        ax = axes[idx]
        sns.heatmap(pivot, annot=True, fmt='.0f', cmap='viridis', ax=ax,
                   cbar_kws={'label': 'Accepted Cells'})
        ax.set_title(f'Plane {plane_idx}')
        ax.set_xlabel('Diameter')
        ax.set_ylabel('anatomical_only')

    plt.tight_layout()

    if save_dir:
        heatmap_path = Path(save_dir) / "parameter_heatmaps.png"
        plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        print(f"✓ Heatmap saved to: {heatmap_path}")

    plt.show()


if __name__ == "__main__":
    # Configuration
    results_root = Path(r"D:\demo\ome_v2\anatomical_gridsearch")

    # Analyze all results
    print(f"Analyzing grid search results in: {results_root}")
    df = analyze_grid_search_results(results_root)

    # Print summary and save CSV
    csv_path = results_root / "grid_search_summary.csv"
    print_summary(df, save_path=csv_path)

    # Generate plots
    plot_results(df, save_dir=results_root)

    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"{'='*80}")
