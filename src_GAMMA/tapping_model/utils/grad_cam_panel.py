"""
Aggregate SHAP feature importance across multiple cross-validation folds.

This module processes per-fold SHAP value CSV files and generates:
    1. Combined visualization grid of all fold-specific plots
    2. Aggregated statistics (mean ± std) across folds
    3. Summary bar plot of top 10 most important features

The aggregation helps identify robust feature importance patterns that
are consistent across different training folds.

Input:
    - Per-fold SHAP CSV files: shap_feature_importance_fold{i}_top10.csv
    - Per-fold SHAP plot files: shap_feature_importance_fold{i}_top10.png

Output:
    - Combined grid PNG: shap_feature_importance_folds_grid.png
    - Aggregated CSV: shap_feature_importance_top10_aggregate.csv
    - Aggregated plot: shap_feature_importance_top10_aggregate.png
"""

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


# Configuration
SHAP_DIR = Path("/mloscratch/users/clerget/NeuroMeditron/src_GAMMA/tapping_model/results/SHAP")
OUTPUT_CSV = SHAP_DIR / "shap_feature_importance_top10_aggregate.csv"
OUTPUT_PLOT = SHAP_DIR / "shap_feature_importance_top10_aggregate.png"
APPENDIX_PNG = SHAP_DIR / "shap_feature_importance_folds_grid.png"
NUM_FOLDS = 5


def combine_pngs_grid(png_paths: List[Path], output_path: Path, ncols: int = 3) -> None:
    """
    Combine multiple PNG images into a single grid layout.
    
    Images are arranged in a grid with specified number of columns.
    The last row may have fewer images if the total number doesn't
    divide evenly by ncols.
    
    Args:
        png_paths: List of paths to PNG files to combine
        output_path: Path where combined image will be saved
        ncols: Number of columns in the grid (default: 3)
    
    Example:
        >>> paths = [Path("fold0.png"), Path("fold1.png"), Path("fold2.png")]
        >>> combine_pngs_grid(paths, Path("combined.png"), ncols=2)
        ✓ Combined PNG grid saved: combined.png
    
    Note:
        Images are arranged left-to-right, top-to-bottom.
        Empty spaces in the last row are filled with white background.
    """
    n = len(png_paths)
    nrows = (n + ncols - 1) // ncols
    
    # Load all images
    imgs = [Image.open(p) for p in png_paths]
    
    # Calculate row heights and column widths
    row_heights = []
    for row in range(nrows):
        row_imgs = imgs[row * ncols : (row + 1) * ncols]
        if row_imgs:
            row_heights.append(max(img.size[1] for img in row_imgs))
    
    col_widths = []
    for col in range(ncols):
        col_imgs = imgs[col::ncols]
        if col_imgs:
            col_widths.append(max(img.size[0] for img in col_imgs))
    
    # Create combined canvas
    total_width = sum(col_widths)
    total_height = sum(row_heights)
    combined = Image.new('RGB', (total_width, total_height), (255, 255, 255))
    
    # Paste images into grid
    y_offset = 0
    for row in range(nrows):
        x_offset = 0
        for col in range(ncols):
            idx = row * ncols + col
            if idx >= n:
                break
            
            img = imgs[idx]
            combined.paste(img, (x_offset, y_offset))
            x_offset += col_widths[col]
        
        y_offset += row_heights[row]
    
    combined.save(output_path)
    print(f"✓ Combined PNG grid saved: {output_path}")


def load_fold_shap_data(shap_dir: Path, num_folds: int) -> List[pd.DataFrame]:
    """
    Load SHAP feature importance data from all cross-validation folds.
    
    Args:
        shap_dir: Directory containing per-fold SHAP CSV files
        num_folds: Number of folds to load
    
    Returns:
        List of DataFrames, each containing SHAP values for one fold
        with 'feature' as index and 'mean_abs_shap' column
    
    Note:
        Missing fold files will generate a warning but won't stop execution.
    """
    dfs = []
    
    for fold in range(num_folds):
        csv_path = shap_dir / f"shap_feature_importance_fold{fold}_top10.csv"
        
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df = df.set_index('feature')
            dfs.append(df)
        else:
            print(f"[WARNING] {csv_path} not found - skipping fold {fold}")
    
    return dfs


def aggregate_shap_values(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Aggregate SHAP values across folds by computing mean and std.
    
    Combines all features from all folds (union), then computes statistics
    for each feature across folds. Features missing in some folds are
    handled by reindexing with NaN values.
    
    Args:
        dfs: List of DataFrames with 'mean_abs_shap' column
    
    Returns:
        DataFrame with columns ['mean_abs_shap', 'std_abs_shap'],
        sorted by mean importance (descending)
    
    Example:
        >>> agg = aggregate_shap_values(fold_dfs)
        >>> print(agg.head(3))
                           mean_abs_shap  std_abs_shap
        feature_x               0.345          0.023
        feature_y               0.298          0.041
        feature_z               0.276          0.019
    """
    # Get union of all features across folds
    all_features = sorted(set().union(*[df.index for df in dfs]))
    
    # Reindex each DataFrame to include all features (fills with NaN)
    reindexed_values = [df.reindex(all_features)['mean_abs_shap'].values for df in dfs]
    
    # Compute statistics across folds
    agg = pd.DataFrame(index=all_features)
    agg['mean_abs_shap'] = np.nanmean(reindexed_values, axis=0)
    agg['std_abs_shap'] = np.nanstd(reindexed_values, axis=0)
    
    # Sort by mean importance
    agg = agg.sort_values('mean_abs_shap', ascending=False)
    
    return agg


def plot_aggregated_shap(
    top_features: pd.DataFrame,
    output_path: Path,
    title: str = 'Aggregated SHAP Feature Importance (Top 10)\nMean ± Std across 5 folds'
) -> None:
    """
    Create horizontal bar plot of aggregated SHAP feature importance.
    
    Features are displayed in ascending order (most important at top).
    Error bars show standard deviation across folds.
    
    Args:
        top_features: DataFrame with 'mean_abs_shap' and 'std_abs_shap' columns
        output_path: Path where plot will be saved
        title: Plot title (supports multi-line with \\n)
    
    Example:
        >>> plot_aggregated_shap(top10_df, Path("shap_summary.png"))
        ✓ Aggregated SHAP top 10 plot saved: shap_summary.png
    """
    plt.figure(figsize=(10, 6))
    
    # Plot bars with error bars (reversed order for top-to-bottom)
    plt.barh(
        top_features.index[::-1],
        top_features['mean_abs_shap'][::-1],
        xerr=top_features['std_abs_shap'][::-1],
        color='#d62728',
        alpha=0.8,
        capsize=3
    )
    
    plt.xlabel('Mean |SHAP value| (across 5 folds)', fontweight='bold')
    plt.title(title, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Aggregated SHAP top 10 plot saved: {output_path}")


def main():
    """
    Main execution function for SHAP aggregation.
    
    Workflow:
        1. Combine per-fold PNG plots into appendix grid figure
        2. Load SHAP data from all fold CSV files
        3. Aggregate statistics (mean ± std) across folds
        4. Select top 10 features by mean importance
        5. Save aggregated CSV
        6. Generate and save summary plot
    """
    print("="*80)
    print("SHAP Feature Importance Aggregation")
    print("="*80)
    
    # Step 1: Combine per-fold PNGs into grid
    print("\n[1/5] Combining per-fold PNG plots...")
    png_paths = [
        SHAP_DIR / f"shap_feature_importance_fold{fold}_top10.png"
        for fold in range(NUM_FOLDS)
    ]
    combine_pngs_grid(png_paths, APPENDIX_PNG, ncols=3)
    
    # Step 2: Load per-fold SHAP data
    print("\n[2/5] Loading per-fold SHAP CSV files...")
    fold_dfs = load_fold_shap_data(SHAP_DIR, NUM_FOLDS)
    print(f"✓ Loaded {len(fold_dfs)} fold(s)")
    
    if not fold_dfs:
        raise ValueError("No SHAP data found - check file paths")
    
    # Step 3: Aggregate across folds
    print("\n[3/5] Aggregating SHAP values across folds...")
    agg_df = aggregate_shap_values(fold_dfs)
    print(f"✓ Aggregated {len(agg_df)} features")
    
    # Step 4: Select top 10 features
    print("\n[4/5] Selecting top 10 features...")
    top10 = agg_df.head(10)
    print(f"✓ Top 10 features selected")
    
    # Step 5: Save aggregated CSV
    print("\n[5/5] Saving outputs...")
    top10.reset_index().to_csv(OUTPUT_CSV, index=False)
    print(f"✓ Aggregated CSV saved: {OUTPUT_CSV}")
    
    # Step 6: Generate summary plot
    plot_aggregated_shap(top10, OUTPUT_PLOT)
    
    print("\n" + "="*80)
    print("✓ SHAP aggregation complete!")
    print("="*80)
    print(f"\nOutputs:")
    print(f"  - Grid PNG:       {APPENDIX_PNG}")
    print(f"  - Aggregated CSV: {OUTPUT_CSV}")
    print(f"  - Summary plot:   {OUTPUT_PLOT}\n")


if __name__ == "__main__":
    main()