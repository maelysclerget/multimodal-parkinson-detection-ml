
"""
Aggregate Top 10 SHAP Feature Importances Across 5 Folds and Plot Summary

This script aggregates the top 10 SHAP feature importances from 5 cross-validation folds of a classical (non-deep learning) model, computes the mean and standard deviation of |SHAP| values for each feature, and visualizes the results. It also combines the per-fold SHAP PNG plots into a single grid for appendix/summary purposes.

Context:
--------
This script is intended for classical machine learning models (e.g., MLPs or tree-based models) where feature importance can be directly interpreted. In the final deep learning pipeline, feature selection and importance were handled differently (e.g., via end-to-end learning or Grad-CAM), but this script is kept for reference and for interpretability studies on classical models.

Expected file structure:
- Per-fold SHAP CSVs: Results/MLP_v2_SHAP/shap_feature_importance_fold{fold}_top10.csv
- Per-fold SHAP PNGs: Results/MLP_v2_SHAP/shap_feature_importance_fold{fold}_top10.png

Outputs:
- Aggregated CSV: shap_feature_importance_top10_aggregate.csv
- Aggregated bar plot: shap_feature_importance_top10_aggregate.png
- Combined PNG grid: shap_feature_importance_folds_grid.png

Usage:
------
Run this script after generating per-fold SHAP CSVs and PNGs. It will aggregate, plot, and save summary files for reporting and interpretation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# Directory where per-fold SHAP CSVs are saved (relative to this script)

shap_dir = Path(__file__).parent / "Results/MLP_v2_SHAP"
output_csv = shap_dir / "shap_feature_importance_top10_aggregate.csv"
output_plot = shap_dir / "shap_feature_importance_top10_aggregate.png"

NUM_FOLDS = 5

# --- Appendix Figure: Combine 5 PNGs side by side ---

# Arrange 3 PNGs on top row, 2 on bottom row
def combine_pngs_grid(png_paths, output_path, ncols=3):
    """
    Combine multiple PNG images into a grid and save as a single image.

    Args:
        png_paths (list): List of paths to PNG files.
        output_path (str or Path): Path to save the combined image.
        ncols (int): Number of columns in the grid (default: 3).

    Returns:
        None. Saves the combined image to output_path.
    """
    from PIL import Image
    
    n = len(png_paths)
    nrows = (n + ncols - 1) // ncols
    imgs = [Image.open(p) for p in png_paths]
    
    # Compute max width per col and max height per row
    widths = [img.size[0] for img in imgs]
    heights = [img.size[1] for img in imgs]
    row_heights = []
    col_widths = []
    
    for row in range(nrows):
        row_imgs = imgs[row*ncols:(row+1)*ncols]
        row_heights.append(max(img.size[1] for img in row_imgs))
    
    for col in range(ncols):
        col_imgs = imgs[col::ncols]
        col_widths.append(max(img.size[0] for img in col_imgs))
    
    total_width = sum(col_widths)
    total_height = sum(row_heights)
    combined = Image.new('RGB', (total_width, total_height), (255,255,255))
    y_offset = 0
    
    for row in range(nrows):
        x_offset = 0
        for col in range(ncols):
            idx = row*ncols + col
            if idx >= n:
                break
            img = imgs[idx]
            combined.paste(img, (x_offset, y_offset))
            x_offset += col_widths[col]
        y_offset += row_heights[row]
    
    combined.save(output_path)
    print(f"\u2713 Combined PNG grid saved: {output_path}")

# Combine the 5 per-fold PNGs into one figure for the Appendix
png_paths = [shap_dir / f"shap_feature_importance_fold{fold}_top10.png" for fold in range(NUM_FOLDS)]
appendix_png = shap_dir / "shap_feature_importance_folds_grid.png"
combine_pngs_grid(png_paths, appendix_png, ncols=3)

# Load all per-fold SHAP top10 CSVs
dfs = []
for fold in range(NUM_FOLDS):
    csv_path = shap_dir / f"shap_feature_importance_fold{fold}_top10.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df = df.set_index('feature')
        dfs.append(df)
    else:
        print(f"Warning: {csv_path} not found.")

# Aggregate mean and std for each feature (union of all top10s)
all_features = sorted(set().union(*[df.index for df in dfs]))
agg = pd.DataFrame(index=all_features)
agg['mean_abs_shap'] = np.mean([df.reindex(all_features)['mean_abs_shap'].values for df in dfs], axis=0)
agg['std_abs_shap'] = np.std([df.reindex(all_features)['mean_abs_shap'].values for df in dfs], axis=0)
agg = agg.sort_values('mean_abs_shap', ascending=False)

# Select top 10 features by mean
top10 = agg.head(10)

# Save aggregated CSV
top10.reset_index().to_csv(output_csv, index=False)
print(f"✓ Aggregated SHAP top 10 CSV saved: {output_csv}")

# Plot summary (bar plot with error bars)
plt.figure(figsize=(10, 6))
plt.barh(
    top10.index[::-1],
    top10['mean_abs_shap'][::-1],
    xerr=top10['std_abs_shap'][::-1],
    color='#d62728',
    alpha=0.8
)
plt.xlabel('Mean |SHAP value| (across 5 folds)', fontweight='bold')
plt.title('Aggregated SHAP Feature Importance (Top 10)\nMean ± Std across 5 folds', fontweight='bold')
plt.tight_layout()
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
print(f"✓ Aggregated SHAP top 10 plot saved: {output_plot}")
plt.close()
