"""
Publication-Quality Visualization: Heatmap B - Spatial Segmentation Errors
Senior Medical Imaging Research & Python Engineering Module for Swin-UMamba
Dataset: Dataset702_AbdomenMR (AMOS 2022 MRI Benchmark)
"""

import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.gridspec import GridSpec


# -------------------------------------------------------------------------
# 1. Anatomical & Palette Definitions
# -------------------------------------------------------------------------

ORGAN_NAMES = {
    1: "Liver",
    2: "Right Kidney",
    3: "Spleen",
    4: "Pancreas",
    5: "Aorta",
    6: "Inferior Vena Cava",
    7: "Right Adrenal Gland",
    8: "Left Adrenal Gland",
    9: "Gallbladder",
    10: "Esophagus",
    11: "Stomach",
    12: "Duodenum",
    13: "Left Kidney",
}

# Authoritative, high-contrast, distinct multiclass color palette for 13 abdominal organs
# (0 = Background / Transparent)
ORGAN_COLORS = {
    0:  (0.0, 0.0, 0.0, 0.0),       # Background (transparent)
    1:  (0.18, 0.62, 0.17, 0.85),   # 1: Liver (Medium Forest Green)
    2:  (0.12, 0.47, 0.71, 0.85),   # 2: Right Kidney (Cobalt Blue)
    3:  (0.58, 0.40, 0.74, 0.85),   # 3: Spleen (Royal Amethyst Purple)
    4:  (1.00, 0.50, 0.05, 0.85),   # 4: Pancreas (Vibrant Orange)
    5:  (0.84, 0.15, 0.16, 0.85),   # 5: Aorta (Crimson Red)
    6:  (0.09, 0.75, 0.81, 0.85),   # 6: Inferior Vena Cava (Cyan / Aqua)
    7:  (0.89, 0.47, 0.76, 0.85),   # 7: Right Adrenal Gland (Hot Pink / Magenta)
    8:  (0.55, 0.34, 0.29, 0.85),   # 8: Left Adrenal Gland (Sienna Brown)
    9:  (0.74, 0.74, 0.13, 0.85),   # 9: Gallbladder (Olive Lime)
    10: (0.50, 0.50, 0.50, 0.85),   # 10: Esophagus (Medium Slate Gray)
    11: (1.00, 0.73, 0.47, 0.85),   # 11: Stomach (Apricot / Salmon)
    12: (0.68, 0.78, 0.91, 0.85),   # 12: Duodenum (Soft Slate Blue)
    13: (0.22, 0.23, 0.47, 0.85),   # 13: Left Kidney (Deep Indigo)
}

# Mutually exclusive spatial error category definitions (Colorblind-conscious palette)
ERROR_COLORS = {
    0: (0.0, 0.0, 0.0, 0.0),       # True Negative / Background: Transparent over MRI
    1: (0.18, 0.55, 0.34, 0.70),   # Correct Organ Classification (Neutral / Muted Jade Green)
    2: (0.00, 0.45, 0.70, 0.88),   # False Negative / Missed Organ (Vibrant Royal Blue)
    3: (0.84, 0.15, 0.16, 0.88),   # False Positive / Spurious Organ (Vermilion Red)
    4: (0.90, 0.60, 0.00, 0.90),   # Inter-Organ Label Confusion (Amber / Deep Goldenrod)
}

ERROR_LABELS = {
    1: "Correct Classification (TP)",
    2: "False Negative / Missed Organ",
    3: "False Positive / Spurious Organ",
    4: "Inter-Organ Label Confusion",
}


def build_discrete_cmap(color_dict):
    """Builds a Matplotlib ListedColormap and BoundaryNorm for discrete integer labels."""
    max_label = max(color_dict.keys())
    colors = [color_dict[i] for i in range(max_label + 1)]
    cmap = ListedColormap(colors)
    bounds = np.arange(max_label + 2) - 0.5
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def reorient_to_radiological(slice_2d):
    """
    Transforms AMOS 2D axial slice array to standard radiological orientation:
    - Anterior on top, Posterior on bottom
    - Patient's Right on viewer's Left, Patient's Left on viewer's Right
    Formula verified against anatomical landmarks: slice.T[::-1, ::-1]
    """
    return slice_2d.T[::-1, ::-1]


def normalize_mri(slice_mri):
    """Robust percentile-based intensity normalization [0.0, 1.0]."""
    p_low, p_high = np.percentile(slice_mri, (0.5, 99.5))
    if p_high > p_low:
        clipped = np.clip(slice_mri, p_low, p_high)
        norm = (clipped - p_low) / (p_high - p_low)
    else:
        norm = np.zeros_like(slice_mri)
    return norm


def compute_spatial_error_map(gt_slice, pred_slice):
    """
    Computes mutually exclusive voxel-wise error categories:
    - 0: Background (GT==0 and Pred==0)
    - 1: Correct Organ Classification (GT == Pred and GT > 0)
    - 2: False Negative (GT > 0 and Pred == 0)
    - 3: False Positive (GT == 0 and Pred > 0)
    - 4: Inter-Organ Label Confusion (GT > 0 and Pred > 0 and GT != Pred)
    """
    error_map = np.zeros_like(gt_slice, dtype=np.uint8)
    
    gt_fg = gt_slice > 0
    pred_fg = pred_slice > 0
    
    # 1. Correct foreground organ
    correct_mask = np.logical_and(gt_fg, pred_slice == gt_slice)
    error_map[correct_mask] = 1
    
    # 2. False negative (missed organ)
    fn_mask = np.logical_and(gt_fg, ~pred_fg)
    error_map[fn_mask] = 2
    
    # 3. False positive (spurious organ)
    fp_mask = np.logical_and(~gt_fg, pred_fg)
    error_map[fp_mask] = 3
    
    # 4. Inter-organ confusion (both foreground, but label mismatch)
    conf_mask = np.logical_and(np.logical_and(gt_fg, pred_fg), gt_slice != pred_slice)
    error_map[conf_mask] = 4
    
    return error_map, {
        "correct_vox": int(correct_mask.sum()),
        "fn_vox": int(fn_mask.sum()),
        "fp_vox": int(fp_mask.sum()),
        "conf_vox": int(conf_mask.sum()),
        "gt_fg_vox": int(gt_fg.sum()),
        "pred_fg_vox": int(pred_fg.sum()),
    }


# -------------------------------------------------------------------------
# 2. Four-Panel Figure Generator
# -------------------------------------------------------------------------

def render_four_panel_figure(
    mri_slice,
    gt_slice,
    pred_slice,
    case_name,
    slice_idx,
    case_dice,
    slice_dice,
    save_path,
    organ_cmap,
    organ_norm,
    error_cmap,
    error_norm,
    dpi=300
):
    """Renders and saves a standalone 4-panel publication-grade figure."""
    mri_rad = reorient_to_radiological(mri_slice)
    gt_rad = reorient_to_radiological(gt_slice)
    pred_rad = reorient_to_radiological(pred_slice)
    
    mri_norm = normalize_mri(mri_rad)
    error_map, err_stats = compute_spatial_error_map(gt_rad, pred_rad)
    
    # Organs present in this slice
    gt_organs = sorted(list(np.unique(gt_rad[gt_rad > 0])))
    pred_organs = sorted(list(np.unique(pred_rad[pred_rad > 0])))
    
    fig = plt.figure(figsize=(20, 7.0), dpi=dpi)
    gs = GridSpec(1, 4, figure=fig, left=0.03, right=0.97, bottom=0.25, top=0.79, wspace=0.06)
    
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[0, 3])
    
    # Global Super-Title with comfortable breathing room
    fig.suptitle(
        f"Heatmap B: Spatial Segmentation Errors — Case {case_name.replace('.nii.gz', '')} (Slice z={slice_idx})\n"
        f"Overall Case Foreground Dice: {case_dice:.4f}  |  Slice Foreground Dice: {slice_dice:.4f}  |  "
        f"Ground-Truth Organs: {len(gt_organs)}  |  Predicted Organs: {len(pred_organs)}",
        fontsize=13.0,
        fontweight="bold",
        y=0.94,
        color="#111111"
    )
    
    # ---------------- PANEL A: Original MRI ----------------
    ax_a.imshow(mri_norm, cmap="gray", interpolation="bilinear")
    ax_a.set_title("Panel A — Original MRI", fontsize=11, fontweight="bold", pad=7)
    # Add subtle radiological anatomical markers
    ax_a.text(0.5, 0.96, "A (Anterior)", transform=ax_a.transAxes, color="#f0f0f0",
              fontsize=8.5, ha="center", va="top", fontweight="semibold",
              bbox=dict(boxstyle="round,pad=0.2", facecolor="#111111", alpha=0.55, edgecolor="none"))
    ax_a.text(0.5, 0.04, "P (Posterior)", transform=ax_a.transAxes, color="#f0f0f0",
              fontsize=8.5, ha="center", va="bottom", fontweight="semibold",
              bbox=dict(boxstyle="round,pad=0.2", facecolor="#111111", alpha=0.55, edgecolor="none"))
    ax_a.text(0.04, 0.5, "R", transform=ax_a.transAxes, color="#f0f0f0",
              fontsize=9, ha="left", va="center", fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.2", facecolor="#111111", alpha=0.55, edgecolor="none"))
    ax_a.text(0.96, 0.5, "L", transform=ax_a.transAxes, color="#f0f0f0",
              fontsize=9, ha="right", va="center", fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.2", facecolor="#111111", alpha=0.55, edgecolor="none"))
    ax_a.set_axis_off()
    
    # ---------------- PANEL B: Ground-Truth Segmentation ----------------
    ax_b.imshow(mri_norm, cmap="gray", interpolation="bilinear")
    ax_b.imshow(gt_rad, cmap=organ_cmap, norm=organ_norm, interpolation="nearest")
    ax_b.set_title("Panel B — Ground-Truth Segmentation", fontsize=11, fontweight="bold", pad=7)
    ax_b.set_axis_off()
    
    # ---------------- PANEL C: Swin-UMamba Prediction ----------------
    ax_c.imshow(mri_norm, cmap="gray", interpolation="bilinear")
    ax_c.imshow(pred_rad, cmap=organ_cmap, norm=organ_norm, interpolation="nearest")
    ax_c.set_title("Panel C — Swin-UMamba Prediction", fontsize=11, fontweight="bold", pad=7)
    ax_c.set_axis_off()
    
    # ---------------- PANEL D: Spatial Segmentation Error Map ----------------
    # Subtle MRI background (dimmed) so error regions pop with maximum anatomical context
    ax_d.imshow(mri_norm * 0.75, cmap="gray", interpolation="bilinear")
    ax_d.imshow(error_map, cmap=error_cmap, norm=error_norm, interpolation="nearest")
    ax_d.set_title("Panel D — Spatial Segmentation Error Map", fontsize=11, fontweight="bold", pad=7)
    ax_d.set_axis_off()
    
    # Add quantitative voxel error box inside Panel D
    err_text = (
        f"FN (Missed): {err_stats['fn_vox']:,} vox\n"
        f"FP (Spurious): {err_stats['fp_vox']:,} vox\n"
        f"Confusion: {err_stats['conf_vox']:,} vox\n"
        f"Concordant: {err_stats['correct_vox']:,} vox"
    )
    ax_d.text(0.03, 0.04, err_text, transform=ax_d.transAxes, fontsize=8,
              verticalalignment="bottom", horizontalalignment="left",
              color="#ffffff", fontfamily="monospace",
              bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1a1a", alpha=0.82, edgecolor="#555555"))

    # ---------------- LEGENDS (Organ Palette & Error Categories) ----------------
    # 1. Error Categories Legend (top row of bottom space)
    error_patches = [
        mpatches.Patch(facecolor=ERROR_COLORS[1][:3], edgecolor="#222222", label=ERROR_LABELS[1]),
        mpatches.Patch(facecolor=ERROR_COLORS[2][:3], edgecolor="#222222", label=ERROR_LABELS[2]),
        mpatches.Patch(facecolor=ERROR_COLORS[3][:3], edgecolor="#222222", label=ERROR_LABELS[3]),
        mpatches.Patch(facecolor=ERROR_COLORS[4][:3], edgecolor="#222222", label=ERROR_LABELS[4]),
    ]
    leg_err = fig.legend(
        handles=error_patches,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.11),
        ncol=4,
        frameon=True,
        facecolor="#f9f9f9",
        edgecolor="#cccccc",
        fontsize=9.2,
        title="Spatial Error Classification (Panel D):",
        title_fontsize=9.5,
    )
    leg_err.get_title().set_fontweight("bold")
    
    # 2. Organs Present in this slice Legend (bottom row)
    all_organs_in_case = sorted(list(set(gt_organs + pred_organs)))
    organ_patches = [
        mpatches.Patch(facecolor=ORGAN_COLORS[idx][:3], edgecolor="#333333", label=ORGAN_NAMES[idx])
        for idx in all_organs_in_case
    ]
    ncol_org = min(len(organ_patches), 7)
    leg_org = fig.legend(
        handles=organ_patches,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=ncol_org,
        frameon=True,
        facecolor="#f9f9f9",
        edgecolor="#cccccc",
        fontsize=8.5,
        title="Anatomical Organs Present in Slice (Panels B & C):",
        title_fontsize=9.0,
    )
    leg_org.get_title().set_fontweight("bold")
    
    # Save figure
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=dpi, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved: {save_path}")
    
    return err_stats


# -------------------------------------------------------------------------
# 3. Master Execution Routine
# -------------------------------------------------------------------------

def main():
    root = Path(__file__).resolve().parent.parent
    img_dir = root / "data" / "nnUNet_raw" / "Dataset702_AbdomenMR" / "imagesTs"
    gt_dir = root / "data" / "nnUNet_raw" / "Dataset702_AbdomenMR" / "labelsTs"
    pred_dir = root / "outputs" / "test_predictions_best"
    metrics_csv = root / "outputs" / "test_metrics_per_case.csv"
    out_dir = root / "outputs" / "heatmap_b_spatial_errors"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=== HEATMAP B: SPATIAL SEGMENTATION ERROR GENERATOR ===")
    print(f"Project root:    {root}")
    print(f"Image directory: {img_dir} (Exists: {img_dir.exists()})")
    print(f"GT directory:    {gt_dir} (Exists: {gt_dir.exists()})")
    print(f"Pred directory:  {pred_dir} (Exists: {pred_dir.exists()})")
    print(f"Output folder:   {out_dir}")
    
    # Load case dice
    df_metrics = pd.read_csv(metrics_csv)
    dice_lookup = dict(zip(df_metrics["case"], df_metrics["dice_foreground"]))
    
    # Build colormaps
    organ_cmap, organ_norm = build_discrete_cmap(ORGAN_COLORS)
    error_cmap, error_norm = build_discrete_cmap(ERROR_COLORS)
    
    # 6 Authoritative representative test cases spanning error modes & performances
    selected_cases = [
        {
            "case": "amos_7813.nii.gz",
            "slice": 45,
            "category": "Lowest Dice / Severe Under-segmentation",
            "description": "Lowest test case (Dice 0.6388). Multi-organ visceral slice with 10 GT organs where Swin-UMamba misses critical organ parenchyma (massive False Negatives)."
        },
        {
            "case": "amos_8178.nii.gz",
            "slice": 41,
            "category": "Low Dice / Spurious Organ & Small-Organ Loss",
            "description": "Second lowest case (Dice 0.6770). Shows significant False Positives (over-segmentation into surrounding fat) alongside missed small organs (adrenal/duodenum)."
        },
        {
            "case": "amos_7894.nii.gz",
            "slice": 16,
            "category": "Low-Mid Dice / Thick-Slice Visceral Confusion",
            "description": "Acquired with 6.6mm thick slices (Dice 0.7287). 11 organs present; demonstrates partial volume artifacts and inter-organ boundary confusion."
        },
        {
            "case": "amos_7438.nii.gz",
            "slice": 20,
            "category": "Moderate Dice / Ultra-High Resolution Boundary Confusion",
            "description": "1024x1024 matrix (Dice 0.8451). High voxel count displaying extensive inter-organ confusion (Amber) at adjacent liver-stomach and pancreas-duodenum borders."
        },
        {
            "case": "amos_7016.nii.gz",
            "slice": 37,
            "category": "High Dice / Rich Multi-Organ Cross-Section",
            "description": "Strong benchmark performer (Dice 0.9061). 11 organs present simultaneously with high anatomical fidelity and localized thin boundary errors."
        },
        {
            "case": "amos_7261.nii.gz",
            "slice": 38,
            "category": "Peak Performer / Exceptional Concordance",
            "description": "Highest-scoring test case (Dice 0.9776). Near-perfect multiclass concordance across 11 organs with minimal sub-voxel discrepancies."
        },
    ]
    
    summary_rows = []
    
    # Generate each 4-panel figure
    for item in selected_cases:
        case_name = item["case"]
        z_idx = item["slice"]
        case_dice = dice_lookup.get(case_name, np.nan)
        
        img_path = img_dir / f"{case_name.replace('.nii.gz', '')}_0000.nii.gz"
        gt_path = gt_dir / case_name
        pred_path = pred_dir / case_name
        
        print(f"\nProcessing {case_name} (z={z_idx})...")
        img_vol = np.asarray(nib.load(str(img_path)).dataobj)
        gt_vol = np.asarray(nib.load(str(gt_path)).dataobj)
        pred_vol = np.asarray(nib.load(str(pred_path)).dataobj)
        
        mri_slice = img_vol[:, :, z_idx]
        gt_slice = gt_vol[:, :, z_idx]
        pred_slice = pred_vol[:, :, z_idx]
        
        # Calculate slice Dice
        g_fg = gt_slice > 0
        p_fg = pred_slice > 0
        intersection = np.logical_and(g_fg, p_fg).sum()
        slice_dice = (2.0 * intersection) / (g_fg.sum() + p_fg.sum()) if (g_fg.sum() + p_fg.sum()) > 0 else 1.0
        
        fig_filename = out_dir / f"heatmap_b_{case_name.replace('.nii.gz', '')}.png"
        
        err_stats = render_four_panel_figure(
            mri_slice=mri_slice,
            gt_slice=gt_slice,
            pred_slice=pred_slice,
            case_name=case_name,
            slice_idx=z_idx,
            case_dice=case_dice,
            slice_dice=slice_dice,
            save_path=fig_filename,
            organ_cmap=organ_cmap,
            organ_norm=organ_norm,
            error_cmap=error_cmap,
            error_norm=error_norm,
            dpi=300
        )
        
        summary_rows.append({
            "case": case_name,
            "slice_z": z_idx,
            "category": item["category"],
            "case_dice": case_dice,
            "slice_dice": slice_dice,
            "gt_fg_voxels": err_stats["gt_fg_vox"],
            "pred_fg_voxels": err_stats["pred_fg_vox"],
            "correct_voxels": err_stats["correct_vox"],
            "fn_voxels": err_stats["fn_vox"],
            "fp_voxels": err_stats["fp_vox"],
            "confusion_voxels": err_stats["conf_vox"],
            "total_error_voxels": err_stats["fn_vox"] + err_stats["fp_vox"] + err_stats["conf_vox"],
            "figure_file": fig_filename.name,
            "description": item["description"],
        })
    
    # Save quantitative metadata table
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "heatmap_b_spatial_errors_metrics.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSaved quantitative summary metrics: {summary_csv}")
    
    # -------------------------------------------------------------------------
    # 4. Generate Combined Master Composite Figure (6x4 Matrix)
    # -------------------------------------------------------------------------
    print("\nGenerating Master Composite Figure across all 6 cases (6 x 4 matrix)...")
    comp_fig = plt.figure(figsize=(22, 28), dpi=300)
    gs_comp = GridSpec(6, 4, figure=comp_fig, left=0.06, right=0.98, bottom=0.08, top=0.935, wspace=0.04, hspace=0.18)
    
    comp_fig.suptitle(
        "Heatmap B: Spatial Segmentation Errors Across Representative Cases\n"
        "Swin-UMamba Multi-Organ Abdominal MRI Segmentation (Dataset702_AbdomenMR)",
        fontsize=18,
        fontweight="bold",
        y=0.978,
        color="#111111"
    )
    
    for row_idx, item in enumerate(selected_cases):
        case_name = item["case"]
        z_idx = item["slice"]
        case_dice = dice_lookup.get(case_name, np.nan)
        
        img_vol = np.asarray(nib.load(str(img_dir / f"{case_name.replace('.nii.gz', '')}_0000.nii.gz")).dataobj)
        gt_vol = np.asarray(nib.load(str(gt_dir / case_name)).dataobj)
        pred_vol = np.asarray(nib.load(str(pred_dir / case_name)).dataobj)
        
        mri_rad = reorient_to_radiological(img_vol[:, :, z_idx])
        gt_rad = reorient_to_radiological(gt_vol[:, :, z_idx])
        pred_rad = reorient_to_radiological(pred_vol[:, :, z_idx])
        
        mri_norm = normalize_mri(mri_rad)
        err_map, e_stats = compute_spatial_error_map(gt_rad, pred_rad)
        
        g_fg = gt_rad > 0
        p_fg = pred_rad > 0
        slice_dice = (2.0 * np.logical_and(g_fg, p_fg).sum()) / (g_fg.sum() + p_fg.sum()) if (g_fg.sum() + p_fg.sum()) > 0 else 1.0
        
        ax0 = comp_fig.add_subplot(gs_comp[row_idx, 0])
        ax1 = comp_fig.add_subplot(gs_comp[row_idx, 1])
        ax2 = comp_fig.add_subplot(gs_comp[row_idx, 2])
        ax3 = comp_fig.add_subplot(gs_comp[row_idx, 3])
        
        # Row label on left
        ax0.text(
            -0.12, 0.5,
            f"Case {case_name.replace('.nii.gz', '')}\n(z={z_idx})\n"
            f"Dice: {case_dice:.3f}\n"
            f"[{item['category'].split('/')[0].strip()}]",
            transform=ax0.transAxes,
            fontsize=9.5,
            fontweight="bold",
            va="center",
            ha="right",
            color="#222222"
        )
        
        # Col 0: MRI
        ax0.imshow(mri_norm, cmap="gray", interpolation="bilinear")
        if row_idx == 0:
            ax0.set_title("Panel A — Original MRI", fontsize=12, fontweight="bold", pad=8)
        ax0.set_axis_off()
        
        # Col 1: GT
        ax1.imshow(mri_norm, cmap="gray", interpolation="bilinear")
        ax1.imshow(gt_rad, cmap=organ_cmap, norm=organ_norm, interpolation="nearest")
        if row_idx == 0:
            ax1.set_title("Panel B — Ground Truth", fontsize=12, fontweight="bold", pad=8)
        ax1.set_axis_off()
        
        # Col 2: Prediction
        ax2.imshow(mri_norm, cmap="gray", interpolation="bilinear")
        ax2.imshow(pred_rad, cmap=organ_cmap, norm=organ_norm, interpolation="nearest")
        if row_idx == 0:
            ax2.set_title("Panel C — Swin-UMamba", fontsize=12, fontweight="bold", pad=8)
        ax2.set_axis_off()
        
        # Col 3: Error Map
        ax3.imshow(mri_norm * 0.75, cmap="gray", interpolation="bilinear")
        ax3.imshow(err_map, cmap=error_cmap, norm=error_norm, interpolation="nearest")
        if row_idx == 0:
            ax3.set_title("Panel D — Spatial Error Map", fontsize=12, fontweight="bold", pad=8)
        ax3.set_axis_off()
        
        # Annotate error metrics in small badge
        badge = f"FN:{e_stats['fn_vox']} | FP:{e_stats['fp_vox']} | Conf:{e_stats['conf_vox']}"
        ax3.text(0.97, 0.04, badge, transform=ax3.transAxes, fontsize=8,
                 ha="right", va="bottom", color="#ffffff", fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="#1a1a1a", alpha=0.8, edgecolor="none"))

    # Master Legends at bottom of composite
    error_patches = [
        mpatches.Patch(facecolor=ERROR_COLORS[1][:3], edgecolor="#222222", label="Correct Foreground Classification (TP)"),
        mpatches.Patch(facecolor=ERROR_COLORS[2][:3], edgecolor="#222222", label="False Negative (Missed Organ Parenchyma)"),
        mpatches.Patch(facecolor=ERROR_COLORS[3][:3], edgecolor="#222222", label="False Positive (Spurious Extraneous Organ)"),
        mpatches.Patch(facecolor=ERROR_COLORS[4][:3], edgecolor="#222222", label="Inter-Organ Label Confusion (Label Swap)"),
    ]
    leg_err = comp_fig.legend(
        handles=error_patches,
        loc="lower center",
        bbox_to_anchor=(0.52, 0.042),
        ncol=4,
        frameon=True,
        facecolor="#f9f9f9",
        edgecolor="#cccccc",
        fontsize=10.5,
        title="Spatial Segmentation Error Map Categories (Panel D):",
        title_fontsize=11.5,
    )
    leg_err.get_title().set_fontweight("bold")
    
    all_organ_patches = [
        mpatches.Patch(facecolor=ORGAN_COLORS[idx][:3], edgecolor="#333333", label=ORGAN_NAMES[idx])
        for idx in range(1, 14)
    ]
    leg_all_org = comp_fig.legend(
        handles=all_organ_patches,
        loc="lower center",
        bbox_to_anchor=(0.52, 0.008),
        ncol=7,
        frameon=True,
        facecolor="#f9f9f9",
        edgecolor="#cccccc",
        fontsize=9.2,
        title="Anatomical Organ Labels (Panels B & C):",
        title_fontsize=10.5,
    )
    leg_all_org.get_title().set_fontweight("bold")
    
    comp_save_path = out_dir / "heatmap_b_spatial_errors_composite.png"
    plt.savefig(comp_save_path, dpi=300, facecolor="white", edgecolor="none")
    plt.close(comp_fig)
    print(f"Saved Master Composite Figure: {comp_save_path}")
    print("\n=== ALL HEATMAP B VISUALIZATIONS GENERATED SUCCESSFULLY ===")


if __name__ == "__main__":
    main()
