"""
Model Focus Heatmap Generator for Swin-UMamba
Explainability & Attribution Analysis for Abdominal MRI Segmentation (AMOS 2022 Benchmark)

Implements class-specific Grad-CAM, Grad-CAM++, and Activation Heatmaps for Swin-UMamba,
visualizing where the trained deep neural network focuses when predicting specific abdominal organs.

Authors: Antigravity AI & Senior Medical Imaging Research Engineer
Dataset: Dataset702_AbdomenMR (AMOS 2022 Challenge)
"""

import os
import sys
from pathlib import Path
import time
import json
import argparse
from typing import Dict, Tuple, Optional, Union, List

import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from scipy.ndimage import binary_dilation

# swin_umamba path setup
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "swin_umamba"))


# -------------------------------------------------------------------------
# 1. Anatomical Taxonomy & Visual Palettes
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

# Synonyms and short names for CLI convenience
ORGAN_ALIASES = {
    "liver": 1,
    "rk": 2,
    "right_kidney": 2,
    "right kidney": 2,
    "spleen": 3,
    "pancreas": 4,
    "panc": 4,
    "aorta": 5,
    "ivc": 6,
    "inferior_vena_cava": 6,
    "inferior vena cava": 6,
    "rag": 7,
    "right_adrenal_gland": 7,
    "right adrenal gland": 7,
    "right adrenal": 7,
    "lag": 8,
    "left_adrenal_gland": 8,
    "left adrenal gland": 8,
    "left adrenal": 8,
    "gallbladder": 9,
    "gb": 9,
    "esophagus": 10,
    "eso": 10,
    "stomach": 11,
    "stom": 11,
    "duodenum": 12,
    "duo": 12,
    "lk": 13,
    "left_kidney": 13,
    "left kidney": 13,
}

# Authoritative multi-organ color palette for 13 abdominal organs
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


def parse_organ_id(organ_input: Union[int, str]) -> Tuple[int, str]:
    """Resolves organ name or ID to (organ_id, organ_display_name)."""
    if isinstance(organ_input, int):
        if organ_input in ORGAN_NAMES:
            return organ_input, ORGAN_NAMES[organ_input]
        raise ValueError(f"Unknown organ ID {organ_input}. Expected 1 to 13.")
    
    clean_str = str(organ_input).strip().lower().replace("-", "_")
    if clean_str.isdigit():
        oid = int(clean_str)
        if oid in ORGAN_NAMES:
            return oid, ORGAN_NAMES[oid]
        raise ValueError(f"Unknown organ ID {oid}. Expected 1 to 13.")
        
    if clean_str in ORGAN_ALIASES:
        oid = ORGAN_ALIASES[clean_str]
        return oid, ORGAN_NAMES[oid]
        
    for oid, name in ORGAN_NAMES.items():
        if clean_str in name.lower():
            return oid, name
            
    raise ValueError(f"Could not resolve target organ from '{organ_input}'. Available organs: {list(ORGAN_NAMES.values())}")


def build_discrete_cmap(color_dict: Dict[int, tuple]) -> Tuple[ListedColormap, BoundaryNorm]:
    """Builds a Matplotlib ListedColormap and BoundaryNorm for integer label volumes."""
    max_label = max(color_dict.keys())
    colors = [color_dict.get(i, (0.0, 0.0, 0.0, 0.0)) for i in range(max_label + 1)]
    cmap = ListedColormap(colors)
    bounds = np.arange(max_label + 2) - 0.5
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def build_heat_overlay_cmap(base_cmap_name: str = "turbo") -> LinearSegmentedColormap:
    """
    Constructs a custom transparency-modulated colormap based on 'turbo' or 'jet'.
    Near zero activation, alpha is 0.0 (transparent, showing clear MRI anatomy).
    Ramps smoothly to alpha 0.70 at peak activation.
    """
    base_cmap = plt.get_cmap(base_cmap_name)
    n_bins = 256
    colors = base_cmap(np.linspace(0, 1, n_bins))
    # Modulate alpha: lower 15% is completely transparent, then ramps smoothly
    alphas = np.zeros(n_bins)
    alphas[int(n_bins * 0.12):] = np.linspace(0.15, 0.72, n_bins - int(n_bins * 0.12)) ** 0.85
    colors[:, 3] = alphas
    return LinearSegmentedColormap.from_list(f"{base_cmap_name}_alpha", colors)


# -------------------------------------------------------------------------
# 2. Image Geometry & Normalization Utilities
# -------------------------------------------------------------------------

def reorient_to_radiological(slice_2d: np.ndarray) -> np.ndarray:
    """
    Transforms AMOS 2D axial slice array to standard radiological orientation:
    - Anterior on top, Posterior on bottom
    - Patient's Right on viewer's Left, Patient's Left on viewer's Right
    Formula verified against anatomical landmarks: slice.T[::-1, ::-1]
    """
    return slice_2d.T[::-1, ::-1]


def normalize_mri(slice_mri: np.ndarray) -> np.ndarray:
    """Robust percentile-based intensity normalization [0.0, 1.0] for MRI visualization."""
    p_low, p_high = np.percentile(slice_mri, (0.5, 99.5))
    if p_high > p_low:
        clipped = np.clip(slice_mri, p_low, p_high)
        norm = (clipped - p_low) / (p_high - p_low)
    else:
        norm = np.zeros_like(slice_mri)
    return norm


def extract_organ_boundary(mask_slice: np.ndarray, organ_id: int) -> np.ndarray:
    """Extracts a 1-pixel boundary contour for the target organ."""
    binary = (mask_slice == organ_id).astype(bool)
    if not np.any(binary):
        return np.zeros_like(binary, dtype=bool)
    dilated = binary_dilation(binary, iterations=1)
    boundary = np.logical_and(dilated, ~binary)
    return boundary


# -------------------------------------------------------------------------
# 3. Model Loading & Feature Extraction Engine
# -------------------------------------------------------------------------

def load_trained_swin_umamba(
    checkpoint_path: Union[str, Path],
    device: torch.device
):
    """Loads the trained Swin-UMamba model with checkpoint weights."""
    from nnunetv2.nets.SwinUMamba import SwinUMamba
    model = SwinUMamba(
        in_chans=1,
        out_chans=14,
        feat_size=[48, 96, 192, 384, 768],
        deep_supervision=False,
        hidden_size=768
    ).to(device)
    
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    weights = ckpt['network_weights'] if 'network_weights' in ckpt else ckpt
    model.load_state_dict(weights)
    model.eval()
    return model


def compute_model_focus_heatmap(
    model: nn.Module,
    slice_mri: np.ndarray,
    target_organ_id: int,
    method: str = "gradcam",
    layer_name: str = "decoder1",
    score_mode: str = "pred_mask",
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> Dict[str, Union[np.ndarray, float, int, str]]:
    """
    Computes class-specific model focus heatmap (Grad-CAM, Grad-CAM++, or Activation).

    Parameters:
    -----------
    model: SwinUMamba network
    slice_mri: 2D numpy array (original MRI axial slice, shape: H x W)
    target_organ_id: Integer class index (1 to 13)
    method: 'gradcam', 'gradcam++', or 'activation'
    layer_name: Feature layer to interrogate ('decoder1', 'decoder2', 'decoder3')
    score_mode: 'pred_mask' (target logits within predicted organ mask) or 'all' (entire slice logit sum)
    device: torch.device

    Returns:
    --------
    Dictionary with normalized activation map, raw min/max, peak coordinates, and logits.
    """
    H_orig, W_orig = slice_mri.shape
    
    # Preprocessing matching nnU-Net: Z-score normalization & pad to 320x320
    mean, std = slice_mri.mean(), slice_mri.std()
    norm_slice = (slice_mri - mean) / max(std, 1e-8)
    
    pad_h = max(0, 320 - H_orig)
    pad_w = max(0, 320 - W_orig)
    padded = np.pad(norm_slice, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
    
    x = torch.from_numpy(padded).float().unsqueeze(0).unsqueeze(0).to(device)
    
    # Stage 1: Fast forward pass through encoder without grad
    with torch.no_grad():
        x1 = model.stem(x)
        vss_outs = model.vssm_encoder(x1)
        enc1 = model.encoder1(x)
        enc2 = model.encoder2(vss_outs[0])
        enc3 = model.encoder3(vss_outs[1])
        enc4 = model.encoder4(vss_outs[2])
        enc5 = model.encoder5(vss_outs[3])
        enc_hidden = vss_outs[4]
        dec4 = model.decoder6(enc_hidden, enc5)
        dec3 = model.decoder5(dec4, enc4)
        dec2 = model.decoder4(dec3, enc3)
        dec1 = model.decoder3(dec2, enc2)
        dec0 = model.decoder2(dec1, enc1)
    
    # Select intermediate feature map to probe
    if layer_name == "decoder1":
        # Target dec0 (shape: 1, 48, 320, 320) which is input to decoder1
        probe_feat = dec0.detach().clone().requires_grad_(True)
        dec_out = model.decoder1(probe_feat)
        logits = model.out_layers[0](dec_out)
    elif layer_name == "decoder2":
        # Target dec1 (shape: 1, 48, 160, 160) which is input to decoder2
        probe_feat = dec1.detach().clone().requires_grad_(True)
        dec0_dyn = model.decoder2(probe_feat, enc1)
        dec_out = model.decoder1(dec0_dyn)
        logits = model.out_layers[0](dec_out)
    else:
        # Default fallback to decoder1 input
        probe_feat = dec0.detach().clone().requires_grad_(True)
        dec_out = model.decoder1(probe_feat)
        logits = model.out_layers[0](dec_out)

    # Segmentation prediction on padded tensor
    pred_labels_padded = torch.argmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
    pred_labels = pred_labels_padded[:H_orig, :W_orig]
    
    # Stage 2: Class-specific Target Score Definition
    organ_logits = logits[0, target_organ_id]
    
    if score_mode == "pred_mask":
        pred_mask = (torch.argmax(logits, dim=1) == target_organ_id)
        if pred_mask.sum() > 0:
            target_score = organ_logits[pred_mask[0]].sum()
        else:
            # If organ not predicted, target top 2% highest logits for that organ
            flat_logits = organ_logits.view(-1)
            k_top = max(1, int(0.02 * flat_logits.numel()))
            topk_vals, _ = torch.topk(flat_logits, k_top)
            target_score = topk_vals.sum()
    else:
        # Global slice sum
        target_score = organ_logits.sum()

    # Stage 3: Attribution Calculation
    model.zero_grad()
    
    if method == "activation":
        # Pure feature activation magnitude (no gradient backward needed)
        cam = torch.mean(torch.abs(probe_feat), dim=1, keepdim=True)
    elif method == "gradcam++":
        # Grad-CAM++: Higher-order weighting
        target_score.backward(retain_graph=True)
        grad = probe_feat.grad
        
        # Second and third order approximations
        grad_2 = grad ** 2
        grad_3 = grad_2 * grad
        sum_feat = torch.sum(probe_feat, dim=(2, 3), keepdim=True)
        denom = 2.0 * grad_2 + sum_feat * grad_3
        denom = torch.where(denom != 0.0, denom, torch.ones_like(denom))
        alpha = grad_2 / denom
        weights = torch.sum(alpha * torch.relu(grad), dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * probe_feat, dim=1, keepdim=True)
        cam = torch.relu(cam)
    else:
        # Standard Grad-CAM
        target_score.backward()
        grad = probe_feat.grad
        weights = torch.mean(grad, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * probe_feat, dim=1, keepdim=True)
        cam = torch.relu(cam)

    # Upsample if probed layer was at lower resolution
    if cam.shape[2:] != (320, 320):
        cam = F.interpolate(cam, size=(320, 320), mode="bilinear", align_corners=False)

    cam_np = cam.squeeze().detach().cpu().numpy()
    
    # Unpad to original slice resolution H_orig x W_orig
    cam_unpad = cam_np[:H_orig, :W_orig]
    
    # Robust normalization to [0.0, 1.0]
    c_min = float(cam_unpad.min())
    c_max = float(cam_unpad.max())
    if c_max > c_min:
        cam_norm = (cam_unpad - c_min) / (c_max - c_min)
    else:
        cam_norm = np.zeros_like(cam_unpad)
        
    peak_y, peak_x = np.unravel_index(np.argmax(cam_norm), cam_norm.shape)

    return {
        "activation_raw": cam_unpad,
        "activation_norm": cam_norm,
        "predicted_slice": pred_labels,
        "min_raw": c_min,
        "max_raw": c_max,
        "peak_y": int(peak_y),
        "peak_x": int(peak_x),
        "method": method,
        "layer": layer_name,
        "score_mode": score_mode,
        "organ_id": target_organ_id,
        "organ_name": ORGAN_NAMES[target_organ_id]
    }


# -------------------------------------------------------------------------
# 4. Publication-Grade Visualization Renderer
# -------------------------------------------------------------------------

def render_model_focus_figures(
    slice_mri: np.ndarray,
    gt_slice: Optional[np.ndarray],
    pred_slice: np.ndarray,
    cam_norm: np.ndarray,
    case_id: str,
    slice_idx: int,
    target_organ_id: int,
    target_organ_name: str,
    method: str,
    layer_name: str,
    output_dir: Path,
    dpi: int = 300
) -> Tuple[Path, Path]:
    """
    Renders and saves:
    1. Standalone single-panel high-res heatmap with organ contour overlay and colorbar.
    2. Comprehensive 4-panel publication-grade figure (A: MRI, B: Focus Heatmap, C: GT, D: Pred + Focus).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Transform all 2D slices to authoritative radiological orientation
    mri_rad = reorient_to_radiological(slice_mri)
    cam_rad = reorient_to_radiological(cam_norm)
    pred_rad = reorient_to_radiological(pred_slice)
    gt_rad = reorient_to_radiological(gt_slice) if gt_slice is not None else None
    
    mri_norm = normalize_mri(mri_rad)
    organ_cmap, organ_norm = build_discrete_cmap(ORGAN_COLORS)
    heat_cmap = build_heat_overlay_cmap("turbo")
    
    # Target organ boundary for crisp contour overlays
    target_gt_boundary = extract_organ_boundary(gt_rad, target_organ_id) if gt_rad is not None else None
    target_pred_boundary = extract_organ_boundary(pred_rad, target_organ_id)
    active_boundary = target_gt_boundary if (target_gt_boundary is not None and np.any(target_gt_boundary)) else target_pred_boundary

    clean_case = case_id.replace(".nii.gz", "").replace("_0000", "")
    safe_organ_name = target_organ_name.lower().replace(" ", "_")
    
    # ---------------------------------------------------------------------
    # Figure 1: Standalone Single-Panel Focus Heatmap
    # ---------------------------------------------------------------------
    single_fig_path = output_dir / f"{clean_case}_z{slice_idx}_{safe_organ_name}_heatmap.png"
    fig_single, ax_s = plt.subplots(figsize=(8.0, 7.5), dpi=dpi)
    
    # Base MRI
    ax_s.imshow(mri_norm, cmap="gray", interpolation="bilinear")
    
    # Overlaid focus heatmap with smooth opacity ramp
    im_heat = ax_s.imshow(cam_rad, cmap=heat_cmap, interpolation="bilinear", vmin=0.0, vmax=1.0)
    
    # Target organ contour in neon orange with dark drop shadow (like reference image)
    if np.any(active_boundary):
        ax_s.contour(active_boundary, levels=[0.5], colors=["#000000"], linewidths=2.8, alpha=0.9)
        ax_s.contour(active_boundary, levels=[0.5], colors=["#FF4500"], linewidths=1.8, alpha=1.0)

    # Mark peak activation hotspot
    peak_y_rad, peak_x_rad = np.unravel_index(np.argmax(cam_rad), cam_rad.shape)
    ax_s.plot(peak_x_rad, peak_y_rad, marker="x", color="#FFFFFF", markersize=10, markeredgewidth=2.2, alpha=0.95)
    ax_s.plot(peak_x_rad, peak_y_rad, marker="+", color="#FF0000", markersize=10, markeredgewidth=1.8, alpha=0.95)

    ax_s.set_title(
        f"Swin-UMamba Focus Heatmap — {target_organ_name}\nCase {clean_case} | Slice z={slice_idx} | Method: {method.upper()} ({layer_name})",
        fontsize=12, fontweight="bold", pad=10
    )
    ax_s.set_axis_off()

    # Radiological directional markers
    ax_s.text(0.5, 0.97, "Anterior (A)", transform=ax_s.transAxes, color="#F0F0F0",
              fontsize=9, ha="center", va="top", fontweight="semibold",
              bbox=dict(boxstyle="round,pad=0.25", facecolor="#111111", alpha=0.6, edgecolor="none"))
    ax_s.text(0.5, 0.03, "Posterior (P)", transform=ax_s.transAxes, color="#F0F0F0",
              fontsize=9, ha="center", va="bottom", fontweight="semibold",
              bbox=dict(boxstyle="round,pad=0.25", facecolor="#111111", alpha=0.6, edgecolor="none"))
    ax_s.text(0.03, 0.5, "Right (R)", transform=ax_s.transAxes, color="#F0F0F0",
              fontsize=9, ha="left", va="center", fontweight="semibold",
              bbox=dict(boxstyle="round,pad=0.25", facecolor="#111111", alpha=0.6, edgecolor="none"))
    ax_s.text(0.97, 0.5, "Left (L)", transform=ax_s.transAxes, color="#F0F0F0",
              fontsize=9, ha="right", va="center", fontweight="semibold",
              bbox=dict(boxstyle="round,pad=0.25", facecolor="#111111", alpha=0.6, edgecolor="none"))

    # Dedicated horizontal colorbar below
    cbar_ax = fig_single.add_axes([0.18, 0.07, 0.64, 0.032])
    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap("turbo"), norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    cbar = fig_single.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Relative Activation (Normalized Model Focus)", fontsize=9.5, fontweight="semibold")
    cbar.set_ticks([0.0, 0.25, 0.50, 0.75, 1.0])
    cbar.set_ticklabels(["0.0 (Min)", "0.25", "0.50", "0.75", "1.0 (Peak)"])
    cbar.ax.tick_params(labelsize=8)

    plt.subplots_adjust(bottom=0.14, top=0.91)
    fig_single.savefig(single_fig_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig_single)

    # ---------------------------------------------------------------------
    # Figure 2: Master Four-Panel Research Visualization
    # ---------------------------------------------------------------------
    four_panel_path = output_dir / f"{clean_case}_z{slice_idx}_{safe_organ_name}_four_panel.png"
    fig_four = plt.figure(figsize=(20, 6.8), dpi=dpi)
    gs = GridSpec(1, 4, figure=fig_four, left=0.03, right=0.97, bottom=0.15, top=0.82, wspace=0.06)
    
    ax_a = fig_four.add_subplot(gs[0, 0])
    ax_b = fig_four.add_subplot(gs[0, 1])
    ax_c = fig_four.add_subplot(gs[0, 2])
    ax_d = fig_four.add_subplot(gs[0, 3])

    fig_four.suptitle(
        f"Model Focus Attribution — Swin-UMamba on Dataset702_AbdomenMR (Case {clean_case}, Slice z={slice_idx})\n"
        f"Target Organ: {target_organ_name} (Class ID: {target_organ_id})  |  Method: {method.upper()}  |  Layer: {layer_name}",
        fontsize=13.5, fontweight="bold", y=0.95, color="#111111"
    )

    # Panel A — Original MRI
    ax_a.imshow(mri_norm, cmap="gray", interpolation="bilinear")
    ax_a.set_title("Panel A — Original MRI", fontsize=11, fontweight="bold", pad=8)
    ax_a.text(0.5, 0.96, "A", transform=ax_a.transAxes, color="#f0f0f0", fontsize=8.5, ha="center", va="top",
              bbox=dict(boxstyle="round,pad=0.2", facecolor="#111111", alpha=0.55, edgecolor="none"))
    ax_a.text(0.5, 0.04, "P", transform=ax_a.transAxes, color="#f0f0f0", fontsize=8.5, ha="center", va="bottom",
              bbox=dict(boxstyle="round,pad=0.2", facecolor="#111111", alpha=0.55, edgecolor="none"))
    ax_a.text(0.04, 0.5, "R", transform=ax_a.transAxes, color="#f0f0f0", fontsize=9, ha="left", va="center",
              bbox=dict(boxstyle="round,pad=0.2", facecolor="#111111", alpha=0.55, edgecolor="none"))
    ax_a.text(0.96, 0.5, "L", transform=ax_a.transAxes, color="#f0f0f0", fontsize=9, ha="right", va="center",
              bbox=dict(boxstyle="round,pad=0.2", facecolor="#111111", alpha=0.55, edgecolor="none"))
    ax_a.set_axis_off()

    # Panel B — Model Focus Heatmap
    ax_b.imshow(mri_norm, cmap="gray", interpolation="bilinear")
    ax_b.imshow(cam_rad, cmap=heat_cmap, interpolation="bilinear", vmin=0.0, vmax=1.0)
    if np.any(active_boundary):
        ax_b.contour(active_boundary, levels=[0.5], colors=["#000000"], linewidths=2.4, alpha=0.85)
        ax_b.contour(active_boundary, levels=[0.5], colors=["#FF4500"], linewidths=1.5, alpha=1.0)
    ax_b.set_title(f"Panel B — Focus Heatmap ({target_organ_name})", fontsize=11, fontweight="bold", pad=8)
    ax_b.set_axis_off()

    # Panel C — Ground-Truth Segmentation
    ax_c.imshow(mri_norm, cmap="gray", interpolation="bilinear")
    if gt_rad is not None:
        ax_c.imshow(gt_rad, cmap=organ_cmap, norm=organ_norm, interpolation="nearest")
        # Target organ outline
        if np.any(target_gt_boundary):
            ax_c.contour(target_gt_boundary, levels=[0.5], colors=["#FFFFFF"], linewidths=1.8, alpha=0.95)
        ax_c.set_title(f"Panel C — Ground Truth (13 Organs)", fontsize=11, fontweight="bold", pad=8)
    else:
        ax_c.set_title("Panel C — Ground Truth (Unavailable)", fontsize=11, fontweight="bold", pad=8)
    ax_c.set_axis_off()

    # Panel D — Prediction & Explanation Overlay
    ax_d.imshow(mri_norm, cmap="gray", interpolation="bilinear")
    ax_d.imshow(pred_rad, cmap=organ_cmap, norm=organ_norm, interpolation="nearest")
    # Outline target organ prediction
    if np.any(target_pred_boundary):
        ax_d.contour(target_pred_boundary, levels=[0.5], colors=["#FF4500"], linewidths=2.0, alpha=1.0)
    # Overlay high activation contour (>0.7 peak focus)
    high_act_mask = (cam_rad >= 0.70)
    if np.any(high_act_mask):
        ax_d.contour(high_act_mask, levels=[0.5], colors=["#FFFF00"], linewidths=1.6, linestyles="dashed", alpha=0.95)
    ax_d.set_title(f"Panel D — Prediction & Focus Alignment", fontsize=11, fontweight="bold", pad=8)
    ax_d.set_axis_off()

    # Shared Legend and Colorbar row below
    # Organ legend patches for present organs
    present_organs = []
    if gt_rad is not None:
        present_organs.extend(list(np.unique(gt_rad[gt_rad > 0])))
    present_organs.extend(list(np.unique(pred_rad[pred_rad > 0])))
    unique_present = sorted(list(set(present_organs)))

    legend_patches = []
    for oid in unique_present:
        rgba = ORGAN_COLORS.get(oid, (0.5, 0.5, 0.5, 0.85))
        is_target = (oid == target_organ_id)
        label_text = f"★ {ORGAN_NAMES[oid]}" if is_target else ORGAN_NAMES[oid]
        patch = mpatches.Patch(
            facecolor=rgba,
            edgecolor="#FF4500" if is_target else "#333333",
            linewidth=2.0 if is_target else 0.8,
            label=label_text
        )
        legend_patches.append(patch)

    if legend_patches:
        fig_four.legend(
            handles=legend_patches,
            loc="lower center",
            bbox_to_anchor=(0.38, 0.03),
            ncol=min(7, len(legend_patches)),
            fontsize=8.5,
            frameon=True,
            facecolor="#FAFAFA",
            edgecolor="#DDDDDD",
            title="Abdominal Organ Classes Present (★ Target Organ)"
        )

    # Colorbar on the right side of the legend
    cbar_ax4 = fig_four.add_axes([0.76, 0.05, 0.18, 0.032])
    sm4 = plt.cm.ScalarMappable(cmap=plt.get_cmap("turbo"), norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm4.set_array([])
    cbar4 = fig_four.colorbar(sm4, cax=cbar_ax4, orientation="horizontal")
    cbar4.set_label("Relative Activation", fontsize=8.5, fontweight="semibold")
    cbar4.set_ticks([0.0, 0.5, 1.0])
    cbar4.set_ticklabels(["0.0 (Min)", "0.5", "1.0 (Peak)"])
    cbar4.ax.tick_params(labelsize=7.5)

    fig_four.savefig(four_panel_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig_four)

    return single_fig_path, four_panel_path


# -------------------------------------------------------------------------
# 5. Main CLI Entry Point
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-quality Model Focus Heatmaps for Swin-UMamba on AMOS MRI."
    )
    parser.add_argument(
        "--case", "--case_id", dest="case_id", type=str, default="amos_7261",
        help="Test case identifier (e.g. amos_7261, amos_7016, amos_7813)"
    )
    parser.add_argument(
        "--slice", "--slice_idx", dest="slice_idx", type=int, default=38,
        help="Axial slice index (z-coordinate, e.g. 38)"
    )
    parser.add_argument(
        "--organ", "--target_organ", dest="target_organ", type=str, default="liver",
        help="Target organ name or integer ID (e.g. liver, pancreas, spleen, right_adrenal_gland, duodenum, 1..13)"
    )
    parser.add_argument(
        "--method", type=str, choices=["gradcam", "gradcam++", "activation"], default="gradcam",
        help="Attribution method: 'gradcam', 'gradcam++', or 'activation'"
    )
    parser.add_argument(
        "--layer", type=str, choices=["decoder1", "decoder2"], default="decoder1",
        help="Target intermediate feature layer: 'decoder1' (high-res 320x320) or 'decoder2' (160x160)"
    )
    parser.add_argument(
        "--score_mode", type=str, choices=["pred_mask", "all"], default="pred_mask",
        help="Target logit aggregation: 'pred_mask' (within predicted organ region) or 'all' (entire slice)"
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default=str(ROOT_DIR / "data/nnUNet_results/Dataset702_AbdomenMR/nnUNetTrainerSwinUMambaScratch__nnUNetPlans__2d/fold_0/checkpoint_best.pth"),
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=str(ROOT_DIR / "outputs/model_focus_heatmaps"),
        help="Output directory to save figures, numpy arrays, and metadata"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Compute device: 'auto', 'cuda', or 'cpu'"
    )
    parser.add_argument(
        "--dpi", type=int, default=300,
        help="Resolution DPI for exported PNG figures"
    )

    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 78)
    print("Swin-UMamba Model Focus Heatmap Generator")
    print(f"Device: {device}")
    print(f"Case: {args.case_id} | Slice z={args.slice_idx}")
    print("=" * 78)

    # 1. Resolve organ
    organ_id, organ_name = parse_organ_id(args.target_organ)
    print(f"Target Organ: {organ_name} (Class ID: {organ_id})")

    # 2. Paths validation
    clean_case = args.case_id.replace(".nii.gz", "").replace("_0000", "")
    img_path = ROOT_DIR / f"data/nnUNet_raw/Dataset702_AbdomenMR/imagesTs/{clean_case}_0000.nii.gz"
    gt_path = ROOT_DIR / f"data/nnUNet_raw/Dataset702_AbdomenMR/labelsTs/{clean_case}.nii.gz"
    pred_path = ROOT_DIR / f"outputs/test_predictions_best/{clean_case}.nii.gz"
    ckpt_path = Path(args.checkpoint)

    if not img_path.exists():
        raise FileNotFoundError(f"Input image not found: {img_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # 3. Load slice data directly
    t0 = time.time()
    img_nii = nib.load(str(img_path))
    z_max = img_nii.shape[2] - 1
    if not (0 <= args.slice_idx <= z_max):
        raise ValueError(f"Slice index {args.slice_idx} out of bounds [0, {z_max}] for case {clean_case}.")

    slice_mri = np.asanyarray(img_nii.dataobj[:, :, args.slice_idx]).astype(np.float32)
    
    gt_slice = None
    if gt_path.exists():
        gt_nii = nib.load(str(gt_path))
        gt_slice = np.asanyarray(gt_nii.dataobj[:, :, args.slice_idx]).astype(np.uint8)

    pred_slice_official = None
    if pred_path.exists():
        pred_nii = nib.load(str(pred_path))
        pred_slice_official = np.asanyarray(pred_nii.dataobj[:, :, args.slice_idx]).astype(np.uint8)

    print(f"Loaded slice {args.slice_idx} in {time.time()-t0:.2f}s (Shape: {slice_mri.shape})")

    # 4. Load trained model
    print(f"Loading Swin-UMamba checkpoint from:\n  {ckpt_path.relative_to(ROOT_DIR) if ckpt_path.is_relative_to(ROOT_DIR) else ckpt_path}")
    t0 = time.time()
    model = load_trained_swin_umamba(ckpt_path, device)
    print(f"Model loaded and verified in {time.time()-t0:.2f}s")

    # 5. Compute attribution heatmap
    print(f"Computing {args.method.upper()} attribution on layer '{args.layer}' (score mode: '{args.score_mode}')...")
    t0 = time.time()
    res = compute_model_focus_heatmap(
        model=model,
        slice_mri=slice_mri,
        target_organ_id=organ_id,
        method=args.method,
        layer_name=args.layer,
        score_mode=args.score_mode,
        device=device
    )
    print(f"Attribution computed in {time.time()-t0:.2f}s!")
    print(f"Raw Activation — Min: {res['min_raw']:.4e}, Max: {res['max_raw']:.4e}")
    print(f"Peak Focus Coordinates: (y={res['peak_y']}, x={res['peak_x']})")

    # Use official prediction volume if available, otherwise newly computed prediction
    pred_to_plot = pred_slice_official if pred_slice_official is not None else res["predicted_slice"]

    # 6. Render figures
    output_dir = Path(args.output_dir)
    print(f"Rendering publication-quality visualizations (DPI={args.dpi})...")
    single_path, four_path = render_model_focus_figures(
        slice_mri=slice_mri,
        gt_slice=gt_slice,
        pred_slice=pred_to_plot,
        cam_norm=res["activation_norm"],
        case_id=clean_case,
        slice_idx=args.slice_idx,
        target_organ_id=organ_id,
        target_organ_name=organ_name,
        method=args.method,
        layer_name=args.layer,
        output_dir=output_dir,
        dpi=args.dpi
    )

    # 7. Save raw activation array as .npy
    safe_organ = organ_name.lower().replace(" ", "_")
    npy_path = output_dir / f"{clean_case}_z{args.slice_idx}_{safe_organ}_activation.npy"
    np.save(str(npy_path), res["activation_norm"])

    # 8. Save metadata JSON
    meta_path = output_dir / f"{clean_case}_z{args.slice_idx}_{safe_organ}_metadata.json"
    metadata = {
        "case_id": clean_case,
        "slice_index": args.slice_idx,
        "target_organ_id": organ_id,
        "target_organ_name": organ_name,
        "attribution_method": args.method,
        "target_layer": args.layer,
        "target_score_mode": args.score_mode,
        "checkpoint_path": str(ckpt_path),
        "raw_min": res["min_raw"],
        "raw_max": res["max_raw"],
        "peak_coordinates": [res["peak_y"], res["peak_x"]],
        "mean_normalized_activation": float(res["activation_norm"].mean()),
        "normalization": "min_max_0_to_1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": {
            "single_panel_heatmap": single_path.name,
            "four_panel_figure": four_path.name,
            "raw_activation_npy": npy_path.name
        }
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print("=" * 78)
    print("OUTPUTS SAVED SUCCESSFULLY:")
    print(f"  Single Heatmap : {single_path}")
    print(f"  Four-Panel     : {four_path}")
    print(f"  Raw Array (.npy): {npy_path}")
    print(f"  Metadata (.json): {meta_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
