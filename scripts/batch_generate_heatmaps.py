"""
Batch Model Focus Heatmap Generator
Evaluates multiple abdominal organs across representative test cases
to produce a comprehensive scientific comparison suite.
"""

import sys
import time
from pathlib import Path
import numpy as np
import nibabel as nib
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "swin_umamba"))
sys.path.insert(0, str(ROOT_DIR / "scripts"))

from generate_model_focus_heatmap import (
    ORGAN_NAMES,
    parse_organ_id,
    load_trained_swin_umamba,
    compute_model_focus_heatmap,
    render_model_focus_figures
)

def run_batch():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Batch Heatmap Generator running on: {device}", flush=True)

    ckpt_path = ROOT_DIR / "data/nnUNet_results/Dataset702_AbdomenMR/nnUNetTrainerSwinUMambaScratch__nnUNetPlans__2d/fold_0/checkpoint_best.pth"
    output_dir = ROOT_DIR / "outputs/model_focus_heatmaps"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Swin-UMamba model checkpoint once...", flush=True)
    t0 = time.time()
    model = load_trained_swin_umamba(ckpt_path, device)
    print(f"Model ready in {time.time()-t0:.2f}s!", flush=True)

    # Suite of test cases and organs to generate:
    # amos_7261 (slice 38): Liver (1), Spleen (3), Pancreas (4), Right Adrenal Gland (7), Duodenum (12), Left Kidney (13)
    # amos_7016 (slice 37): Spleen (3), Liver (1), Pancreas (4)
    tasks = [
        {"case": "amos_7261", "slice": 38, "organs": [1, 3, 4, 7, 12, 13], "method": "gradcam"},
        {"case": "amos_7261", "slice": 38, "organs": [4, 7], "method": "gradcam++"}, # Compare Grad-CAM++ on small organs
        {"case": "amos_7016", "slice": 37, "organs": [1, 3, 4], "method": "gradcam"},
    ]

    for task in tasks:
        clean_case = task["case"]
        slice_idx = task["slice"]
        method = task["method"]

        img_path = ROOT_DIR / f"data/nnUNet_raw/Dataset702_AbdomenMR/imagesTs/{clean_case}_0000.nii.gz"
        gt_path = ROOT_DIR / f"data/nnUNet_raw/Dataset702_AbdomenMR/labelsTs/{clean_case}.nii.gz"
        pred_path = ROOT_DIR / f"outputs/test_predictions_best/{clean_case}.nii.gz"

        img_nii = nib.load(str(img_path))
        slice_mri = np.asanyarray(img_nii.dataobj[:, :, slice_idx]).astype(np.float32)

        gt_slice = None
        if gt_path.exists():
            gt_nii = nib.load(str(gt_path))
            gt_slice = np.asanyarray(gt_nii.dataobj[:, :, slice_idx]).astype(np.uint8)

        pred_slice = None
        if pred_path.exists():
            pred_nii = nib.load(str(pred_path))
            pred_slice = np.asanyarray(pred_nii.dataobj[:, :, slice_idx]).astype(np.uint8)

        print(f"\nProcessing Case {clean_case} (Slice z={slice_idx}, Method: {method.upper()})...", flush=True)

        for organ_id in task["organs"]:
            organ_name = ORGAN_NAMES[organ_id]
            print(f"  --> Target Organ: {organ_name:22s} (ID {organ_id:2d})...", end="", flush=True)
            t_org = time.time()

            res = compute_model_focus_heatmap(
                model=model,
                slice_mri=slice_mri,
                target_organ_id=organ_id,
                method=method,
                layer_name="decoder1",
                score_mode="pred_mask",
                device=device
            )

            pred_to_plot = pred_slice if pred_slice is not None else res["predicted_slice"]

            # Method suffix for filename if not standard gradcam
            method_suffix = f"_{method}" if method != "gradcam" else ""
            safe_organ = organ_name.lower().replace(" ", "_")

            single_path, four_path = render_model_focus_figures(
                slice_mri=slice_mri,
                gt_slice=gt_slice,
                pred_slice=pred_to_plot,
                cam_norm=res["activation_norm"],
                case_id=clean_case,
                slice_idx=slice_idx,
                target_organ_id=organ_id,
                target_organ_name=f"{organ_name} ({method.upper()})" if method != "gradcam" else organ_name,
                method=method,
                layer_name="decoder1",
                output_dir=output_dir,
                dpi=300
            )

            # Save raw array
            npy_path = output_dir / f"{clean_case}_z{slice_idx}_{safe_organ}{method_suffix}_activation.npy"
            np.save(str(npy_path), res["activation_norm"])

            print(f" Done in {time.time()-t_org:.2f}s | Peak: (y={res['peak_y']}, x={res['peak_x']}), max raw: {res['max_raw']:.4f}", flush=True)

    print("\nBatch generation complete! All figures and arrays saved to outputs/model_focus_heatmaps/", flush=True)

if __name__ == "__main__":
    run_batch()
