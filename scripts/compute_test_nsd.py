"""
Normalized Surface Dice (NSD) & Official Benchmark Evaluation
Evaluates 1000-Epoch Swin-UMamba Predictions without Retraining
Dataset: Dataset702_AbdomenMR (AMOS 2022 MRI Benchmark - 50 Test Cases)
Protocol: Official AMOS / Swin-UMamba Surface Distance Evaluation
"""

import sys
from pathlib import Path
import os
import time
from collections import OrderedDict
import numpy as np
import pandas as pd
import nibabel as nb
from concurrent.futures import ProcessPoolExecutor, as_completed

# Ensure evaluation module is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluation"))
from SurfaceDice import (
    compute_surface_distances,
    compute_surface_dice_at_tolerance,
    compute_dice_coefficient
)

# Official AMOS benchmark tolerances in millimeters
LABEL_TOLERANCE = OrderedDict({
    "Liver": 5,
    "RK": 3,
    "Spleen": 3,
    "Pancreas": 5,
    "Aorta": 2,
    "IVC": 2,
    "RAG": 2,
    "LAG": 2,
    "Gallbladder": 2,
    "Esophagus": 3,
    "Stomach": 5,
    "Duodenum": 7,
    "LK": 3
})

ORGAN_ID_MAP = {
    1: "Liver",
    2: "RK",
    3: "Spleen",
    4: "Pancreas",
    5: "Aorta",
    6: "IVC",
    7: "RAG",
    8: "LAG",
    9: "Gallbladder",
    10: "Esophagus",
    11: "Stomach",
    12: "Duodenum",
    13: "LK"
}


def find_lower_upper_zbound(organ_mask):
    """Returns the lower and upper axial slice indices where the organ is labeled."""
    organ_mask = np.uint8(organ_mask)
    z_index = np.where(organ_mask > 0)[2]
    if len(z_index) == 0:
        return 0, organ_mask.shape[2]
    z_lower = np.min(z_index)
    z_upper = np.max(z_index) + 1  # inclusive upper bound for slicing
    return z_lower, z_upper


def evaluate_single_case(case_name, gt_path, seg_path):
    """Computes per-organ NSD and DSC for a single case following the official protocol."""
    gt_file = gt_path / case_name
    seg_file = seg_path / case_name

    gt_nii = nb.load(str(gt_file))
    seg_nii = nb.load(str(seg_file))
    case_spacing = gt_nii.header.get_zooms()

    gt_data = np.uint8(gt_nii.get_fdata())
    seg_data = np.uint8(seg_nii.get_fdata())

    case_nsd = OrderedDict({"Name": case_name})
    case_dsc = OrderedDict({"Name": case_name})

    for i, (organ, tol_mm) in enumerate(LABEL_TOLERANCE.items(), 1):
        gt_sum = np.sum(gt_data == i)
        seg_sum = np.sum(seg_data == i)

        if gt_sum == 0 and seg_sum == 0:
            # True negative case (e.g. absent gallbladder)
            dsc_i = 1.0
            nsd_i = 1.0
        elif gt_sum == 0 and seg_sum > 0:
            # Spurious detection where organ is absent
            dsc_i = 0.0
            nsd_i = 0.0
        elif gt_sum > 0 and seg_sum == 0:
            # Completely missed organ
            dsc_i = 0.0
            nsd_i = 0.0
        else:
            # Tubular structures (Aorta, IVC, Esophagus): evaluate within labeled z-range
            if i in (5, 6, 10):
                z_lower, z_upper = find_lower_upper_zbound(gt_data == i)
                organ_i_gt = (gt_data[:, :, z_lower:z_upper] == i)
                organ_i_seg = (seg_data[:, :, z_lower:z_upper] == i)
            else:
                organ_i_gt = (gt_data == i)
                organ_i_seg = (seg_data == i)

            dsc_i = compute_dice_coefficient(organ_i_gt, organ_i_seg)
            surface_distances = compute_surface_distances(organ_i_gt, organ_i_seg, case_spacing)
            nsd_i = compute_surface_dice_at_tolerance(surface_distances, tol_mm)

        case_nsd[f"{organ}_NSD"] = round(float(nsd_i), 4)
        case_dsc[f"{organ}_DSC"] = round(float(dsc_i), 4)

    return case_nsd, case_dsc


def main():
    root = Path(__file__).resolve().parent.parent
    gt_dir = root / "data" / "nnUNet_raw" / "Dataset702_AbdomenMR" / "labelsTs"
    seg_dir = root / "outputs" / "test_predictions_best"
    out_dir = root / "outputs"

    print("=" * 75)
    print("SWIN-UMAMBA OFFICIAL NSD & DSC EVALUATION (1000-EPOCH MODEL)")
    print("=" * 75)
    print(f"Ground Truth Directory: {gt_dir}")
    print(f"Predictions Directory:  {seg_dir}")
    print(f"Output Directory:       {out_dir}")

    filenames = sorted([f.name for f in seg_dir.glob("*.nii.gz") if (gt_dir / f.name).exists()])
    print(f"Total Test Cases to Evaluate: {len(filenames)}")

    start_time = time.time()
    all_nsd_rows = []
    all_dsc_rows = []

    # Parallel evaluation across CPU cores
    max_workers = min(os.cpu_count() or 4, 10)
    print(f"Running parallel surface distance evaluation on {max_workers} worker processes...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_case = {
            executor.submit(evaluate_single_case, fname, gt_dir, seg_dir): fname
            for fname in filenames
        }
        completed = 0
        for future in as_completed(future_to_case):
            case_name = future_to_case[future]
            try:
                nsd_row, dsc_row = future.result()
                all_nsd_rows.append(nsd_row)
                all_dsc_rows.append(dsc_row)
                completed += 1
                if completed % 5 == 0 or completed == len(filenames):
                    print(f"  [{completed:2d}/{len(filenames)}] Evaluated: {case_name}")
            except Exception as e:
                print(f"  Error processing {case_name}: {e}")

    elapsed = time.time() - start_time
    print(f"\nEvaluation completed in {elapsed:.1f} seconds ({elapsed / len(filenames):.2f} s/case).")

    # Sort results by case name
    all_nsd_rows.sort(key=lambda x: x["Name"])
    all_dsc_rows.sort(key=lambda x: x["Name"])

    df_nsd = pd.DataFrame(all_nsd_rows)
    df_dsc = pd.DataFrame(all_dsc_rows)

    # Compute mean case score across organs
    organ_nsd_cols = [c for c in df_nsd.columns if c.endswith("_NSD")]
    organ_dsc_cols = [c for c in df_dsc.columns if c.endswith("_DSC")]

    df_nsd["Mean_NSD"] = df_nsd[organ_nsd_cols].mean(axis=1)
    df_dsc["Mean_DSC"] = df_dsc[organ_dsc_cols].mean(axis=1)

    # Save per-case files
    nsd_csv_path = out_dir / "test_metric_NSD.csv"
    dsc_csv_path = out_dir / "test_metric_DSC.csv"
    df_nsd.to_csv(nsd_csv_path, index=False)
    df_dsc.to_csv(dsc_csv_path, index=False)
    print(f"Saved: {nsd_csv_path}")
    print(f"Saved: {dsc_csv_path}")

    # Build per-organ summary table
    summary_rows = []
    for organ in LABEL_TOLERANCE.keys():
        nsd_col = f"{organ}_NSD"
        dsc_col = f"{organ}_DSC"
        tol_mm = LABEL_TOLERANCE[organ]

        summary_rows.append({
            "Organ": organ,
            "Tolerance_mm": tol_mm,
            "Mean_NSD": df_nsd[nsd_col].mean(),
            "Std_NSD": df_nsd[nsd_col].std(),
            "Median_NSD": df_nsd[nsd_col].median(),
            "Mean_DSC": df_dsc[dsc_col].mean(),
            "Std_DSC": df_dsc[dsc_col].std(),
            "Median_DSC": df_dsc[dsc_col].median(),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "test_nsd_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved: {summary_csv}")

    # Overall macro metrics
    macro_nsd = summary_df["Mean_NSD"].mean()
    macro_dsc = summary_df["Mean_DSC"].mean()

    print("\n" + "=" * 75)
    print("FINAL 1000-EPOCH OFFICIAL TEST BENCHMARK RESULTS")
    print("=" * 75)
    print(f"Evaluated Test Cases:       {len(df_nsd)}")
    print(f"Overall Mean NSD:           {macro_nsd:.4f} ({macro_nsd * 100:.2f}%)")
    print(f"Overall Mean DSC (Official):{macro_dsc:.4f} ({macro_dsc * 100:.2f}%)")
    print("\nPer-Organ Breakdown (Tolerance in mm):")
    for idx, r in summary_df.iterrows():
        print(f"  {r['Organ']:<18s} (tol={r['Tolerance_mm']}mm): NSD = {r['Mean_NSD']:.4f} +/- {r['Std_NSD']:.4f} | DSC = {r['Mean_DSC']:.4f} +/- {r['Std_DSC']:.4f}")

    # Comparison with base paper
    print("\n" + "=" * 75)
    print("COMPARISON WITH BASE PAPER (Swin-UMamba, arXiv:2402.03302)")
    print("=" * 75)
    paper_comparison = [
        {"Model": "UNETR", "Pretraining": "None", "Paper_DSC": 0.5747, "Paper_NSD": 0.6309},
        {"Model": "SwinUNETR", "Pretraining": "None", "Paper_DSC": 0.7028, "Paper_NSD": 0.7669},
        {"Model": "nnFormer", "Pretraining": "None", "Paper_DSC": 0.7279, "Paper_NSD": 0.7963},
        {"Model": "SegResNet", "Pretraining": "None", "Paper_DSC": 0.7317, "Paper_NSD": 0.8034},
        {"Model": "nnU-Net (2D)", "Pretraining": "None", "Paper_DSC": 0.7450, "Paper_NSD": 0.8153},
        {"Model": "U-Mamba_Bot", "Pretraining": "None", "Paper_DSC": 0.7588, "Paper_NSD": 0.8285},
        {"Model": "U-Mamba_Enc", "Pretraining": "None", "Paper_DSC": 0.7625, "Paper_NSD": 0.8327},
        {"Model": "Swin-UMamba (Scratch)", "Pretraining": "None (Paper)", "Paper_DSC": 0.7054, "Paper_NSD": 0.7647},
        {"Model": "This Run (1000 Epochs, Fold 0)", "Pretraining": "None (Ours)", "Paper_DSC": round(macro_dsc, 4), "Paper_NSD": round(macro_nsd, 4)},
        {"Model": "Swin-UMamba (Pretrained)", "Pretraining": "ImageNet (Paper)", "Paper_DSC": 0.7760, "Paper_NSD": 0.8421},
    ]
    df_comp = pd.DataFrame(paper_comparison)
    comp_csv = out_dir / "nsd_comparison_with_paper.csv"
    df_comp.to_csv(comp_csv, index=False)
    print(df_comp.to_string(index=False))
    print(f"\nSaved Comparison Table: {comp_csv}")


if __name__ == "__main__":
    main()
