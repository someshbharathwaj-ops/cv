from pathlib import Path
import json
import nibabel as nib
import numpy as np
import pandas as pd

pred_dir = Path("outputs/test_predictions_best")
gt_dir = Path("data/nnUNet_raw/Dataset702_AbdomenMR/labelsTs")
dataset_file = Path("data/nnUNet_raw/Dataset702_AbdomenMR/dataset.json")

labels = json.loads(dataset_file.read_text())["labels"]
classes = [(name, int(idx)) for name, idx in labels.items()
           if int(idx) != 0]

case_rows = []

for gt_path in sorted(gt_dir.glob("*.nii.gz")):
    pred_path = pred_dir / gt_path.name
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing prediction: {pred_path}")

    gt = np.asarray(nib.load(str(gt_path)).dataobj)
    pred = np.asarray(nib.load(str(pred_path)).dataobj)

    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch: {gt_path.name}")

    for name, idx in classes:
        g = gt == idx
        p = pred == idx
        ng, npred = int(g.sum()), int(p.sum())
        intersection = int(np.logical_and(g, p).sum())

        # Ignore a class when absent in both masks.
        # Count a false positive as zero Dice/IoU.
        if ng == 0 and npred == 0:
            dice = iou = np.nan
        else:
            dice = 2 * intersection / (ng + npred)
            union = int(np.logical_or(g, p).sum())
            iou = intersection / union if union else np.nan

        case_rows.append({
            "case": gt_path.name,
            "class": name,
            "label_id": idx,
            "gt_voxels": ng,
            "pred_voxels": npred,
            "dice": dice,
            "iou": iou,
        })

per_case = pd.DataFrame(case_rows)
out = Path("outputs")
out.mkdir(exist_ok=True)
per_case.to_csv(out / "test_per_class_per_case.csv", index=False)

summary = (
    per_case.groupby(["class", "label_id"], as_index=False)
    .agg(
        evaluated_cases=("dice", "count"),
        mean_dice=("dice", "mean"),
        std_dice=("dice", "std"),
        mean_iou=("iou", "mean"),
        std_iou=("iou", "std"),
    )
    .sort_values("label_id")
)
summary.to_csv(out / "test_per_class_summary.csv", index=False)

print("\nPER-ORGAN TEST RESULTS")
print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nSaved:")
print("outputs/test_per_class_per_case.csv")
print("outputs/test_per_class_summary.csv")
