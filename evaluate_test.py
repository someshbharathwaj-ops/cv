from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd

pred_dir = Path("outputs/test_predictions_best")
gt_dir = Path("data/nnUNet_raw/Dataset702_AbdomenMR/labelsTs")

rows = []

for gt_path in sorted(gt_dir.glob("*.nii.gz")):
    pred_path = pred_dir / gt_path.name

    if not pred_path.exists():
        raise FileNotFoundError(f"Missing prediction: {pred_path}")

    gt = np.asarray(nib.load(str(gt_path)).dataobj)
    pred = np.asarray(nib.load(str(pred_path)).dataobj)

    if gt.shape != pred.shape:
        raise ValueError(
            f"Shape mismatch for {gt_path.name}: "
            f"GT {gt.shape}, prediction {pred.shape}"
        )

    gt_fg = gt > 0
    pred_fg = pred > 0

    intersection = np.logical_and(gt_fg, pred_fg).sum()
    gt_count = gt_fg.sum()
    pred_count = pred_fg.sum()
    union = np.logical_or(gt_fg, pred_fg).sum()

    dice = (
        2 * intersection / (gt_count + pred_count)
        if gt_count + pred_count > 0 else 1.0
    )
    iou = intersection / union if union > 0 else 1.0

    rows.append({
        "case": gt_path.name,
        "dice_foreground": dice,
        "iou_foreground": iou,
    })

df = pd.DataFrame(rows)
Path("outputs").mkdir(exist_ok=True)
df.to_csv("outputs/test_metrics_per_case.csv", index=False)

print("\nTEST SET RESULTS")
print("================")
print("Test cases:", len(df))
print(f"Mean foreground Dice: {df['dice_foreground'].mean():.4f}")
print(f"Std foreground Dice:  {df['dice_foreground'].std():.4f}")
print(f"Mean foreground IoU:  {df['iou_foreground'].mean():.4f}")
print(f"Std foreground IoU:   {df['iou_foreground'].std():.4f}")

print("\nLowest-Dice cases:")
print(df.nsmallest(5, "dice_foreground").to_string(index=False))
print("\nSaved: outputs/test_metrics_per_case.csv")
