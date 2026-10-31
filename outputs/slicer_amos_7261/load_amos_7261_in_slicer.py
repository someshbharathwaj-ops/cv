# =========================================================================
# 3D Slicer Automated Loader: amos_7261 Spatial Segmentation Error Map
# Run in 3D Slicer: Copy & Paste into Slicer Python Console (Ctrl + `)
# =========================================================================

import os
import slicer

data_dir = "C:/Users/SOMESH BHARATHWAJ/Documents/Swin-UMamba/outputs/slicer_amos_7261"
print(f"Loading Swin-UMamba Case amos_7261 from: {data_dir}")

# 1. Clear scene
slicer.mrmlScene.Clear(0)

# 2. Load Original MRI as Background Volume
mri_file = os.path.join(data_dir, "amos_7261_image.nii.gz")
mri_node = slicer.util.loadVolume(mri_file)
mri_node.SetName("amos_7261_MRI_Original")
print(f"Loaded MRI Volume: {mri_node.GetName()}")

# 3. Load Spatial Error Map as Slicer Segmentation Node
seg_file = os.path.join(data_dir, "amos_7261_error_map.seg.nrrd")
if os.path.exists(seg_file):
    seg_node = slicer.util.loadSegmentation(seg_file)
    seg_node.SetName("amos_7261_Spatial_Error_Map")
    print(f"Loaded Error Map Segmentation: {seg_node.GetName()}")

# 4. Load Ground-Truth Segmentation
gt_seg_file = os.path.join(data_dir, "amos_7261_ground_truth.seg.nrrd")
if os.path.exists(gt_seg_file):
    gt_seg_node = slicer.util.loadSegmentation(gt_seg_file)
    gt_seg_node.SetName("amos_7261_Ground_Truth")
    # Hide by default so error map is visible first
    gt_seg_node.GetDisplayNode().SetVisibility(False)

# 5. Load Prediction Segmentation
pred_seg_file = os.path.join(data_dir, "amos_7261_prediction.seg.nrrd")
if os.path.exists(pred_seg_file):
    pred_seg_node = slicer.util.loadSegmentation(pred_seg_file)
    pred_seg_node.SetName("amos_7261_Prediction")
    pred_seg_node.GetDisplayNode().SetVisibility(False)

# 6. Configure Slice Views to target slice z=38
# In physical space (Origin: -50.33mm, Spacing: 3.0mm): z=38 corresponds to Z = +63.67 mm
target_z_physical = -50.33195495605469 + (38 * 3.0)

# Navigate Red (Axial) slice view to z=38
red_slice = slicer.app.layoutManager().sliceWidget("Red").sliceLogic()
red_slice.SetSliceOffset(target_z_physical)

# Set Background Volume across all slice views
slicer.util.setSliceViewerLayers(background=mri_node, fit=True)

print("=" * 65)
print("amos_7261 Interactive Exploration Ready in 3D Slicer!")
print(f"Axial View positioned at slice z=38 (Z = {target_z_physical:.2f} mm)")
print("Toggle Segmentations in the Data Module (Left Panel):")
print("  - 'amos_7261_Spatial_Error_Map' (Green=TP, Blue=FN, Red=FP, Orange=Confusion)")
print("  - 'amos_7261_Ground_Truth' (13 Multiclass Organs)")
print("  - 'amos_7261_Prediction' (Swin-UMamba Multiclass Output)")
print("=" * 65)
