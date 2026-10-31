import time
import sys
import os
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

print("1. Python starting...", flush=True)
sys.path.insert(0, r"c:\Users\SOMESH BHARATHWAJ\Documents\Swin-UMamba\swin_umamba")
from nnunetv2.nets.SwinUMamba import SwinUMamba

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"2. Using device: {device}", flush=True)

t0 = time.time()
model = SwinUMamba(
    in_chans=1,
    out_chans=14,
    feat_size=[48, 96, 192, 384, 768],
    deep_supervision=False,
    hidden_size=768
).to(device)
print(f"3. Model instantiated in {time.time()-t0:.2f}s", flush=True)

ckpt_path = r"c:\Users\SOMESH BHARATHWAJ\Documents\Swin-UMamba\data\nnUNet_results\Dataset702_AbdomenMR\nnUNetTrainerSwinUMambaScratch__nnUNetPlans__2d\fold_0\checkpoint_best.pth"
t0 = time.time()
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt['network_weights'])
model.eval()
print(f"4. Checkpoint loaded in {time.time()-t0:.2f}s", flush=True)

# Load real MRI slice directly without decompressing the entire volume
img_path = r"c:\Users\SOMESH BHARATHWAJ\Documents\Swin-UMamba\data\nnUNet_raw\Dataset702_AbdomenMR\imagesTs\amos_7261_0000.nii.gz"
t0 = time.time()
img_nii = nib.load(img_path)
slice_idx = 38
# Read only slice 38 from dataobj
slice_mri = np.asanyarray(img_nii.dataobj[:, :, slice_idx]).astype(np.float32)
print(f"5. Slice {slice_idx} loaded in {time.time()-t0:.2f}s, shape: {slice_mri.shape}", flush=True)

# Normalization & Padding to patch size 320x320
mean, std = slice_mri.mean(), slice_mri.std()
norm_slice = (slice_mri - mean) / max(std, 1e-8)
H, W = slice_mri.shape
pad_h = max(0, 320 - H)
pad_w = max(0, 320 - W)
padded = np.pad(norm_slice, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
x = torch.from_numpy(padded).float().unsqueeze(0).unsqueeze(0).to(device)

print(f"6. Running encoder forward pass...", flush=True)
t0 = time.time()
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
print(f"7. Encoder forward pass complete in {time.time()-t0:.2f}s, dec0 shape: {dec0.shape}", flush=True)

test_organs = {
    1: "Liver",
    3: "Spleen",
    4: "Pancreas",
    7: "Right Adrenal Gland",
    13: "Left Kidney"
}

results = {}
for organ_id, organ_name in test_organs.items():
    t0 = time.time()
    # Target feature from dec0
    target_feat = dec0.detach().clone().requires_grad_(True)
    dec_out = model.decoder1(target_feat)
    logits = model.out_layers[0](dec_out)
    
    # Target score: sum of logits for the organ over predicted region or slice
    pred_mask = (torch.argmax(logits, dim=1) == organ_id)
    if pred_mask.sum() > 0:
        target_score = logits[0, organ_id][pred_mask[0]].sum()
    else:
        target_score = logits[0, organ_id].sum()
        
    model.zero_grad()
    target_score.backward()
    
    grad = target_feat.grad
    alpha = torch.mean(grad, dim=(2, 3), keepdim=True)
    cam = torch.sum(alpha * target_feat, dim=1, keepdim=True)
    cam = torch.relu(cam)
    cam = cam.squeeze().detach().cpu().numpy()
    
    # Unpad to original slice dimensions
    cam_unpad = cam[:H, :W]
    
    # Robust normalization [0, 1]
    c_min, c_max = cam_unpad.min(), cam_unpad.max()
    if c_max > c_min:
        cam_norm = (cam_unpad - c_min) / (c_max - c_min)
    else:
        cam_norm = np.zeros_like(cam_unpad)
        
    peak_y, peak_x = np.unravel_index(np.argmax(cam_norm), cam_norm.shape)
    results[organ_name] = {
        "min": float(c_min),
        "max": float(c_max),
        "peak": (int(peak_y), int(peak_x)),
        "mean_act": float(cam_norm.mean()),
        "time": time.time() - t0
    }
    print(f"Organ {organ_name:20s} (ID {organ_id:2d}): Peak at (y={peak_y}, x={peak_x}), max raw: {c_max:.4f}, mean norm: {cam_norm.mean():.4f} (calc in {time.time()-t0:.3f}s)", flush=True)

print("All test organs evaluated successfully!", flush=True)
