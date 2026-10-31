import time
import sys
import torch

print("Testing efficient Grad-CAM on SwinUMamba...", flush=True)
sys.path.insert(0, r"c:\Users\SOMESH BHARATHWAJ\Documents\Swin-UMamba\swin_umamba")
from nnunetv2.nets.SwinUMamba import SwinUMamba

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SwinUMamba(
    in_chans=1,
    out_chans=14,
    feat_size=[48, 96, 192, 384, 768],
    deep_supervision=False,
    hidden_size=768
).to(device)

ckpt_path = r"c:\Users\SOMESH BHARATHWAJ\Documents\Swin-UMamba\data\nnUNet_results\Dataset702_AbdomenMR\nnUNetTrainerSwinUMambaScratch__nnUNetPlans__2d\fold_0\checkpoint_best.pth"
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt['network_weights'])
model.eval()

x = torch.randn(1, 1, 320, 320, device=device)

# We want feature activation at decoder2 output (dec0) or decoder1 output (dec_out)
# Let's inspect SwinUMamba forward method:
# x1 = self.stem(x_in)
# vss_outs = self.vssm_encoder(x1)
# enc1 = self.encoder1(x_in)
# enc2 = self.encoder2(vss_outs[0])
# enc3 = self.encoder3(vss_outs[1])
# enc4 = self.encoder4(vss_outs[2])
# enc5 = self.encoder5(vss_outs[3])
# enc_hidden = vss_outs[4]
# dec4 = self.decoder6(enc_hidden, enc5)
# dec3 = self.decoder5(dec4, enc4)
# dec2 = self.decoder4(dec3, enc3)
# dec1 = self.decoder3(dec2, enc2)
# dec0 = self.decoder2(dec1, enc1)
# dec_out = self.decoder1(dec0)
# out = self.out_layers[0](dec_out)

print("Running forward pass up to dec0 without grad...", flush=True)
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
print(f"Computed dec0 in {time.time()-t0:.2f}s! dec0 shape: {dec0.shape}", flush=True)

# Now enable gradient on dec0 (or decoder1) for Grad-CAM
target_feat = dec0.detach().clone().requires_grad_(True)
dec_out = model.decoder1(target_feat)
logits = model.out_layers[0](dec_out)

# Target: Organ 1 (Liver)
organ_id = 1
# Target score: sum of logits over positive prediction, or total logit sum
target_score = logits[0, organ_id].sum()

t0 = time.time()
target_score.backward()
print(f"Backward pass from logits to dec0 completed in {time.time()-t0:.4f}s!", flush=True)

grad = target_feat.grad
print(f"Grad shape: {grad.shape}, min: {grad.min().item():.4e}, max: {grad.max().item():.4e}", flush=True)

# Grad-CAM: Global average pooling over spatial dimensions
alpha = torch.mean(grad, dim=(2, 3), keepdim=True) # (1, C, 1, 1)
cam = torch.sum(alpha * target_feat, dim=1, keepdim=True) # (1, 1, H, W)
cam = torch.relu(cam) # positive contributions
cam = cam.squeeze().detach().cpu().numpy()
print(f"Grad-CAM shape: {cam.shape}, min: {cam.min():.4f}, max: {cam.max():.4f}", flush=True)
