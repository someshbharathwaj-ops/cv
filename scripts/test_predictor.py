import sys
import torch
sys.path.insert(0, r"c:\Users\SOMESH BHARATHWAJ\Documents\Swin-UMamba\swin_umamba")
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

model_dir = r"c:\Users\SOMESH BHARATHWAJ\Documents\Swin-UMamba\data\nnUNet_results\Dataset702_AbdomenMR\nnUNetTrainerSwinUMambaScratch__nnUNetPlans__2d"

predictor = nnUNetPredictor(
    tile_step_size=0.5,
    use_gaussian=True,
    use_mirroring=False,
    perform_everything_on_device=True,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

print("Initializing predictor from trained model folder...", flush=True)
predictor.initialize_from_trained_model_folder(
    model_dir,
    use_folds=(0,),
    checkpoint_name="checkpoint_best.pth"
)
print("Predictor successfully initialized!", flush=True)
print(f"Network: {type(predictor.network)}", flush=True)
print(f"Plans: {predictor.plans_manager.plans_name}, Patch size: {predictor.configuration_manager.patch_size}", flush=True)
