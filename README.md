# [Swin-UMamba: Mamba-based UNet with ImageNet-based pretraining](https://arxiv.org/abs/2402.03302)

## Key Features

This repository provides the official implementation of: *[Swin-UMamba: Mamba-based UNet with ImageNet-based pretraining](https://arxiv.org/abs/2402.03302)*

- The first attempt to discover the impact of ImageNet pretrained Mamba-based networks in medical image segmentation.
- Provides two Mamba-based networks for medical image segmentation with different computation requirements.
- Swin-UMamba can outperform previous segmentation models including CNNs, ViTs, and the latest Mamba-based models with notable margin. 

## Links

- [Paper](https://arxiv.org/abs/2402.03302)
- [Model](https://drive.google.com/drive/folders/1zOt0ZfQPjoPdY37NfLKevYs4x5eClThN?usp=sharing)
- [Code](https://github.com/JiarunLiu/Swin-UMamba)

## Details

<!-- Insert a pipeline of your algorithm here if got one -->
<div align="center">
    <a href="https://"><img width="1000px" height="auto" src="https://github.com/JiarunLiu/Swin-UMamba/blob/main/assets/swin-umamba.png"></a>
</div>

Accurate medical image segmentation demands the integration of multi-scale information, spanning from local features to global dependencies. However, it is challenging for existing methods to model long-range global information, where convolutional neural networks are constrained by their local receptive fields, and vision transformers suffer from high quadratic complexity of their attention mechanism. Recently, Mamba-based models have gained great attention for their impressive ability in long sequence modeling. Several studies have demonstrated that these models can outperform popular vision models in various tasks, offering higher accuracy, lower memory consumption, and less computational burden. However, existing Mamba-based models are mostly trained from scratch and do not explore the power of pretraining, which has been proven to be quite effective for data-efficient medical image analysis. This paper introduces a novel Mamba-based model, Swin-UMamba, designed specifically for medical image segmentation tasks, leveraging the advantages of ImageNet-based pretraining. Our experimental results reveal the vital role of ImageNet-based training in enhancing the performance of Mamba-based models. Swin-UMamba demonstrates superior performance with a large margin compared to CNNs, ViTs, and latest Mamba-based models. Notably, on AbdomenMRI, Endoscopy, and Microscopy datasets, Swin-UMamba outperforms its closest counterpart U-Mamba by an average score of 2.72%.



**Main Results**

- AbdomenMRI
<img src="https://github.com/JiarunLiu/Swin-UMamba/blob/main/assets/abdomenmr.png" width="50%" />

- Endoscopy
<img src="https://github.com/JiarunLiu/Swin-UMamba/blob/main/assets/endoscopy.png" width="50%" />

- Microscopy
<img src="https://github.com/JiarunLiu/Swin-UMamba/blob/main/assets/microscopy.png" width="50%" />


## Dataset Links

All three datasets can be downloaded from [U-Mamba](https://github.com/bowang-lab/U-Mamba).

## Get Started

**Main Requirements**  
> torch==2.0.1  
> torchvision==0.15.2  
> causal-conv1d==1.1.1  
> mamba-ssm  
> torchinfo   
> timm  
> numba  


**Installation**
```shell
# create a new conda env
conda create -n swin_umamba python=3.10
conda activate swin_umamba

# install requirements
pip install torch==2.0.1 torchvision==0.15.2
pip install causal-conv1d==1.1.1
pip install mamba-ssm
pip install torchinfo timm numba

# install swin_umamba
git clone https://github.com/JiarunLiu/Swin-UMamba
cd Swin-UMamba/swin_umamba
pip install -e .
```

**Download Model**

We use the ImageNet pretrained VMamba-Tiny model from [VMamba](https://github.com/MzeroMiko/VMamba). You need to download the model checkpoint and put it into `data/pretrained/vmamba/vmamba_tiny_e292.pth`

```
wget https://github.com/MzeroMiko/VMamba/releases/download/%2320240218/vssmtiny_dp01_ckpt_epoch_292.pth
mv vssmtiny_dp01_ckpt_epoch_292.pth data/pretrained/vmamba/vmamba_tiny_e292.pth
```

**Preprocess**

We use the same data & processing strategy following U-Mamba. Download dataset from [U-Mamba](https://github.com/bowang-lab/U-Mamba) and put them into the data folder. Then preprocess the dataset with following command:

```shell
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```


**Training & Testing**

Using the following command to train & test Swin-UMamba

```shell
# AbdomenMR dataset
bash scripts/train_AbdomenMR.sh MODEL_NAME
# Endoscopy dataset
bash scripts/train_Endoscopy.sh MODEL_NAME
# Microscopy dataset 
bash scripts/train_Microscopy.sh MODEL_NAME
```

Here  `MODEL_NAME` can be:

- `nnUNetTrainerSwinUMamba`: Swin-UMamba model with ImageNet pretraining
- `nnUNetTrainerSwinUMambaD`: Swin-UMamba$\dagger$  model with ImageNet pretraining
- `nnUNetTrainerSwinUMambaScratch`: Swin-UMamba model without ImageNet pretraining
- `nnUNetTrainerSwinUMambaDScratch`: Swin-UMamba$\dagger$  model without ImageNet pretraining

You can download our model checkpoints [here](https://drive.google.com/drive/folders/1zOt0ZfQPjoPdY37NfLKevYs4x5eClThN?usp=sharing).

## 🔬 Model Focus Heatmaps (Interpretability & Explainability)

This repository includes a publication-quality model interpretability suite designed specifically for Swin-UMamba on multi-organ abdominal MRI segmentation (`Dataset702_AbdomenMR`). It allows researchers to compute and visualize **where the trained Swin-UMamba network focuses its activations** when making class-specific organ predictions.

### Key Features
- **Class-Specific Attribution**: Explanations can be targeted to any of the 13 abdominal organ classes (e.g., Liver, Pancreas, Spleen, Right Adrenal Gland, Duodenum, Kidneys).
- **Supported Methods**:
  - `gradcam`: Gradient-weighted Class Activation Mapping targeting intermediate decoder feature layers.
  - `gradcam++`: Higher-order gradient-weighted CAM providing sharper pixel-level localization for small or dispersed anatomical structures.
  - `activation`: Channel-averaged raw forward activations across the intermediate layer.
- **Scientific Figure Generation**:
  - **Single-panel Focus Heatmap**: Grayscale MRI with translucent turbo heatmap overlay, target organ contour, and peak focus marker.
  - **Four-panel Research Figure**:
    - **Panel A**: Grayscale input MRI with anatomical orientation markers (A, P, R, L).
    - **Panel B**: Class-specific Model Focus Heatmap with relative activation colorbar.
    - **Panel C**: Multi-organ ground-truth segmentation masks.
    - **Panel D**: Model prediction overlay with target organ outline and peak focus alignment contour.
- **Reproducible Exports**: Automatically exports high-resolution 300 DPI PNG figures, raw normalized 2D numpy arrays (`.npy`), and structured provenance records (`.json`).

### Quick Start: Generate Heatmap via CLI

Run the standalone CLI script from the repository root:

```bash
# Generate focus heatmap for Liver (Organ ID 1) on slice z=38
python generate_model_focus_heatmap.py --case amos_7261 --slice 38 --organ 1

# Generate focus heatmap for Pancreas (Organ ID 4)
python generate_model_focus_heatmap.py --case amos_7261 --slice 38 --organ pancreas

# Generate focus heatmap for Right Adrenal Gland using Grad-CAM++
python generate_model_focus_heatmap.py --case amos_7261 --slice 38 --organ "Right Adrenal Gland" --method gradcam++

# Run on a different test case and slice
python generate_model_focus_heatmap.py --case amos_7016 --slice 37 --organ spleen
```

### CLI Options

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--case` | `str` | `amos_7261` | Case identifier (e.g. `amos_7261`, `amos_7016`). |
| `--slice` | `int` | `38` | Axial slice index ($z$). |
| `--organ` | `str/int` | `1` (Liver) | Target organ by name or ID (1: Liver, 2: R Kidney, 3: Spleen, 4: Pancreas, 5: Aorta, 6: IVC, 7: R Adrenal, 8: L Adrenal, 9: Gallbladder, 10: Esophagus, 11: Stomach, 12: Duodenum, 13: L Kidney). |
| `--method` | `str` | `gradcam` | Attribution method: `gradcam`, `gradcam++`, or `activation`. |
| `--layer` | `str` | `decoder1` | Target intermediate layer probed (`decoder1` or `decoder2`). |
| `--score_mode` | `str` | `pred_mask` | Target score pooling: `pred_mask`, `gt_mask`, or `mean`. |
| `--checkpoint` | `path` | Best ckpt | Path to trained `.pth` checkpoint. |
| `--output_dir` | `path` | `outputs/model_focus_heatmaps` | Directory where figures and data are saved. |
| `--device` | `str` | `cuda` / auto | Compute device (`cuda` or `cpu`). |
| `--dpi` | `int` | `300` | Output image resolution for publication / presentations. |

### Batch Multi-Organ Comparison Suite

To generate heatmaps across multiple test cases, organs, and attribution methods simultaneously:

```bash
python scripts/batch_generate_heatmaps.py
```

### Outputs & File Structure

Generated artifacts are saved directly to `outputs/model_focus_heatmaps/`:
- `<case>_z<slice>_<organ>_heatmap.png`: High-resolution standalone focus heatmap overlay.
- `<case>_z<slice>_<organ>_four_panel.png`: 4-panel publication-ready comparison figure.
- `<case>_z<slice>_<organ>_activation.npy`: Raw normalized activation array ($[0, 1]$).
- `<case>_z<slice>_<organ>_metadata.json`: Full execution metadata (peak coordinates, layer, method, checkpoint).

### Scientific Interpretation & Methodological Notes
> [!IMPORTANT]
> **Attribution vs. Internal Attention**: Grad-CAM computes the gradient-weighted linear combination of forward feature maps with respect to a specific organ's output logits:
> $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right), \quad \alpha_k^c = \frac{1}{Z}\sum_i \sum_j \frac{\partial S_c}{\partial A_{i,j}^k}$$
> It highlights regions in the feature representations that positively contribute to maximizing the target organ's predicted logits. It does **not** measure cognitive or transformer self-attention, nor does it guarantee clinical causal sufficiency. A high activation region indicates that features at that location were influential under the linear approximation of the selected layer.

## 🙋‍♀️ Feedback and Contact

For further questions, please feel free to contact [Jiarun Liu](jr.liu@siat.ac.cn)


## 🛡️ License

This project is under the Apache License 2.0 license. See [LICENSE](LICENSE) for details.


## 🙏 Acknowledgement
 
Our code is based on [nnU-Net](https://github.com/MIC-DKFZ/nnUNet), [Mamba](https://github.com/state-spaces/mamba), [UMamba](https://github.com/bowang-lab/U-Mamba), [VMamba](https://github.com/MzeroMiko/VMamba), and [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet). We thank the authors for making their valuable code & data publicly available.


## 📝 Citation

If you find this repository useful, please consider citing this paper:
```
@article{Swin-UMamba,
    title={Swin-UMamba: Mamba-based UNet with ImageNet-based pretraining},
    author={Jiarun Liu and Hao Yang and Hong-Yu Zhou and Yan Xi and Lequan Yu and Yizhou Yu and Yong Liang and Guangming Shi and Shaoting Zhang and Hairong Zheng and Shanshan Wang},
    journal={arXiv preprint arXiv:2402.03302},
    year={2024}
}
```
