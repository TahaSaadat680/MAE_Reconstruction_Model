---
title: MAE Reconstruction
emoji: 🎭
colorFrom: purple
colorTo: cyan
sdk: docker
pinned: false
license: mit
---

# Masked Autoencoder (MAE) — Image Reconstruction Demo

Interactive demo of a **ViT-Base MAE** trained on Tiny ImageNet for 20 epochs.

Upload any image → the model masks a configurable percentage of 16×16 patches → reconstructs the full image from the remaining visible patches.

## Architecture

| Component  | Config            |
|------------|-------------------|
| Backbone   | ViT-Base          |
| Image size | 224 × 224         |
| Patch size | 16 × 16 → 196 patches |
| Mask ratio | 75% (default)     |
| Encoder    | dim=768, depth=12, heads=12 |
| Decoder    | dim=384, depth=12, heads=6  |

## Results (Validation set — 7,680 images)

| Metric | Score  |
|--------|--------|
| PSNR   | 24.60 dB |
| SSIM   | 0.7338   |

## Usage

1. Upload an image (JPG / PNG / WebP)
2. Adjust the **Mask Ratio** slider (0.1 – 0.95)
3. Click **Reconstruct**
4. Compare Original · Masked · Reconstructed panels
5. Download the output or inspect the pixel difference map"# MAE_Reconstruction_Model" 
