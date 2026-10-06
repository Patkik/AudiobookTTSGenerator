---
name: kokoro-model-trainer
description: Use when fine-tuning, training, configuring, or exporting Kokoro TTS models and emotion voicepacks on GPU runtimes
---

# Kokoro Model Trainer

## Overview
A reference workflow for executing two-stage fine-tuning of the Kokoro-82M (StyleTTS2) architecture on cloud GPUs (Google Colab / Kaggle T4), extracting emotion voicepacks, and exporting quantized ONNX models for CPU inference.

## When to Use
- Fine-tuning Kokoro on custom voices or emotional speech datasets
- Managing Google Colab or Kaggle T4 training sessions and checkpoints
- Extracting per-emotion 3D style tensors (`emotions/<emotion>.npy`)
- Exporting trained PyTorch checkpoints to INT8 quantized ONNX models

**When NOT to use:**
- When preparing, cleaning, or phonemizing raw audio (use `kokoro-dataset-curator` instead)
- When running local CPU inference with existing models (use `app/synthesis/engine.py`)

## Two-Stage Training Architecture

```text
Pretrained Kokoro-82M
         │
         ▼
[ Stage 1: Acoustic Pre-training ]
• Train: TextEncoder + ProsodyPredictor
• Freeze: iSTFTNet vocoder + PL-BERT
• Losses: Mel L1 + Multi-Res STFT + Duration + F0/Energy
• Duration: ~4-6 hours on free Colab T4
         │
         ▼
[ Stage 2: Adversarial Prosody Training ]
• Train: All encoders + ProsodyPredictor
• Freeze: iSTFTNet vocoder only
• Losses: MPD + MRD GAN losses + WavLM SLM perceptual loss
• Duration: ~8-12 hours across Colab sessions
         │
         ▼
[ Voicepack Extraction & ONNX Quantization ]
• extract_voicepacks.py ──▶ emotions/<emotion>.npy (shape: [510, 1, 256])
• export_onnx.py        ──▶ kokoro-expressive.int8.onnx (~80 MB)
```

## Step-by-Step Training Workflow

### 1. Launch Cloud Environment
Open `training/train_kokoro.ipynb` in [Google Colab](https://colab.research.google.com/) or Kaggle.
- Ensure runtime type is set to **T4 GPU** (or V100/A100).
- Mount Google Drive so checkpoints persist across 12-hour disconnects.

### 2. Stage 1: Acoustic Training
Run `train_first.py` with batch size 4 on T4:
```bash
python train_first.py \
  --config_path Configs/config.yml \
  --train_data train_list.txt \
  --val_data val_list.txt \
  --pretrained_model models/kokoro-v1_0.pth \
  --save_dir /content/drive/MyDrive/checkpoints/stage1 \
  --batch_size 4
```
**Stop condition:** Mel loss stabilizes below ~0.35–0.40.

### 3. Stage 2: Adversarial Prosody Training
Run `train_second.py` with WavLM perceptual loss active:
```bash
python train_second.py \
  --config_path Configs/config.yml \
  --train_data train_list.txt \
  --val_data val_list.txt \
  --pretrained_model /content/drive/MyDrive/checkpoints/stage1/best_model.pth \
  --save_dir /content/drive/MyDrive/checkpoints/stage2 \
  --batch_size 4
```
**Stop condition:** Perceptual loss flattens and discriminator balance is achieved.

### 4. Extract Emotion Voicepacks
Run the voicepack extraction script to generate 3D tensors:
```bash
python training/extract_voicepacks.py \
  --voices_bin "models/voices-v1.0.bin" \
  --out_dir "emotions"
```
Verify generated `.npy` files have shape `(510, 1, 256)` float32.

### 5. Export and Quantize ONNX
Convert the trained PyTorch checkpoint into a lightweight INT8 ONNX model:
```bash
python training/export_onnx.py \
  --input "kokoro-expressive.onnx" \
  --output "models/kokoro-expressive.int8.onnx"
```

## Common Mistakes & Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| Colab Disconnect loses training | Checkpoints stored in ephemeral `/content` | Mount Google Drive and save checkpoints to `/content/drive/MyDrive/...` |
| CUDA Out of Memory (OOM) | Batch size too large for 16GB T4 | Set `--batch_size 4` and enable gradient accumulation (`accum_steps: 2`) |
| Unstable GAN training in Stage 2 | Learning rate too high | Reduce generator LR to 1e-5 and discriminator LR to 5e-5 |
| Dimension mismatch in `VoicepackStore` | Voicepack array flattened to 2D | Ensure saved `.npy` tensor retains 3D shape `(510, 1, 256)` |
