---
name: kokoro-dataset-curator
description: Use when downloading, preparing, cleaning, phonemizing, or formatting audio speech datasets for Kokoro TTS fine-tuning or training manifests
---

# Kokoro Dataset Curator

## Overview
A reference workflow for turning raw, uncurated audio clips (from open datasets like ESD, RAVDESS, Expresso, or user recordings) into standardized, Kokoro-compliant training manifests and 24 kHz WAV files.

## When to Use
- Preparing custom voice recordings or emotional datasets for fine-tuning Kokoro TTS
- Converting audio to the strict acoustic specification required by StyleTTS2 and iSTFTNet
- Phonemizing text transcripts into Kokoro's 178-token IPA vocabulary
- Generating `train_list.txt` and `val_list.txt` manifests

**When NOT to use:**
- When running model training on GPUs (use `kokoro-model-trainer` instead)
- When synthesizing speech for end users (use `SynthesisEngine` in `app/synthesis/`)

## Acoustic Specifications (Kokoro Standard)
All audio must strictly adhere to these parameters before being included in training manifests:

| Parameter | Required Value | Rationale |
|-----------|----------------|-----------|
| Sample Rate | **24,000 Hz** | Kokoro iSTFTNet vocoder output frequency |
| Channels | **Mono (1 channel)** | Multi-channel audio causes dimension mismatch |
| Bit Depth | **float32 or int16** | 32-bit float preferred for internal processing |
| Loudness | **-23 LUFS / Peak 0.95** | Prevents gradient explosion and clipping |
| Duration | **1.0s – 12.0s** | StyleTTS2 alignment fails on <1s; VRAM blows up on >12s |
| Silence | **Trimmed (top_db=30)** | Dead air distorts duration predictor training |

## Workflow & Implementation

### 1. Batch Audio Standardization
Run the curation script against the folder of raw audio clips:

```bash
python training/prepare_dataset.py \
  --input_dir "data/raw_audio" \
  --output_dir "data/processed_24k" \
  --target_sr 24000 \
  --min_sec 1.0 \
  --max_sec 12.0
```

### 2. Manifest & Phoneme Generation
Prepare a CSV metadata file with columns: `filename, transcript, speaker_id, emotion` and run:

```bash
python training/phonemize_dataset.py \
  --csv "data/metadata.csv" \
  --wav_dir "data/processed_24k" \
  --output_dir "data/manifests" \
  --val_ratio 0.05
```

This creates:
- `data/manifests/train_list.txt`
- `data/manifests/val_list.txt`

### 3. Manifest Line Format
Ensure every row follows this pipe-delimited structure:
```text
data/processed_24k/clip_001.wav|həlˈoʊ wˈɜːld|speaker_01|happy
```

## Common Mistakes & Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| Out of Vocabulary (OOV) tokens | Non-standard characters or emojis in text | Tokenizer automatically strips them; verify `phonemize_text()` returns non-empty string |
| CUDA OOM in Stage 1 training | Audio clips > 12 seconds in dataset | Re-run `prepare_dataset.py` with `--max_sec 10.0` |
| Monotonic Alignment Search (MAS) failure | Audio has long silences (>0.5s) at boundaries | Run silence trimming with `top_db=30` |
| Robotic voice artifacts | Clip contains background noise or music | Filter training clips to ensure SNR > 20 dB |
