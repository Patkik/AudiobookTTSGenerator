# Expressive Kokoro fine-tuning contract

## Curated input

Use `prepare_dataset.py` on filtered Meta Expresso and a verified, consented
single-speaker corpus (1.5–3 hours). Do not include ESD or RAVDESS: the
manifest tool rejects rows where `source` or `dataset` is `esd` or `ravdess`.
Metadata must provide `filename`, `transcript`, `speaker_id`, and `emotion`.
Provide `alignment_path` (or `mfa_alignment`) for Montreal Forced Aligner output
when available; preserve annotated `[breath]` and `[gasp]` events in the
alignment source.

```powershell
python training\prepare_dataset.py --input_dir data\raw --output_dir data\processed_24k
python training\phonemize_dataset.py --csv data\metadata.csv --wav_dir data\processed_24k --output_dir data\manifests
```

The commands produce compatible four-field `train_list.txt` and `val_list.txt`
files plus `conditioning.jsonl`. The sidecar retains the tag-specific duration
targets without breaking a stock Kokoro manifest reader.

## Required harness integration

This repository does not vendor the upstream `train_first.py` and
`train_second.py` harnesses. When running them in Colab/Kaggle, load the JSONL
sidecar by audio path and feed `duration_targets` plus the emotion ID to the
DurationPredictor. Add the following loss to both stage-one and stage-two
generator objectives at the style projection head:

```python
from training.emotion_conditioning import style_classification_loss

aux_loss = style_classification_loss(
    speaker_logits, emotion_logits, speaker_ids, emotion_ids,
    speaker_weight=1.0, emotion_weight=1.0,
)
loss = base_loss + 0.1 * aux_loss
```

Keep the text encoder, duration predictor, emotion embedding, and style
projection in FP16/FP32. Only generator/vocoder-compatible graph nodes may be
INT8 quantized. Supply actual exported ONNX node names to
`training/export_onnx.py --conditioning_nodes`.

## Runtime parameter contract

| Tag | Total duration | Phoneme target | Pause |
| --- | ---: | --- | ---: |
| whisper | 1.15× | unvoiced stops/fricatives 1.25× | 150 ms trailing |
| sad | 1.20× | vowels 1.30× | profile pause |
| fearful | unchanged total | vowels 0.85× | 40 ms before plosives |
| happy | 0.95× | pitch variance learned by acoustic head | profile pause |

The current ONNX runtime accepts only global `speed`; it applies the calibrated
inverse total-duration values where available. The phoneme-class fields above
are training targets and become inference controls only after an exported model
exposes them.
