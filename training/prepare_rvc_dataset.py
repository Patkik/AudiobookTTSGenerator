"""
Prepares and augments RVC training datasets following the Three-P Framework
(Pitch, Pace, and Projection) to prevent neutral overfitting when training > 300 epochs.
"""
import os
import re
import math
import argparse
import numpy as np
import soundfile as sf
import scipy.signal

try:
    import pyloudnorm as pyln
except ImportError:
    pyln = None


def classify_projection_type(tag_or_text: str) -> str:
    """
    Categorize audio into Three-P projection categories:
    - 'whisper': Unvoiced, breathy speech with lower harmonic-to-noise ratio
    - 'high_excitation': High-amplitude, wider F0 pitch excursions (shouts, screams)
    - 'emotional_inflection': Crying, laughing, breathless dialogue, gasps
    - 'neutral': Ordinary conversational talking
    """
    if not tag_or_text:
        return "neutral"

    text = tag_or_text.lower()

    if re.search(r'\b(whisper|whispered|breathy|soft|sigh|quiet)\b', text):
        return "whisper"
    if re.search(r'\b(shout|shouting|scream|screamed|screaming|yell|yelled|roar|roared)\b', text):
        return "high_excitation"
    if re.search(r'\b(cry|crying|sob|sobbed|tear|tears|weep|laugh|laughing|chuckle|gasp|breathless)\b', text):
        return "emotional_inflection"

    return "neutral"


def resample_waveform(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample 1D float32 audio waveform."""
    if orig_sr == target_sr or len(audio) == 0:
        return np.asarray(audio, dtype=np.float32)
    gcd = math.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    return scipy.signal.resample_poly(audio, up, down).astype(np.float32)


def normalize_loudness(audio: np.ndarray, sr: int, target_lufs: float = -23.0) -> np.ndarray:
    """
    Normalize audio loudness to target LUFS (EBU R128 standard).
    Falls back to RMS target scaling if signal is too short or pyloudnorm unmeterable.
    """
    if len(audio) == 0:
        return audio

    audio = np.asarray(audio, dtype=np.float32)

    # Try pyloudnorm first (needs at least ~0.4s to meter accurately)
    if pyln is not None and len(audio) >= int(sr * 0.4):
        try:
            meter = pyln.Meter(sr)
            loudness = meter.integrated_loudness(audio)
            if np.isfinite(loudness) and loudness > -70.0:
                normalized = pyln.normalize.loudness(audio, loudness, target_lufs)
                peak = np.max(np.abs(normalized))
                if peak > 0.98:
                    normalized = normalized * (0.98 / peak)
                return normalized.astype(np.float32)
        except Exception:
            pass

    # RMS Fallback
    rms = np.sqrt(np.mean(audio ** 2) + 1e-9)
    # Approximate target RMS for target LUFS: LUFS ~= 20 * log10(rms) - 0.6
    target_rms = 10.0 ** ((target_lufs + 0.6) / 20.0)
    scale = target_rms / rms
    normalized = audio * scale
    peak = np.max(np.abs(normalized))
    if peak > 0.98:
        normalized = normalized * (0.98 / peak)
    return normalized.astype(np.float32)


def process_rvc_clip(
    in_path: str,
    out_path: str,
    tag: str,
    target_sr: int = 40_000,
    target_lufs: float = -23.0,
) -> dict | None:
    """
    Resample, loudness-normalize, and export an audio clip for RVC training.
    """
    if not os.path.exists(in_path):
        return None

    try:
        data, sr = sf.read(in_path, dtype="float32")
    except Exception:
        return None

    # Mono mixdown if stereo
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    if len(data) == 0:
        return None

    # Resample
    if sr != target_sr:
        data = resample_waveform(data, sr, target_sr)

    # Loudness normalize
    data = normalize_loudness(data, target_sr, target_lufs=target_lufs)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    sf.write(out_path, data, target_sr, subtype="PCM_16")

    projection = classify_projection_type(tag)
    duration_sec = float(len(data) / target_sr)

    return {
        "path": out_path,
        "projection": projection,
        "duration_sec": duration_sec,
        "sample_rate": target_sr,
    }


def build_rvc_manifest(entries: list[dict], manifest_path: str) -> None:
    """
    Save RVC training manifest in format: path|speaker|projection
    """
    os.makedirs(os.path.dirname(os.path.abspath(manifest_path)), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        for e in entries:
            line = f"{e['path']}|{e.get('speaker', 'rudeus')}|{e.get('projection', 'neutral')}\n"
            f.write(line)


def main():
    parser = argparse.ArgumentParser(description="Prepare augmented dataset for RVC training.")
    parser.add_argument("--input_dir", default="data/raw_datasets/NonverbalTTS", help="Source audio folder")
    parser.add_argument("--out_dir", default="data/rvc_augmented", help="Destination folder")
    parser.add_argument("--speaker", default="rudeus", help="Speaker identifier")
    parser.add_argument("--sr", type=int, default=40000, help="Target sample rate (40000 or 48000)")
    parser.add_argument("--lufs", type=float, default=-23.0, help="Target loudness in LUFS")
    args = parser.parse_args()

    print(f"Preparing RVC dataset from {args.input_dir} -> {args.out_dir} at {args.sr}Hz...")
    entries = []
    if os.path.exists(args.input_dir):
        for root, _, files in os.walk(args.input_dir):
            for file in files:
                if file.endswith((".wav", ".mp3", ".flac")):
                    in_p = os.path.join(root, file)
                    out_p = os.path.join(args.out_dir, "wavs", file)
                    tag = os.path.splitext(file)[0]
                    meta = process_rvc_clip(in_p, out_p, tag, target_sr=args.sr, target_lufs=args.lufs)
                    if meta:
                        meta["speaker"] = args.speaker
                        entries.append(meta)

    manifest_p = os.path.join(args.out_dir, "rvc_train_manifest.txt")
    build_rvc_manifest(entries, manifest_p)
    print(f"Done! Processed {len(entries)} clips. Manifest saved to {manifest_p}")


if __name__ == "__main__":
    main()
