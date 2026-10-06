#!/usr/bin/env python3
"""
prepare_dataset.py - Acoustic Normalization & Curation for Kokoro-82M Fine-Tuning.
Performs:
1. 24 kHz mono resampling
2. Boundary silence trimming (top_db=30)
3. Duration filtering (1.0s to 12.0s)
4. EBU R128 loudness normalization (-23 LUFS / peak 0.95 limit)
5. Export to 24 kHz 16-bit PCM WAV
"""

import argparse
from pathlib import Path
import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf


def normalize_audio(
    audio_path: Path,
    output_path: Path,
    target_sr: int = 24000,
    target_lufs: float = -23.0,
    peak_limit: float = 0.95,
    top_db: int = 30,
    min_sec: float = 1.0,
    max_sec: float = 12.0,
) -> bool:
    """Processes a single audio file to strict Kokoro/StyleTTS 2 acoustic specifications."""
    try:
        # 1. Load audio & resample to 24 kHz mono
        y, sr = librosa.load(audio_path, sr=target_sr, mono=True)

        # 2. Boundary Silence Trimming (top_db=30)
        # Long boundary silences cause Monotonic Alignment Search (MAS) failures
        intervals = librosa.effects.split(y, top_db=top_db)
        if len(intervals) == 0:
            return False
        y_trimmed = np.concatenate([y[start:end] for start, end in intervals])

        # 3. Duration Filtering (1.0s - 12.0s)
        # Clips < 1.0s break duration predictors; clips > 12.0s cause CUDA OOM
        duration = len(y_trimmed) / target_sr
        if duration < min_sec or duration > max_sec:
            return False

        # 4. EBU R128 Loudness Normalization (-23 LUFS, Peak 0.95)
        meter = pyln.Meter(target_sr)
        loudness = meter.integrated_loudness(y_trimmed)
        
        # Guard against silent/near-silent files
        if np.isinf(loudness) or np.isnan(loudness):
            return False

        y_norm = pyln.normalize.loudness(y_trimmed, loudness, target_lufs)
        
        # Peak limiting to prevent digital clipping in iSTFTNet
        max_peak = np.max(np.abs(y_norm))
        if max_peak > peak_limit:
            y_norm = y_norm * (peak_limit / max_peak)

        # 5. Export 24 kHz 16-bit PCM WAV
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, y_norm.astype(np.float32), target_sr, subtype="PCM_16")
        return True

    except Exception as e:
        print(f"Error processing {audio_path.name}: {e}")
        return False


def _process_single_clip(args_tuple):
    """Worker helper for parallel execution."""
    file_path, out_path, target_sr, min_sec, max_sec, skip_existing = args_tuple
    if skip_existing and out_path.exists():
        return True
    return normalize_audio(
        file_path, out_path, target_sr=target_sr, min_sec=min_sec, max_sec=max_sec
    )


def main():
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    parser = argparse.ArgumentParser(description="Batch process audio for Kokoro TTS.")
    parser.add_argument("--input_dir", type=Path, required=True, help="Path to raw audio clips")
    parser.add_argument("--output_dir", type=Path, required=True, help="Target 24kHz directory")
    parser.add_argument("--target_sr", type=int, default=24000, help="Sampling rate (default: 24000)")
    parser.add_argument("--min_sec", type=float, default=1.0)
    parser.add_argument("--max_sec", type=float, default=12.0)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2), help="Parallel workers")
    parser.add_argument("--skip_existing", action="store_true", default=True, help="Skip already processed files")
    args = parser.parse_args()

    audio_files = (
        list(args.input_dir.rglob("*.wav")) +
        list(args.input_dir.rglob("*.flac")) +
        list(args.input_dir.rglob("*.mp3")) +
        list(args.input_dir.rglob("*.ogg"))
    )
    total = len(audio_files)
    print(f"Found {total} raw audio files. Starting normalization with {args.workers} worker processes...")

    tasks = []
    for file_path in audio_files:
        rel_path = file_path.relative_to(args.input_dir)
        out_path = args.output_dir / rel_path.with_suffix(".wav")
        tasks.append((file_path, out_path, args.target_sr, args.min_sec, args.max_sec, args.skip_existing))

    passed = 0
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for res in executor.map(_process_single_clip, tasks, chunksize=32):
            if res:
                passed += 1
            done += 1
            if done % 1000 == 0 or done == total:
                print(f"  Processed {done}/{total} clips ({passed} valid, {done/total:.1%})...")

    if total > 0:
        print(f"\nFinished normalization: {passed}/{total} clips passed criteria ({passed/total:.1%}).")
        print(f"Standardized WAVs saved to: {args.output_dir}")
    else:
        print("No audio files found in input directory.")


if __name__ == "__main__":
    main()
