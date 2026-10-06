"""
Prepares raw audio files for Kokoro TTS fine-tuning:
1. Resamples to 24,000 Hz mono using high-accuracy NumPy interpolation
2. Trims leading and trailing silence
3. Normalizes peak amplitude to prevent clipping and ensure consistent volume
4. Filters clips by duration (rejects clips too short or too long)
"""
import os
import argparse
import numpy as np
import soundfile as sf
from typing import Tuple


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int = 24_000) -> np.ndarray:
    """Resample audio array to target sample rate using linear interpolation."""
    if orig_sr == target_sr:
        return audio
    duration = len(audio) / orig_sr
    num_target_samples = int(duration * target_sr)
    orig_times = np.linspace(0, duration, len(audio), endpoint=False)
    target_times = np.linspace(0, duration, num_target_samples, endpoint=False)
    resampled = np.interp(target_times, orig_times, audio)
    return resampled.astype(np.float32)


def trim_silence(audio: np.ndarray, sample_rate: int = 24_000, top_db: float = 30.0) -> np.ndarray:
    """Trim leading and trailing silence based on energy threshold."""
    if len(audio) == 0:
        return audio
    
    frame_length = int(sample_rate * 0.02)  # 20ms
    hop_length = int(sample_rate * 0.01)    # 10ms
    
    if len(audio) < frame_length:
        return audio
        
    num_frames = 1 + (len(audio) - frame_length) // hop_length
    frames = np.lib.stride_tricks.as_strided(
        audio,
        shape=(num_frames, frame_length),
        strides=(audio.strides[0] * hop_length, audio.strides[0]),
    )
    
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    max_rms = np.max(rms)
    if max_rms == 0:
        return audio
        
    db = 20 * np.log10(rms / max_rms + 1e-12)
    active = np.where(db > -top_db)[0]
    
    if len(active) == 0:
        return audio
        
    start_frame = max(0, active[0] - 2)
    end_frame = min(num_frames, active[-1] + 3)
    
    start_sample = start_frame * hop_length
    end_sample = min(len(audio), end_frame * hop_length + frame_length)
    
    return audio[start_sample:end_sample]


def normalize_audio(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Normalize peak amplitude to target_peak (e.g. 0.95)."""
    peak = np.max(np.abs(audio))
    if peak == 0:
        return audio
    return (audio * (target_peak / peak)).astype(np.float32)


def process_audio_file(
    in_path: str,
    out_path: str,
    target_sr: int = 24_000,
    min_sec: float = 1.0,
    max_sec: float = 12.0,
) -> Tuple[bool, str]:
    """
    Process a single audio file and write to out_path.
    Returns (success, message).
    """
    try:
        data, sr = sf.read(in_path, dtype="float32", always_2d=False)
    except Exception as e:
        return False, f"Failed to read {in_path}: {e}"
        
    # Convert stereo to mono
    if data.ndim > 1:
        data = np.mean(data, axis=1)
        
    # Resample if needed
    if sr != target_sr:
        data = resample_audio(data, sr, target_sr)
        
    # Trim silence
    data = trim_silence(data, target_sr, top_db=30.0)
    
    # Check duration bounds
    dur = len(data) / target_sr
    if dur < min_sec:
        return False, f"Audio too short ({dur:.2f}s < {min_sec}s)"
    if dur > max_sec:
        return False, f"Audio too long ({dur:.2f}s > {max_sec}s)"
        
    # Normalize peak
    data = normalize_audio(data, target_peak=0.95)
    
    # Ensure directory exists and write
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sf.write(out_path, data, target_sr)
    return True, f"Processed ({dur:.2f}s)"


def batch_prepare(input_dir: str, output_dir: str, target_sr: int = 24_000, min_sec: float = 1.0, max_sec: float = 12.0):
    valid_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    os.makedirs(output_dir, exist_ok=True)
    
    processed = 0
    skipped = 0
    
    for root, _, files in os.walk(input_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in valid_exts:
                in_file = os.path.join(root, f)
                rel_path = os.path.relpath(in_file, input_dir)
                out_name = os.path.splitext(rel_path)[0] + ".wav"
                out_file = os.path.join(output_dir, out_name)
                
                success, msg = process_audio_file(in_file, out_file, target_sr, min_sec, max_sec)
                if success:
                    processed += 1
                else:
                    skipped += 1
                    
    print(f"Dataset preparation complete: {processed} files processed, {skipped} files skipped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare audio files for Kokoro TTS fine-tuning.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with raw audio files")
    parser.add_argument("--output_dir", type=str, required=True, help="Destination directory for processed 24kHz WAVs")
    parser.add_argument("--target_sr", type=int, default=24000, help="Target sample rate (default 24000)")
    parser.add_argument("--min_sec", type=float, default=1.0, help="Minimum duration in seconds")
    parser.add_argument("--max_sec", type=float, default=12.0, help="Maximum duration in seconds")
    args = parser.parse_args()
    
    batch_prepare(args.input_dir, args.output_dir, args.target_sr, args.min_sec, args.max_sec)
