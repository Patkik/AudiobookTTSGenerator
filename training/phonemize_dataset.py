"""
Converts text transcripts into Kokoro IPA phonemes, validates against the
Kokoro vocabulary, and outputs training manifests (train_list.txt / val_list.txt).

Format:
  path/to/audio.wav|phoneme_string|speaker_id|emotion_id
"""
import os
import random
import argparse
import csv
from typing import List, Tuple
from kokoro_onnx.tokenizer import Tokenizer

_tokenizer = None

def get_tokenizer() -> Tokenizer:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = Tokenizer()
    return _tokenizer


def phonemize_text(text: str, lang: str = "en-us") -> str:
    """Convert text to Kokoro IPA phonemes."""
    t = get_tokenizer()
    phonemes = t.phonemize(text, lang=lang)
    return phonemes.strip()


def build_manifest_entry(
    wav_path: str,
    text: str,
    speaker_id: str = "speaker_00",
    emotion: str = "neutral",
    lang: str = "en-us",
) -> str:
    """Generate a single formatted manifest row."""
    phonemes = phonemize_text(text, lang=lang)
    # Forward slashes for portable paths
    norm_path = wav_path.replace("\\", "/")
    return f"{norm_path}|{phonemes}|{speaker_id}|{emotion}"


def split_dataset(entries: List[str], val_ratio: float = 0.05, seed: int = 42) -> Tuple[List[str], List[str]]:
    """Deterministically split manifest entries into train and validation sets."""
    rng = random.Random(seed)
    shuffled = list(entries)
    rng.shuffle(shuffled)
    
    val_size = int(len(shuffled) * val_ratio)
    val_set = shuffled[:val_size]
    train_set = shuffled[val_size:]
    return train_set, val_set


def process_metadata_csv(
    csv_path: str,
    output_dir: str,
    wav_dir: str = "",
    val_ratio: float = 0.05,
):
    """
    Reads a CSV metadata file with columns:
    [filename, transcript, speaker_id, emotion]
    and writes train_list.txt and val_list.txt to output_dir.
    """
    entries = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.get("filename") or row.get("wav") or row.get("audio")
            text = row.get("transcript") or row.get("text")
            speaker = row.get("speaker_id") or row.get("speaker") or "speaker_00"
            emotion = row.get("emotion") or "neutral"
            
            if not filename or not text:
                continue
                
            full_wav_path = os.path.join(wav_dir, filename) if wav_dir else filename
            entry = build_manifest_entry(full_wav_path, text, speaker, emotion)
            entries.append(entry)
            
    train_entries, val_entries = split_dataset(entries, val_ratio=val_ratio)
    
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train_list.txt")
    val_path = os.path.join(output_dir, "val_list.txt")
    
    with open(train_path, "w", encoding="utf-8") as f:
        f.write("\n".join(train_entries) + "\n")
        
    with open(val_path, "w", encoding="utf-8") as f:
        f.write("\n".join(val_entries) + "\n")
        
    print(f"Manifests generated:")
    print(f"  Train: {len(train_entries)} rows -> {train_path}")
    print(f"  Val:   {len(val_entries)} rows -> {val_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Kokoro TTS training manifests.")
    parser.add_argument("--csv", type=str, required=True, help="Path to metadata CSV (cols: filename, transcript, speaker_id, emotion)")
    parser.add_argument("--wav_dir", type=str, default="", help="Prefix directory for audio files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save train_list.txt and val_list.txt")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="Validation set ratio (default: 0.05)")
    args = parser.parse_args()
    
    process_metadata_csv(args.csv, args.output_dir, args.wav_dir, args.val_ratio)
