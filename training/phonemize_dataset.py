#!/usr/bin/env python3
"""
phonemize_dataset.py - G2P Conversion, Vocabulary Reconciliation, and Manifest Generation.
Uses Kokoro's G2P front-end (backed by espeakng-loader),
applies vocabulary reconciliation rules (substituting OOV IPA characters like ʏ -> y, ASCII g -> IPA ɡ),
and generates pipe-delimited training manifests (train_list.txt and val_list.txt).
"""

import argparse
import csv
from pathlib import Path
import random
from typing import List
from kokoro_onnx.tokenizer import Tokenizer

_tokenizer = None


def get_tokenizer() -> Tokenizer:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = Tokenizer()
    return _tokenizer


def reconcile_vocabulary(ipa_str: str) -> str:
    """
    Reconciles out-of-vocabulary IPA characters with Kokoro's fixed 178-token vocabulary.
    Applies phonetic distance mapping (e.g., short-ü 'ʏ' -> close front 'y', ASCII g -> IPA ɡ).
    """
    replacements = {
        "ʏ": "y",  # German short open-mid front rounded vowel -> close front
        "g": "ɡ",  # ASCII g -> IPA Latin small letter script g
        "‒": "-",  # Normalize dash variants
        "—": "-",
        "–": "-",
    }
    for OOV_char, valid_char in replacements.items():
        ipa_str = ipa_str.replace(OOV_char, valid_char)
    return ipa_str


from functools import lru_cache


@lru_cache(maxsize=100000)
def phonemize_text(text: str, lang: str = "en-us") -> str:
    """Phonemize text to IPA and reconcile vocabulary (cached)."""
    t = get_tokenizer()
    phonemes = t.phonemize(text, lang=lang)
    return reconcile_vocabulary(phonemes).strip()


def build_manifest_entry(
    wav_file: Path | str = None,
    raw_text: str = "",
    speaker: str = "hero_speaker",
    emotion: str = "neutral",
    lang: str = "en-us",
    path_prefix: str | None = None,
    wav_rel_path: str | None = None,
) -> str | None:
    """Creates a formatted manifest line: filepath|phonemes|speaker_id|emotion."""
    target_path = wav_file if wav_file is not None else wav_rel_path
    if target_path is None:
        return None
    phonemes = phonemize_text(raw_text, lang=lang)
    if not phonemes:
        return None
    if isinstance(target_path, Path) and path_prefix is None:
        wav_str = str(target_path.resolve()).replace("\\", "/")
    else:
        clean_rel = str(target_path).replace("\\", "/").lstrip("/")
        wav_str = f"{path_prefix.rstrip('/')}/{clean_rel}" if path_prefix and not clean_rel.startswith(path_prefix.rstrip('/')) else clean_rel
    return f"{wav_str}|{phonemes}|{speaker}|{emotion}"


def main():
    import zipfile

    parser = argparse.ArgumentParser(description="Phonemize transcriptions and generate manifests.")
    parser.add_argument("--csv", type=Path, default=Path("data/metadata.csv"), help="Metadata CSV")
    parser.add_argument("--wav_dir", type=Path, default=Path("data/processed_24k"), help="Directory containing processed 24kHz WAVs")
    parser.add_argument("--output_dir", type=Path, default=Path("data/manifests"), help="Manifest output directory")
    parser.add_argument("--path_prefix", type=str, default="/content/data/processed_24k", help="Path prefix for wav files in Colab")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries: List[str] = []
    
    with open(args.csv, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=",")
        for row in reader:
            filename = row.get("filename") or row.get("wav") or row.get("audio")
            if not filename:
                continue

            wav_file = args.wav_dir / filename
            if not wav_file.exists():
                continue

            raw_text = (row.get("transcript") or row.get("text") or "").strip()
            speaker = (row.get("speaker_id") or row.get("speaker") or "hero_speaker").strip()
            emotion = (row.get("emotion") or "neutral").strip()

            line = build_manifest_entry(
                filename, raw_text, speaker, emotion, path_prefix=args.path_prefix
            )
            if line:
                manifest_entries.append(line)
            if len(manifest_entries) % 1000 == 0 and line:
                print(f"  Phonemized {len(manifest_entries)} entries...")

    # Shuffle and split into train / val
    random.shuffle(manifest_entries)
    val_count = int(len(manifest_entries) * args.val_ratio)
    val_lines = manifest_entries[:val_count]
    train_lines = manifest_entries[val_count:]

    train_path = args.output_dir / "train_list.txt"
    val_path = args.output_dir / "val_list.txt"

    with open(train_path, "w", encoding="utf-8") as f:
        f.write("\n".join(train_lines) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        f.write("\n".join(val_lines) + "\n")

    print(f"\nManifests generated successfully in {args.output_dir}:")
    print(f"  - Train samples: {len(train_lines)} ({train_path})")
    print(f"  - Val samples:   {len(val_lines)} ({val_path})")

    # Also package into data/manifests.zip
    zip_path = Path("data/manifests.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(train_path, arcname="train_list.txt")
        zf.write(val_path, arcname="val_list.txt")
    print(f"Packaged {zip_path} ({zip_path.stat().st_size} bytes).")


if __name__ == "__main__":
    main()
