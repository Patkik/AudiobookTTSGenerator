#!/usr/bin/env python3
"""
download_datasets.py - Automated downloader and unpacker for Kokoro fine-tuning datasets.

Supports:
1. RAVDESS (Ryerson Emotional Speech and Song Database) from birgermoell/ravdess
2. ESD (Emotional Speech Database) from jspaulsen/esd
3. Meta Expresso (expressive read speech) from ylacombe/expresso

Saves extracted WAV files to data/raw_datasets/<DATASET>/ and compiles
a unified data/metadata.csv for training.
"""

import argparse
import csv
import io
import os
from pathlib import Path
import sys
from typing import Dict, List

# Suppress HF symlink warnings on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

from huggingface_hub import hf_hub_download, snapshot_download
import pyarrow.parquet as pq


RAVDESS_EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

RAVDESS_STATEMENTS = {
    "01": "Kids are talking by the door.",
    "02": "Dogs are sitting by the door.",
}


def download_ravdess(raw_dir: Path) -> List[Dict[str, str]]:
    """Downloads RAVDESS dataset from Hugging Face and extracts metadata."""
    ravdess_dir = raw_dir / "RAVDESS"
    ravdess_dir.mkdir(parents=True, exist_ok=True)
    print("\n[1/3] Downloading RAVDESS from birgermoell/ravdess...")

    snapshot_download(
        repo_id="birgermoell/ravdess",
        repo_type="dataset",
        allow_patterns="Actor_*/*.wav",
        local_dir=ravdess_dir,
    )

    records = []
    wav_files = sorted(ravdess_dir.rglob("*.wav"))
    print(f"  Found {len(wav_files)} RAVDESS WAV files. Indexing...")

    for wav_path in wav_files:
        stem = wav_path.stem
        parts = stem.split("-")
        if len(parts) != 7:
            continue

        modality, vocal_chan, emotion_code, intensity_code, stmt_code, rep_code, actor_code = parts
        # Modality: 03 is audio-only; 01 is full AV audio
        emotion = RAVDESS_EMOTIONS.get(emotion_code, "neutral")
        if intensity_code == "02":
            emotion = f"strong_{emotion}"
        statement = RAVDESS_STATEMENTS.get(stmt_code, "Kids are talking by the door.")
        speaker = f"Actor_{int(actor_code):02d}"

        rel_path = wav_path.relative_to(raw_dir).as_posix()
        records.append({
            "filename": rel_path,
            "transcript": statement,
            "speaker_id": speaker,
            "emotion": emotion,
            "dataset": "RAVDESS",
        })

    print(f"  Indexed {len(records)} RAVDESS recordings.")
    return records


def download_esd(raw_dir: Path, english_only: bool = True) -> List[Dict[str, str]]:
    """Downloads ESD dataset shards and unpacks audio WAVs."""
    esd_dir = raw_dir / "ESD"
    esd_dir.mkdir(parents=True, exist_ok=True)
    lang_desc = "English only" if english_only else "English + Chinese"
    print(f"\n[2/3] Downloading ESD from jspaulsen/esd ({lang_desc})...")

    records = []
    total_shards = 7

    for shard_idx in range(total_shards):
        shard_name = f"data/train-{shard_idx:05d}-of-{total_shards:05d}.parquet"
        print(f"  Downloading shard {shard_idx + 1}/{total_shards}: {shard_name}...")
        try:
            parquet_file = hf_hub_download(
                repo_id="jspaulsen/esd",
                filename=shard_name,
                repo_type="dataset",
            )
        except Exception as e:
            print(f"  Warning: Failed to download shard {shard_name}: {e}")
            continue

        pf = pq.ParquetFile(parquet_file)
        table = pf.read()

        row_count = len(table)
        for i in range(row_count):
            lang = table["language"][i].as_py()
            if english_only and lang != "en":
                continue

            speaker_id = str(table["speaker_id"][i].as_py()).zfill(4)
            emotion = str(table["emotion"][i].as_py()).lower()
            transcript = str(table["transcript"][i].as_py()).strip()
            audio_obj = table["audio"][i].as_py()

            if not audio_obj or not audio_obj.get("bytes"):
                continue

            spk_dir = esd_dir / f"Speaker_{speaker_id}"
            spk_dir.mkdir(parents=True, exist_ok=True)

            out_filename = f"{speaker_id}_{emotion}_{shard_idx:02d}_{i:05d}.wav"
            out_path = spk_dir / out_filename

            if not out_path.exists():
                with open(out_path, "wb") as f:
                    f.write(audio_obj["bytes"])

            rel_path = out_path.relative_to(raw_dir).as_posix()
            records.append({
                "filename": rel_path,
                "transcript": transcript,
                "speaker_id": f"ESD_{speaker_id}",
                "emotion": emotion,
                "dataset": "ESD",
            })

        print(f"    Shard {shard_idx + 1} processed. Total ESD records so far: {len(records)}")

    print(f"  Successfully extracted {len(records)} ESD utterances.")
    return records


def download_expresso(raw_dir: Path, max_shards: int = 12) -> List[Dict[str, str]]:
    """Downloads Expresso read-speech dataset shards and unpacks audio WAVs."""
    expresso_dir = raw_dir / "Expresso"
    expresso_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[3/3] Downloading Expresso from ylacombe/expresso...")

    records = []
    total_shards = 12
    shards_to_fetch = min(max_shards, total_shards)

    for shard_idx in range(shards_to_fetch):
        shard_name = f"read/train-{shard_idx:05d}-of-{total_shards:05d}.parquet"
        print(f"  Downloading shard {shard_idx + 1}/{shards_to_fetch}: {shard_name}...")
        try:
            parquet_file = hf_hub_download(
                repo_id="ylacombe/expresso",
                filename=shard_name,
                repo_type="dataset",
            )
        except Exception as e:
            print(f"  Warning: Failed to download shard {shard_name}: {e}")
            continue

        pf = pq.ParquetFile(parquet_file)
        table = pf.read()

        row_count = len(table)
        for i in range(row_count):
            speaker_id = str(table["speaker_id"][i].as_py())
            style = str(table["style"][i].as_py()).lower()
            text = str(table["text"][i].as_py()).strip()
            item_id = str(table["id"][i].as_py()) if "id" in table.column_names else f"{speaker_id}_{style}_{i:05d}"
            audio_obj = table["audio"][i].as_py()

            if not audio_obj or not audio_obj.get("bytes"):
                continue

            spk_dir = expresso_dir / speaker_id
            spk_dir.mkdir(parents=True, exist_ok=True)

            out_filename = f"{item_id}.wav"
            out_path = spk_dir / out_filename

            if not out_path.exists():
                with open(out_path, "wb") as f:
                    f.write(audio_obj["bytes"])

            rel_path = out_path.relative_to(raw_dir).as_posix()
            records.append({
                "filename": rel_path,
                "transcript": text,
                "speaker_id": f"Expresso_{speaker_id}",
                "emotion": style,
                "dataset": "Expresso",
            })

        print(f"    Shard {shard_idx + 1} processed. Total Expresso records so far: {len(records)}")

    print(f"  Successfully extracted {len(records)} Expresso utterances.")
    return records


def save_metadata(records: List[Dict[str, str]], output_csv: Path):
    """Saves records to unified metadata CSV."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["filename", "transcript", "speaker_id", "emotion", "dataset"]

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"\nMetadata CSV compiled successfully: {output_csv} ({len(records)} entries)")


def main():
    parser = argparse.ArgumentParser(description="Download fine-tuning datasets for Kokoro TTS.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["ravdess", "esd", "expresso"],
        choices=["ravdess", "esd", "expresso"],
        help="Datasets to download",
    )
    parser.add_argument(
        "--raw_dir",
        type=Path,
        default=Path("data/raw_datasets"),
        help="Target raw datasets directory",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("data/metadata.csv"),
        help="Path for compiled metadata CSV",
    )
    parser.add_argument(
        "--include_chinese",
        action="store_true",
        help="Include Chinese samples from ESD dataset (default: English only)",
    )
    parser.add_argument(
        "--expresso_shards",
        type=int,
        default=12,
        help="Number of Expresso shards to download (default: 12)",
    )
    args = parser.parse_args()

    all_records: List[Dict[str, str]] = []
    
    # Load existing metadata if present to allow incremental downloads
    if args.output_csv.exists():
        with open(args.output_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_map = {r["filename"]: r for r in reader}
            print(f"Loaded {len(existing_map)} existing metadata records.")
    else:
        existing_map = {}

    if "ravdess" in args.datasets:
        ravdess_records = download_ravdess(args.raw_dir)
        for r in ravdess_records:
            existing_map[r["filename"]] = r

    if "esd" in args.datasets:
        esd_records = download_esd(args.raw_dir, english_only=not args.include_chinese)
        for r in esd_records:
            existing_map[r["filename"]] = r

    if "expresso" in args.datasets:
        expresso_records = download_expresso(args.raw_dir, max_shards=args.expresso_shards)
        for r in expresso_records:
            existing_map[r["filename"]] = r

    all_records = list(existing_map.values())
    save_metadata(all_records, args.output_csv)
    print("\nDataset preparation finished! Ready for prepare_dataset.py and phonemize_dataset.py.")


if __name__ == "__main__":
    main()
