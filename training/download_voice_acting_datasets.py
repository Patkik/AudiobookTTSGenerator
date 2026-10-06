#!/usr/bin/env python3
"""
download_voice_acting_datasets.py - Voice Acting & Expressive Dataset Downloader for Kokoro TTS.

Downloads and unpacks curated datasets designed specifically for expressive character acting:
1. deepvk/NonverbalTTS (Apache-2.0): 17h English speech with aligned non-verbal tags (breaths 🌬️, laughs 🤣).
2. mythicinfinity/libritts_r (CC-BY-4.0): High-quality clean neutral speech to anchor speaker identities.
3. ylacombe/expresso (CC-BY-NC-4.0): Expressive studio styles (whisper, laughing, sad, confused).

Compiles unified metadata to data/voice_acting_metadata.csv ready for curation & phonemization.
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Ensure UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq


def download_nonverbal_tts(
    raw_dir: Path,
    shards_to_fetch: int = 2,
    min_dnsmos: float = 3.0,
) -> List[Dict[str, str]]:
    """
    Downloads shards from deepvk/NonverbalTTS and extracts audio + nonverbal annotations.
    """
    nonverbal_dir = raw_dir / "NonverbalTTS"
    nonverbal_dir.mkdir(parents=True, exist_ok=True)
    print("\n[1/2] Downloading NonverbalTTS from deepvk/NonverbalTTS (breaths, gasps, laughs)...")

    # Available shards: dev/0000.parquet, train/0000.parquet, train/0001.parquet ...
    candidate_shards = ["default/dev/0000.parquet"] + [
        f"default/train/{i:04d}.parquet" for i in range(shards_to_fetch)
    ]

    records: List[Dict[str, str]] = []

    for shard_idx, shard_name in enumerate(candidate_shards):
        print(f"  Fetching shard {shard_idx + 1}/{len(candidate_shards)}: {shard_name}...")
        try:
            parquet_path = hf_hub_download(
                repo_id="deepvk/NonverbalTTS",
                filename=shard_name,
                repo_type="dataset",
            )
        except Exception as e:
            print(f"  Warning: Could not download {shard_name}: {e}")
            continue

        table = pq.read_table(parquet_path)
        row_count = len(table)
        print(f"    Loaded {row_count} rows from {shard_name}. Extracting audio...")

        extracted_in_shard = 0
        for i in range(row_count):
            dnsmos = table["dnsmos"][i].as_py()
            if dnsmos is not None and dnsmos < min_dnsmos:
                continue

            speaker_id = str(table["speaker_id"][i].as_py() or "unknown")
            emotion = str(table["Emotion"][i].as_py() or "neutral").lower()
            gender = str(table["gender"][i].as_py() or "f").lower()
            text_result = str(table["Result"][i].as_py() or "").strip()
            audio_obj = table["audio"][i].as_py()

            if not audio_obj or not audio_obj.get("bytes") or not text_result:
                continue

            # Identify non-verbal vocalizations
            tags = []
            if "🌬️" in text_result or "breath" in text_result.lower():
                tags.append("breath")
            if "🤣" in text_result or "laugh" in text_result.lower():
                tags.append("laugh")

            # Clean emoji markers for phonemizer while noting emotion
            clean_text = text_result.replace("🌬️", "").replace("🤣", "").strip()
            if not clean_text:
                continue

            spk_dir = nonverbal_dir / speaker_id
            spk_dir.mkdir(parents=True, exist_ok=True)

            out_filename = f"{speaker_id}_{emotion}_{shard_idx:02d}_{i:05d}.wav"
            out_path = spk_dir / out_filename

            if not out_path.exists():
                with open(out_path, "wb") as f:
                    f.write(audio_obj["bytes"])

            rel_path = out_path.relative_to(raw_dir.parent).as_posix()
            records.append({
                "filename": rel_path,
                "transcript": clean_text,
                "speaker_id": f"Nonverbal_{speaker_id}",
                "emotion": emotion if not tags else f"{emotion}_{tags[0]}",
                "dataset": "NonverbalTTS",
            })
            extracted_in_shard += 1

        print(f"    Extracted {extracted_in_shard} high-quality clips from {shard_name}. Total: {len(records)}")

    return records


def download_libritts_r_clean(
    raw_dir: Path,
    max_shards: int = 1,
) -> List[Dict[str, str]]:
    """
    Downloads clean neutral speech from mythicinfinity/libritts_r dev.clean.
    Anchors speaker identities and prevents voice drift.
    """
    libri_dir = raw_dir / "LibriTTS_R"
    libri_dir.mkdir(parents=True, exist_ok=True)
    print("\n[2/2] Downloading clean neutral baseline from mythicinfinity/libritts_r...")

    records: List[Dict[str, str]] = []

    for shard_idx in range(max_shards):
        shard_name = f"data/dev.clean/dev.clean-{shard_idx:05d}-of-00004.parquet"
        print(f"  Fetching LibriTTS-R shard {shard_idx + 1}/{max_shards}: {shard_name}...")
        try:
            parquet_path = hf_hub_download(
                repo_id="mythicinfinity/libritts_r",
                filename=shard_name,
                repo_type="dataset",
            )
        except Exception as e:
            print(f"  Warning: Could not download {shard_name}: {e}")
            continue

        table = pq.read_table(parquet_path)
        row_count = len(table)
        print(f"    Loaded {row_count} rows from {shard_name}. Extracting...")

        extracted_in_shard = 0
        for i in range(row_count):
            speaker_id = str(table["speaker_id"][i].as_py())
            text = str(
                table["text_normalized"][i].as_py()
                if "text_normalized" in table.column_names
                else table["text"][i].as_py()
            ).strip()
            audio_obj = table["audio"][i].as_py()

            if not audio_obj or not audio_obj.get("bytes") or not text:
                continue

            spk_dir = libri_dir / f"Speaker_{speaker_id}"
            spk_dir.mkdir(parents=True, exist_ok=True)

            out_filename = f"{speaker_id}_neutral_{shard_idx:02d}_{i:05d}.wav"
            out_path = spk_dir / out_filename

            if not out_path.exists():
                with open(out_path, "wb") as f:
                    f.write(audio_obj["bytes"])

            rel_path = out_path.relative_to(raw_dir.parent).as_posix()
            records.append({
                "filename": rel_path,
                "transcript": text,
                "speaker_id": f"Libri_{speaker_id}",
                "emotion": "neutral",
                "dataset": "LibriTTS_R",
            })
            extracted_in_shard += 1

        print(f"    Extracted {extracted_in_shard} neutral baseline clips. Total: {len(records)}")

    return records


def main():
    parser = argparse.ArgumentParser(description="Download voice acting datasets for Kokoro TTS.")
    parser.add_argument("--output_dir", type=Path, default=Path("data/raw_datasets"), help="Raw datasets directory")
    parser.add_argument("--metadata_csv", type=Path, default=Path("data/voice_acting_metadata.csv"), help="Output metadata CSV")
    parser.add_argument("--nonverbal_shards", type=int, default=1, help="Number of NonverbalTTS train shards to download")
    parser.add_argument("--libritts_shards", type=int, default=1, help="Number of LibriTTS-R dev shards to download")
    parser.add_argument("--skip_libritts", action="store_true", help="Skip LibriTTS-R")
    parser.add_argument("--skip_nonverbal", action="store_true", help="Skip NonverbalTTS")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_records: List[Dict[str, str]] = []

    if not args.skip_nonverbal:
        nonverbal_records = download_nonverbal_tts(args.output_dir, shards_to_fetch=args.nonverbal_shards)
        all_records.extend(nonverbal_records)

    if not args.skip_libritts:
        libri_records = download_libritts_r_clean(args.output_dir, max_shards=args.libritts_shards)
        all_records.extend(libri_records)

    if all_records:
        args.metadata_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.metadata_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "transcript", "speaker_id", "emotion", "dataset"])
            writer.writeheader()
            writer.writerows(all_records)
        print(f"\n[SUCCESS] Saved {len(all_records)} voice acting records to {args.metadata_csv}")
    else:
        print("\nNo records extracted.")


if __name__ == "__main__":
    main()
