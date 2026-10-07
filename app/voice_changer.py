"""Recorded-speech conversion and consented training-data bundle helpers."""
from __future__ import annotations

import csv
import os
from pathlib import Path
import re
import tempfile
from typing import Protocol
import zipfile

import numpy as np
import soundfile as sf

from app.characters.registry import CharacterRegistry
from app.parser.tag_parser import TagParser
from app.synthesis.engine import SynthesisEngine

SAMPLE_RATE = 24_000
DEFAULT_TRANSCRIPTION_MODEL = "base.en"
_SAFE_NAME = re.compile(r"[^a-zA-Z0-9_-]+")


class Transcriber(Protocol):
    """Minimal protocol that keeps the UI independent of the ASR implementation."""

    def transcribe(self, recording_path: str) -> str:
        """Return a transcript for a recorded speech file."""


class FasterWhisperTranscriber:
    """Lazy CPU ASR adapter used only when a user requests transcription."""

    def __init__(self, model_name: str = DEFAULT_TRANSCRIPTION_MODEL) -> None:
        self._model_name = model_name
        self._model = None

    def transcribe(self, recording_path: str) -> str:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "Speech transcription requires faster-whisper. Install the project "
                "requirements and restart the app."
            ) from exc

        if self._model is None:
            self._model = WhisperModel(
                self._model_name,
                device="cpu",
                compute_type="int8",
            )

        segments, _ = self._model.transcribe(recording_path, language="en", vad_filter=True)
        transcript = " ".join(segment.text.strip() for segment in segments).strip()
        if not transcript:
            raise ValueError("No intelligible speech was detected in the recording.")
        return transcript


class VoiceChanger:
    """Converts recorded speech content into a selected Kokoro voice."""

    def __init__(
        self,
        engine: SynthesisEngine,
        transcriber: Transcriber | None = None,
    ) -> None:
        self._engine = engine
        self._transcriber = transcriber or FasterWhisperTranscriber()
        self._parser = TagParser()

    def transcribe(self, recording_path: str | None) -> str:
        """Transcribe a microphone/upload recording after basic input validation."""
        self._validate_recording(recording_path)
        return self._transcriber.transcribe(recording_path)

    def convert(
        self,
        recording_path: str | None,
        transcript: str,
        voice_id: str,
        emotion: str = "neutral",
        speed: float = 1.0,
    ) -> str:
        """Render a confirmed transcript in the chosen voice and return a WAV path."""
        self._validate_recording(recording_path)
        if not transcript or not transcript.strip():
            raise ValueError("Transcribe the recording or enter a transcript before converting.")
        if not voice_id:
            raise ValueError("Choose a target voice before converting.")
        if not 0.5 <= speed <= 2.0:
            raise ValueError("Speed must be between 0.5 and 2.0.")

        target_registry = CharacterRegistry()
        target_registry.add("NARRATOR", voice_id)
        segments = self._parser.parse(f"[{emotion}] {transcript.strip()}")
        if not segments:
            raise ValueError("The transcript did not produce any speakable segments.")

        previous_speed_scale = getattr(self._engine, "speed_scale", 1.0)
        try:
            self._engine.speed_scale = speed
            audio, sample_rate = self._engine.synthesize_segments(segments, target_registry)
        finally:
            self._engine.speed_scale = previous_speed_scale

        if len(audio) == 0:
            raise RuntimeError("Voice conversion produced no audio.")
        return self._write_temp_wav(audio, sample_rate)

    @staticmethod
    def _validate_recording(recording_path: str | None) -> None:
        if not recording_path or not os.path.isfile(recording_path):
            raise ValueError("Record or upload an audio file first.")
        try:
            info = sf.info(recording_path)
        except RuntimeError as exc:
            raise ValueError("The supplied recording is not a readable audio file.") from exc
        if info.duration <= 0:
            raise ValueError("The supplied recording is empty.")
        if info.duration > 120:
            raise ValueError("Recordings must be two minutes or shorter.")

    @staticmethod
    def _write_temp_wav(audio: np.ndarray, sample_rate: int) -> str:
        output = tempfile.NamedTemporaryFile(prefix="kokoro-converted-", suffix=".wav", delete=False)
        output.close()
        sf.write(output.name, np.asarray(audio, dtype=np.float32), sample_rate)
        return output.name


def create_training_bundle(
    recording_path: str | None,
    transcript: str,
    speaker_id: str,
    emotion: str,
    consent_confirmed: bool,
) -> str:
    """Package an original recording and reviewed metadata for later curation.

    Generated audio is deliberately excluded: synthetic outputs should not be
    used as ground-truth voice-training material.
    """
    VoiceChanger._validate_recording(recording_path)
    if not consent_confirmed:
        raise ValueError("Confirm that you have permission to use this recording for training.")
    if not transcript or not transcript.strip():
        raise ValueError("A reviewed transcript is required for a training bundle.")

    normalized_speaker = _SAFE_NAME.sub("_", (speaker_id or "").strip()).strip("_")
    if not normalized_speaker:
        raise ValueError("Enter a speaker ID using letters, numbers, hyphens, or underscores.")

    audio, source_rate = sf.read(recording_path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = _resample(audio, source_rate, SAMPLE_RATE)

    bundle_dir = Path(tempfile.mkdtemp(prefix="kokoro-training-"))
    wav_name = f"{normalized_speaker}_{Path(recording_path).stem}.wav"
    wav_path = bundle_dir / wav_name
    sf.write(wav_path, audio, SAMPLE_RATE, subtype="PCM_16")

    metadata_path = bundle_dir / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as metadata_file:
        writer = csv.DictWriter(
            metadata_file,
            fieldnames=["filename", "transcript", "speaker_id", "emotion", "source"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "filename": wav_name,
                "transcript": transcript.strip(),
                "speaker_id": normalized_speaker,
                "emotion": (emotion or "neutral").strip().lower(),
                "source": "user_recording",
            }
        )

    readme_path = bundle_dir / "README.txt"
    readme_path.write_text(
        "This bundle contains an original, consented recording and reviewed metadata.\n"
        "Run training/prepare_dataset.py before adding it to a Kokoro manifest.\n"
        "The generated conversion preview is intentionally excluded from training data.\n",
        encoding="utf-8",
    )

    archive = tempfile.NamedTemporaryFile(prefix="kokoro-training-", suffix=".zip", delete=False)
    archive.close()
    with zipfile.ZipFile(archive.name, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for path in (wav_path, metadata_path, readme_path):
            output.write(path, path.name)
    return archive.name


def _resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample mono audio without adding a runtime dependency to the web UI."""
    if source_rate == target_rate:
        return np.asarray(audio, dtype=np.float32)
    target_length = round(len(audio) * target_rate / source_rate)
    positions = np.linspace(0, len(audio) - 1, target_length, dtype=np.float64)
    return np.interp(positions, np.arange(len(audio)), audio).astype(np.float32)
