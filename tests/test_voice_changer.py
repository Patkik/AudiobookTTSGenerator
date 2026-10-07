import csv
from pathlib import Path
import wave
import zipfile

import numpy as np
import pytest
import soundfile as sf

from app.voice_changer import VoiceChanger, create_training_bundle


class FakeTranscriber:
    def transcribe(self, recording_path: str) -> str:
        assert Path(recording_path).is_file()
        return "Hello from the microphone."


class FakeEngine:
    speed_scale = 1.0

    def synthesize_segments(self, segments, registry):
        assert segments[0].text == "Hello from the microphone."
        assert registry.get_voice("NARRATOR") == "af_heart"
        return np.full(240, 0.2, dtype=np.float32), 24_000


@pytest.fixture
def recording(tmp_path):
    path = tmp_path / "recording.wav"
    sf.write(path, np.linspace(-0.25, 0.25, 16_000, dtype=np.float32), 16_000)
    return str(path)


def test_transcribe_and_convert_recording(recording):
    changer = VoiceChanger(FakeEngine(), FakeTranscriber())
    transcript = changer.transcribe(recording)
    output = changer.convert(recording, transcript, "af_heart")
    assert transcript == "Hello from the microphone."
    assert Path(output).is_file()
    info = sf.info(output)
    assert info.samplerate == 24_000
    assert info.frames == 240


def test_training_bundle_contains_original_recording_and_metadata(recording):
    bundle = create_training_bundle(recording, "Reviewed speech.", "speaker_01", "happy", True)
    with zipfile.ZipFile(bundle) as archive:
        names = archive.namelist()
        assert "metadata.csv" in names
        assert "README.txt" in names
        wav_name = next(name for name in names if name.endswith(".wav"))
        assert wav_name.startswith("speaker_01_")
        metadata = list(csv.DictReader(archive.read("metadata.csv").decode().splitlines()))
        assert metadata == [{
            "filename": wav_name,
            "transcript": "Reviewed speech.",
            "speaker_id": "speaker_01",
            "emotion": "happy",
            "source": "user_recording",
        }]


def test_training_bundle_requires_permission_and_reviewed_transcript(recording):
    with pytest.raises(ValueError, match="permission"):
        create_training_bundle(recording, "Reviewed speech.", "speaker_01", "neutral", False)
    with pytest.raises(ValueError, match="reviewed transcript"):
        create_training_bundle(recording, "", "speaker_01", "neutral", True)
