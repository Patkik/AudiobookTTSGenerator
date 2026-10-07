# Voice changer workflow

The **Voice Changer** tab supports short microphone recordings and audio uploads.
It is an offline speech-to-speech rendering workflow:

1. Record or upload up to two minutes of speech.
2. Select **Transcribe recording** and review or correct the resulting text.
3. Choose a built-in Kokoro target voice, expression, and output speed.
4. Select **Convert to selected voice** to preview and download the rendered WAV.

The feature uses `faster-whisper` for local transcription and Kokoro for output
synthesis. It does not preserve source timing, accent, or vocal identity, and it
is not a low-latency waveform-preserving voice-conversion model.

## Creating a training-data bundle

Use **Bundle original recording for training** only for recordings you are
permitted to use. A reviewed transcript, speaker ID, expression tag, and consent
confirmation are required. The ZIP contains:

- a mono 24 kHz PCM WAV converted from the original recording;
- `metadata.csv` with `filename`, `transcript`, `speaker_id`, `emotion`, and
  `source`; and
- instructions for the existing dataset-preparation workflow.

The synthetic conversion preview is intentionally omitted from this bundle. It
is not ground-truth data and should not be used to train a voice model.
