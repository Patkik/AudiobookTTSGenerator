# Kokoro TTS + RVC Expressive Voice Pipeline — Design Specification

**Date:** 2026-10-07  
**Status:** Under Review  
**Target Architecture:** Kokoro-82M (Acting Source) + RVC v2 (Timbre Converter) + Narrative Context Framing  

---

## 1. Executive Summary & Problem Analysis

### 1.1 The Overfitting & Expression Dilemma
When fine-tuning or training Retrieval-based Voice Conversion (RVC) models beyond 300 epochs on standard conversational anime datasets:
- The model overfits to flat, calm, neutral speech.
- When paired with an expressive TTS generator producing extreme pitch inflections or dynamic projections (whispers, shouts, crying), an overfitted RVC model:
  - Fails to track high or unvoiced $F_0$ trajectories.
  - Generates severe metallic phase distortion, audio warbling, and voice cracks.
  - Flattens the actor's intended dynamic range.

### 1.2 The Two-Layer Solution
1. **The Acting Base (Kokoro Layer):** Acting originates in Kokoro. We use **Narrative Context Framing** (injecting dramatic stage directions and hesitation cues into the prosody/timing engine) and **Differential Style Vector Arithmetic** ($\Delta s$) so Kokoro delivers genuine emotional inflections, pauses, and pitch movement.
2. **The Timbre Conversion (RVC Layer):** RVC converts the voice identity (e.g., to Rudeus Greyrat) while faithfully preserving Kokoro's pitch trajectory ($F_0$) and phoneme timing using tuned inference hyperparameters:
   - `index_rate = 0.40` (prevents flattening Kokoro's dynamic acting)
   - `protect = 0.35` (protects unvoiced whisper breath and voiceless consonants `s`, `t`, `sh`)
   - `filter_radius = 3` (smooths pitch transitions to prevent cracks)
   - `volume_envelope = 1.0` (preserves whisper vs. shout volume dynamics)
   - `f0_method = rmvpe` (Robust Multiscale Voice Pitch Estimation)
3. **Dataset Augmentation for Training:**
   - A dedicated dataset preparation pipeline (`training/prepare_rvc_dataset.py`) structuring audio around the **Three-P Framework** (Pitch, Pace, Projection) using our curated NonverbalTTS dataset (whispers, laughter, screams, crying) to enable retraining/fine-tuning without overfitting.

---

## 2. Architecture & Data Flow

```text
User Script (Screenplay / Tags / Dialogue)
  │
  ▼
[ Tag Parser & Narrative Context Resolver ]
  │  • Separates spoken dialogue from acting stage directions
  │  • Resolves hesitation pauses (pause_before_ms), punctuation cadence, speed, volume
  │  • Maps characters to voice identity + RVC model configuration
  │
  ▼
[ Kokoro Expressive Synthesis Engine ]
  │  • Computes identity-preserving style vector (Base timbre dims 0..127 + Emotion offset dims 128..255)
  │  • Synthesizes 24 kHz float32 raw audio with dynamic prosody
  │
  ▼
[ Character Post-Processing Router ]
  ├── Character has NO RVC model (e.g. SYLPHIETTE: af_sky) ────┐
  └── Character HAS RVC model (e.g. RUDEUS: am_michael)        │
        │                                                     │
        ▼                                                     │
  [ RVC Pipeline Engine ]                                     │
    • Loads Rudeus.pth (VITS weights) + Rudeus.index (FAISS)  │
    • Runs RMVPE pitch extraction on Kokoro audio             │
    • Applies index_rate=0.40, protect=0.35, filter_radius=3   │
    • Preserves volume envelope (1.0)                         │
    • Resamples to target rate (24 kHz / 40 kHz)              │
        │                                                     │
        └──────────────────────┬──────────────────────────────┘
                               │
                               ▼
                   [ Audio Mastering & DSP ]
                     • Multiband Compression & Warm Presence EQ
                     • Inter-segment natural crossfades & pauses
                     • Studio limiter (zero clipping)
                               │
                               ▼
                   Final Mastered Audio (WAV / MP3)
```

---

## 3. Subsystem Specifications

### 3.1 Narrative Context Framing (`app/parser/narrative_framing.py`)
- **Purpose:** Contextualize spoken lines so Kokoro renders emotional subtext without speaking stage cues aloud.
- **API:**
  ```python
  class DefaultNarrativeContextResolver:
      def resolve(self, narrative_cue: str, segment: SpeechSegment) -> NarrativeContextFrame:
          ...
  ```
- **Rules:**
  - Stage directions in screenplay headers (e.g. `RUDEUS (desperate, shouting): ...`) or explicit scene descriptions are parsed for acting keywords (e.g. `crying`, `whispering`, `shouting`, `hesitant`, `terrified`, `relieved`).
  - Automatically calculates:
    - `pause_before_ms`: 150–400ms for hesitant/emotional lines.
    - `speed`: 0.85–0.92 for intimate/sad lines; 1.15–1.25 for panicked/urgent lines.
    - `volume_db`: -3dB to -6dB for whispers; +2dB to +4dB for shouts.
    - `alpha`: scales emotional delta intensity up to 0.60 for extreme scenes.
  - Spoken text remains strictly dialogue; context is purely prosodic metadata.

### 3.2 RVC Pipeline Post-Processor (`app/synthesis/rvc_pipeline.py`)
- **Data Model:**
  ```python
  @dataclass
  class RVCConfig:
      model_path: str
      index_path: Optional[str] = None
      index_rate: float = 0.40       # Blueprint: 0.30 - 0.50
      protect: float = 0.35          # Blueprint: 0.33 - 0.50
      filter_radius: int = 3         # Blueprint: 3
      volume_envelope: float = 1.0   # Blueprint: 1.0 (retain dynamics)
      f0_method: str = "rmvpe"       # Blueprint: RMVPE
      f0_up_key: int = 0             # Semitone pitch shift
      target_sample_rate: int = 24_000
  ```
- **Pipeline Architecture:**
  - `RVCPipeline.convert(audio: np.ndarray, config: RVCConfig) -> np.ndarray`:
    1. Validates inputs and paths (`Rudeus.pth`, `Rudeus.index`).
    2. Runs conversion via:
       - Direct runner (via Python 3.11 sidecar with installed torch/rvc or ONNX runtime).
       - External process wrapper with automatic batching and timeout protection.
       - Graceful fallback: if RVC runtime is temporarily offline, yields Kokoro's high-expressiveness base audio with an informative status flag.

### 3.3 Character Registry Upgrades (`app/characters/registry.py`)
- Character definitions support optional `rvc_model` and `rvc_config` overrides.
- Pre-configured defaults:
  - `"SYLPHIETTE"`: `voice_id = "af_sky"`, `gender = "female"`, `rvc_model = None`
  - `"RUDEUS"`: `voice_id = "am_michael"`, `gender = "male"`, `rvc_model = "rudeus"`
  - `"ALICE"`, `"BOB"`, `"NARRATOR"` remain standard Kokoro voices.

### 3.4 Dataset Augmentation Utility (`training/prepare_rvc_dataset.py`)
- **Purpose:** Enables retraining or fine-tuning RVC models without neutral overfitting.
- **Features:**
  - Scans `data/raw_datasets/NonverbalTTS` (666 downloaded clips) and local character audio clips.
  - Categorizes clips into Three-P buckets:
    - *Projection / Unvoiced:* Whispers, soft-spoken, sighs (low HNR).
    - *High Excitation:* Shouts, screams, wide $F_0$ excursions.
    - *Emotional Inflections:* Crying, laughing, breathless lines.
  - Automatically slices, loudness normalizes (-23 LUFS / EBU R128), and converts to 40 kHz WAV mono.
  - Produces ready-to-train file manifests (`rvc_train_manifest.txt`).

### 3.5 Gradio UI Integration (`app/app.py`)
- **Editor Tab:**
  - Added "🎙️ RVC Voice Conversion" accordion.
  - Checkbox to toggle RVC post-processing.
  - Interactive sliders pre-set to blueprint defaults:
    - Index Rate (0.40)
    - Consonant Protection (0.35)
    - Filter Radius (3)
    - Volume Envelope (1.0)
    - Pitch Shift semitones (0)
- **Characters Tab:**
  - Shows RVC Model assignment column (`rudeus`, `none`).

---

## 4. Verification & Testing Plan

1. **Unit Tests (`tests/synthesis/test_rvc_pipeline.py`):**
   - Verify `RVCConfig` validation and default hyperparameter ranges matching blueprint.
   - Verify audio format conversion, resampling, and fallback behavior.
2. **Unit Tests (`tests/parser/test_narrative_framing.py`):**
   - Verify narrative cue parsing extracts emotional frames (whisper, shout, desperation) without leaking text into dialogue.
   - Verify punctuation and hesitation pause calculation.
3. **Registry Tests (`tests/test_registry.py`):**
   - Verify Rudeus is registered with `rvc_model="rudeus"`.
   - Verify backwards-compatibility with existing characters.
4. **End-to-End Dialogue Test:**
   - Synthesize sample dialogue between Sylphiette and Rudeus.
   - Verify audio generation, zero clipping, and clean pipeline execution.
