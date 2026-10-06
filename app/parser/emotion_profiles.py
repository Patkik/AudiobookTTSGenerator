"""
Emotion tag definitions: maps tag names to synthesis parameter bundles.
EmotionProfile fields:
  speed         - speech rate multiplier (1.0 = normal)
  ipa_prefix    - IPA pitch direction marker prepended to phonemes (↗ ↘ → etc.) or None
  volume_db     - post-processing volume offset in dB (0.0 = no change)
  pause_before_ms - silence injected BEFORE the segment (milliseconds)
  pause_after_ms  - silence injected AFTER the segment (milliseconds)
  sfx_file      - path to SFX WAV to inject before this segment, or None
  silence_ms    - if set, this profile is ONLY silence (no TTS synthesis)
  emotion_token - integer ID used by a trained emotion-conditioned model
  duration_scale - target total duration multiplier retained for training metadata
  unvoiced_duration_scale, vowel_duration_scale, pre_plosive_pause_ms -
      phoneme-level targets consumed by the retrained duration predictor
"""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class EmotionProfile:
    speed: float = 1.0
    ipa_prefix: Optional[str] = None
    volume_db: float = 0.0
    pause_before_ms: int = 0
    pause_after_ms: int = 50
    sfx_file: Optional[str] = None
    silence_ms: Optional[int] = None
    emotion_token: int = 178  # default: neutral
    duration_scale: float = 1.0
    unvoiced_duration_scale: float = 1.0
    vowel_duration_scale: float = 1.0
    pre_plosive_pause_ms: int = 0
    f0_variance_scale: float = 1.0


EMOTION_TOKENS: dict[str, int] = {
    "neutral":   178,
    "happy":     179,
    "sad":       180,
    "angry":     181,
    "fearful":   182,
    "excited":   183,
    "calm":      184,
    "nervous":   185,
    "whisper":   186,
    "shout":     187,
    "laugh":     188,
    "cry":       189,
    "sigh":      190,
    "gasp":      191,
    "dramatic":  192,
    "sarcastic": 193,
    "monotone":  194,
    "hesitant":  195,
    "confident": 196,
}

EMOTION_PROFILES: dict[str, EmotionProfile] = {
    "neutral":    EmotionProfile(speed=1.0,  ipa_prefix="→",  volume_db=0.0,  pause_before_ms=0,   pause_after_ms=50,  emotion_token=178),
    "happy":      EmotionProfile(speed=1.0 / 0.95, f0_variance_scale=1.4, ipa_prefix="↗", volume_db=0.0, pause_before_ms=0, pause_after_ms=50, emotion_token=179, duration_scale=0.95),
    "sad":        EmotionProfile(speed=1.0 / 1.20, ipa_prefix="↘", volume_db=-2.0, pause_before_ms=100, pause_after_ms=150, emotion_token=180, duration_scale=1.20, vowel_duration_scale=1.30),
    "angry":      EmotionProfile(speed=1.15, ipa_prefix="↑",  volume_db=3.0,  pause_before_ms=0,   pause_after_ms=80,  emotion_token=181),
    "fearful":    EmotionProfile(speed=1.25, ipa_prefix="↗",  volume_db=-1.0, pause_before_ms=50,  pause_after_ms=100, emotion_token=182, duration_scale=0.80, vowel_duration_scale=0.80, pre_plosive_pause_ms=40),
    "excited":    EmotionProfile(speed=1.2,  ipa_prefix="↗",  volume_db=1.0,  pause_before_ms=0,   pause_after_ms=50,  emotion_token=183),
    "calm":       EmotionProfile(speed=0.8,  ipa_prefix="→",  volume_db=-1.0, pause_before_ms=50,  pause_after_ms=100, emotion_token=184),
    "nervous":    EmotionProfile(speed=1.1,  ipa_prefix="↗",  volume_db=0.0,  pause_before_ms=0,   pause_after_ms=50,  emotion_token=185),
    "whisper":    EmotionProfile(speed=1.0 / 1.25, ipa_prefix=None, volume_db=-8.0, pause_before_ms=50, pause_after_ms=150, emotion_token=186, duration_scale=1.25, unvoiced_duration_scale=1.25),
    "shout":      EmotionProfile(speed=1.2,  ipa_prefix="↑",  volume_db=4.0,  pause_before_ms=0,   pause_after_ms=100, emotion_token=187),
    "laugh":      EmotionProfile(speed=1.15, ipa_prefix=None, volume_db=0.0,  pause_before_ms=50,  pause_after_ms=100, sfx_file="sfx/laugh.wav", emotion_token=188),
    "cry":        EmotionProfile(speed=0.8,  ipa_prefix="↘",  volume_db=-2.0, pause_before_ms=100, pause_after_ms=200, sfx_file="sfx/cry.wav",   emotion_token=189),
    "sigh":       EmotionProfile(speed=0.85, ipa_prefix="↘",  volume_db=-3.0, pause_before_ms=50,  pause_after_ms=150, sfx_file="sfx/sigh.wav",  emotion_token=190),
    "gasp":       EmotionProfile(speed=1.0,  ipa_prefix=None, volume_db=0.0,  pause_before_ms=0,   pause_after_ms=50,  sfx_file="sfx/gasp.wav",  emotion_token=191),
    "dramatic":   EmotionProfile(speed=0.8,  ipa_prefix="↘",  volume_db=0.0,  pause_before_ms=100, pause_after_ms=200, emotion_token=192),
    "sarcastic":  EmotionProfile(speed=0.95, ipa_prefix="↗",  volume_db=0.0,  pause_before_ms=0,   pause_after_ms=50,  emotion_token=193),
    "monotone":   EmotionProfile(speed=1.0,  ipa_prefix="→",  volume_db=-1.0, pause_before_ms=0,   pause_after_ms=30,  emotion_token=194),
    "hesitant":   EmotionProfile(speed=0.8,  ipa_prefix=None, volume_db=0.0,  pause_before_ms=50,  pause_after_ms=100, emotion_token=195),
    "confident":  EmotionProfile(speed=1.05, ipa_prefix="→",  volume_db=1.0,  pause_before_ms=0,   pause_after_ms=50,  emotion_token=196),
    # Pacing-only (no TTS synthesis, just silence injection)
    "pause":      EmotionProfile(silence_ms=500,  emotion_token=178),
    "long pause": EmotionProfile(silence_ms=1200, emotion_token=178),
    "short pause":EmotionProfile(silence_ms=250,  emotion_token=178),
}

# Aliases: normalize variant forms to canonical tag name
_ALIASES: dict[str, str] = {
    "whispering": "whisper",
    "shouting":   "shout",
    "shouted":    "shout",
    "crying":     "cry",
    "sobbing":    "cry",
    "laughing":   "laugh",
    "laughs":     "laugh",
    "chuckles":   "laugh",
    "giggles":    "laugh",
    "sighing":    "sigh",
    "sighs":      "sigh",
    "gasping":    "gasp",
    "gasps":      "gasp",
    "scared":     "fearful",
    "frightened": "fearful",
    "panicked":   "fearful",
    "energetic":  "excited",
    "cheerful":   "happy",
    "cheerfully": "happy",
    "sorrowful":  "sad",
    "melancholic":"sad",
    "furious":    "angry",
    "serene":     "calm",
    "quietly":    "calm",
    "slowly":     "dramatic",
    "deadpan":    "monotone",
    "flatly":     "monotone",
    "trembling":  "fearful",
    "shaky":      "nervous",
}


def resolve_profile(tag: str) -> EmotionProfile:
    """
    Return the EmotionProfile for a tag string.
    Handles case-insensitivity, alias resolution, and unknown tags (-> neutral).
    
    Args:
        tag: Raw tag string from parser (e.g. "HAPPY", "whispering", "long pause")
    
    Returns:
        The matching EmotionProfile, or the neutral profile if tag is unknown.
    """
    normalized = tag.strip().lower()
    canonical = _ALIASES.get(normalized, normalized)
    return EMOTION_PROFILES.get(canonical, EMOTION_PROFILES["neutral"])
