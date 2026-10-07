"""Kokoro voice-pack metadata derived from the voice id (e.g. ``af_heart``)."""

_ACCENTS = {"a": "American", "b": "British", "e": "Spanish", "f": "French",
            "h": "Hindi", "i": "Italian", "j": "Japanese", "p": "Portuguese", "z": "Mandarin"}
_GENDERS = {"f": "female", "m": "male"}

FALLBACK_VOICES = ["af_heart", "af_bella", "am_michael", "bm_george"]


def voice_gender(voice_id: str) -> str:
    """'female' or 'male' from the voice prefix; defaults to female if unknown."""
    return _GENDERS.get(voice_id[1:2], "female") if len(voice_id) > 1 else "female"


def voice_accent(voice_id: str) -> str:
    return _ACCENTS.get(voice_id[:1], "Other")


def voice_label(voice_id: str) -> str:
    """Human label, e.g. 'af_heart' -> 'Heart · Female · American'."""
    name = voice_id.split("_", 1)[1] if "_" in voice_id else voice_id
    return f"{name.replace('_', ' ').title()} · {voice_gender(voice_id).title()} · {voice_accent(voice_id)}"


def voice_choices(voice_ids: list[str]) -> list[tuple[str, str]]:
    """Dropdown (label, value) pairs sorted by gender, accent, name."""
    ordered = sorted(voice_ids, key=lambda v: (voice_gender(v), voice_accent(v), v))
    return [(voice_label(v), v) for v in ordered]
