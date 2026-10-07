"""
Manages the mapping of character names (from script/screenplay) to
Kokoro voice IDs. Persists in memory for the duration of a session.
"""

from typing import Optional

CHARACTER_ALIASES: dict[str, str] = {
    "RUDDY": "RUDEUS",
    "RUDY": "RUDEUS",
    "SILFIE": "SYLPHIETTE",
    "SYLPHIE": "SYLPHIETTE",
}

DEFAULT_CHARACTERS: dict[str, dict] = {
    "NARRATOR":   {"voice_id": "af_heart",   "gender": "female", "rvc_model": None},
    "ALICE":      {"voice_id": "af_bella",   "gender": "female", "rvc_model": None},
    "BOB":        {"voice_id": "am_michael", "gender": "male",   "rvc_model": None},
    "SYLPHIETTE": {"voice_id": "af_bella",   "gender": "female", "rvc_model": None},
    "RUDEUS":     {"voice_id": "am_michael", "gender": "male",   "rvc_model": "rudeus"},
}


class CharacterRegistry:
    """
    Maps character names -> Kokoro voice IDs and optional RVC models.
    
    Character names are stored and looked up case-insensitively.
    Unknown characters fall back to the NARRATOR voice.
    """

    def __init__(self) -> None:
        self._chars: dict[str, dict] = {
            k.upper(): dict(v) for k, v in DEFAULT_CHARACTERS.items()
        }

    def _resolve_name(self, name: str) -> str:
        key = name.upper().strip()
        return CHARACTER_ALIASES.get(key, key)

    def add(
        self,
        name: str,
        voice_id: str,
        gender: str = "female",
        rvc_model: Optional[str] = None,
    ) -> None:
        """Add or update a character."""
        key = self._resolve_name(name)
        existing = self._chars.get(key, {})
        self._chars[key] = {
            "voice_id": voice_id,
            "gender": gender,
            "rvc_model": rvc_model if rvc_model is not None else existing.get("rvc_model"),
        }

    def remove(self, name: str) -> None:
        """Remove a character (NARRATOR cannot be removed)."""
        key = self._resolve_name(name)
        if key != "NARRATOR":
            self._chars.pop(key, None)

    def get_voice(self, name: str) -> str:
        """Return the Kokoro voice ID for a character, or NARRATOR's voice if unknown."""
        key = self._resolve_name(name)
        entry = self._chars.get(key)
        if entry:
            return entry["voice_id"]
        return self._chars["NARRATOR"]["voice_id"]

    def get_gender(self, name: str) -> str:
        """Return 'male' or 'female' for SFX selection."""
        key = self._resolve_name(name)
        entry = self._chars.get(key)
        return entry["gender"] if entry else "female"

    def get_rvc_model(self, name: str) -> Optional[str]:
        """Return optional RVC model name (e.g. 'rudeus') or None."""
        key = self._resolve_name(name)
        entry = self._chars.get(key)
        return entry.get("rvc_model") if entry else None

    def set_rvc_model(self, name: str, rvc_model: Optional[str]) -> None:
        """Set or clear the RVC model for a character."""
        key = self._resolve_name(name)
        if key in self._chars:
            self._chars[key]["rvc_model"] = rvc_model

    def all_characters(self) -> list[dict]:
        """Return all registered characters as a list of dicts."""
        return [
            {"name": name, **info}
            for name, info in self._chars.items()
        ]

