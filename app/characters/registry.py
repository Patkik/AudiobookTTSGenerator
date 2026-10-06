"""
Manages the mapping of character names (from script/screenplay) to
Kokoro voice IDs. Persists in memory for the duration of a session.
"""

DEFAULT_CHARACTERS: dict[str, dict] = {
    "NARRATOR": {"voice_id": "af_heart",   "gender": "female"},
    "ALICE":    {"voice_id": "af_bella",   "gender": "female"},
    "BOB":      {"voice_id": "am_michael", "gender": "male"},
}


class CharacterRegistry:
    """
    Maps character names -> Kokoro voice IDs.
    
    Character names are stored and looked up case-insensitively.
    Unknown characters fall back to the NARRATOR voice.
    """

    def __init__(self) -> None:
        self._chars: dict[str, dict] = {
            k.upper(): dict(v) for k, v in DEFAULT_CHARACTERS.items()
        }

    def add(self, name: str, voice_id: str, gender: str = "female") -> None:
        """Add or update a character."""
        self._chars[name.upper()] = {"voice_id": voice_id, "gender": gender}

    def remove(self, name: str) -> None:
        """Remove a character (NARRATOR cannot be removed)."""
        key = name.upper()
        if key != "NARRATOR":
            self._chars.pop(key, None)

    def get_voice(self, name: str) -> str:
        """Return the Kokoro voice ID for a character, or NARRATOR's voice if unknown."""
        entry = self._chars.get(name.upper())
        if entry:
            return entry["voice_id"]
        return self._chars["NARRATOR"]["voice_id"]

    def get_gender(self, name: str) -> str:
        """Return 'male' or 'female' for SFX selection."""
        entry = self._chars.get(name.upper())
        return entry["gender"] if entry else "female"

    def all_characters(self) -> list[dict]:
        """Return all registered characters as a list of dicts."""
        return [
            {"name": name, **info}
            for name, info in self._chars.items()
        ]
