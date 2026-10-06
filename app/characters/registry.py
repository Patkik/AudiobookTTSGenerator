"""
Manages the mapping of character names (from script/screenplay) to
Kokoro voice IDs. Optionally persisted to a JSON file.
"""
import json
import os

from app.characters.voices import voice_gender

DEFAULT_CHARACTERS: dict[str, dict] = {
    "NARRATOR": {"voice_id": "af_heart"},
    "ALICE":    {"voice_id": "af_bella"},
    "BOB":      {"voice_id": "am_michael"},
}


class CharacterRegistry:
    """
    Maps character names -> Kokoro voice IDs.

    Character names are stored and looked up case-insensitively.
    Unknown characters fall back to the NARRATOR voice. Gender is derived
    from the voice id rather than stored.
    """

    def __init__(self, path: str | None = None) -> None:
        self._path = path
        self._chars: dict[str, dict] = {}
        self.reset(save=False)
        if path and os.path.exists(path):
            self._load()

    def _load(self) -> None:
        try:
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)
            loaded = {str(k).upper(): {"voice_id": str(v["voice_id"])} for k, v in data.items()}
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return
        if "NARRATOR" not in loaded:
            loaded["NARRATOR"] = dict(DEFAULT_CHARACTERS["NARRATOR"])
        self._chars = loaded

    def _save(self) -> None:
        if not self._path:
            return
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self._chars, f, indent=2)
        except OSError:
            pass

    def add(self, name: str, voice_id: str, gender: str | None = None) -> None:
        """Add or update a character. ``gender`` is ignored (derived from voice)."""
        self._chars[name.upper()] = {"voice_id": voice_id}
        self._save()

    def rename(self, old: str, new: str) -> bool:
        """Rename a character; NARRATOR can't be renamed and the target must be free."""
        old_k, new_k = old.upper(), new.upper()
        if old_k == "NARRATOR" or old_k not in self._chars:
            return False
        if new_k != old_k and new_k in self._chars:
            return False
        self._chars = {(new_k if k == old_k else k): v for k, v in self._chars.items()}
        self._save()
        return True

    def remove(self, name: str) -> bool:
        """Remove a character (NARRATOR cannot be removed). Returns True if removed."""
        key = name.upper()
        if key == "NARRATOR" or key not in self._chars:
            return False
        del self._chars[key]
        self._save()
        return True

    def reset(self, save: bool = True) -> None:
        self._chars = {k.upper(): dict(v) for k, v in DEFAULT_CHARACTERS.items()}
        if save:
            self._save()

    def has(self, name: str) -> bool:
        return name.upper() in self._chars

    def get_voice(self, name: str) -> str:
        """Return the Kokoro voice ID for a character, or NARRATOR's voice if unknown."""
        entry = self._chars.get(name.upper())
        if entry:
            return entry["voice_id"]
        return self._chars["NARRATOR"]["voice_id"]

    def get_gender(self, name: str) -> str:
        """Return 'male' or 'female', derived from the character's voice."""
        return voice_gender(self.get_voice(name))

    def all_characters(self) -> list[dict]:
        """Return all registered characters as a list of dicts."""
        return [
            {"name": name, "voice_id": info["voice_id"], "gender": voice_gender(info["voice_id"])}
            for name, info in self._chars.items()
        ]
