"""
Manages the mapping of character names (from script/screenplay) to
Kokoro voice IDs. Optionally persisted to a JSON file.
"""
import json
import os
from typing import Optional

from app.characters.voices import voice_gender

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
    Unknown characters fall back to the NARRATOR voice. Gender is derived
    from the voice id rather than stored.
    """

    def __init__(self, path: str | None = None) -> None:
        self._path = path
        self._chars: dict[str, dict] = {}
        self.reset(save=False)
        if path and os.path.exists(path):
            self._load()

    def _resolve_name(self, name: str) -> str:
        key = name.upper().strip()
        return CHARACTER_ALIASES.get(key, key)

    def _load(self) -> None:
        try:
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)
            loaded = {
                str(k).upper(): {
                    "voice_id": str(v["voice_id"]),
                    "rvc_model": v.get("rvc_model"),
                }
                for k, v in data.items()
            }
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

    def add(
        self,
        name: str,
        voice_id: str,
        gender: Optional[str] = None,
        rvc_model: Optional[str] = None,
    ) -> None:
        """Add or update a character."""
        key = self._resolve_name(name)
        existing = self._chars.get(key, {})
        self._chars[key] = {
            "voice_id": voice_id,
            "gender": voice_gender(voice_id),
            "rvc_model": rvc_model if rvc_model is not None else existing.get("rvc_model"),
        }
        self._save()

    def rename(self, old: str, new: str) -> bool:
        """Rename a character; NARRATOR can't be renamed and the target must be free."""
        old_k, new_k = self._resolve_name(old), self._resolve_name(new)
        if old_k == "NARRATOR" or old_k not in self._chars:
            return False
        if new_k != old_k and new_k in self._chars:
            return False
        self._chars = {(new_k if k == old_k else k): v for k, v in self._chars.items()}
        self._save()
        return True

    def remove(self, name: str) -> bool:
        """Remove a character (NARRATOR cannot be removed). Returns True if removed."""
        key = self._resolve_name(name)
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
        key = self._resolve_name(name)
        return key in self._chars

    def get_voice(self, name: str) -> str:
        """Return the Kokoro voice ID for a character, or NARRATOR's voice if unknown."""
        key = self._resolve_name(name)
        entry = self._chars.get(key)
        if entry:
            return entry["voice_id"]
        return self._chars["NARRATOR"]["voice_id"]

    def get_gender(self, name: str) -> str:
        """Return 'male' or 'female', derived from the character's voice."""
        return voice_gender(self.get_voice(name))

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
            self._save()

    def all_characters(self) -> list[dict]:
        """Return all registered characters as a list of dicts."""
        return [
            {
                "name": name,
                "voice_id": info["voice_id"],
                "gender": voice_gender(info["voice_id"]),
                "rvc_model": info.get("rvc_model"),
            }
            for name, info in self._chars.items()
        ]
