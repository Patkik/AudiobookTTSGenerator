"""Gradio-free character CRUD operations returning (rows, message, level)."""
from app.characters.registry import CharacterRegistry
from app.characters.voices import voice_accent, voice_gender

HEADERS = ["Name", "Voice", "Gender", "Accent"]


def table_rows(reg: CharacterRegistry) -> list[list[str]]:
    return [
        [c["name"], c["voice_id"], c["gender"].title(), voice_accent(c["voice_id"])]
        for c in reg.all_characters()
    ]


def _clean(name: str | None) -> str:
    return " ".join((name or "").split()).upper()


def add_character(reg: CharacterRegistry, name: str, voice: str) -> tuple[bool, str]:
    key = _clean(name)
    if not key:
        return False, "Enter a character name."
    if not voice:
        return False, "Choose a voice."
    if reg.has(key):
        return False, f"{key} already exists. Select it and use Update."
    reg.add(key, voice)
    return True, f"Added {key} ({voice_gender(voice)})."


def update_character(reg: CharacterRegistry, original: str, name: str, voice: str) -> tuple[bool, str]:
    orig, key = _clean(original), _clean(name)
    if not orig or not reg.has(orig):
        return False, "Select a character in the table first."
    if not key:
        return False, "Enter a character name."
    if not voice:
        return False, "Choose a voice."
    if key != orig and not reg.rename(orig, key):
        reason = "NARRATOR can't be renamed." if orig == "NARRATOR" else f"{key} already exists."
        return False, reason
    reg.add(key, voice)
    return True, f"Updated {key}."


def delete_character(reg: CharacterRegistry, name: str) -> tuple[bool, str]:
    key = _clean(name)
    if not key or not reg.has(key):
        return False, "Select a character in the table first."
    if not reg.remove(key):
        return False, "NARRATOR can't be deleted."
    return True, f"Deleted {key}."
