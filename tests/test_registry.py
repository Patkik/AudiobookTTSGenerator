import pytest
from app.characters.registry import CharacterRegistry

def test_default_narrator_voice():
    reg = CharacterRegistry()
    assert reg.get_voice("NARRATOR") == "af_heart"

def test_add_and_get_character():
    reg = CharacterRegistry()
    reg.add("ALICE", "af_bella", gender="female")
    assert reg.get_voice("ALICE") == "af_bella"

def test_get_unknown_returns_narrator_voice():
    reg = CharacterRegistry()
    assert reg.get_voice("UNKNOWN_CHARACTER") == reg.get_voice("NARRATOR")

def test_case_insensitive_lookup():
    reg = CharacterRegistry()
    reg.add("BOB", "am_michael", gender="male")
    assert reg.get_voice("bob") == reg.get_voice("BOB")

def test_all_characters_includes_defaults():
    reg = CharacterRegistry()
    chars = reg.all_characters()
    names = [c["name"] for c in chars]
    assert "NARRATOR" in names

def test_remove_character():
    reg = CharacterRegistry()
    reg.add("TEMP", "af_bella", gender="female")
    reg.remove("TEMP")
    assert reg.get_voice("TEMP") == reg.get_voice("NARRATOR")

def test_update_voice():
    reg = CharacterRegistry()
    reg.add("ALICE", "af_bella", gender="female")
    reg.add("ALICE", "af_heart", gender="female")  # update
    assert reg.get_voice("ALICE") == "af_heart"
