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


def test_rudeus_default_character_with_rvc():
    reg = CharacterRegistry()
    assert reg.get_voice("RUDEUS") == "am_michael"
    assert reg.get_gender("RUDEUS") == "male"
    assert reg.get_rvc_model("RUDEUS") == "rudeus"


def test_sylphiette_default_character_without_rvc():
    reg = CharacterRegistry()
    assert reg.get_voice("SYLPHIETTE") == "af_nicole"
    assert reg.get_gender("SYLPHIETTE") == "female"
    assert reg.get_rvc_model("SYLPHIETTE") is None


def test_add_and_set_rvc_model():
    reg = CharacterRegistry()
    reg.add("ERIS", "af_bella", gender="female", rvc_model="eris")
    assert reg.get_rvc_model("ERIS") == "eris"
    reg.set_rvc_model("ERIS", None)
    assert reg.get_rvc_model("ERIS") is None


def test_all_characters_includes_rvc_model():
    reg = CharacterRegistry()
    chars = reg.all_characters()
    rudeus = next(c for c in chars if c["name"] == "RUDEUS")
    assert rudeus["rvc_model"] == "rudeus"
    narrator = next(c for c in chars if c["name"] == "NARRATOR")
    assert narrator.get("rvc_model") is None


def test_ruddy_and_silfie_aliases():
    reg = CharacterRegistry()
    # Ruddy should resolve to Rudeus settings
    assert reg.get_voice("RUDDY") == "am_michael"
    assert reg.get_gender("RUDDY") == "male"
    assert reg.get_rvc_model("RUDDY") == "rudeus"

    # Silfie should resolve to Sylphiette settings
    assert reg.get_voice("SILFIE") == "af_nicole"
    assert reg.get_gender("SILFIE") == "female"
    assert reg.get_rvc_model("SILFIE") is None


