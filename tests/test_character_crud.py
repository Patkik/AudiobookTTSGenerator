import json

from app.characters import crud
from app.characters.registry import CharacterRegistry
from app.characters.voices import voice_accent, voice_choices, voice_gender, voice_label


def test_voice_label_and_gender():
    assert voice_label("af_heart") == "Heart · Female · American"
    assert voice_gender("bm_george") == "male"
    assert voice_accent("bm_george") == "British"


def test_voice_choices_group_by_gender():
    ids = [v for _, v in voice_choices(["am_michael", "af_bella", "bf_emma"])]
    assert ids == ["af_bella", "bf_emma", "am_michael"]


def test_gender_derived_from_voice():
    reg = CharacterRegistry()
    reg.add("X", "am_adam", gender="female")
    assert reg.get_gender("X") == "male"


def test_crud_add_update_delete_rename():
    reg = CharacterRegistry()
    assert crud.add_character(reg, " zed ", "am_adam")[0]
    assert not crud.add_character(reg, "ZED", "am_adam")[0]
    assert not crud.add_character(reg, "", "am_adam")[0]
    assert crud.update_character(reg, "ZED", "ZOE", "af_bella")[0]
    assert not reg.has("ZED") and reg.get_voice("ZOE") == "af_bella"
    assert not crud.update_character(reg, "ZOE", "ALICE", "af_bella")[0]
    assert crud.delete_character(reg, "ZOE")[0]
    assert not crud.delete_character(reg, "NARRATOR")[0]
    assert not crud.update_character(reg, "NARRATOR", "HOST", "af_heart")[0]
    assert crud.update_character(reg, "NARRATOR", "NARRATOR", "af_bella")[0]


def test_persistence_roundtrip(tmp_path):
    path = str(tmp_path / "chars.json")
    reg = CharacterRegistry(path)
    reg.add("ZED", "am_adam")
    assert CharacterRegistry(path).get_voice("ZED") == "am_adam"
    reg.reset()
    assert not CharacterRegistry(path).has("ZED")


def test_corrupt_file_falls_back(tmp_path):
    p = tmp_path / "chars.json"
    p.write_text("not json")
    assert CharacterRegistry(str(p)).has("NARRATOR")
    p.write_text(json.dumps({"A": {"voice_id": "af_bella"}}))
    assert CharacterRegistry(str(p)).has("NARRATOR")
