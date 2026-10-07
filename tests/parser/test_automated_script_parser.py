import pytest
from app.parser.automated_script_parser import AutomatedScriptParser
from app.characters.registry import CharacterRegistry


def test_parse_beat_headers_and_speakers():
    parser = AutomatedScriptParser()
    script = """
# Beat 1 & 2: Ruddy (Introspective & Gentle)
[whispers, softly] I had a dream once... [sighs] In my dream, there was a child... crying in the dark.
[warmly] "Silfie... I'm right here."

# Beat 3: Silfie (Breathless Relief)
[gasp, tearful relief] "Ruddy... [soft exhale] thank goodness!"
"""
    segments = parser.parse(script)
    assert len(segments) >= 3

    # Beat 1 segments should be Ruddy
    assert segments[0].character == "RUDDY"
    assert segments[0].emotion == "whisper"
    assert "I had a dream once" in segments[0].text

    # Silfie line
    silfie_segs = [s for s in segments if s.character == "SILFIE"]
    assert len(silfie_segs) >= 1
    assert "thank goodness" in silfie_segs[-1].text


def test_inter_speaker_turn_pause():
    parser = AutomatedScriptParser()
    script = """
# Beat 1: Ruddy
"Silfie... I'm right here."

# Beat 2: Silfie
"Ruddy... thank goodness!"
"""
    segments = parser.parse(script)
    assert len(segments) == 2
    assert segments[0].character == "RUDDY"
    assert segments[1].character == "SILFIE"

    # Speaker turn pause: when switching from Ruddy to Silfie, pause_before_ms should be at least 400ms
    assert segments[1].profile.pause_before_ms >= 400


def test_punctuation_cadence_pauses():
    parser = AutomatedScriptParser()
    script = """
RUDDY: I had a dream once...
SILFIE: What was it?
RUDDY: It was dark—
"""
    segments = parser.parse(script)
    assert len(segments) == 3
    # Ellipsis pause
    assert segments[0].profile.pause_after_ms >= 450
    # Question pause
    assert segments[1].profile.pause_after_ms >= 220
    # Dash pause
    assert segments[2].profile.pause_after_ms >= 350


def test_compound_tag_resolution():
    parser = AutomatedScriptParser()
    script = """
[whispers, softly] "I had a dream."
[gasp, tearful relief] "Thank goodness!"
[low voice, rising resolve] "I swear it."
"""
    segments = parser.parse(script)
    assert len(segments) == 3
    assert segments[0].emotion == "whisper"
    assert segments[1].emotion == "relieved"
    assert segments[2].emotion in {"dramatic", "whisper"}
