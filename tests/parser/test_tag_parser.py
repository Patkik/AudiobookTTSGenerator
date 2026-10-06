import pytest
from app.parser.tag_parser import TagParser, SpeechSegment
from app.parser.emotion_profiles import EMOTION_TOKENS

parser = TagParser()

def test_plain_text_no_tags():
    segments = parser.parse("Hello world.")
    assert len(segments) == 1
    assert segments[0].text == "Hello world."
    assert segments[0].emotion == "neutral"
    assert segments[0].character == "NARRATOR"

def test_inline_tag_before_text():
    segments = parser.parse("[happy] She smiled warmly.")
    assert len(segments) == 1
    assert segments[0].emotion == "happy"
    assert segments[0].text == "She smiled warmly."

def test_inline_tag_changes_between_sentences():
    text = "[happy] Good morning! [sad] But then she cried."
    segments = parser.parse(text)
    assert len(segments) == 2
    assert segments[0].emotion == "happy"
    assert segments[0].text == "Good morning!"
    assert segments[1].emotion == "sad"
    assert segments[1].text == "But then she cried."

def test_compound_tag_uses_first():
    segments = parser.parse("[whispering, nervous] Did you hear that?")
    assert segments[0].emotion == "whisper"

def test_pause_tag_produces_silence_segment():
    segments = parser.parse("Wait. [pause] Then she spoke.")
    pause_segs = [s for s in segments if s.is_silence]
    assert len(pause_segs) == 1
    assert pause_segs[0].profile.silence_ms == 500

def test_screenplay_format_character_and_emotion():
    text = "ALICE (angry): I can't believe this!"
    segments = parser.parse(text)
    assert len(segments) == 1
    assert segments[0].character == "ALICE"
    assert segments[0].emotion == "angry"
    assert segments[0].text == "I can't believe this!"

def test_screenplay_multi_emotion():
    text = "BOB (sad, quietly): I'm sorry."
    segments = parser.parse(text)
    assert segments[0].character == "BOB"
    assert segments[0].emotion == "sad"

def test_screenplay_narrator():
    text = "NARRATOR: The clock struck midnight."
    segments = parser.parse(text)
    assert segments[0].character == "NARRATOR"
    assert segments[0].emotion == "neutral"

def test_mixed_formats_in_one_block():
    text = "NARRATOR: It was quiet.\n[sad] She sighed.\nBOB (angry): Enough!"
    segments = parser.parse(text)
    assert len(segments) == 3
    assert segments[0].character == "NARRATOR"
    assert segments[1].emotion == "sad"
    assert segments[2].character == "BOB"

def test_empty_text_returns_empty():
    assert parser.parse("") == []
    assert parser.parse("   ") == []

def test_segment_has_profile():
    segments = parser.parse("[happy] Hello!")
    assert segments[0].profile.speed == 1.1
    assert segments[0].profile.emotion_token == EMOTION_TOKENS["happy"]
