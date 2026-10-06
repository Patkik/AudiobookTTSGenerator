import pytest
from app.parser.script_resolver import ScriptResolver

resolver = ScriptResolver()

def test_plain_text_passthrough():
    text = "[happy] Hello world."
    result = resolver.load_text(text)
    assert result == text

def test_strips_excessive_blank_lines():
    text = "Line one.\n\n\n\nLine two."
    result = resolver.load_text(text)
    assert "\n\n\n" not in result

def test_detects_txt_by_filename():
    result = resolver.load_text(b"Hello.\n[sad] Goodbye.", filename="book.txt")
    assert "Hello." in result
    assert "[sad] Goodbye." in result

def test_epub_bytes_returns_text(tmp_path):
    result = resolver.load_text(b"not an epub", filename="book.epub")
    assert isinstance(result, str)

def test_empty_input_returns_empty():
    assert resolver.load_text("") == ""
    assert resolver.load_text(b"") == ""
