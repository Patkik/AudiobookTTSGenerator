"""
Loads text from various sources (plain string, .txt bytes, .epub bytes)
and normalizes it into a clean string for the TagParser.
"""
import re
from typing import Union


class ScriptResolver:
    """
    Accepts text as a raw string or uploaded file bytes.
    Returns a normalized string ready for TagParser.
    """

    def load_text(self, source: Union[str, bytes], filename: str = "") -> str:
        """
        Load and normalize text from a source.

        Args:
            source: Raw text string, or bytes from an uploaded file.
            filename: Original filename (used to detect format from extension).

        Returns:
            Normalized text string. Returns "" for empty or unreadable input.
        """
        if not source:
            return ""

        if isinstance(source, bytes):
            if filename.lower().endswith(".epub"):
                return self._load_epub(source)
            else:
                # Assume UTF-8 encoded text file
                try:
                    text = source.decode("utf-8")
                except UnicodeDecodeError:
                    text = source.decode("latin-1", errors="replace")
                return self._normalize(text)

        # Already a string
        return self._normalize(source)

    def _normalize(self, text: str) -> str:
        """Collapse 3+ consecutive blank lines to 2, strip trailing whitespace."""
        text = re.sub(r'\n{3,}', '\n\n', text)
        lines = [line.rstrip() for line in text.splitlines()]
        return "\n".join(lines).strip()

    def _load_epub(self, epub_bytes: bytes) -> str:
        """Extract plain text from EPUB bytes using ebooklib."""
        try:
            import io
            import ebooklib
            from ebooklib import epub
            from html.parser import HTMLParser

            class _HTMLTextExtractor(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self._chunks: list[str] = []
                    self._skip_tags = {"script", "style"}
                    self._in_skip = False

                def handle_starttag(self, tag, attrs):
                    if tag in self._skip_tags:
                        self._in_skip = True

                def handle_endtag(self, tag):
                    if tag in self._skip_tags:
                        self._in_skip = False
                    if tag in {"p", "div", "h1", "h2", "h3", "br"}:
                        self._chunks.append("\n")

                def handle_data(self, data):
                    if not self._in_skip:
                        self._chunks.append(data)

                def get_text(self) -> str:
                    return "".join(self._chunks)

            book = epub.read_epub(io.BytesIO(epub_bytes))
            parts: list[str] = []
            for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                content = item.get_content().decode("utf-8", errors="replace")
                extractor = _HTMLTextExtractor()
                extractor.feed(content)
                parts.append(extractor.get_text())

            return self._normalize("\n\n".join(parts))

        except Exception:
            return ""
