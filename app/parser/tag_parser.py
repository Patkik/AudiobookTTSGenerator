"""
Parses text containing expression tags into a list of SpeechSegments.

Supported formats:
  A) Inline tags:    [happy] She smiled. [sad] He wept.
  B) Screenplay:     ALICE (angry): I won't stand for this!
  C) Mixed:          any combination of A and B, line by line
"""
import re
from dataclasses import dataclass
from typing import Optional

from app.parser.emotion_profiles import EmotionProfile, resolve_profile, _ALIASES

# Regex: [tag text] - captures everything inside brackets
_INLINE_TAG = re.compile(r'\[([^\]]+)\]')

# Regex: CHARACTER (emotion, optional emotion2): dialogue text
# Character must start with uppercase letter, rest uppercase/space
_SCREENPLAY = re.compile(
    r'^(?P<character>[A-Z][A-Z0-9 ]{0,30}?)\s*'
    r'(?:\((?P<emotions>[^)]+)\))?\s*:\s*(?P<text>.+)$'
)


@dataclass
class SpeechSegment:
    """One unit of speech: text to synthesize + how to synthesize it."""
    text: str
    emotion: str
    character: str
    profile: EmotionProfile
    is_silence: bool = False  # True for [pause] / [long pause] tags


class TagParser:
    """
    Parses raw text (any supported format) into a list of SpeechSegments.
    
    Line-by-line strategy:
    1. Try to match a screenplay line (CHAR (emotion): text)
    2. Otherwise scan for inline [tags] and split text at each tag boundary
    3. Track "current emotion" state across lines (last seen tag persists)
    """

    def parse(self, raw_text: str, default_character: str = "NARRATOR") -> list[SpeechSegment]:
        """
        Parse raw_text into speech segments.

        Args:
            raw_text: Input text with optional expression tags.
            default_character: Character name used when no screenplay header is present.

        Returns:
            List of SpeechSegment objects, in order.
        """
        if not raw_text or not raw_text.strip():
            return []

        segments: list[SpeechSegment] = []
        current_emotion = "neutral"
        current_character = default_character

        for line in raw_text.splitlines():
            line = line.strip()
            if not line:
                continue

            # Try screenplay format first
            m = _SCREENPLAY.match(line)
            if m:
                character = m.group("character").strip()
                raw_emotions = m.group("emotions") or "neutral"
                # Take first emotion from compound tags like "sad, quietly"
                first_tag = raw_emotions.split(",")[0].strip().lower()
                canonical_emotion = _ALIASES.get(first_tag, first_tag)
                text = m.group("text").strip()
                profile = resolve_profile(canonical_emotion)
                segments.append(SpeechSegment(
                    text=text,
                    emotion=canonical_emotion,
                    character=character,
                    profile=profile,
                    is_silence=profile.silence_ms is not None,
                ))
                current_character = character
                current_emotion = canonical_emotion
                continue

            # Inline tag parsing: split line at [tag] boundaries
            line_segments = self._parse_inline(line, current_emotion, current_character)
            if line_segments:
                # Update state from last segment
                current_emotion = line_segments[-1].emotion
                current_character = line_segments[-1].character
                segments.extend(line_segments)

        return segments

    def _parse_inline(
        self,
        line: str,
        current_emotion: str,
        current_character: str,
    ) -> list[SpeechSegment]:
        """Split one line at inline [tag] boundaries."""
        segments: list[SpeechSegment] = []
        pos = 0
        active_emotion = current_emotion

        for match in _INLINE_TAG.finditer(line):
            # Text before this tag (use active_emotion)
            before = line[pos:match.start()].strip()
            if before:
                profile = resolve_profile(active_emotion)
                segments.append(SpeechSegment(
                    text=before,
                    emotion=active_emotion,
                    character=current_character,
                    profile=profile,
                    is_silence=profile.silence_ms is not None,
                ))

            # Update active emotion from this tag
            raw_tag = match.group(1)
            first_tag = raw_tag.split(",")[0].strip().lower()
            canonical_tag = _ALIASES.get(first_tag, first_tag)

            profile = resolve_profile(canonical_tag)
            # Check if this tag is a silence-only tag
            if profile.silence_ms is not None:
                segments.append(SpeechSegment(
                    text="",
                    emotion=canonical_tag,
                    character=current_character,
                    profile=profile,
                    is_silence=True,
                ))
                # Note: silence tag does NOT change active_emotion for following text
            else:
                active_emotion = canonical_tag

            pos = match.end()

        # Remaining text after last tag
        remaining = line[pos:].strip()
        if remaining:
            profile = resolve_profile(active_emotion)
            segments.append(SpeechSegment(
                text=remaining,
                emotion=active_emotion,
                character=current_character,
                profile=profile,
                is_silence=profile.silence_ms is not None,
            ))

        return segments
