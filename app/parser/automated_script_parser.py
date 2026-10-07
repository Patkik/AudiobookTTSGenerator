"""
Automated script parser for expressive multi-speaker dialogue.
Automatically extracts beat headers, character assignments, compound acting tags,
inter-speaker turn pauses (400ms), and punctuation micro-cadence breaks.
"""
import re
import dataclasses
from typing import Optional

from app.parser.emotion_profiles import EmotionProfile, resolve_profile, _ALIASES, EMOTION_PROFILES
from app.parser.tag_parser import SpeechSegment, calculate_punctuation_pause

_BEAT_HEADER = re.compile(
    r'^#\s*Beat\s*[^:]*:\s*(?P<character>[A-Za-z0-9_ ]+?)(?:\s*\((?P<subtext>[^)]+)\))?\s*$',
    re.IGNORECASE
)

_SCREENPLAY = re.compile(
    r'^(?P<character>[A-Z][A-Za-z0-9_ ]{0,30}?)\s*'
    r'(?:\((?P<emotions>[^)]+)\))?\s*:\s*(?P<text>.+)$'
)

_INLINE_TAG = re.compile(r'\[([^\]]+)\]')


def calculate_cadence_pause(text: str, base_pause_ms: int = 50) -> int:
    """Calculates natural human cadence pause based on ending punctuation."""
    t = text.rstrip()
    if not t:
        return base_pause_ms
    if t.endswith("..."):
        return max(base_pause_ms, 450)
    if t.endswith("—") or t.endswith("--") or t.endswith("-"):
        return max(base_pause_ms, 350)
    if t.endswith("?") or t.endswith("!"):
        return max(base_pause_ms, 250)
    if t.endswith("."):
        return max(base_pause_ms, 220)
    if t.endswith(":") or t.endswith(";"):
        return max(base_pause_ms, 250)
    if t.endswith(","):
        return max(base_pause_ms, 180)
    return base_pause_ms


class AutomatedScriptParser:
    """
    Parses scripts annotated with beat comments, compound emotion tags,
    and dialogue cues into SpeechSegments with automated pacing.
    """

    def parse(self, raw_text: str, default_character: str = "NARRATOR") -> list[SpeechSegment]:
        if not raw_text or not raw_text.strip():
            return []

        segments: list[SpeechSegment] = []
        current_character = default_character
        active_emotion = "neutral"
        last_speaker: Optional[str] = None

        for line in raw_text.splitlines():
            line = line.strip()
            if not line:
                continue

            # 1. Check for Beat Header: e.g. # Beat 1 & 2: Ruddy (Introspective & Gentle)
            m_beat = _BEAT_HEADER.match(line)
            if m_beat:
                char_name = m_beat.group("character").strip().upper()
                current_character = char_name
                subtext = m_beat.group("subtext")
                if subtext:
                    sub_first = subtext.split(",")[0].split("&")[0].strip().lower()
                    active_emotion = _ALIASES.get(sub_first, sub_first)
                continue

            # 2. Check for Screenplay line: e.g. RUDDY (gentle): Silfie...
            m_screen = _SCREENPLAY.match(line)
            if m_screen:
                char_name = m_screen.group("character").strip().upper()
                current_character = char_name
                raw_emotions = m_screen.group("emotions") or active_emotion
                first_tag = raw_emotions.split(",")[0].strip().lower()
                canonical_emotion = _ALIASES.get(first_tag, first_tag)
                active_emotion = canonical_emotion
                text = m_screen.group("text").strip()
                # Strip wrapping quotes if present
                if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
                    text = text[1:-1].strip()

                line_segs = self._parse_line_with_tags(
                    text, active_emotion, current_character, last_speaker
                )
                if line_segs:
                    last_speaker = line_segs[-1].character
                    segments.extend(line_segs)
                continue

            # 3. Standard line with potential inline tags & quotes
            line_segs = self._parse_line_with_tags(
                line, active_emotion, current_character, last_speaker
            )
            if line_segs:
                last_speaker = line_segs[-1].character
                active_emotion = line_segs[-1].emotion
                segments.extend(line_segs)

        return segments

    def _parse_line_with_tags(
        self,
        line: str,
        current_emotion: str,
        current_character: str,
        last_speaker: Optional[str],
    ) -> list[SpeechSegment]:
        segments: list[SpeechSegment] = []
        pos = 0
        active_emotion = current_emotion
        is_first_in_line = True

        for match in _INLINE_TAG.finditer(line):
            before = line[pos:match.start()].strip()
            # Strip quotes
            before_clean = before.strip('"\'')
            if before_clean:
                profile = resolve_profile(active_emotion)
                pause_after = calculate_cadence_pause(before_clean, profile.pause_after_ms)
                pause_before = profile.pause_before_ms

                # If switching speaker on this line, enforce inter-speaker turn pause
                if is_first_in_line and last_speaker is not None and last_speaker != current_character:
                    pause_before = max(pause_before, 400)
                    is_first_in_line = False

                profile = dataclasses.replace(
                    profile,
                    pause_after_ms=pause_after,
                    pause_before_ms=pause_before,
                )

                embedding_scale = self._resolve_embedding_scale(active_emotion)
                segments.append(SpeechSegment(
                    text=before_clean,
                    emotion=active_emotion,
                    character=current_character,
                    profile=profile,
                    is_silence=profile.silence_ms is not None,
                    embedding_scale=embedding_scale,
                ))

            # Parse tag
            raw_tag = match.group(1).strip()
            parts = [p.strip().lower() for p in raw_tag.split(",")]
            if len(parts) > 1:
                # Compound acting tag (e.g. 'gasp, tearful relief' or 'whispers, softly')
                found_emotion = None
                for p in parts:
                    can = _ALIASES.get(p, p)
                    if can not in {"gasp", "sigh", "pause"} and can in EMOTION_PROFILES:
                        found_emotion = can
                        break
                if found_emotion is None:
                    found_emotion = _ALIASES.get(parts[0], parts[0])
                active_emotion = found_emotion
            else:
                canonical_tag = _ALIASES.get(parts[0], parts[0])
                profile = resolve_profile(canonical_tag)
                # Single standalone sound / silence tag (e.g. [sighs], [pause], [soft exhale])
                if profile.silence_ms is not None or canonical_tag in {"sigh", "gasp", "pause", "long pause"}:
                    segments.append(SpeechSegment(
                        text="",
                        emotion=canonical_tag,
                        character=current_character,
                        profile=profile,
                        is_silence=True,
                    ))
                else:
                    active_emotion = canonical_tag

            pos = match.end()

        # Remaining text after last tag
        remaining = line[pos:].strip()
        remaining_clean = remaining.strip('"\'')
        if remaining_clean:
            profile = resolve_profile(active_emotion)
            pause_after = calculate_cadence_pause(remaining_clean, profile.pause_after_ms)
            pause_before = profile.pause_before_ms

            if is_first_in_line and last_speaker is not None and last_speaker != current_character:
                pause_before = max(pause_before, 400)

            profile = dataclasses.replace(
                profile,
                pause_after_ms=pause_after,
                pause_before_ms=pause_before,
            )

            embedding_scale = self._resolve_embedding_scale(active_emotion)
            segments.append(SpeechSegment(
                text=remaining_clean,
                emotion=active_emotion,
                character=current_character,
                profile=profile,
                is_silence=profile.silence_ms is not None,
                embedding_scale=embedding_scale,
            ))

        return segments

    def _resolve_embedding_scale(self, emotion: str) -> float:
        """Assign prosodic embedding scale multiplier based on emotional intensity."""
        if emotion in {"whisper", "shout", "angry", "dramatic"}:
            return 1.3
        if emotion in {"happy", "relieved", "excited", "tender"}:
            return 1.2
        return 1.0
