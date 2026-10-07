"""
Narrative Context Resolver: extracts emotional and prosodic acting cues
from descriptive scene directions and stage notes without reading them aloud.
"""
import re
from typing import Optional

from app.parser.emotion_profiles import EMOTION_PROFILES, _ALIASES
from app.parser.tag_parser import SpeechSegment
from app.synthesis.engine import NarrativeContextFrame


class DefaultNarrativeContextResolver:
    """
    Default resolver that translates natural language narrative directions
    into Kokoro acoustic parameters (hesitation pauses, speed micro-timing,
    volume boosts/cuts, and emotion delta scale alpha).
    """

    def __call__(self, narrative_context: str, segment: SpeechSegment) -> Optional[NarrativeContextFrame]:
        return self.resolve(narrative_context, segment)

    def resolve(self, narrative_context: str, segment: SpeechSegment) -> Optional[NarrativeContextFrame]:
        """
        Analyze narrative context and return a NarrativeContextFrame.
        Returns None or empty frame if no acting cues are recognized.
        """
        if not narrative_context or not narrative_context.strip():
            return None

        ctx = narrative_context.lower()
        text = segment.text.strip() if segment.text else ""

        # Tracking variables
        emotion: Optional[str] = None
        speed: Optional[float] = None
        alpha: Optional[float] = None
        volume_db: Optional[float] = None
        pause_before_ms: Optional[int] = None
        pause_after_ms: Optional[int] = None

        matched = False

        # 1. Shouting / Screaming / Roaring
        if re.search(r'\b(shout|shouted|shouting|scream|screamed|screaming|yell|yelled|roared)\b', ctx):
            matched = True
            emotion = "shout"
            speed = 1.10
            volume_db = 3.0
            alpha = 0.55
            pause_before_ms = 150

        # 2. Whispering / Breathy / Murmuring
        elif re.search(r'\b(whisper|whispered|whispering|breathy|murmur|murmured|softly)\b', ctx):
            matched = True
            emotion = "whisper"
            speed = 0.92
            volume_db = -3.5
            alpha = 0.45
            pause_before_ms = 100

        # 3. Crying / Sobbing / Weeping
        elif re.search(r'\b(cry|crying|sob|sobbed|sobbing|tear|tears|weep|wept)\b', ctx):
            matched = True
            emotion = "sad"
            speed = 0.90
            volume_db = -1.0
            alpha = 0.50
            pause_before_ms = 200

        # 4. Fear / Terrified / Trembling
        elif re.search(r'\b(fear|fearful|terrifi|terrified|scared|frightened|trembl|trembling)\b', ctx):
            matched = True
            emotion = "fearful"
            speed = 1.12
            volume_db = 1.0
            alpha = 0.50
            pause_before_ms = 180

        # 5. Rage / Anger / Furious
        elif re.search(r'\b(furious|fury|rage|enraged|angry|snapped)\b', ctx):
            matched = True
            emotion = "angry"
            speed = 1.10
            volume_db = 2.5
            alpha = 0.55
            pause_before_ms = 120

        # 6. Laughing / Amused
        elif re.search(r'\b(laugh|laughed|laughing|chuckle|chuckled|giggle|giggled)\b', ctx):
            matched = True
            emotion = "happy"
            speed = 1.05
            volume_db = 1.0
            alpha = 0.45

        # 7. Tender / Gentle / Affectionate
        elif re.search(r'\b(tender|tenderly|gentle|gently|loving|affectionate)\b', ctx):
            matched = True
            emotion = "tender"
            speed = 0.92
            volume_db = -1.5
            alpha = 0.40

        # Check for secondary desperation modifier
        if re.search(r'\b(desperat|desperate|desperation|desperately|cracking)\b', ctx):
            matched = True
            if emotion is None:
                emotion = "fearful"
            alpha = max(alpha or 0.35, 0.55)
            pause_before_ms = max(pause_before_ms or 0, 200)

        # Check for hesitation cues
        if re.search(r'\b(hesitat|hesitated|hesitating|paused|wavered|stammer|falter|faltered)\b', ctx):
            matched = True
            pause_before_ms = max(pause_before_ms or 0, 350)
            if speed is None:
                speed = 0.95

        # Text-level prosodic enhancements (e.g. ellipsis hesitation)
        if text.startswith("...") or "..." in text[:8]:
            pause_before_ms = max(pause_before_ms or 0, 300)

        if not matched and emotion is None and speed is None and pause_before_ms is None:
            return None

        # Validate canonical emotion against supported profiles
        if emotion is not None:
            canonical = _ALIASES.get(emotion.strip().lower(), emotion.strip().lower())
            if canonical in EMOTION_PROFILES and EMOTION_PROFILES[canonical].silence_ms is None:
                emotion = canonical
            else:
                emotion = None

        return NarrativeContextFrame(
            emotion=emotion,
            speed=speed,
            alpha=alpha,
            volume_db=volume_db,
            pause_before_ms=pause_before_ms,
            pause_after_ms=pause_after_ms,
        )
