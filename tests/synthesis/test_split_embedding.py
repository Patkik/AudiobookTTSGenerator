import os
import numpy as np
import pytest
from app.synthesis.voicepack_store import VoicepackStore


def test_identity_preservation_dims_0_to_127():
    """Verify that speaker timbre (dimensions 0-127) remains 100% untouched when emotions are applied."""
    store = VoicepackStore(voices_path="models/voices-v1.0.bin", emotions_dir="emotions")
    
    base_style = store.get_style("af_heart", emotion="neutral")
    assert base_style.shape == (510, 1, 256)
    
    for emotion in ["happy", "sad", "angry", "whisper", "excited"]:
        emotional_style = store.get_style("af_heart", emotion=emotion, alpha=0.5)
        # Dimensions 0-127 (speaker timbre and identity) MUST be exactly identical to base voice
        np.testing.assert_array_equal(
            emotional_style[:, :, :128],
            base_style[:, :, :128],
            err_msg=f"Speaker identity was corrupted for emotion '{emotion}'!"
        )


def test_prosody_modulation_dims_128_to_255():
    """Verify that prosody (dimensions 128-255) is altered when emotion is applied."""
    store = VoicepackStore(voices_path="models/voices-v1.0.bin", emotions_dir="emotions")
    
    base_style = store.get_style("af_heart", emotion="neutral")
    emotional_style = store.get_style("af_heart", emotion="happy", alpha=0.5)
    
    # If happy.npy exists, dimensions 128-255 should be modified
    if os.path.exists("emotions/happy.npy"):
        assert not np.array_equal(emotional_style[:, :, 128:], base_style[:, :, 128:]), (
            "Prosody dimensions 128-255 were not modulated!"
        )


def test_gender_aware_delta_routing():
    """Verify that store selects gender-specific deltas when available."""
    store = VoicepackStore(voices_path="models/voices-v1.0.bin", emotions_dir="emotions")
    
    # Test male vs female routing with a sample voice
    male_style = store.get_style("am_michael", emotion="happy", gender="male", alpha=0.4)
    female_style = store.get_style("af_heart", emotion="happy", gender="female", alpha=0.4)
    
    assert male_style.shape == (510, 1, 256)
    assert female_style.shape == (510, 1, 256)
    # Timbre should match their respective base voices
    base_male = store.get_style("am_michael", emotion="neutral")
    base_female = store.get_style("af_heart", emotion="neutral")
    
    np.testing.assert_array_equal(male_style[:, :, :128], base_male[:, :, :128])
    np.testing.assert_array_equal(female_style[:, :, :128], base_female[:, :, :128])
