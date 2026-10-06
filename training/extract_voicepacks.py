"""
Extracts and generates emotion voicepack tensors for Kokoro TTS.

Modes:
1. Extraction from reference audio (used in Colab with PyTorch StyleEncoder).
2. Latent space synthesis & blending from base Kokoro voices (runs locally on CPU).
3. Saves resulting 3D arrays to `emotions/<emotion>.npy` for immediate use by `VoicepackStore`.
"""
import os
import argparse
import numpy as np
from typing import List, Tuple, Dict


def blend_voice_arrays(weighted_voices: List[Tuple[np.ndarray, float]]) -> np.ndarray:
    """
    Linear interpolation of multiple voice arrays.
    Each element is (tensor, weight).
    Weights are normalized to sum to 1.0.
    """
    total_weight = sum(w for _, w in weighted_voices)
    if total_weight <= 0:
        raise ValueError("Total weight must be positive.")
        
    shape = weighted_voices[0][0].shape
    blended = np.zeros(shape, dtype=np.float32)
    
    for tensor, weight in weighted_voices:
        norm_w = weight / total_weight
        blended += tensor.astype(np.float32) * norm_w
        
    return blended.astype(np.float32)


def save_emotion_voicepack(emotion: str, tensor: np.ndarray, out_dir: str = "emotions") -> str:
    """Save an emotion voicepack array to emotions/<emotion>.npy."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{emotion.lower()}.npy")
    np.save(out_path, tensor.astype(np.float32))
    print(f"Saved emotion voicepack: {out_path} {tensor.shape}")
    return out_path


def generate_preset_emotions_from_bin(voices_bin_path: str = "models/voices-v1.0.bin", out_dir: str = "emotions"):
    """
    Generates high-contrast emotion voicepacks by interpolating characteristic
    base voices from Kokoro's latent space.
    """
    if not os.path.exists(voices_bin_path):
        raise FileNotFoundError(f"Missing {voices_bin_path}")
        
    voices = np.load(voices_bin_path)
    
    # Emotional blending recipe
    recipes: Dict[str, List[Tuple[str, float]]] = {
        "happy":     [("af_bella", 0.6), ("af_heart", 0.4)],
        "excited":   [("af_bella", 0.7), ("af_nicole", 0.3)],
        "sad":       [("af_sarah", 0.7), ("bf_emma", 0.3)],
        "calm":      [("af_heart", 0.8), ("af_sky", 0.2)],
        "dramatic":  [("bm_george", 0.6), ("am_michael", 0.4)],
        "sarcastic": [("af_alloy", 0.5), ("af_bella", 0.5)],
        "angry":     [("am_adam", 0.7), ("am_michael", 0.3)],
    }
    
    for emotion, weights in recipes.items():
        # Check if all voices exist
        voice_tensors = []
        for voice_name, w in weights:
            if voice_name in voices:
                voice_tensors.append((voices[voice_name], w))
            elif "af_heart" in voices:
                voice_tensors.append((voices["af_heart"], w))
                
        if voice_tensors:
            blended = blend_voice_arrays(voice_tensors)
            save_emotion_voicepack(emotion, blended, out_dir=out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract or blend emotion voicepacks.")
    parser.add_argument("--voices_bin", type=str, default="models/voices-v1.0.bin", help="Path to voices-v1.0.bin")
    parser.add_argument("--out_dir", type=str, default="emotions", help="Destination folder for .npy packs")
    args = parser.parse_args()
    
    generate_preset_emotions_from_bin(args.voices_bin, args.out_dir)
