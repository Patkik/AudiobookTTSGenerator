import numpy as np
from app.synthesis.engine import blend_styles
from app.synthesis.audio_pipeline import AudioPipeline


def test_blend_styles():
    a = np.ones((3, 1, 4), dtype=np.float32)
    b = np.zeros_like(a)
    assert np.allclose(blend_styles(a, b, 0.25), 0.25)
    assert blend_styles(None, b, 0.25) is b
    assert blend_styles(a, b, 0.0) is b


def test_f0_variance_noop_without_scale():
    x = np.random.randn(24000).astype(np.float32)
    assert np.array_equal(AudioPipeline.expand_f0_variance(x, 1.0), x)
