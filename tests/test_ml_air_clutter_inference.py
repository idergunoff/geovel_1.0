import numpy as np
import pytest

from ml_air_clutter.inference import InferenceConfig, blend_inference_result, run_full_profile_inference
from ml_air_clutter.model import ModelConfig


torch = pytest.importorskip("torch")


class IdentityCleanModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(()))

    def forward(self, x):
        return x[:, :1] + self.bias


def test_full_profile_inference_covers_tail_and_blends_alpha():
    noisy = np.arange(10 * 512, dtype=np.float32).reshape(10, 512) % 256
    model = IdentityCleanModel()
    config = ModelConfig(input_channels=("raw",))
    result = run_full_profile_inference(model, config, noisy, InferenceConfig(patch_width=4, stride=3, alpha=0.25, window="uniform"))

    assert result["clean_pred"].shape == noisy.shape
    np.testing.assert_allclose(result["clean_pred"], noisy, atol=1e-5)
    np.testing.assert_allclose(result["cleaned"], noisy, atol=1e-5)
    assert result["meta"]["window_starts"] == [0, 3, 6]


def test_blend_inference_result_uses_alpha_between_noisy_and_prediction():
    noisy = np.full((2, 3), 10.0, dtype=np.float32)
    clean_pred = np.full((2, 3), 2.0, dtype=np.float32)
    blended = blend_inference_result(noisy, clean_pred, 0.5)

    np.testing.assert_allclose(blended["cleaned"], 6.0)
    np.testing.assert_allclose(blended["residual"], 8.0)
    assert blended["alpha"] == 0.5
