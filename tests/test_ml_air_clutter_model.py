import importlib.util
import json
from pathlib import Path

import pytest

_MODEL_PATH = Path(__file__).resolve().parents[1] / "ml_air_clutter" / "model.py"
_SPEC = importlib.util.spec_from_file_location("ml_air_clutter_model_under_test", _MODEL_PATH)
model_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(model_module)

ModelConfig = model_module.ModelConfig
validate_model_config = model_module.validate_model_config


def test_model_config_serializes_tuple_channels_as_json_list():
    config = ModelConfig(model_type="small_unet", input_channels=("raw", "grad_x"), base_channels=8)

    data = config.to_dict()

    assert data["model_type"] == "small_unet"
    assert data["input_channels"] == ["raw", "grad_x"]
    json.dumps(data)


def test_model_config_rejects_residual_mode_for_mvp():
    config = ModelConfig(output_mode="residual")

    with pytest.raises(ValueError, match="direct_clean"):
        validate_model_config(config)


def test_model_config_rejects_unknown_optional_channel():
    config = ModelConfig(input_channels=("raw", "phase"))

    with pytest.raises(ValueError, match="Unsupported input channels"):
        validate_model_config(config)
