import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cdp.models.policy import CDPPolicy


def _condition_builder_without_weights():
    model = object.__new__(CDPPolicy)
    object.__setattr__(model, "branch_count", 4)
    return model


def test_four_conditions_preserve_factor_order_and_masks():
    model = _condition_builder_without_weights()
    pixels = np.full((16, 16, 3), 200, dtype=np.uint8)
    image = Image.fromarray(pixels)
    mask = np.zeros((16, 16), dtype=bool)
    mask[4:12, 4:12] = True
    example = {
        "image": [image],
        "lang": "pick the object from the source and place it in the target",
        "semantic_factors": [
            {"factor_id": 0, "factor_text": "the object", "view_masks": [mask]},
            {"factor_id": 1, "factor_text": "the source", "view_masks": [None]},
            {"factor_id": 2, "factor_text": "the target", "view_masks": [mask]},
        ],
    }
    example["semantic_factors"].reverse()
    images, texts, masks = model._build_learned_condition_payload([example])
    assert len(images) == len(texts) == len(masks) == 4
    assert texts == [example["lang"], "the object", "the source", "the target"]
    assert all(np.array_equal(np.asarray(views[0]), pixels) for views in images)
    assert masks[0] == masks[2] == [None]
    assert np.array_equal(masks[1][0], mask)
    assert np.array_equal(masks[3][0], mask)


def test_conditions_support_variable_factor_counts():
    model = _condition_builder_without_weights()
    image = Image.fromarray(np.full((8, 8, 3), 128, dtype=np.uint8))
    examples = [
        {"image": [image], "lang": "look", "semantic_factors": []},
        {"image": [image], "lang": "pick the cup", "semantic_factors": [
            {"factor_id": 0, "factor_text": "the cup", "view_masks": [None]}
        ]},
    ]
    images, texts, masks = model._build_learned_condition_payload(examples)
    assert len(images) == len(texts) == len(masks) == 8
    assert texts[4:6] == ["pick the cup", "the cup"]
    assert model._branch_mask(examples, "cpu").tolist() == [
        [True, False, False, False], [True, True, False, False]
    ]


def test_vlm_backend_uses_configuration_in_an_arbitrary_model_directory():
    from cdp.models.vlm import qwen2_5, qwen3, get_vlm_model

    with tempfile.TemporaryDirectory() as directory:
        model_path = Path(directory) / "weights"
        model_path.mkdir()
        config = OmegaConf.create({"framework": {"qwenvl": {"base_vlm": str(model_path)}}})
        for model_type, module, class_name in [
            ("qwen2_5_vl", qwen2_5, "_QWen_VL_Interface"),
            ("qwen3_vl", qwen3, "_QWen3_VL_Interface"),
        ]:
            (model_path / "config.json").write_text(json.dumps({"model_type": model_type}))
            expected = object()
            with mock.patch.object(module, class_name, return_value=expected):
                assert get_vlm_model(config) is expected


if __name__ == "__main__":
    test_four_conditions_preserve_factor_order_and_masks()
    test_conditions_support_variable_factor_counts()
    test_vlm_backend_uses_configuration_in_an_arbitrary_model_directory()
    print("test_compositional_conditions: PASS")
