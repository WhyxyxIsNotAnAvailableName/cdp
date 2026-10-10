import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cdp.models.policy import CDPPolicy
from cdp.perception import pipeline
from cdp.perception.grounding import ground_images
from cdp.perception.instruction_parser import FactorCandidate


class Inputs(dict):
    input_ids = None

    def to(self, device):
        return self


class Processor:
    def __init__(self, labels, label_key):
        self.labels = labels
        self.label_key = label_key

    def __call__(self, images, **kwargs):
        self.batch_size = len(images)
        return Inputs(images=images)

    def post_process_grounded_object_detection(self, outputs, input_ids, **kwargs):
        count = len(self.labels)
        return [dict(boxes=torch.ones(count, 4), scores=torch.ones(count),
                     **{self.label_key: self.labels}) for _ in outputs]


class Grounder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, images):
        return images


class Predictor:
    def set_image_batch(self, images):
        self.images = images

    def predict_batch(self, box_batch, **kwargs):
        masks, scores = [], []
        for image, boxes in zip(self.images, box_batch):
            height, width = image.shape[:2]
            masks.append(np.full((len(boxes), 1, height, width), image[0, 0, 0] % 2, dtype=bool))
            scores.append(np.ones(len(boxes)))
        return masks, scores, None


def make_runtime(factors, label_key):
    return dict(device=torch.device("cpu"), processor=Processor([f.factor_text for f in factors], label_key),
                grounding_model=Grounder(), sam_predictor=Predictor(), threshold_key="threshold")


def test_shared_grounding_and_factors():
    factors = [FactorCandidate(i, name, name, i, i + 1, "dobj")
               for i, name in enumerate(("cup", "bowl", "plate", "tray", "spoon"))]
    images = [Image.new("RGB", (9, 5), (1, 1, 1)), Image.new("RGB", (13, 7), (2, 2, 2))]
    config = SimpleNamespace(batch_size=4, box_threshold=0.3, text_threshold=0.25)
    for label_key in ("labels", "text_labels"):
        runtime = make_runtime(factors, label_key)
        offline = ground_images(images, factors, config, runtime)
        perception = pipeline.SemanticPerception(config, "cpu")
        with mock.patch.object(pipeline, "parse_instruction", return_value=factors) as parse, \
             mock.patch.object(pipeline, "build_runtime", return_value=runtime) as build:
            online = perception(images, "pick the cup", 4)
            changed = perception([Image.new("RGB", (9, 5), (2, 2, 2)), images[1]], "pick the cup", 4)
        assert parse.call_count == 2 and build.call_count == 1
        assert len(online) == 3 and [f["factor_id"] for f in online] == [0, 1, 2]
        assert runtime["processor"].batch_size == 4 and len(runtime["sam_predictor"].images) == 2
        for index, factor in enumerate(online):
            assert factor["view_valid"] == [True, True]
            for view, mask in enumerate(factor["view_masks"]):
                np.testing.assert_array_equal(mask, offline[view]["masks"][index])
        assert online[0]["view_masks"][0].all() and not changed[0]["view_masks"][0].any()
    with mock.patch.object(pipeline, "parse_instruction", return_value=[]), \
         mock.patch.object(pipeline, "build_runtime") as build:
        assert pipeline.SemanticPerception(config, "cpu")(images, "stop", 4) == []
        build.assert_not_called()


def test_predict_action_builds_current_conditions():
    factors = [FactorCandidate(i, name, name, i, i + 1, "dobj")
               for i, name in enumerate(("cup", "bowl", "plate", "tray"))]
    model = object.__new__(CDPPolicy)
    torch.nn.Module.__init__(model)
    model.config = OmegaConf.create({"perception": {"batch_size": 4, "box_threshold": 0.3, "text_threshold": 0.25},
        "datasets": {"vla_data": {"obs_image_size": [12, 8]}},
        "framework": {"action_model": {"inference_noise_schedule": "cosine"}}})
    model._semantic_perception = None
    model.branch_count = 4
    model.branch_weights = [0.25] * 4
    model.qwen_vl_interface = SimpleNamespace(model=SimpleNamespace(device=torch.device("cpu")))
    model.action_model = torch.nn.Module()
    model.action_model.projection = torch.nn.Linear(2, 2)

    def sample(embeddings, **kwargs):
        return model.action_model.projection(torch.zeros(1, 3, 2, dtype=embeddings[0].dtype))

    sampler = mock.Mock(side_effect=sample)
    model.action_model.predict_action_compositional = sampler
    captured = []

    def encode(examples):
        captured.extend(examples)
        return [torch.zeros(4, 1, 2, dtype=torch.bfloat16)], None

    model._encode_conditions = encode
    runtime = make_runtime(factors, "text_labels")
    example = {"image": [Image.new("RGB", (9, 5), (1, 1, 1))], "lang": "pick the cup"}
    with mock.patch.object(pipeline, "parse_instruction", return_value=factors) as parse, \
         mock.patch.object(pipeline, "build_runtime", return_value=runtime):
        result = model.predict_action(example)
        model.predict_action(dict(example, semantic_factors=[{"factor_id": 99}]))
    assert parse.call_count == 2
    assert result["normalized_actions"].shape == (1, 3, 2)
    assert result["normalized_actions"].dtype == np.float32
    assert all(len(example["semantic_factors"]) == 3 for example in captured)
    assert captured[0]["image"][0].size == (12, 8)
    assert captured[0]["semantic_factors"][0]["view_masks"][0].shape == (5, 9)
    assert sampler.call_args.kwargs["branch_mask"].tolist() == [[True, True, True, True]]
    assert set(model.state_dict()) == {"action_model.projection.weight", "action_model.projection.bias"}
    assert "semantic_factors" not in example


if __name__ == "__main__":
    test_shared_grounding_and_factors()
    test_predict_action_builds_current_conditions()
    print("test_online_perception: PASS")
