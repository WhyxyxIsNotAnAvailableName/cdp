import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from omegaconf import OmegaConf
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cdp.models.base import BasePolicy
from cdp.models.config import read_model_config
from cdp.models.utils import FrameworkTools
from cdp.data.lerobot.transform.state_action import Normalizer
from cdp.data.lerobot.transform.state_action import StateActionTransform
from cdp.data.lerobot.datasets import LeRobotMixtureDataset


def test_checkpoint_loading():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        config = OmegaConf.create({"framework": {"name": "test", "qwenvl": {"base_vlm": "old"}}, "trainer": {}})
        OmegaConf.save(config, root / "config.yaml")
        statistics = {"test": {"action": {"min": [0, 0], "max": [1, 1]}}}
        (root / "dataset_statistics.json").write_text(json.dumps(statistics))
        source = torch.nn.Linear(2, 1)
        final = root / "final_model"
        final.mkdir()
        torch.save(source.state_dict(), final / "pytorch_model.pt")
        state = root / "checkpoints/state_steps_4"
        state.mkdir(parents=True)
        torch.save(source.state_dict(), state / "pytorch_model.bin")
        for checkpoint in (final, final / "pytorch_model.pt", state):
            with mock.patch("cdp.models.base.build_policy", side_effect=lambda cfg: torch.nn.Linear(2, 1)) as builder:
                loaded = BasePolicy.from_pretrained(checkpoint, config_overrides=["framework.qwenvl.base_vlm=new"])
            assert builder.call_args.kwargs["cfg"].framework.qwenvl.base_vlm == "new"
            assert loaded.norm_stats == statistics and not loaded.training
            for key, value in source.state_dict().items():
                torch.testing.assert_close(loaded.state_dict()[key], value)
        save_file(source.state_dict(), final / "model.safetensors")
        with mock.patch("cdp.models.base.build_policy", side_effect=lambda cfg: torch.nn.Linear(2, 1)):
            loaded = BasePolicy.from_pretrained(final)
        torch.testing.assert_close(loaded.weight, source.weight)
        config.framework.name = "full"
        OmegaConf.save(config, root / "config.full.yaml")
        assert read_model_config(final)[0]["framework"]["name"] == "full"


def test_action_inverse_matches_training():
    stats = {"min": [-4, 0], "max": [6, 1], "q01": [-2, 0.1], "q99": [3, 0.9],
             "mean": [1, 0.5], "std": [2, 0.25]}
    values = np.array([[[-1.0, -1.0], [1.0, 1.0], [1.5, 0.25]]], dtype=np.float32)
    for mode in ("min_max", "q99", "mean_std", "binary"):
        saved = dict(stats, normalization_modes=[mode, mode])
        actual = FrameworkTools.unnormalize_actions(values, saved)
        expected = Normalizer(mode, dict(stats)).inverse(torch.tensor(values)).numpy()
        np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(FrameworkTools.unnormalize_actions(values, stats)[0, :2, 1], [0, 1])
    mixed = dict(stats, normalization_modes=["min_max", "binary"], mask=[True, False],
                 binary_thresholds=[0.5, 0.7])
    np.testing.assert_allclose(FrameworkTools.unnormalize_actions(values, mixed)[0, :2], [[-4, 0], [6, 1]])
    np.testing.assert_array_equal(values, np.array([[[-1, -1], [1, 1], [1.5, 0.25]]], dtype=np.float32))


def test_saved_normalization_order_and_modes():
    class Dataset:
        tag = "test"
        trajectory_ids = [0]
        modality_keys = {"action": ["action.gripper", "action.position"], "state": []}
        transforms = SimpleNamespace(transforms=[StateActionTransform(
            apply_to=["action.gripper", "action.position"],
            normalization_modes={"action.gripper": "binary", "action.position": "min_max"},
            binary_threshold=0.7,
        )])

        def __len__(self):
            return 2

    def statistics(low, high):
        return SimpleNamespace(min=low, max=high, q01=low, q99=high,
                               mean=[0.0] * len(low), std=[1.0] * len(low))

    mixture = object.__new__(LeRobotMixtureDataset)
    mixture.datasets = [Dataset()]
    mixture.merged_metadata = {"test": SimpleNamespace(
        statistics=SimpleNamespace(action={"position": statistics([-2, 1], [2, 3]),
                                          "gripper": statistics([0], [1])}, state={}),
        modalities=SimpleNamespace(action={"position": SimpleNamespace(shape=(2,)),
                                           "gripper": SimpleNamespace(shape=(1,))}),
    )}
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "statistics.json"
        mixture.save_dataset_statistics(path)
        saved = json.loads(path.read_text())["test"]["action"]
    assert saved["action_keys"] == ["gripper", "position"]
    assert saved["action_dims"] == [1, 2]
    assert saved["normalization_modes"] == ["binary", "min_max", "min_max"]
    assert saved["binary_thresholds"] == [0.7, 0.7, 0.7]
    restored = FrameworkTools.unnormalize_actions(np.array([[1, -1, -1], [0, 1, 1]]), saved)
    np.testing.assert_allclose(restored, [[1, -2, 1], [0, 2, 3]])


if __name__ == "__main__":
    test_checkpoint_loading()
    test_action_inverse_matches_training()
    test_saved_normalization_order_and_modes()
    print("test_policy_io: PASS")
