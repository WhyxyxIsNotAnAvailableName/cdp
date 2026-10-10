# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modified by [Fangjing Wang/ SUST University] in [2025].
# Modification: [return raw data and suport multi-dataset mixture].
# Modified by [Jinhui YE/ HKUST University] in [2025].
# Modification: [suport topdowm processing, suport param from config].


from cdp.data.config import dataset_specs
from cdp.data.lerobot.datasets import LeRobotSingleDataset, LeRobotMixtureDataset, ModalityConfig
from cdp.data.lerobot.embodiment_tags import EmbodimentTag
from cdp.data.lerobot.transform.base import ComposedModalityTransform
from cdp.data.lerobot.transform.state_action import StateActionToTensor, StateActionTransform


def collate_fn(batch):
    return batch


def make_LeRobotSingleDataset(dataset_name, dataset_path, data_cfg):
    horizon = data_cfg["action_horizon"]
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError("action_horizon must be a positive integer")
    action_keys = [f"action.{key}" for key in data_cfg["action_keys"]]
    state_keys = [f"state.{key}" for key in data_cfg["state_keys"]]
    modality_config = {
        "video": ModalityConfig(delta_indices=[0], modality_keys=[f"video.{key}" for key in data_cfg["keys"]]),
        "action": ModalityConfig(delta_indices=list(range(horizon)), modality_keys=action_keys),
        "language": ModalityConfig(delta_indices=[0], modality_keys=[f"annotation.{data_cfg['language_key']}"]),
    }
    transforms = []
    if state_keys:
        modality_config["state"] = ModalityConfig(delta_indices=[0], modality_keys=state_keys)
        transforms.append(StateActionToTensor(apply_to=state_keys))
    transforms.extend([
        StateActionToTensor(apply_to=action_keys),
        StateActionTransform(
            apply_to=action_keys,
            normalization_modes={key: data_cfg["action_normalization"] for key in action_keys},
        ),
    ])
    return LeRobotSingleDataset(
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        modality_configs=modality_config,
        transforms=ComposedModalityTransform(transforms=transforms),
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        video_backend=data_cfg.get("video_backend", "torchvision_av"),
        delete_pause_frame=data_cfg.get("delete_pause_frame", False),
        data_cfg=data_cfg,
    )


def get_vla_dataset(data_cfg, mode="train", balance_dataset_weights=False,
                    balance_trajectory_weights=False, seed=42, **kwargs):
    dataset_mixture = [
        (make_LeRobotSingleDataset(name, path, data_cfg), weight)
        for name, path, weight in dataset_specs(data_cfg)
    ]
    return LeRobotMixtureDataset(
        dataset_mixture,
        mode=mode,
        balance_dataset_weights=balance_dataset_weights,
        balance_trajectory_weights=balance_trajectory_weights,
        seed=seed,
        data_cfg=data_cfg,
        **kwargs,
    )
