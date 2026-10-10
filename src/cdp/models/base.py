import importlib
from pathlib import Path
from typing import Any, Dict, List, Sequence
from omegaconf import OmegaConf

import numpy as np
import torch
from transformers import PretrainedConfig, PreTrainedModel

from cdp.models.config import dict_to_namespace, read_model_config, resolve_checkpoint_path
from cdp.models.utils import POLICY_REGISTRY, FrameworkTools, auto_get_trainable_modules
from cdp.training.logging import initialize_overwatch

logger = initialize_overwatch(__name__)


def merge_config_overrides(model_config: dict, config_overrides: Sequence[str] | None = None) -> dict:
    if isinstance(config_overrides, str):
        raise TypeError("config_overrides must be a sequence of KEY=VALUE strings, not a bare string.")
    if not config_overrides:
        return model_config

    overrides = list(config_overrides)
    non_string_overrides = [item for item in overrides if not isinstance(item, str)]
    if non_string_overrides:
        raise TypeError(
            "config_overrides must be a sequence of KEY=VALUE strings; "
            f"got non-string entries: {non_string_overrides}"
        )

    bad_overrides = [item for item in overrides if "=" not in item]
    if bad_overrides:
        raise ValueError("Invalid config_overrides entries. Expected KEY=VALUE dotlist items, " f"got: {bad_overrides}")

    try:
        override_cfg = OmegaConf.from_dotlist(overrides)
        return OmegaConf.to_container(
            OmegaConf.merge(OmegaConf.create(model_config), override_cfg),
            resolve=False,
        )
    except Exception as exc:
        raise ValueError(f"Failed to parse config_overrides={overrides!r}: {exc}") from exc


def _import_policy():
    importlib.import_module("cdp.models.policy")


def build_policy(cfg):
    _import_policy()
    return POLICY_REGISTRY[cfg.framework.name](cfg)


class BasePolicy(PreTrainedModel):
    def __init__(self, hf_config: PretrainedConfig | None = None) -> None:
        if hf_config is None:
            hf_config = PretrainedConfig()
        super().__init__(hf_config)

    def forward(self, examples: List[dict], **kwargs) -> dict:
        raise NotImplementedError(
            f"{type(self).__name__} must implement forward(examples) -> dict with 'action_loss' key."
        )

    def predict_action(self, examples: List[dict], **kwargs) -> dict:
        raise NotImplementedError(
            f"{type(self).__name__} must implement predict_action(examples) -> dict with 'normalized_actions' key."
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_checkpoint: str,
        config_overrides: Sequence[str] | None = None,
        **kwargs,
    ):
        checkpoint = resolve_checkpoint_path(pretrained_checkpoint)
        model_config, norm_stats = read_model_config(checkpoint)
        config = dict_to_namespace(merge_config_overrides(model_config, config_overrides))
        config.trainer.pretrained_checkpoint = None
        model = build_policy(cfg=config)
        model.norm_stats = norm_stats
        if checkpoint.suffix == ".safetensors":
            from safetensors.torch import load_file
            state_dict = load_file(str(checkpoint), device="cpu")
        else:
            state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return model
