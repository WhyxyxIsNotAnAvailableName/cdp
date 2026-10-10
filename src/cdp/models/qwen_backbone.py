# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by Jinhui YE / HKUST University] in [2025].
from typing import List, Optional

import torch
import torch.nn as nn

from cdp.models.base import BasePolicy
from cdp.models.config import populate_layerwise_dit_cfg
from cdp.models.flow_matching import LayerwiseFlowmatchingActionHead, get_action_model
from cdp.models.vlm import get_vlm_model
from cdp.data.config import infer_action_dim


class QwenActionBackbone(BasePolicy):
    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        horizon = self.config.framework.action_model.action_horizon
        if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
            raise ValueError("action_horizon must be a positive integer")
        if self.config.framework.action_model.get("action_dim") is None:
            self.config.framework.action_model.action_dim = infer_action_dim(self.config.datasets.vla_data)
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        vlm_hf_cfg = self.qwen_vl_interface.model.config
        text_cfg = getattr(vlm_hf_cfg, "text_config", vlm_hf_cfg)
        num_vl_layers = int(text_cfg.num_hidden_layers)
        llm_hidden_size = int(vlm_hf_cfg.hidden_size)
        self.config.framework.qwenvl.vl_hidden_dim = llm_hidden_size
        self.config.framework.qwenvl.num_vl_layers = num_vl_layers

        diffusion_model_cfg = self.config.framework.action_model.diffusion_model_cfg
        action_dit_hidden_dim = diffusion_model_cfg.get("action_dit_hidden_dim", None)
        if action_dit_hidden_dim is None:
            action_dit_hidden_dim = llm_hidden_size
        self.action_dit_hidden_dim = int(action_dit_hidden_dim)

        populate_layerwise_dit_cfg(
            self.config,
            dit_hidden_dim=self.action_dit_hidden_dim,
            num_dit_layers=num_vl_layers,
        )

        self.action_model: LayerwiseFlowmatchingActionHead = get_action_model(config=self.config)
        self.num_action_dit_layers = len(self.action_model.model.transformer_blocks)

        self.project_layers = nn.ModuleList(
            [
                (
                    nn.Identity()
                    if llm_hidden_size == self.action_dit_hidden_dim
                    else nn.Sequential(
                        nn.LayerNorm(llm_hidden_size),
                        nn.Linear(llm_hidden_size, self.action_dit_hidden_dim),
                    )
                )
                for _ in range(self.num_action_dit_layers)
            ]
        )

        self.action_horizon = int(self.config.framework.action_model.action_horizon)

    def _project_vl_hidden_for_action(self, vl_embs_list: List[torch.Tensor]) -> List[torch.Tensor]:
        if len(vl_embs_list) != len(self.project_layers):
            raise ValueError(
                f"Layer number mismatch: got {len(vl_embs_list)} VL layers, "
                f"but project_layers has {len(self.project_layers)} layers."
            )
        return [proj(vl_h) for proj, vl_h in zip(self.project_layers, vl_embs_list)]
