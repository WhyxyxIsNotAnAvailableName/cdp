from __future__ import annotations

from typing import List

import numpy as np
import torch
from PIL import Image

from cdp.models.images import to_pil_preserve
from cdp.models.qwen_backbone import QwenActionBackbone
from cdp.models.highlighting import LearnablePackedBackground
from cdp.models.utils import POLICY_REGISTRY
from cdp.training.utils import resize_images


@POLICY_REGISTRY.register("CDPPolicy")
class CDPPolicy(QwenActionBackbone):
    branch_names = ("global", "factor_1", "factor_2", "factor_3")

    def __init__(self, config=None, **kwargs):
        super().__init__(config=config, **kwargs)
        self._semantic_perception = None
        composition = self.config.framework.get("composition", {})
        self.branch_count = int(self.config.datasets.vla_data.get("semantic_factor_budget", 4))
        self.branch_names = ("global",) + tuple(f"factor_{index}" for index in range(1, self.branch_count))
        weights = composition.get("branch_weights", [1.0 / self.branch_count] * self.branch_count)
        if len(weights) != self.branch_count:
            raise ValueError(f"Expected {self.branch_count} branch weights, got {weights}")
        self.branch_weights = [float(value) for value in weights]
        self.magnitude_alpha = float(composition.get("magnitude_alpha", 0.4))
        self.magnitude_loss_weight = float(composition.get("magnitude_loss_weight", 1.0))
        image_processor = self.qwen_vl_interface.processor.image_processor
        low_frequency_grid = composition.get("low_frequency_grid", [8, 8])
        self.packed_background = LearnablePackedBackground(
            patch_size=int(image_processor.patch_size),
            temporal_patch_size=int(image_processor.temporal_patch_size),
            max_views=len(self.config.datasets.vla_data["keys"]),
            background_size=tuple(composition.get("background_size", [224, 224])),
            low_frequency_grid=tuple(int(value) for value in low_frequency_grid),
            merge_size=int(image_processor.merge_size),
            image_mean=tuple(image_processor.image_mean),
            image_std=tuple(image_processor.image_std),
        )

        if bool(self.config.framework.qwenvl.get("gradient_checkpointing", False)):
            self.qwen_vl_interface.model.config.use_cache = False
            self.qwen_vl_interface.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

    def _build_learned_condition_payload(
        self, examples: List[dict]
    ) -> tuple[list, list[str], list[list[np.ndarray | None]]]:
        condition_images = []
        condition_texts = []
        condition_masks = []
        for example in examples:
            images = [image.convert("RGB") for image in example["image"]]
            factors = sorted(example.get("semantic_factors", []), key=lambda item: item["factor_id"])
            if len(factors) >= self.branch_count:
                raise ValueError("Semantic factor count exceeds the branch budget")
            condition_images.append(images)
            condition_texts.append(example["lang"])
            condition_masks.append([None] * len(images))
            for factor in factors:
                if len(factor["view_masks"]) != len(images):
                    raise ValueError(
                        f"Factor/image view mismatch: masks={len(factor['view_masks'])} "
                        f"images={len(images)}"
                    )
                condition_images.append(images)
                condition_texts.append(factor["factor_text"])
                condition_masks.append(list(factor["view_masks"]))
            for _ in range(self.branch_count - 1 - len(factors)):
                condition_images.append(images)
                condition_texts.append(example["lang"])
                condition_masks.append([None] * len(images))
        return condition_images, condition_texts, condition_masks

    def _branch_mask(self, examples, device):
        counts = torch.tensor(
            [1 + len(example.get("semantic_factors", [])) for example in examples], device=device
        )
        return torch.arange(self.branch_count, device=device)[None] < counts[:, None]

    def _pack_semantic_masks(self, condition_images, condition_masks):
        flat_masks = []
        apply_background = []
        view_ids = []
        for images, masks in zip(condition_images, condition_masks, strict=True):
            for view_id, (image, mask) in enumerate(zip(images, masks, strict=True)):
                if mask is None:
                    mask_array = np.ones((image.height, image.width), dtype=np.uint8) * 255
                    apply_background.append(False)
                else:
                    mask_image = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).resize(
                        image.size, resample=Image.Resampling.NEAREST
                    )
                    mask_array = np.asarray(mask_image, dtype=np.uint8)
                    apply_background.append(True)
                flat_masks.append(Image.fromarray(mask_array).convert("RGB"))
                view_ids.append(view_id)

        mask_inputs = self.qwen_vl_interface.processor.image_processor(
            images=flat_masks,
            do_rescale=True,
            do_normalize=False,
            return_tensors="pt",
        )
        return mask_inputs, apply_background, view_ids

    def _encode_conditions(self, examples: List[dict]):
        images, texts, masks = self._build_learned_condition_payload(examples)
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images, texts)
        mask_inputs, apply_background, view_ids = self._pack_semantic_masks(images, masks)
        qwen_inputs["pixel_values"] = self.packed_background(
            qwen_inputs["pixel_values"],
            qwen_inputs["image_grid_thw"],
            mask_inputs["pixel_values"].to(qwen_inputs["pixel_values"].device),
            mask_inputs["image_grid_thw"],
            apply_background,
            view_ids,
        )
        attention_mask = qwen_inputs.get("attention_mask", None)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            embeddings = list(outputs.hidden_states[-self.num_action_dit_layers :])
            embeddings = self._project_vl_hidden_for_action(embeddings)
        return embeddings, attention_mask

    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        if not examples:
            raise ValueError("examples must be a non-empty list")
        vl_embs_list, attention_mask = self._encode_conditions(examples)
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=torch.bool)
        actions = torch.as_tensor(
            np.asarray([example["action"] for example in examples]),
            device=vl_embs_list[-1].device,
            dtype=vl_embs_list[-1].dtype,
        )
        repeated = int(self.config.framework.action_model.get("repeated_diffusion_steps", 8))
        result = self.action_model.forward_compositional(
            vl_embs_list,
            actions,
            branch_count=self.branch_count,
            repeated_diffusion_steps=repeated,
            branch_weights=self.branch_weights,
            branch_mask=self._branch_mask(examples, actions.device),
            magnitude_alpha=self.magnitude_alpha,
            magnitude_loss_weight=self.magnitude_loss_weight,
            encoder_attention_mask=attention_mask,
        )
        output = {
            "action_loss": result["loss"],
            "flow_matching_loss": result["flow_matching_loss"].detach(),
            "magnitude_loss": result["magnitude_loss"].detach(),
            "branch_norm": result["branch_norm"].detach(),
            "composed_norm": result["composed_norm"].detach(),
            "branch_cosine": result["branch_cosine"].detach(),
            "magnitude_active_fraction": result["magnitude_active_fraction"].detach(),
        }
        valid_by_factor = []
        for factor_id in range(self.branch_count - 1):
            flags = [
                factor_id < len(example.get("semantic_factors", []))
                and bool(any(example["semantic_factors"][factor_id]["view_valid"]))
                for example in examples
            ]
            valid_by_factor.append(sum(flags) / len(flags))
        output["mask_valid_by_factor"] = torch.tensor(
            valid_by_factor, device=result["loss"].device, dtype=torch.float32
        )
        if self.packed_background is not None:
            output["background_residual_norm"] = (
                self.packed_background.low_frequency_residual.detach().float().norm()
            )
        return output

    @torch.inference_mode()
    def predict_action(self, examples: List[dict] = None, **kwargs) -> dict:
        from cdp.perception.pipeline import SemanticPerception

        if type(examples) is not list:
            examples = [examples]
        if self._semantic_perception is None:
            self._semantic_perception = SemanticPerception(
                self.config.perception, self.qwen_vl_interface.model.device
            )
        normalized_examples = []
        target_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        for example in examples:
            copied = dict(example)
            images = to_pil_preserve(example["image"])
            copied["semantic_factors"] = self._semantic_perception(
                images, example["lang"], self.branch_count
            )
            if target_size:
                images = resize_images([images], target_size=target_size)[0]
            copied["image"] = images
            normalized_examples.append(copied)

        vl_embs_list, attention_mask = self._encode_conditions(normalized_examples)
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=torch.bool)
        with torch.autocast(vl_embs_list[-1].device.type, dtype=torch.bfloat16):
            actions = self.action_model.predict_action_compositional(
                vl_embs_list,
                branch_count=self.branch_count,
                branch_weights=self.branch_weights,
                branch_mask=self._branch_mask(normalized_examples, vl_embs_list[-1].device),
                noise_schedule=str(self.config.framework.action_model.get("inference_noise_schedule", "cosine")),
                encoder_attention_mask=attention_mask,
            )
        return {"normalized_actions": actions.detach().float().cpu().numpy()}
