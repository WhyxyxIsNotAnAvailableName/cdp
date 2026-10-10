from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from cdp.perception.instruction_parser import FactorCandidate


def normalize_tokens(text: str) -> set[str]:
    return {token for token in text.lower().replace("-", " ").split() if token not in {"the", "a", "an"}}


def label_match_score(label: str, entity: str) -> float:
    label_tokens = normalize_tokens(label)
    entity_tokens = normalize_tokens(entity)
    if not label_tokens or not entity_tokens:
        return 0.0
    overlap = len(label_tokens & entity_tokens)
    if overlap == 0:
        return 0.0
    return overlap / len(label_tokens | entity_tokens)


def select_factor_detections(result: dict, factors: list[FactorCandidate]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = list(result["text_labels"] if "text_labels" in result else result["labels"])
    boxes = result["boxes"].float().cpu().numpy()
    scores = result["scores"].float().cpu().numpy()
    selected_boxes = np.zeros((len(factors), 4), dtype=np.float32)
    selected_scores = np.zeros(len(factors), dtype=np.float32)
    selected_valid = np.zeros(len(factors), dtype=bool)
    factors_by_entity: dict[str, list[int]] = {}
    for factor_idx, factor in enumerate(factors):
        factors_by_entity.setdefault(factor.entity_text, []).append(factor_idx)

    for entity_text, factor_indices in factors_by_entity.items():
        candidates = []
        for detection_idx, label in enumerate(labels):
            match_score = label_match_score(label, entity_text)
            if match_score > 0:
                candidates.append((match_score, float(scores[detection_idx]), detection_idx))
        if not candidates:
            continue
        candidates.sort(reverse=True)
        chosen = candidates[: len(factor_indices)]
        chosen.sort(
            key=lambda item: (
                float((boxes[item[2], 0] + boxes[item[2], 2]) * 0.5),
                float((boxes[item[2], 1] + boxes[item[2], 3]) * 0.5),
            )
        )
        for factor_idx, (_, score, detection_idx) in zip(factor_indices, chosen):
            selected_boxes[factor_idx] = boxes[detection_idx]
            selected_scores[factor_idx] = score
            selected_valid[factor_idx] = True
    return selected_boxes, selected_scores, selected_valid


def build_runtime(config, device="cuda:0"):
    sam2_root = Path(__file__).resolve().parents[3] / "third_party/sam2"
    sys.path.insert(0, str(sam2_root))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = torch.device(device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    processor = AutoProcessor.from_pretrained(str(config.grounding_model))
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        str(config.grounding_model)
    ).to(device).eval()
    sam_model = build_sam2(config.sam_config, str(config.sam_checkpoint), device=str(device))
    signature = inspect.signature(processor.post_process_grounded_object_detection)
    return {
        "device": device,
        "processor": processor,
        "grounding_model": grounding_model,
        "sam_predictor": SAM2ImagePredictor(sam_model),
        "threshold_key": "box_threshold" if "box_threshold" in signature.parameters else "threshold",
    }


def ground_images(images, factors, config, runtime):
    if config.batch_size < 1:
        raise ValueError("batch_size must be positive")
    count = len(factors)
    records = [dict(
        masks=[None] * count,
        mask_valid=np.zeros(count, dtype=bool),
        boxes=np.zeros((count, 4), dtype=np.float32),
        dino_scores=np.zeros(count, dtype=np.float32),
        sam_scores=np.zeros(count, dtype=np.float32),
    ) for _ in images]
    if not factors:
        return records
    query = ". ".join(dict.fromkeys(factor.factor_text for factor in factors)) + "."
    device = torch.device(runtime["device"])
    processor = runtime["processor"]
    predictor = runtime["sam_predictor"]
    batch_size = int(config.batch_size)
    for start in range(0, len(images), batch_size):
        batch = [image.convert("RGB") for image in images[start:start + batch_size]]
        inference_images = batch + [batch[-1]] * (batch_size - len(batch))
        inputs = processor(
            images=inference_images,
            text=[query] * len(inference_images),
            return_tensors="pt",
            padding=True,
        ).to(device)
        with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            outputs = runtime["grounding_model"](**inputs)
        detections = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            text_threshold=config.text_threshold,
            target_sizes=[image.size[::-1] for image in inference_images],
            **{runtime["threshold_key"]: config.box_threshold},
        )[:len(batch)]
        sam_images, sam_boxes, sam_maps = [], [], []
        for local_index, detection in enumerate(detections):
            boxes, scores, valid = select_factor_detections(detection, factors)
            record = records[start + local_index]
            record["boxes"] = boxes
            record["dino_scores"] = scores
            indices = np.flatnonzero(valid)
            if len(indices):
                sam_images.append(np.asarray(batch[local_index]).copy())
                sam_boxes.append(boxes[indices])
                sam_maps.append((start + local_index, indices))
        if not sam_images:
            continue
        with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            predictor.set_image_batch(sam_images)
            masks_batch, scores_batch, _ = predictor.predict_batch(
                point_coords_batch=None,
                point_labels_batch=None,
                box_batch=sam_boxes,
                multimask_output=False,
            )
        for masks, scores, (image_index, factor_ids) in zip(masks_batch, scores_batch, sam_maps):
            masks = np.asarray(masks)
            if masks.ndim == 4 and masks.shape[1] == 1:
                masks = masks[:, 0]
            elif masks.ndim == 2:
                masks = masks[None]
            elif masks.ndim != 3:
                raise ValueError(f"Unexpected SAM mask shape: {masks.shape}")
            masks = masks.astype(bool)
            scores = np.asarray(scores).reshape(-1)
            record = records[image_index]
            for mask_index, factor_id in enumerate(factor_ids):
                record["masks"][factor_id] = masks[mask_index]
                record["mask_valid"][factor_id] = True
                record["sam_scores"][factor_id] = scores[mask_index]
    return records
