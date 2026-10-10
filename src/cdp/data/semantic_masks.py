from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
from cdp.perception.factors import select_semantic_factors


class SemanticSidecarReader:
    def __init__(self, root: str | Path, dataset_name: str, cache_size: int = 8, factor_budget: int = 4):
        self.root = Path(root)
        self.dataset_name = dataset_name
        self.cache_size = max(1, int(cache_size))
        self.factor_budget = int(factor_budget)
        if self.factor_budget < 1:
            raise ValueError("factor_budget must include the global branch")
        self.dataset_root = self.root / "masks" / dataset_name
        if not self.dataset_root.is_dir():
            raise FileNotFoundError(f"Semantic sidecar dataset not found: {self.dataset_root}")
        self._cache: OrderedDict[int, dict] = OrderedDict()

    def _episode_dir(self, episode_index: int) -> Path:
        return self.dataset_root / f"episode_{int(episode_index):06d}"

    def _load_episode(self, episode_index: int) -> dict:
        episode_index = int(episode_index)
        cached = self._cache.pop(episode_index, None)
        if cached is not None:
            self._cache[episode_index] = cached
            return cached

        episode_dir = self._episode_dir(episode_index)
        metadata_path = episode_dir / "episode.json"
        arrays_path = episode_dir / "meta.npz"
        masks_path = episode_dir / "masks_packbits.npy"
        for path in (metadata_path, arrays_path, masks_path):
            if not path.is_file():
                raise FileNotFoundError(f"Incomplete semantic sidecar episode: missing {path}")

        metadata = json.loads(metadata_path.read_text())
        if int(metadata.get("schema_version", -1)) != 3:
            raise ValueError(
                f"Unsupported semantic sidecar schema for {episode_dir}: "
                f"{metadata.get('schema_version')}"
            )
        with np.load(arrays_path, allow_pickle=False) as values:
            arrays = {key: values[key] for key in values.files}
        masks = np.load(masks_path, mmap_mode="r")
        if masks.ndim != 5:
            raise ValueError(f"Expected [T,F,V,H,W8] masks, got {masks.shape} at {masks_path}")
        record = {"metadata": metadata, "arrays": arrays, "masks": masks}
        self._cache[episode_index] = record
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return record

    def get_frame(self, episode_index: int, frame_index: int) -> dict:
        record = self._load_episode(episode_index)
        metadata = record["metadata"]
        arrays = record["arrays"]
        masks = record["masks"]
        frame_index = int(frame_index)
        if frame_index < 0 or frame_index >= masks.shape[0]:
            raise IndexError(
                f"Frame {frame_index} outside semantic episode length {masks.shape[0]} "
                f"for episode {episode_index}"
            )

        view_count = masks.shape[2]
        widths = np.broadcast_to(np.atleast_1d(arrays["image_width"]), (view_count,))
        heights = np.broadcast_to(
            np.atleast_1d(arrays.get("image_height", masks.shape[3])), (view_count,)
        )
        valid = arrays["mask_valid"][frame_index]
        factors = []
        selected = select_semantic_factors(metadata["factors"], self.factor_budget)
        for factor in selected:
            factor_id = int(factor["factor_id"])
            view_masks = []
            view_valid = []
            for view_index in range(masks.shape[2]):
                is_valid = bool(valid[factor_id, view_index])
                view_valid.append(is_valid)
                if is_valid:
                    unpacked = np.unpackbits(
                        masks[frame_index, factor_id, view_index, :int(heights[view_index])],
                        axis=-1,
                        count=int(widths[view_index]),
                    ).astype(bool)
                    view_masks.append(unpacked)
                else:
                    view_masks.append(None)
            factors.append(
                {
                    "factor_id": factor_id,
                    "factor_text": factor["factor_text"],
                    "entity_text": factor["entity_text"],
                    "view_masks": view_masks,
                    "view_valid": view_valid,
                    "dino_scores": arrays["dino_scores"][frame_index, factor_id].astype(np.float32),
                    "sam_scores": arrays["sam_scores"][frame_index, factor_id].astype(np.float32),
                }
            )
        return {
            "schema_version": 3,
            "episode_index": int(episode_index),
            "frame_index": frame_index,
            "views": list(metadata.get("views", [])),
            "factors": factors,
            "all_semantic_masks_invalid": not any(any(factor["view_valid"]) for factor in factors),
        }
