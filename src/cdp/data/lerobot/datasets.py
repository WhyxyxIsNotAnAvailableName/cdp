# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import hashlib
import logging
import io
import json, torch
import copy
from collections import defaultdict
from pathlib import Path
from typing import Sequence
import os, random
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
import torch.distributed as dist

from cdp.data.lerobot.video import get_all_frames, get_frames_by_timestamps

from cdp.data.lerobot.embodiment_tags import EmbodimentTag
from cdp.data.lerobot.schema import (
    DatasetMetadata,
    DatasetStatisticalValues,
    LeRobotModalityMetadata,
    LeRobotStateActionMetadata,
)
from cdp.data.lerobot.transform import ComposedModalityTransform
from cdp.data.lerobot.transform.state_action import StateActionTransform

from functools import partial
from typing import Tuple, List
import pickle
import gc


logger = logging.getLogger(__name__)


LE_ROBOT_MODALITY_FILENAME = "meta/modality.json"
LE_ROBOT_EPISODE_FILENAME = "meta/episodes.jsonl"
LE_ROBOT_TASKS_FILENAME = "meta/tasks.jsonl"
LE_ROBOT_INFO_FILENAME = "meta/info.json"
LE_ROBOT_STATS_FILENAME = "meta/stats_cdp.json"
LE_ROBOT_DATA_FILENAME = "data/*/*.parquet"
LE_ROBOT_STEPS_FILENAME = "meta/steps.pkl"
LE_ROBOT_STATS_FORMAT_VERSION = 2
EPSILON = 5e-4


LE_ROBOT3_TASKS_FILENAME = "meta/tasks.parquet"
LE_ROBOT3_EPISODE_FILENAME = "meta/episodes/*/*.parquet"


def calculate_dataset_statistics(parquet_paths: list[Path]) -> dict:
    all_low_dim_data_list = []

    for parquet_path in tqdm(
        sorted(list(parquet_paths)),
        desc="Collecting all parquet files...",
    ):

        parquet_data = pd.read_parquet(parquet_path)
        parquet_data = parquet_data
        all_low_dim_data_list.append(parquet_data)

    if not all_low_dim_data_list:
        raise FileNotFoundError(
            f"No parquet files found under the provided paths: {[str(p) for p in parquet_paths[:3]]}..."
            f" — make sure the dataset has been downloaded/converted before training."
        )

    all_low_dim_data = pd.concat(all_low_dim_data_list, axis=0)

    dataset_statistics = {}
    for le_modality in tqdm(all_low_dim_data.columns, desc="Processing modalities"):
        if "task_info" in le_modality:
            continue
        try:
            np_data = np.vstack(
                [np.asarray(x, dtype=np.float32) for x in all_low_dim_data[le_modality]]
            )
        except Exception as e:
            logger.warning(f'Failed to process modality {le_modality} due to error: {e}')
            continue

        dataset_statistics[le_modality] = {
            "mean": np.mean(np_data, axis=0).tolist(),
            "std": np.std(np_data, axis=0).tolist(),
            "min": np.min(np_data, axis=0).tolist(),
            "max": np.max(np_data, axis=0).tolist(),
            "q01": np.quantile(np_data, 0.01, axis=0).tolist(),
            "q99": np.quantile(np_data, 0.99, axis=0).tolist(),
        }
    return dataset_statistics


def _normalize_action_mode(mode: str) -> str:
    mode = str(mode).lower()
    if mode in {"absolute", "raw"}:
        mode = "abs"
    if mode not in {"abs", "delta", "rel"}:
        mode = "abs"
    return mode


def _normalize_action_mode_apply_keys(
    action_mode_apply_keys: Sequence[str] | None,
    fallback_keys: Sequence[str] | None = None,
) -> list[str]:
    source_keys = action_mode_apply_keys if action_mode_apply_keys else (fallback_keys or [])
    normalized = []
    for key in source_keys:
        key = str(key)
        if not key.startswith("action."):
            key = f"action.{key}"
        normalized.append(key)
    return normalized


def _normalize_action_mode_state_map(action_mode_state_map: dict[str, str] | None) -> dict[str, str]:
    normalized = {}
    for action_key, state_key in (action_mode_state_map or {}).items():
        action_key = str(action_key)
        state_key = str(state_key)
        if not action_key.startswith("action."):
            action_key = f"action.{action_key}"
        if not state_key.startswith("state."):
            state_key = f"state.{state_key}"
        normalized[action_key] = state_key
    return normalized


def _build_stats_cache_config(
    action_mode: str,
) -> dict:
    return {
        "mode": action_mode,
    }


def _invalidate_legacy_stats_cache(stats_path: Path, reason: str) -> None:
    if not stats_path.exists():
        return
    logger.warning(f'Removing stale dataset statistics cache at {stats_path}: {reason}')
    stats_path.unlink()


def _load_stats_cache(
    stats_path: Path,
    expected_config: dict,
    *,
    invalidate_legacy: bool,
) -> dict | None:
    if not stats_path.exists():
        return None

    try:
        with open(stats_path, "r") as f:
            payload = json.load(f)
    except Exception as exc:
        if invalidate_legacy:
            _invalidate_legacy_stats_cache(stats_path, f"failed to load JSON ({exc})")
        return None

    if not isinstance(payload, dict):
        if invalidate_legacy:
            _invalidate_legacy_stats_cache(stats_path, "unexpected top-level format")
        return None

    format_version = payload.get("__format_version")
    cache_config = payload.get("__cache_config")
    statistics = payload.get("statistics")
    if format_version != LE_ROBOT_STATS_FORMAT_VERSION or cache_config is None or statistics is None:
        if invalidate_legacy:
            _invalidate_legacy_stats_cache(stats_path, "legacy statistics format detected")
        return None

    if cache_config != expected_config:
        if invalidate_legacy:
            _invalidate_legacy_stats_cache(stats_path, "statistics config mismatch, rebuilding cache")
        return None

    return statistics


def _save_stats_cache(stats_path: Path, cache_config: dict, statistics: dict) -> None:
    payload = {
        "__format_version": LE_ROBOT_STATS_FORMAT_VERSION,
        "__cache_config": cache_config,
        "statistics": statistics,
    }
    tmp_path = stats_path.with_suffix(".tmp")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=4)
    os.replace(tmp_path, stats_path)


def _compute_statistics_for_mode(
    parquet_paths: list[Path],
    dataset_name: str,
    action_mode: str,
    lerobot_modality_meta: "LeRobotModalityMetadata",
    action_keys_full: list[str],
    state_keys_full: list[str],
    action_indices: list[int] | None,
    state_indices: list[int] | None,
    action_mode_apply_keys: list[str] | None,
    action_mode_state_map: dict[str, str] | None,
) -> dict:
    if int(os.environ.get("RANK", "0")) == 0:
        logger.info(f'Calculating dataset statistics for {dataset_name} (mode={action_mode})')

    base_stats = calculate_dataset_statistics(parquet_paths)

    if action_mode == "abs":
        return base_stats

    if action_indices is None or state_indices is None:
        raise ValueError(
            "Both action and state modalities are required to compute "
            f"{action_mode} action mode statistics."
        )

    if action_mode == "delta":
        return calculate_delta_action_statistics(
            parquet_paths=parquet_paths,
            lerobot_modality_meta=lerobot_modality_meta,
            action_keys_full=action_keys_full,
            state_keys_full=state_keys_full,
            action_indices=action_indices,
            state_indices=state_indices,
            action_mode_apply_keys=action_mode_apply_keys,
            action_mode_state_map=action_mode_state_map,
            base_stats=base_stats,
        )
    if action_mode == "rel":
        return calculate_rel_action_statistics(
            parquet_paths=parquet_paths,
            lerobot_modality_meta=lerobot_modality_meta,
            action_keys_full=action_keys_full,
            state_keys_full=state_keys_full,
            action_indices=action_indices,
            state_indices=state_indices,
            action_mode_apply_keys=action_mode_apply_keys,
            action_mode_state_map=action_mode_state_map,
            base_stats=base_stats,
        )
    raise ValueError(f"Unsupported action mode for statistics: {action_mode}")


def _load_or_compute_statistics(
    stats_path: Path,
    stats_cache_config: dict,
    parquet_paths: list[Path],
    dataset_name: str,
    action_mode: str,
    lerobot_modality_meta: "LeRobotModalityMetadata",
    action_keys_full: list[str],
    state_keys_full: list[str],
    action_indices: list[int] | None,
    state_indices: list[int] | None,
    action_mode_apply_keys: list[str] | None,
    action_mode_state_map: dict[str, str] | None,
) -> dict:
    le_statistics = _load_stats_cache(
        stats_path,
        stats_cache_config,
        invalidate_legacy=True,
    )
    if le_statistics is not None:
        return le_statistics

    le_statistics = _compute_statistics_for_mode(
        parquet_paths=parquet_paths,
        dataset_name=dataset_name,
        action_mode=action_mode,
        lerobot_modality_meta=lerobot_modality_meta,
        action_keys_full=action_keys_full,
        state_keys_full=state_keys_full,
        action_indices=action_indices,
        state_indices=state_indices,
        action_mode_apply_keys=action_mode_apply_keys,
        action_mode_state_map=action_mode_state_map,
    )
    _save_stats_cache(stats_path, stats_cache_config, le_statistics)
    return le_statistics


def _get_action_col_slices(
    lerobot_modality_meta: "LeRobotModalityMetadata",
    action_keys_full: list[str],
    state_keys_full: list[str],
    action_mode_apply_keys: list[str] | None = None,
    action_mode_state_map: dict[str, str] | None = None,
) -> dict[str, list[tuple[tuple[int, int], str, tuple[int, int], str, str]]]:
    apply_keys = _normalize_action_mode_apply_keys(action_mode_apply_keys, action_keys_full)
    action_mode_state_map = _normalize_action_mode_state_map(action_mode_state_map)

    action_meta = lerobot_modality_meta.action
    state_meta = lerobot_modality_meta.state

    action_col_slices: dict[str, list[tuple[tuple[int, int], str, tuple[int, int]]]] = {}
    for action_key in apply_keys:
        if not action_key.startswith("action."):
            raise ValueError(f"Invalid action key {action_key}. Expected prefix 'action.'.")
        state_key = action_mode_state_map.get(action_key, action_key.replace("action.", "state.", 1))
        if state_key not in state_keys_full:
            raise ValueError(
                f"State key {state_key} not found for action key {action_key}. "
                f"Add it to action_mode_state_map or remove {action_key} from action_mode_apply_keys."
            )

        action_subkey = action_key.replace("action.", "", 1)
        state_subkey = state_key.replace("state.", "", 1)
        if action_subkey not in action_meta or state_subkey not in state_meta:
            raise ValueError(f"Action/state key missing in metadata: {action_key} -> {state_key}")

        action_cfg = action_meta[action_subkey]
        state_cfg = state_meta[state_subkey]
        action_col = action_cfg.original_key or action_subkey
        state_col = state_cfg.original_key or state_subkey
        action_slice = (action_cfg.start, action_cfg.end)
        state_slice = (state_cfg.start, state_cfg.end)
        action_padding = "first_last" if action_cfg.absolute else "zero"
        state_padding = "first_last" if state_cfg.absolute else "zero"
        action_col_slices.setdefault(action_col, []).append(
            (action_slice, state_col, state_slice, action_padding, state_padding)
        )

    return action_col_slices


def calculate_delta_action_statistics(
    parquet_paths: list[Path],
    lerobot_modality_meta: "LeRobotModalityMetadata",
    action_keys_full: list[str],
    state_keys_full: list[str],
    action_indices: list[int],
    state_indices: list[int],
    action_mode_apply_keys: list[str] | None = None,
    action_mode_state_map: dict[str, str] | None = None,
    base_stats: dict | None = None,
) -> dict:
    if base_stats is None:
        base_stats = calculate_dataset_statistics(parquet_paths)

    action_col_slices = _get_action_col_slices(
        lerobot_modality_meta, action_keys_full, state_keys_full, action_mode_apply_keys, action_mode_state_map
    )
    if not action_col_slices:
        raise ValueError("No action columns found in the dataset.")

    def _get_chunk(array: np.ndarray, step_indices: np.ndarray, padding_strategy: str) -> np.ndarray:
        max_length = array.shape[0]
        front_padding = step_indices < 0
        end_padding = step_indices >= max_length
        padding_positions = np.logical_or(front_padding, end_padding)
        output = np.zeros((len(step_indices), array.shape[1]), dtype=array.dtype)
        if (~padding_positions).any():
            output[~padding_positions] = array[step_indices[~padding_positions]]
        if padding_positions.any():
            if padding_strategy == "first_last":
                output[front_padding] = array[0]
                output[end_padding] = array[-1]
            elif padding_strategy == "zero":
                output[padding_positions] = 0
            else:
                raise ValueError(f"Invalid padding strategy: {padding_strategy}")
        return output

    accum: dict[str, list[np.ndarray]] = {col: [] for col in action_col_slices.keys()}
    for parquet_path in tqdm(sorted(list(parquet_paths)), desc="Collecting delta action stats"):
        data = pd.read_parquet(parquet_path)
        trajectory_length = len(data)
        for action_col, slice_list in action_col_slices.items():
            if action_col not in data.columns:
                raise ValueError(f"{action_col} not found in parquet columns.")
            action_matrix = np.stack(data[action_col])
            action_padding_ref = slice_list[0][3]
            prepared_slices = []
            for a_slice, state_col, s_slice, action_padding, state_padding in slice_list:
                if state_col not in data.columns:
                    raise ValueError(f"{state_col} not found in parquet columns.")
                state_matrix = np.stack(data[state_col])
                state_part_full = state_matrix[:, s_slice[0] : s_slice[1]]
                prepared_slices.append((a_slice, state_part_full, state_padding))
            for base_index in range(trajectory_length):
                action_steps = np.array(action_indices) + base_index
                action_chunk_full = _get_chunk(action_matrix, action_steps, action_padding_ref)

                for a_slice, state_part_full, state_padding in prepared_slices:
                    action_part_chunk = action_chunk_full[:, a_slice[0] : a_slice[1]]
                    state_chunk = _get_chunk(state_part_full, np.array(state_indices) + base_index, state_padding)
                    if action_part_chunk.shape[1] != state_chunk.shape[1]:
                        raise ValueError(f"Action/state dim mismatch for {action_col}:{a_slice}")

                    out = action_part_chunk.copy()
                    if len(out) > 1:
                        out[1:] = action_part_chunk[1:] - action_part_chunk[:-1]
                    out[0] = action_part_chunk[0] - state_chunk[0]
                    action_chunk_full[:, a_slice[0] : a_slice[1]] = out

                accum[action_col].append(action_chunk_full)

    delta_stats = copy.deepcopy(base_stats)
    for action_col, series_list in accum.items():
        if not series_list:
            continue
        all_values = np.concatenate(series_list, axis=0).astype(np.float32)
        delta_stats[action_col] = {
            "mean": np.mean(all_values, axis=0).tolist(),
            "std": np.std(all_values, axis=0).tolist(),
            "min": np.min(all_values, axis=0).tolist(),
            "max": np.max(all_values, axis=0).tolist(),
            "q01": np.quantile(all_values, 0.01, axis=0).tolist(),
            "q99": np.quantile(all_values, 0.99, axis=0).tolist(),
        }
    return delta_stats


def calculate_rel_action_statistics(
    parquet_paths: list[Path],
    lerobot_modality_meta: "LeRobotModalityMetadata",
    action_keys_full: list[str],
    state_keys_full: list[str],
    action_indices: list[int],
    state_indices: list[int],
    action_mode_apply_keys: list[str] | None = None,
    action_mode_state_map: dict[str, str] | None = None,
    base_stats: dict | None = None,
) -> dict:
    if base_stats is None:
        base_stats = calculate_dataset_statistics(parquet_paths)

    action_col_slices = _get_action_col_slices(
        lerobot_modality_meta, action_keys_full, state_keys_full, action_mode_apply_keys, action_mode_state_map
    )
    if not action_col_slices:
        raise ValueError("No action columns found in the dataset.")

    def _get_chunk(array: np.ndarray, step_indices: np.ndarray, padding_strategy: str) -> np.ndarray:
        max_length = array.shape[0]
        front_padding = step_indices < 0
        end_padding = step_indices >= max_length
        padding_positions = np.logical_or(front_padding, end_padding)
        output = np.zeros((len(step_indices), array.shape[1]), dtype=array.dtype)
        if (~padding_positions).any():
            output[~padding_positions] = array[step_indices[~padding_positions]]
        if padding_positions.any():
            if padding_strategy == "first_last":
                output[front_padding] = array[0]
                output[end_padding] = array[-1]
            elif padding_strategy == "zero":
                output[padding_positions] = 0
            else:
                raise ValueError(f"Invalid padding strategy: {padding_strategy}")
        return output

    accum: dict[str, list[np.ndarray]] = {col: [] for col in action_col_slices.keys()}
    for parquet_path in tqdm(sorted(list(parquet_paths)), desc="Collecting rel action stats"):
        data = pd.read_parquet(parquet_path)
        trajectory_length = len(data)
        for action_col, slice_list in action_col_slices.items():
            if action_col not in data.columns:
                raise ValueError(f"{action_col} not found in parquet columns.")
            action_matrix = np.stack(data[action_col])
            action_padding_ref = slice_list[0][3]
            prepared_slices = []
            for a_slice, state_col, s_slice, action_padding, state_padding in slice_list:
                if state_col not in data.columns:
                    raise ValueError(f"{state_col} not found in parquet columns.")
                state_matrix = np.stack(data[state_col])
                state_part_full = state_matrix[:, s_slice[0] : s_slice[1]]
                prepared_slices.append((a_slice, state_part_full, state_padding))
            for base_index in range(trajectory_length):
                action_steps = np.array(action_indices) + base_index
                action_chunk_full = _get_chunk(action_matrix, action_steps, action_padding_ref)

                for a_slice, state_part_full, state_padding in prepared_slices:
                    action_part_chunk = action_chunk_full[:, a_slice[0] : a_slice[1]]
                    state_chunk = _get_chunk(state_part_full, np.array(state_indices) + base_index, state_padding)
                    if action_part_chunk.shape[1] != state_chunk.shape[1]:
                        raise ValueError(f"Action/state dim mismatch for {action_col}:{a_slice}")

                    out = action_part_chunk - state_chunk[0]
                    action_chunk_full[:, a_slice[0] : a_slice[1]] = out

                accum[action_col].append(action_chunk_full)

    rel_stats = copy.deepcopy(base_stats)
    for action_col, series_list in accum.items():
        if not series_list:
            continue
        all_values = np.concatenate(series_list, axis=0).astype(np.float32)
        rel_stats[action_col] = {
            "mean": np.mean(all_values, axis=0).tolist(),
            "std": np.std(all_values, axis=0).tolist(),
            "min": np.min(all_values, axis=0).tolist(),
            "max": np.max(all_values, axis=0).tolist(),
            "q01": np.quantile(all_values, 0.01, axis=0).tolist(),
            "q99": np.quantile(all_values, 0.99, axis=0).tolist(),
        }
    return rel_stats

class ModalityConfig(BaseModel):
    delta_indices: list[int]
    """Delta indices to sample relative to the current index. The returned data will correspond to the original data at a sampled base index + delta indices."""
    modality_keys: list[str]
    """The keys to load for the modality in the dataset."""


class LeRobotSingleDataset(Dataset):
    def __init__(
        self,
        dataset_path: Path | str,
        modality_configs: dict[str, ModalityConfig],
        embodiment_tag: str | EmbodimentTag,
        video_backend: str = "decord",
        video_backend_kwargs: dict | None = None,
        transforms: ComposedModalityTransform | None = None,
        delete_pause_frame: bool = False,
        data_cfg = None,
        dataset_name: str | None = None,
        **kwargs,
    ):
        self.data_cfg = data_cfg
        if not Path(dataset_path).exists():
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist")

        self._lerobot_version =  self.data_cfg.get("lerobot_version", "v2.0")

        self._action_mode = None
        self._action_mode_state_map = {}
        self._action_mode_apply_keys = None

        self.delete_pause_frame = delete_pause_frame

        self.modality_configs = modality_configs
        self.video_backend = video_backend
        self.video_backend_kwargs = video_backend_kwargs if video_backend_kwargs is not None else {}
        self.transforms = (
            transforms if transforms is not None else ComposedModalityTransform(transforms=[])
        )

        self._dataset_path = Path(dataset_path)
        self._dataset_name = dataset_name if dataset_name is not None else self._dataset_path.name
        self._semantic_sidecar = None
        semantic_sidecar_root = self.data_cfg.get("semantic_sidecar_root", None) if self.data_cfg else None
        if semantic_sidecar_root:
            from cdp.data.semantic_masks import SemanticSidecarReader

            self._semantic_sidecar = SemanticSidecarReader(
                root=semantic_sidecar_root,
                dataset_name=self._dataset_name,
                cache_size=int(self.data_cfg.get("semantic_sidecar_cache_size", 8)),
                factor_budget=int(self.data_cfg.get("semantic_factor_budget", 4)),
            )
        if isinstance(embodiment_tag, EmbodimentTag):
            self.tag = embodiment_tag.value
        else:
            self.tag = embodiment_tag

        self._init_action_mode()
        self._metadata = self._get_metadata(EmbodimentTag(self.tag))

        self._lerobot_modality_meta = self._get_lerobot_modality_meta()
        self._lerobot_info_meta = self._get_lerobot_info_meta()
        self._data_path_pattern = self._get_data_path_pattern()
        self._video_path_pattern = self._get_video_path_pattern()
        self._chunk_size = self._get_chunk_size()
        self._tasks = self._get_tasks()

        self.curr_traj_data = None
        self.curr_traj_id = None

        self._trajectory_ids, self._trajectory_lengths = self._get_trajectories()
        self._modality_keys = self._get_modality_keys()
        self._delta_indices = self._get_delta_indices()
        self._all_steps = self._get_all_steps()
        self.set_transforms_metadata(self.metadata)
        self.set_epoch(0)


        self._check_integrity()

    @property
    def dataset_path(self) -> Path:
        return self._dataset_path

    @property
    def metadata(self) -> DatasetMetadata:
        return self._metadata

    @property
    def trajectory_ids(self) -> np.ndarray:
        return self._trajectory_ids

    @property
    def trajectory_lengths(self) -> np.ndarray:
        return self._trajectory_lengths

    @property
    def all_steps(self) -> list[tuple[int, int]]:
        return self._all_steps

    @property
    def modality_keys(self) -> dict:
        return self._modality_keys

    @property
    def delta_indices(self) -> dict[str, np.ndarray]:
        return self._delta_indices

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    @property
    def lerobot_modality_meta(self) -> LeRobotModalityMetadata:
        return self._lerobot_modality_meta

    @property
    def lerobot_info_meta(self) -> dict:
        return self._lerobot_info_meta

    @property
    def data_path_pattern(self) -> str:
        return self._data_path_pattern

    @property
    def video_path_pattern(self) -> str:
        return self._video_path_pattern

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def tasks(self) -> pd.DataFrame:
        return self._tasks

    def _get_metadata(self, embodiment_tag: EmbodimentTag) -> DatasetMetadata:
        modality_meta_path = self.dataset_path / LE_ROBOT_MODALITY_FILENAME
        assert (
            modality_meta_path.exists()
        ), f"Please provide a {LE_ROBOT_MODALITY_FILENAME} file in {self.dataset_path}"

        simplified_modality_meta: dict[str, dict] = {}
        with open(modality_meta_path, "r") as f:
            le_modality_meta = LeRobotModalityMetadata.model_validate(json.load(f))
        for modality in ["state", "action"]:
            simplified_modality_meta[modality] = {}
            le_state_action_meta: dict[str, LeRobotStateActionMetadata] = getattr(
                le_modality_meta, modality
            )
            for subkey in le_state_action_meta:
                state_action_dtype = np.dtype(le_state_action_meta[subkey].dtype)
                if np.issubdtype(state_action_dtype, np.floating):
                    continuous = True
                else:
                    continuous = False
                simplified_modality_meta[modality][subkey] = {
                    "absolute": le_state_action_meta[subkey].absolute,
                    "rotation_type": le_state_action_meta[subkey].rotation_type,
                    "shape": [
                        le_state_action_meta[subkey].end - le_state_action_meta[subkey].start
                    ],
                    "continuous": continuous,
                }

        le_info_path = self.dataset_path / LE_ROBOT_INFO_FILENAME
        assert (
            le_info_path.exists()
        ), f"Please provide a {LE_ROBOT_INFO_FILENAME} file in {self.dataset_path}"
        with open(le_info_path, "r") as f:
            le_info = json.load(f)
        simplified_modality_meta["video"] = {}
        for new_key in le_modality_meta.video:
            original_key = le_modality_meta.video[new_key].original_key
            if original_key is None:
                original_key = new_key
            le_video_meta = le_info["features"][original_key]
            height = le_video_meta["shape"][le_video_meta["names"].index("height")]
            width = le_video_meta["shape"][le_video_meta["names"].index("width")]

            try:
                channels = le_video_meta["shape"][le_video_meta["names"].index("channel")]
                fps = le_video_meta["video_info"]["video.fps"]
            except (ValueError, KeyError):
                try:
                    channels = le_video_meta["info"]["video.channels"]
                    fps = le_video_meta["info"]["video.fps"]
                except (ValueError, KeyError):

                    channels = 3
                    fps = le_info.get("fps", 30)
            simplified_modality_meta["video"][new_key] = {
                "resolution": [width, height],
                "channels": channels,
                "fps": fps,
            }

        def is_main():
            return (not dist.is_initialized()) or dist.get_rank() == 0

        action_mode = _normalize_action_mode(self.data_cfg.get("action_mode", "abs") if self.data_cfg else "abs")

        stats_path = self.dataset_path / LE_ROBOT_STATS_FILENAME
        action_cfg = self.modality_configs.get("action")
        state_cfg = self.modality_configs.get("state")
        action_keys_full = list(action_cfg.modality_keys) if action_cfg else []
        state_keys_full = list(state_cfg.modality_keys) if state_cfg else []
        action_indices = list(action_cfg.delta_indices) if action_cfg else None
        state_indices = list(state_cfg.delta_indices) if state_cfg else None

        apply_keys = _normalize_action_mode_apply_keys(
            self.data_cfg.get("action_mode_apply_keys", None) if self.data_cfg else None,
            action_keys_full,
        )
        normalized_state_map = _normalize_action_mode_state_map(
            self.data_cfg.get("action_mode_state_map", {}) if self.data_cfg else {}
        )
        stats_cache_config = _build_stats_cache_config(
            action_mode=action_mode,
        )
        parquet_files = list(self.dataset_path.glob(LE_ROBOT_DATA_FILENAME))

        if is_main():
            le_statistics = _load_or_compute_statistics(
                stats_path,
                stats_cache_config=stats_cache_config,
                parquet_paths=parquet_files,
                dataset_name=self.dataset_name,
                action_mode=action_mode,
                lerobot_modality_meta=le_modality_meta,
                action_keys_full=action_keys_full,
                state_keys_full=state_keys_full,
                action_indices=action_indices,
                state_indices=state_indices,
                action_mode_apply_keys=apply_keys,
                action_mode_state_map=normalized_state_map,
            )
        else:
            le_statistics = None

        if dist.is_initialized():
            dist.barrier()

        if le_statistics is None:
            le_statistics = _load_stats_cache(
                stats_path,
                stats_cache_config,
                invalidate_legacy=False,
            )
            if le_statistics is None:
                raise RuntimeError(
                    f"Dataset statistics cache is missing or invalid after sync: {stats_path}"
                )

        for stat in le_statistics.values():
            DatasetStatisticalValues.model_validate(stat)

        dataset_statistics = {}
        for our_modality in ["state", "action"]:
            dataset_statistics[our_modality] = {}
            for subkey in simplified_modality_meta[our_modality]:
                dataset_statistics[our_modality][subkey] = {}
                state_action_meta = le_modality_meta.get_key_meta(f"{our_modality}.{subkey}")
                assert isinstance(state_action_meta, LeRobotStateActionMetadata)
                le_modality = state_action_meta.original_key
                for stat_name in le_statistics[le_modality]:
                    indices = np.arange(
                        state_action_meta.start,
                        state_action_meta.end,
                    )
                    stat = np.array(le_statistics[le_modality][stat_name])
                    dataset_statistics[our_modality][subkey][stat_name] = stat[indices].tolist()

        metadata = DatasetMetadata(
            statistics=dataset_statistics,  # type: ignore
            modalities=simplified_modality_meta,  # type: ignore
            embodiment_tag=embodiment_tag,
        )

        return metadata

    def _get_trajectories(self) -> tuple[np.ndarray, np.ndarray]:
        if self._lerobot_version == "v2.0":
            file_path = self.dataset_path / LE_ROBOT_EPISODE_FILENAME
            with open(file_path, "r") as f:
                episode_metadata = [json.loads(line) for line in f]
            trajectory_ids = []
            trajectory_lengths = []
            for episode in episode_metadata:
                trajectory_ids.append(episode["episode_index"])
                trajectory_lengths.append(episode["length"])
            return np.array(trajectory_ids), np.array(trajectory_lengths)

        elif self._lerobot_version == "v3.0":
            file_paths = sorted(list((self.dataset_path).glob(LE_ROBOT3_EPISODE_FILENAME)))
            trajectory_ids = []
            trajectory_lengths = []

            self.trajectory_ids_to_metadata = {}
            for file_path in file_paths:
                episodes_data = pd.read_parquet(file_path)
                timestamp_cols = [
                    c
                    for c in episodes_data.columns
                    if str(c).startswith("videos/") and str(c).endswith("/from_timestamp")
                ]
                for index, episode in episodes_data.iterrows():
                    trajectory_ids.append(episode["episode_index"])
                    trajectory_lengths.append(episode["length"])

                    from_timestamps = {}
                    for col in timestamp_cols:
                        value = episode[col]
                        if pd.isna(value):
                            continue

                        video_key = str(col)[len("videos/") : -len("/from_timestamp")]
                        from_timestamps[video_key] = float(value)

                    video_file_indices = {}
                    for col in timestamp_cols:
                        video_key = str(col)[len("videos/") : -len("/from_timestamp")]
                        chunk_col = f"videos/{video_key}/chunk_index"
                        file_col = f"videos/{video_key}/file_index"
                        if chunk_col in episode and file_col in episode:
                            video_file_indices[video_key] = {
                                "chunk_index": int(episode[chunk_col]),
                                "file_index": int(episode[file_col]),
                            }
                    episode_meta = {
                        "data/chunk_index": episode["data/chunk_index"],
                        "data/file_index": episode["data/file_index"],
                        "data/file_from_index": index,
                        "videos/from_timestamps": from_timestamps,
                        "videos/file_indices": video_file_indices,
                    }

                    self.trajectory_ids_to_metadata[trajectory_ids[-1]] = episode_meta

            return np.array(trajectory_ids), np.array(trajectory_lengths)

    def _get_all_steps(self) -> list[tuple[int, int]]:
        def is_main():
            return (not dist.is_initialized()) or dist.get_rank() == 0

        config_key = self._get_steps_config_key()
        steps_filename = "steps_data_index.pkl"
        steps_path = self.dataset_path / "meta" / steps_filename

        if steps_path.exists():
            try:
                with open(steps_path, "rb") as f:
                    cached_data = pickle.load(f)
                return cached_data["steps"]
            except Exception as e:

                logger.info(f"[RANK {os.environ.get('RANK', 'NA')}] Failed to load cached steps ({e}), will rebuild.")

        if is_main():
            all_steps = self._get_all_steps_single_process()

            cache_data = {
                "config_key": config_key,
                "steps": all_steps,
                "num_trajectories": len(self.trajectory_ids),
                "total_steps": len(all_steps),
                "computed_timestamp": pd.Timestamp.now().isoformat(),
                "delete_pause_frame": self.delete_pause_frame,
            }

            steps_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = steps_path.with_suffix(".tmp")

            with open(tmp_path, "wb") as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, steps_path)

            logger.info(f'Cached steps saved to {steps_path}')

        if dist.is_initialized():
            dist.barrier()

        with open(steps_path, "rb") as f:
            cached_data = pickle.load(f)

        return cached_data["steps"]

    def _get_steps_config_key(self) -> str:
        config_dict = {
            "delete_pause_frame": self.delete_pause_frame,
            "dataset_name": self.dataset_name,
        }

        config_str = str(sorted(config_dict.items()))
        return hashlib.md5(config_str.encode()).hexdigest()[:12]

    def _get_all_steps_single_process(self) -> list[tuple[int, int]]:
        all_steps: list[tuple[int, int]] = []
        skipped_trajectories = 0
        processed_trajectories = 0

        has_language_modality = 'language' in self.modality_keys and len(self.modality_keys['language']) > 0

        for trajectory_id, trajectory_length in tqdm(zip(self.trajectory_ids, self.trajectory_lengths), total=len(self.trajectory_ids), desc="Getting All Step"):
            try:
                if self._lerobot_version == "v2.0":
                    data = self.get_trajectory_data(trajectory_id)
                elif self._lerobot_version == "v3.0":
                    data = self.get_trajectory_data_lerobot_v3(trajectory_id)

                trajectory_skipped = False

                if has_language_modality:
                    self.curr_traj_data = data

                    language_instruction = self.get_language(trajectory_id, self.modality_keys['language'][0], 0)
                    if not language_instruction or language_instruction[0] == "":
                        logger.warning("Trajectory %s skipped: empty instruction", trajectory_id)
                        skipped_trajectories += 1
                        trajectory_skipped = True
                        continue

            except Exception as e:
                logger.warning("Trajectory %s skipped: %s", trajectory_id, e)
                skipped_trajectories += 1
                trajectory_skipped = True
                continue

            if not trajectory_skipped:
                processed_trajectories += 1

            for base_index in range(trajectory_length):
                all_steps.append((trajectory_id, base_index))

        logger.info(
            "Dataset %s: trajectories=%d skipped=%d steps=%d",
            self.dataset_name, processed_trajectories, skipped_trajectories, len(all_steps),
        )

        return all_steps

    def _get_position_and_gripper_values(self, data: pd.DataFrame) -> tuple[list, list]:
        action_keys = self.modality_keys.get('action', [])

        delta_position_values = None
        position_candidates = ['delta_eef_position']
        coordinate_candidates = ['x', 'y', 'z']

        for pos_key in position_candidates:
            full_key = f"action.{pos_key}"
            if full_key in action_keys:
                try:

                    le_action_cfg = self.lerobot_modality_meta.action
                    subkey = pos_key
                    if subkey in le_action_cfg:
                        le_key = le_action_cfg[subkey].original_key or subkey
                        if le_key in data.columns:
                            data_array = np.stack(data[le_key])
                            le_indices = np.arange(le_action_cfg[subkey].start, le_action_cfg[subkey].end)
                            filtered_data = data_array[:, le_indices]
                            delta_position_values = filtered_data.tolist()
                            break
                except Exception:
                    continue

        if delta_position_values is None:
            x_data, y_data, z_data = None, None, None
            for coord in coordinate_candidates:
                full_key = f"action.{coord}"
                if full_key in action_keys:
                    try:
                        le_action_cfg = self.lerobot_modality_meta.action
                        if coord in le_action_cfg:
                            le_key = le_action_cfg[coord].original_key or coord
                            if le_key in data.columns:
                                data_array = np.stack(data[le_key])
                                le_indices = np.arange(le_action_cfg[coord].start, le_action_cfg[coord].end)
                                coord_data = data_array[:, le_indices].flatten()
                                if coord == 'x':
                                    x_data = coord_data
                                elif coord == 'y':
                                    y_data = coord_data
                                elif coord == 'z':
                                    z_data = coord_data
                    except Exception:
                        continue

            if x_data is not None and y_data is not None and z_data is not None:
                delta_position_values = np.column_stack((x_data, y_data, z_data)).tolist()

        if delta_position_values is None:

            if 'action.delta_eef_position' in data.columns:
                delta_position_values = data['action.delta_eef_position'].to_numpy().tolist()
            elif all(col in data.columns for col in ['action.x', 'action.y', 'action.z']):
                x_vals = data['action.x'].to_numpy()
                y_vals = data['action.y'].to_numpy()
                z_vals = data['action.z'].to_numpy()
                delta_position_values = np.column_stack((x_vals, y_vals, z_vals)).tolist()
            else:
                raise ValueError(f"No suitable position columns found. Available columns: {data.columns.tolist()}")

        gripper_values = None
        gripper_candidates = ['gripper_close', 'gripper']

        for grip_key in gripper_candidates:
            full_key = f"action.{grip_key}"
            if full_key in action_keys:
                try:
                    le_action_cfg = self.lerobot_modality_meta.action
                    if grip_key in le_action_cfg:
                        le_key = le_action_cfg[grip_key].original_key or grip_key
                        if le_key in data.columns:
                            data_array = np.stack(data[le_key])
                            le_indices = np.arange(le_action_cfg[grip_key].start, le_action_cfg[grip_key].end)
                            gripper_data = data_array[:, le_indices].flatten()
                            gripper_values = gripper_data.tolist()
                            break
                except Exception:
                    continue

        if gripper_values is None:

            if 'action.gripper_close' in data.columns:
                gripper_values = data['action.gripper_close'].to_numpy().tolist()
            elif 'action.gripper' in data.columns:
                gripper_values = data['action.gripper'].to_numpy().tolist()
            else:
                raise ValueError(f"No suitable gripper columns found. Available columns: {data.columns.tolist()}")

        return delta_position_values, gripper_values

    def _get_modality_keys(self) -> dict:
        modality_keys = defaultdict(list)
        for modality, config in self.modality_configs.items():
            modality_keys[modality] = config.modality_keys
        return modality_keys

    def _get_delta_indices(self) -> dict[str, np.ndarray]:
        delta_indices: dict[str, np.ndarray] = {}
        for config in self.modality_configs.values():
            for key in config.modality_keys:
                delta_indices[key] = np.array(config.delta_indices)
        return delta_indices

    def _init_action_mode(self) -> None:
        if self.data_cfg is None:
            self._action_mode = "abs"
            return

        action_mode = self.data_cfg.get("action_mode", "abs")
        if action_mode is None:
            action_mode = "abs"
        action_mode = _normalize_action_mode(action_mode)
        if action_mode not in {"abs", "delta", "rel"}:
            raise ValueError(f"Invalid action_mode: {action_mode}. Expected one of: abs, delta, rel.")
        self._action_mode = action_mode

        apply_keys = _normalize_action_mode_apply_keys(self.data_cfg.get("action_mode_apply_keys", None))
        if apply_keys:
            self._action_mode_apply_keys = apply_keys

        self._action_mode_state_map = _normalize_action_mode_state_map(
            self.data_cfg.get("action_mode_state_map", {}) or {}
        )

    def _infer_state_key_for_action(self, action_key: str) -> str | None:
        if action_key in self._action_mode_state_map:
            return self._action_mode_state_map[action_key]

        if not action_key.startswith("action."):
            return None
        base = action_key.replace("action.", "", 1)
        if f"state.{base}" in self.modality_keys.get("state", []):
            return f"state.{base}"
        return None

    def _apply_action_mode(self, data: dict) -> dict:
        if self._action_mode in (None, "abs"):
            return data

        action_keys = self._action_mode_apply_keys or self.modality_keys.get("action", [])
        for action_key in action_keys:
            if action_key not in data:
                logger.info(f'[WARNING] Action key {action_key} not found in data')
                continue
            state_key = self._infer_state_key_for_action(action_key)

            if state_key is None or state_key not in data:
                continue

            action_values = np.asarray(data[action_key])
            state_values = np.asarray(data[state_key])
            if action_values.ndim != 2 or state_values.ndim != 2:
                raise ValueError(
                    f"Expected 2D arrays for action/state, got {action_key}: {action_values.shape}, {state_key}: {state_values.shape}"
                )
            if action_values.shape[1] != state_values.shape[1]:
                raise ValueError(
                    f"Action/state dim mismatch for {action_key} vs {state_key}: {action_values.shape} vs {state_values.shape}"
                )

            state0 = state_values[0]
            if self._action_mode == "delta":
                out = action_values.copy()
                if len(out) > 1:
                    out[1:] = action_values[1:] - action_values[:-1]
                out[0] = action_values[0] - state0
            elif self._action_mode == "rel":
                out = action_values - state0
            else:
                out = action_values

            data[action_key] = out

        return data

    def _get_lerobot_modality_meta(self) -> LeRobotModalityMetadata:
        modality_meta_path = self.dataset_path / LE_ROBOT_MODALITY_FILENAME
        assert (
            modality_meta_path.exists()
        ), f"Please provide a {LE_ROBOT_MODALITY_FILENAME} file in {self.dataset_path}"
        with open(modality_meta_path, "r") as f:
            modality_meta = LeRobotModalityMetadata.model_validate(json.load(f))
        return modality_meta

    def _get_lerobot_info_meta(self) -> dict:
        info_meta_path = self.dataset_path / LE_ROBOT_INFO_FILENAME
        with open(info_meta_path, "r") as f:
            info_meta = json.load(f)
        return info_meta

    def _get_data_path_pattern(self) -> str:
        return self.lerobot_info_meta["data_path"]

    def _get_video_path_pattern(self) -> str:
        return self.lerobot_info_meta["video_path"]

    def _get_chunk_size(self) -> int:
        return self.lerobot_info_meta["chunks_size"]

    def _get_tasks(self) -> pd.DataFrame:
        if self._lerobot_version == "v2.0":
            tasks_path = self.dataset_path / LE_ROBOT_TASKS_FILENAME
            with open(tasks_path, "r") as f:
                tasks = [json.loads(line) for line in f]
            df = pd.DataFrame(tasks)
            return df.set_index("task_index")

        elif self._lerobot_version == "v3.0":
            tasks_path = self.dataset_path / LE_ROBOT3_TASKS_FILENAME
            df = pd.read_parquet(tasks_path)
            df = df.reset_index()
            df = df.rename(columns={'index': 'task'})
            df = df[['task_index', 'task']]
            return df
    def _check_integrity(self):
        ERROR_MSG_HEADER = f"Error occurred in initializing dataset {self.dataset_name}:\n"

        for modality_config in self.modality_configs.values():
            for key in modality_config.modality_keys:
                if key == "lapa_action" or key == "dream_actions":
                    continue

                try:
                    self.lerobot_modality_meta.get_key_meta(key)
                except Exception as e:
                    raise ValueError(
                        ERROR_MSG_HEADER + f"Unable to find key {key} in modality metadata:\n{e}"
                    )

    def set_transforms_metadata(self, metadata: DatasetMetadata):
        self.transforms.set_metadata(metadata)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.all_steps)

    def __str__(self) -> str:
        return f"{self.dataset_name} ({len(self)} steps)"

    def __getitem__(self, index: int) -> dict:
        trajectory_id, base_index = self.all_steps[index]
        raw_data = self.get_step_data(trajectory_id, base_index)
        data = self.transforms(raw_data)
        sample = self._pack_sample(data)
        return self._attach_semantic_sidecar(sample, trajectory_id, base_index)

    def _attach_semantic_sidecar(
        self, sample: dict, trajectory_id: int, base_index: int
    ) -> dict:
        if self._semantic_sidecar is not None:
            semantic = self._semantic_sidecar.get_frame(trajectory_id, base_index)
            image_keys = [key.removeprefix("video.") for key in self.modality_keys["video"]]
            if semantic["views"] != image_keys:
                raise ValueError(
                    f"Semantic/image view mismatch for {self.dataset_name} episode={trajectory_id} "
                    f"frame={base_index}: semantic={semantic['views']} images={len(sample['image'])}"
                )
            sample["semantic_factors"] = semantic["factors"]
            sample["semantic_quality"] = {
                "all_semantic_masks_invalid": semantic["all_semantic_masks_invalid"],
                "episode_index": semantic["episode_index"],
                "frame_index": semantic["frame_index"],
            }
        return sample

    def _pack_sample(self, data: dict) -> dict:
        step_images = []
        image_size = tuple(self.data_cfg["obs_image_size"])
        for video_key in self.modality_keys["video"]:
            image = data[video_key][0]
            image = Image.fromarray(image).resize(image_size)
            step_images.append(image)

        language = data[self.modality_keys["language"][0]][0]
        action = []
        for action_key in self.modality_keys["action"]:
            action.append(data[action_key])
        action = np.concatenate(action, axis=1).astype(np.float16)

        sample = {
            "action": action,
            "image": step_images,
            "lang": language,
            "robot_tag": self.tag
        }

        if self.data_cfg is not None and self.data_cfg.get("include_state", False) not in ["False", False]:
            state = []
            for state_key in self.modality_keys.get("state", []):
                state.append(data[state_key])
            if not state:
                import warnings
                warnings.warn(
                    "include_state=True but no state modality keys found in modality_configs "
                    "(state modality may be disabled in the dataset's DataConfig). "
                    "Skipping state packing.",
                    stacklevel=2,
                )
            else:
                state = np.concatenate(state, axis=1).astype(np.float16)
                sample["state"] = state

        return sample

    def get_step_data(self, trajectory_id: int, base_index: int) -> dict:
        data = {}

        self.curr_traj_data = self.get_trajectory_data(trajectory_id)

        for modality in self.modality_keys:

            for key in self.modality_keys[modality]:
                data[key] = self.get_data_by_modality(trajectory_id, modality, key, base_index)
        data = self._apply_action_mode(data)
        return data

    def get_trajectory_data(self, trajectory_id: int) -> pd.DataFrame:
        if self._lerobot_version == "v2.0":

            if self.curr_traj_id == trajectory_id and self.curr_traj_data is not None:
                return self.curr_traj_data
            else:
                chunk_index = self.get_episode_chunk(trajectory_id)
                parquet_path = self.dataset_path / self.data_path_pattern.format(
                    episode_chunk=chunk_index, episode_index=trajectory_id
                )
                assert parquet_path.exists(), f"Parquet file not found at {parquet_path}"
                return pd.read_parquet(parquet_path)
        elif self._lerobot_version == "v3.0":
            return self.get_trajectory_data_lerobot_v3(trajectory_id)

    def get_trajectory_data_lerobot_v3(self, trajectory_id: int) -> pd.DataFrame:
        if self.curr_traj_id == trajectory_id and self.curr_traj_data is not None:
            return self.curr_traj_data
        else:
            episode_meta = self.trajectory_ids_to_metadata[trajectory_id]
            chunk_index = episode_meta["data/chunk_index"]
            file_index = self.get_episode_file_index(trajectory_id)

            parquet_path = self.dataset_path / self.data_path_pattern.format(
                chunk_index=chunk_index, file_index=file_index
            )
            assert parquet_path.exists(), f"Parquet file not found at {parquet_path}"
            file_data = pd.read_parquet(parquet_path)

            episode_data = file_data.loc[file_data["episode_index"] == trajectory_id].copy()
            return episode_data

    def get_trajectory_index(self, trajectory_id: int) -> int:
        trajectory_indices = np.where(self.trajectory_ids == trajectory_id)[0]
        if len(trajectory_indices) != 1:
            raise ValueError(
                f"Error finding trajectory index for {trajectory_id}, found {trajectory_indices=}"
            )
        return trajectory_indices[0]

    def get_episode_chunk(self, ep_index: int) -> int:
        return ep_index // self.chunk_size
    def get_episode_file_index(self, ep_index: int) -> int:
        episode_meta = self.trajectory_ids_to_metadata[ep_index]
        return episode_meta["data/file_index"]

    def get_episode_file_from_index(self, ep_index: int) -> int:
        episode_meta = self.trajectory_ids_to_metadata[ep_index]
        return episode_meta["data/file_from_index"]

    def retrieve_data_and_pad(
        self,
        array: np.ndarray,
        step_indices: np.ndarray,
        max_length: int,
        padding_strategy: str = "first_last",
    ) -> np.ndarray:
        front_padding_indices = step_indices < 0
        end_padding_indices = step_indices >= max_length
        padding_positions = np.logical_or(front_padding_indices, end_padding_indices)

        raw_data = array[step_indices[~padding_positions]]
        assert isinstance(raw_data, np.ndarray), f"{type(raw_data)=}"

        if raw_data.ndim == 1:
            expected_shape = (len(step_indices),)
        else:
            expected_shape = (len(step_indices), *array.shape[1:])

        output = np.zeros(expected_shape)

        output[~padding_positions] = raw_data

        if padding_positions.any():
            if padding_strategy == "first_last":

                front_padding_data = array[0]
                end_padding_data = array[-1]
                output[front_padding_indices] = front_padding_data
                output[end_padding_indices] = end_padding_data
            elif padding_strategy == "zero":

                output[padding_positions] = 0
            else:
                raise ValueError(f"Invalid padding strategy: {padding_strategy}")
        return output

    def get_video_path(self, trajectory_id: int, key: str) -> Path:
        chunk_index = self.get_episode_chunk(trajectory_id)
        original_key = self.lerobot_modality_meta.video[key].original_key
        if original_key is None:
            original_key = key
        if self._lerobot_version == "v2.0":
            video_filename = self.video_path_pattern.format(
                episode_chunk=chunk_index, episode_index=trajectory_id, video_key=original_key
            )
        elif self._lerobot_version == "v3.0":
            episode_meta = self.trajectory_ids_to_metadata[trajectory_id]

            video_file_indices = episode_meta.get("videos/file_indices", {})

            if original_key in video_file_indices:
                video_chunk_index = video_file_indices[original_key]["chunk_index"]
                video_file_index = video_file_indices[original_key]["file_index"]
            else:
                video_chunk_index = episode_meta["data/chunk_index"]
                video_file_index = episode_meta["data/file_index"]
            video_filename = self.video_path_pattern.format(
                video_key=original_key,
                chunk_index=episode_meta["data/chunk_index"],
                file_index=episode_meta["data/file_index"],
            )
        return self.dataset_path / video_filename

    def get_video(
        self,
        trajectory_id: int,
        key: str,
        base_index: int,
    ) -> np.ndarray:
        step_indices = self.delta_indices[key] + base_index

        trajectory_index = self.get_trajectory_index(trajectory_id)

        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, self.trajectory_lengths[trajectory_index] - 1)
        assert key.startswith("video."), f"Video key must start with 'video.', got {key}"

        key = key.replace("video.", "")

        original_key = self.lerobot_modality_meta.video[key].original_key
        if original_key is None:
            original_key = key
        if self.curr_traj_data is not None and original_key in self.curr_traj_data.columns:
            image_entries = self.curr_traj_data[original_key].tolist()

            def _decode_image_entry(entry):
                if isinstance(entry, np.ndarray):
                    return entry
                if isinstance(entry, Image.Image):
                    return np.array(entry)
                if isinstance(entry, dict):
                    img_bytes = entry.get("bytes", None)
                    img_path = entry.get("path", None)

                    if img_bytes is not None:
                        return np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))

                    if img_path is not None:
                        path_obj = Path(img_path)
                        if not path_obj.is_absolute():
                            path_obj = self.dataset_path / path_obj
                        return np.array(Image.open(path_obj).convert("RGB"))

                raise TypeError(f"Unsupported image entry type: {type(entry)}")

            frames = []
            for idx in step_indices:
                safe_idx = int(min(max(idx, 0), len(image_entries) - 1))
                frames.append(_decode_image_entry(image_entries[safe_idx]))

            return np.stack(frames)

        video_path = self.get_video_path(trajectory_id, key)

        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        assert "timestamp" in self.curr_traj_data.columns, f"No timestamp found in {trajectory_id=}"
        timestamp: np.ndarray = self.curr_traj_data["timestamp"].to_numpy()

        video_timestamp = timestamp[step_indices]
        if self._lerobot_version == "v3.0":
            episode_meta = self.trajectory_ids_to_metadata.get(trajectory_id, {})
            from_timestamps = episode_meta.get("videos/from_timestamps", {})
            original_video_key = self.lerobot_modality_meta.video[key].original_key
            if original_video_key is None:
                original_video_key = key
            from_timestamp = float(from_timestamps.get(original_video_key, 0.0))
            video_timestamp = video_timestamp + from_timestamp

        return get_frames_by_timestamps(
            video_path.as_posix(),
            video_timestamp,
            video_backend=self.video_backend,
            video_backend_kwargs=self.video_backend_kwargs,
        )

    def get_state_or_action(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        base_index: int,
    ) -> np.ndarray:
        step_indices = self.delta_indices[key] + base_index

        trajectory_index = self.get_trajectory_index(trajectory_id)

        max_length = self.trajectory_lengths[trajectory_index]
        assert key.startswith(modality + "."), f"{key} must start with {modality + '.'}, got {key}"

        key = key.replace(modality + ".", "")

        le_state_or_action_cfg = getattr(self.lerobot_modality_meta, modality)
        le_key = le_state_or_action_cfg[key].original_key
        if le_key is None:
            le_key = key

        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        assert le_key in self.curr_traj_data.columns, f"No {le_key} found in {trajectory_id=}"
        data_array: np.ndarray = np.stack(self.curr_traj_data[le_key])  # type: ignore
        assert data_array.ndim == 2, f"Expected 2D array, got key {le_key} is{data_array.shape} array"
        le_indices = np.arange(
            le_state_or_action_cfg[key].start,
            le_state_or_action_cfg[key].end,
        )
        data_array = data_array[:, le_indices]

        state_or_action_cfg = getattr(self.metadata.modalities, modality)[key]

        return self.retrieve_data_and_pad(
            array=data_array,
            step_indices=step_indices,
            max_length=max_length,
            padding_strategy="first_last" if state_or_action_cfg.absolute else "zero",

        )

    def get_language(
        self,
        trajectory_id: int,
        key: str,
        base_index: int,
    ) -> list[str]:
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"

        step_indices = self.delta_indices[key] + base_index

        trajectory_index = self.get_trajectory_index(trajectory_id)

        max_length = self.trajectory_lengths[trajectory_index]

        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, max_length - 1)

        task_indices: list[int] = []
        assert key.startswith(
            "annotation."
        ), f"Language key must start with 'annotation.', got {key}"
        subkey = key.replace("annotation.", "")
        annotation_meta = self.lerobot_modality_meta.annotation
        assert annotation_meta is not None, f"Annotation metadata is None for {subkey}"
        assert (
            subkey in annotation_meta
        ), f"Annotation key {subkey} not found in metadata, available annotation keys: {annotation_meta.keys()}"
        subkey_meta = annotation_meta[subkey]
        original_key = subkey_meta.original_key
        if original_key is None:
            original_key = key
        for i in range(len(step_indices)):

            value = self.curr_traj_data[original_key].iloc[step_indices[i]]
            task_indices.append(value if isinstance(value, (int, float)) else value.item())

        return self.tasks.loc[task_indices]["task"].tolist()

    def get_data_by_modality(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        base_index: int,
    ):
        if modality == "video":
            return self.get_video(trajectory_id, key, base_index)
        elif modality == "state" or modality == "action":
            return self.get_state_or_action(trajectory_id, modality, key, base_index)
        elif modality == "language":
            return self.get_language(trajectory_id, key, base_index)
        else:
            raise ValueError(f"Invalid modality: {modality}")

    def _save_dataset_statistics_(self, save_path: Path | str, format: str = "json") -> None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        statistics_data = {}

        used_action_keys, used_state_keys = get_used_modality_keys(self.modality_keys)

        tag = self.tag
        tag_stats = {}

        if hasattr(self.metadata.statistics, 'action') and self.metadata.statistics.action:
            action_stats = self.metadata.statistics.action

            non_gripper_keys = []
            gripper_keys = []

            for key in action_stats.keys():
                if key in used_action_keys:
                    if "gripper" in key.lower():
                        gripper_keys.append(key)
                    else:
                        non_gripper_keys.append(key)

            reordered_keys = non_gripper_keys + gripper_keys

            filtered_action_stats = {}
            for key in reordered_keys:
                filtered_action_stats[key] = action_stats[key]

            if filtered_action_stats:

                combined_action_stats = combine_modality_stats(filtered_action_stats)

                _action_norm_modes = _extract_action_normalization_modes(self.transforms)
                mask = generate_action_mask_for_used_keys(
                    self.metadata.modalities.action, filtered_action_stats.keys(),
                    normalization_modes=_action_norm_modes,
                )
                combined_action_stats["mask"] = mask

                tag_stats["action"] = combined_action_stats

        if hasattr(self.metadata.statistics, 'state') and self.metadata.statistics.state:
            state_stats = self.metadata.statistics.state

            non_gripper_keys = []
            gripper_keys = []

            for key in state_stats.keys():
                if key in used_state_keys:
                    if "gripper" in key.lower():
                        gripper_keys.append(key)
                    else:
                        non_gripper_keys.append(key)

            reordered_keys = non_gripper_keys + gripper_keys

            filtered_state_stats = {}
            for key in reordered_keys:
                filtered_state_stats[key] = state_stats[key]

            if filtered_state_stats:
                combined_state_stats = combine_modality_stats(filtered_state_stats)
                tag_stats["state"] = combined_state_stats

        tag_stats["num_transitions"] = len(self)
        tag_stats["num_trajectories"] = len(self.trajectory_ids)

        statistics_data[tag] = tag_stats

        if format.lower() == "json":
            if not str(save_path).endswith('.json'):
                save_path = save_path.with_suffix('.json')
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(statistics_data, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}. Currently only 'json' is supported.")

        logger.info(f'Single dataset statistics saved to: {save_path}')


class CachedLeRobotSingleDataset(LeRobotSingleDataset):
    def __init__(self, img_resize: tuple[int, int] | None = None, *args, **kwargs):
        if img_resize is not None and not isinstance(img_resize, tuple):
            img_resize = tuple(img_resize)
            assert len(img_resize) == 2, f"Expected tuple of length 2, got {img_resize}"
        self.img_resize = img_resize

        super().__init__(*args, **kwargs)
        cached_frames: dict[str, np.ndarray] = {}

        for key in self.modality_keys["video"]:
            all_frames = []
            original_key = key
            key = key.replace("video.", "")
            for trajectory_id, trajectory_length in tqdm(
                zip(self.trajectory_ids, self.trajectory_lengths),
                total=len(self.trajectory_ids),
                desc=f"Caching {key} frames",
            ):
                video_path = self.get_video_path(trajectory_id, key)
                frames = get_all_frames(
                    video_path.as_posix(),
                    video_backend=self.video_backend,
                    video_backend_kwargs=self.video_backend_kwargs,
                    resize_size=img_resize,
                )
                assert frames.ndim == 4, f"Expected 4D array, got {frames.shape} array"
                assert frames.shape[3] == 3, f"Expected 3 channels, got {frames.shape[3]} channels"

                all_frames.append(frames)
            cached_frames[key] = np.concatenate(all_frames, axis=0)
        self.cached_frames = cached_frames
        self.start_indices = np.cumsum(self.trajectory_lengths) - self.trajectory_lengths

    def get_video(self, trajectory_id: int, key: str, base_index: int) -> np.ndarray:
        step_indices = self.delta_indices[key] + base_index

        trajectory_index = self.get_trajectory_index(trajectory_id)

        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, self.trajectory_lengths[trajectory_index] - 1)
        assert key.startswith("video."), f"Video key must start with 'video.', got {key}"

        key = key.replace("video.", "")

        absolute_indices = self.start_indices[trajectory_index] + step_indices
        return self.cached_frames[key][absolute_indices]

    def get_step_data(self, trajectory_id: int, base_index: int) -> dict:
        data = {}
        self.curr_traj_data = self.get_trajectory_data(trajectory_id)

        for modality in self.modality_keys:

            for key in self.modality_keys[modality]:
                data[key] = self.get_data_by_modality(trajectory_id, modality, key, base_index)
        return data

    def set_transforms_metadata(self, metadata: DatasetMetadata):
        if self.img_resize is not None:
            all_video_keys = [key for key in self.modality_keys["video"]]
            for key in metadata.modalities.video:
                if key in all_video_keys:
                    metadata.modalities.video[key].resolution = self.img_resize
        super().set_transforms_metadata(metadata)


def safe_hash(input_tuple):
    tuple_string = repr(input_tuple).encode("utf-8")
    sha256 = hashlib.sha256()
    sha256.update(tuple_string)

    seed = int(sha256.hexdigest(), 16)

    return seed & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF


class MixtureSpecElement(BaseModel):
    dataset_path: list[Path] | Path = Field(..., description="The path to the dataset.")
    dataset_weight: float = Field(..., description="The weight of the dataset in the mixture.")
    distribute_weights: bool = Field(
        default=False,
        description="Whether to distribute the weights of the dataset across all the paths. If True, the weights will be evenly distributed across all the paths.",
    )


def combine_modality_stats(modality_stats: dict) -> dict:
    combined_stats = {
        "mean": [],
        "std": [],
        "max": [],
        "min": [],
        "q01": [],
        "q99": []
    }

    for subkey in modality_stats.keys():
        subkey_stats = modality_stats[subkey]

        for stat_name in ["mean", "std", "max", "min", "q01", "q99"]:
            stat_value = getattr(subkey_stats, stat_name)
            if isinstance(stat_value, (list, tuple)):
                combined_stats[stat_name].extend(stat_value)
            else:

                if hasattr(stat_value, 'tolist'):
                    combined_stats[stat_name].extend(stat_value.tolist())
                else:
                    combined_stats[stat_name].append(float(stat_value))

    return combined_stats

def _extract_action_normalization_modes(transforms) -> dict:
    modes = {}
    for t in transforms.transforms:
        if isinstance(t, StateActionTransform):
            for key, mode in t.normalization_modes.items():
                if key.startswith("action."):
                    subkey = key[len("action."):]
                    modes[subkey] = mode
    return modes


def generate_action_mask_for_used_keys(
    action_modalities: dict,
    used_action_keys_ordered,
    normalization_modes: dict | None = None,
) -> list[bool]:
    mask = []

    for subkey in used_action_keys_ordered:
        if subkey in action_modalities:
            subkey_config = action_modalities[subkey]

            if hasattr(subkey_config, 'shape') and len(subkey_config.shape) > 0:
                dim_count = subkey_config.shape[0]
            else:
                dim_count = 1

            is_binary = (
                normalization_modes.get(subkey) == "binary"
                if normalization_modes is not None
                else False
            )

            for _ in range(dim_count):
                mask.append(not is_binary)

    return mask

def get_used_modality_keys(modality_keys: dict) -> tuple[list, list]:
    used_action_keys = []
    used_state_keys = []

    for action_key in modality_keys.get("action", []):
        if action_key.startswith("action."):
            clean_key = action_key.replace("action.", "")
            used_action_keys.append(clean_key)

    for state_key in modality_keys.get("state", []):
        if state_key.startswith("state."):
            clean_key = state_key.replace("state.", "")
            used_state_keys.append(clean_key)

    return used_action_keys, used_state_keys

class LeRobotMixtureDataset(Dataset):
    def __init__(
        self,
        data_mixture: Sequence[tuple[LeRobotSingleDataset, float]],
        mode: str,
        balance_dataset_weights: bool = True,
        balance_trajectory_weights: bool = True,
        seed: int = 42,
        metadata_config: dict = {
            "percentile_mixing_method": "min_max",
        },
        **kwargs,
    ):
        datasets: list[LeRobotSingleDataset] = []
        dataset_sampling_weights: list[float] = []
        for dataset, weight in data_mixture:

            if len(dataset) == 0:
                logger.warning(f'Skipping empty dataset {dataset.dataset_name}')
                continue
            datasets.append(dataset)
            dataset_sampling_weights.append(weight)

        if len(datasets) == 0:
            raise ValueError("No valid datasets found in the mixture. All datasets are empty.")

        self.datasets = datasets
        self.balance_dataset_weights = balance_dataset_weights
        self.balance_trajectory_weights = balance_trajectory_weights
        self.seed = seed
        self.mode = mode
        self.data_cfg = kwargs["data_cfg"] if "data_cfg" in kwargs else None

        self._dataset_lengths = np.array([len(dataset) for dataset in self.datasets])
        self._getitem_count = 0

        self._dataset_sampling_weights = np.array(dataset_sampling_weights)

        if self.balance_dataset_weights:
            self._dataset_sampling_weights *= self._dataset_lengths

        if np.any(self._dataset_sampling_weights <= 0):
            logger.warning(f'Found zero or negative sampling weights: {self._dataset_sampling_weights}')

            self._dataset_sampling_weights = np.maximum(self._dataset_sampling_weights, 1e-8)

        weights_sum = self._dataset_sampling_weights.sum()
        if weights_sum == 0 or np.isnan(weights_sum):
            logger.warning("Invalid dataset weight sum %s; using equal weights", weights_sum)

            self._dataset_sampling_weights = np.ones(len(self.datasets)) / len(self.datasets)
        else:
            self._dataset_sampling_weights /= weights_sum

        self._trajectory_sampling_weights: list[np.ndarray] = []
        for i, dataset in enumerate(self.datasets):
            trajectory_sampling_weights = np.ones(len(dataset.trajectory_lengths))
            if self.balance_trajectory_weights:
                trajectory_sampling_weights *= dataset.trajectory_lengths

            if np.any(trajectory_sampling_weights <= 0):
                logger.warning(f'Dataset {i} has zero or negative trajectory weights')
                trajectory_sampling_weights = np.maximum(trajectory_sampling_weights, 1e-8)

            weights_sum = trajectory_sampling_weights.sum()
            if weights_sum == 0 or np.isnan(weights_sum):
                logger.warning(f'Dataset {i} has invalid trajectory weights sum: {weights_sum}')

                trajectory_sampling_weights = np.ones(len(dataset.trajectory_lengths)) / len(dataset.trajectory_lengths)
            else:
                trajectory_sampling_weights /= weights_sum

            self._trajectory_sampling_weights.append(trajectory_sampling_weights)

        self._primary_dataset_indices = np.array(dataset_sampling_weights) == 1.0
        if not np.any(self._primary_dataset_indices):

            max_weight = max(dataset_sampling_weights)
            self._primary_dataset_indices = np.array(dataset_sampling_weights) == max_weight

        if not np.any(self._primary_dataset_indices):

            logger.warning('Still no primary dataset found. Using first dataset as primary.')
            self._primary_dataset_indices = np.zeros(len(self.datasets), dtype=bool)
            self._primary_dataset_indices[0] = True

        self._epoch = torch.zeros((), dtype=torch.int64).share_memory_()

        self._fixed_samples = None
        fixed_subset_size = int(
            self.data_cfg.get("fixed_sample_subset_size", 0) if self.data_cfg is not None else 0
        )
        if fixed_subset_size < 0:
            raise ValueError("fixed_sample_subset_size must be non-negative")
        if fixed_subset_size:
            self._fixed_samples = [self.sample_step(index) for index in range(fixed_subset_size)]
            logger.info(f'Using fixed training subset with {fixed_subset_size} sampled steps')

        self.update_metadata(metadata_config)

    @property
    def dataset_lengths(self) -> np.ndarray:
        return self._dataset_lengths

    @property
    def dataset_sampling_weights(self) -> np.ndarray:
        return self._dataset_sampling_weights

    @property
    def trajectory_sampling_weights(self) -> list[np.ndarray]:
        return self._trajectory_sampling_weights

    @property
    def primary_dataset_indices(self) -> np.ndarray:
        return self._primary_dataset_indices

    def __str__(self) -> str:
        dataset_descriptions = []
        for dataset, weight in zip(self.datasets, self.dataset_sampling_weights):
            dataset_description = {
                "Dataset": str(dataset),
                "Sampling weight": float(weight),
            }
            dataset_descriptions.append(dataset_description)
        return json.dumps({"Mixture dataset": dataset_descriptions}, indent=2)

    @property
    def epoch(self):
        return self._epoch.item()

    def set_epoch(self, epoch: int):
        self._epoch.fill_(epoch)

    def sample_step(self, index: int) -> tuple[LeRobotSingleDataset, int, int]:
        if self._fixed_samples is not None:
            return self._fixed_samples[index % len(self._fixed_samples)]

        seed = index if self.mode != "train" else safe_hash((self.epoch, index, self.seed))
        rng = np.random.default_rng(seed)

        dataset_index = rng.choice(len(self.datasets), p=self.dataset_sampling_weights)
        dataset = self.datasets[dataset_index]

        trajectory_index = rng.choice(
            len(dataset.trajectory_ids), p=self.trajectory_sampling_weights[dataset_index]
        )
        trajectory_id = dataset.trajectory_ids[trajectory_index]

        base_index = rng.choice(dataset.trajectory_lengths[trajectory_index])
        return dataset, trajectory_id, base_index

    def __getitem__(self, index: int) -> dict:
        self._getitem_count += 1
        if self._getitem_count % 1000 == 0:
            gc.collect()

        max_retries = 10
        last_exception = None

        for attempt in range(max_retries):
            try:
                sample_tries = 0
                max_sample_tries = 200
                while True:
                    sample_tries += 1
                    if sample_tries > max_sample_tries:
                        raise RuntimeError(
                            f"Unable to sample a valid item after {max_sample_tries} attempts. "
                            f"dataset={self.datasets[0].dataset_name if len(self.datasets)>0 else 'unknown'}"
                        )

                    dataset, trajectory_id, step = self.sample_step(index)

                    total_videos = int(dataset.lerobot_info_meta.get("total_videos", 0))
                    if total_videos == 0:
                        break

                    key = dataset.modality_keys["video"][0].replace("video.", "")
                    video_path = dataset.get_video_path(trajectory_id, key)
                    if os.path.exists(video_path):
                        break
                    index = random.randint(0, len(self) - 1)

                raw_data = dataset.get_step_data(trajectory_id, step)
                data = dataset.transforms(raw_data)
                sample = dataset._pack_sample(data)
                return dataset._attach_semantic_sidecar(sample, trajectory_id, step)

            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:

                    logger.warning(f'Attempt {attempt + 1}/{max_retries} failed for index {index}: {e}')

                    index = random.randint(0, len(self) - 1)
                else:


                    raise last_exception

    def __len__(self) -> int:
        if self._fixed_samples is not None:
            return len(self._fixed_samples)

        if len(self.datasets) == 0:
            return 0

        if np.any(self.dataset_lengths == 0) or np.any(np.isnan(self.dataset_lengths)):
            logger.warning(f'Found zero or NaN dataset lengths: {self.dataset_lengths}')

            valid_indices = (self.dataset_lengths > 0) & (~np.isnan(self.dataset_lengths))
            if not np.any(valid_indices):
                logger.warning('All datasets have zero or NaN length')
                return 0
        else:
            valid_indices = np.ones(len(self.datasets), dtype=bool)

        if np.any(self.dataset_sampling_weights == 0) or np.any(np.isnan(self.dataset_sampling_weights)):
            logger.warning(f'Found zero or NaN sampling weights: {self.dataset_sampling_weights}')

            valid_weights = (self.dataset_sampling_weights > 0) & (~np.isnan(self.dataset_sampling_weights))
            valid_indices = valid_indices & valid_weights
            if not np.any(valid_indices):
                logger.warning('All sampling weights are zero or NaN')
                return 0

        primary_and_valid = self.primary_dataset_indices & valid_indices
        if not np.any(primary_and_valid):
            logger.warning(f'No valid primary datasets found. Primary indices: {self.primary_dataset_indices}, Valid indices: {valid_indices}')

            if np.any(valid_indices):
                max_length = self.dataset_lengths[valid_indices].max()
                logger.info(f'Fallback: Using maximum dataset length: {max_length}')
                return int(max_length)
            else:
                return 0

        ratios = (self.dataset_lengths / self.dataset_sampling_weights)[primary_and_valid]

        if np.any(np.isnan(ratios)) or np.any(np.isinf(ratios)):
            logger.warning(f'Found NaN or inf in ratios: {ratios}')
            logger.info(f'Dataset lengths: {self.dataset_lengths[primary_and_valid]}')
            logger.info(f'Sampling weights: {self.dataset_sampling_weights[primary_and_valid]}')

            valid_ratios = ratios[~np.isnan(ratios) & ~np.isinf(ratios)]
            if len(valid_ratios) == 0:
                logger.warning('All ratios are NaN or inf')
                return 0
            max_ratio = valid_ratios.max()
        else:
            max_ratio = ratios.max()

        result = int(max_ratio)
        if result == 0:
            logger.warning(f'Dataset mixture length is 0')
        return result

    @staticmethod
    def compute_overall_statistics(
        per_task_stats: list[dict[str, dict[str, list[float] | np.ndarray]]],
        dataset_sampling_weights: list[float] | np.ndarray,
        percentile_mixing_method: str = "weighted_average",
    ) -> dict[str, dict[str, list[float]]]:
        dataset_sampling_weights = np.array(dataset_sampling_weights)
        normalized_weights = dataset_sampling_weights / dataset_sampling_weights.sum()

        overall_stats: dict[str, dict[str, list[float]]] = {}

        modality_keys = per_task_stats[0].keys()

        for modality in modality_keys:

            num_dims = len(per_task_stats[0][modality]["mean"])

            weighted_means = np.zeros(num_dims)
            weighted_squares = np.zeros(num_dims)

            min_list = []
            max_list = []
            q01_list = []
            q99_list = []

            for task_idx, task_stats in enumerate(per_task_stats):
                w_i = normalized_weights[task_idx]
                stats = task_stats[modality]
                means = np.array(stats["mean"])
                stds = np.array(stats["std"])

                weighted_means += w_i * means
                weighted_squares += w_i * (stds**2 + means**2)

                min_list.append(stats["min"])
                max_list.append(stats["max"])
                q01_list.append(stats["q01"])
                q99_list.append(stats["q99"])

            overall_mean = weighted_means.tolist()

            overall_variance = weighted_squares - weighted_means**2
            overall_std = np.sqrt(overall_variance).tolist()

            overall_min = np.min(np.array(min_list), axis=0).tolist()
            overall_max = np.max(np.array(max_list), axis=0).tolist()

            q01_array = np.array(q01_list)
            q99_array = np.array(q99_list)
            if percentile_mixing_method == "weighted_average":
                weighted_q01 = np.average(q01_array, axis=0, weights=normalized_weights).tolist()
                weighted_q99 = np.average(q99_array, axis=0, weights=normalized_weights).tolist()

            elif percentile_mixing_method == "min_max":
                weighted_q01 = np.min(q01_array, axis=0).tolist()
                weighted_q99 = np.max(q99_array, axis=0).tolist()
            else:
                raise ValueError(f"Invalid percentile mixing method: {percentile_mixing_method}")

            overall_stats[modality] = {
                "min": overall_min,
                "max": overall_max,
                "mean": overall_mean,
                "std": overall_std,
                "q01": weighted_q01,
                "q99": weighted_q99,
            }

        return overall_stats

    @staticmethod
    def merge_metadata(
        metadatas: list[DatasetMetadata],
        dataset_sampling_weights: list[float],
        percentile_mixing_method: str,
    ) -> DatasetMetadata:
        metadata_dicts = [metadata.model_dump(mode="json") for metadata in metadatas]

        merged_metadata = {}

        assert all(
            metadata.embodiment_tag == metadatas[0].embodiment_tag for metadata in metadatas
        ), "All metadata must have the same embodiment tag"
        merged_metadata["embodiment_tag"] = metadatas[0].embodiment_tag

        dataset_statistics = {}
        dataset_statistics["state"] = LeRobotMixtureDataset.compute_overall_statistics(
            per_task_stats=[m["statistics"]["state"] for m in metadata_dicts],
            dataset_sampling_weights=dataset_sampling_weights,
            percentile_mixing_method=percentile_mixing_method,
        )
        dataset_statistics["action"] = LeRobotMixtureDataset.compute_overall_statistics(
            per_task_stats=[m["statistics"]["action"] for m in metadata_dicts],
            dataset_sampling_weights=dataset_sampling_weights,
            percentile_mixing_method=percentile_mixing_method,
        )
        merged_metadata["statistics"] = dataset_statistics

        modality_configs = defaultdict(set)
        for metadata in metadata_dicts:
            for modality, configs in metadata["modalities"].items():
                modality_configs[modality].add(json.dumps(configs))
        merged_metadata["modalities"] = {}
        for modality, configs in modality_configs.items():

            assert (
                len(configs) == 1
            ), f"Multiple modality configs for modality {modality}: {list(configs)}"
            merged_metadata["modalities"][modality] = json.loads(configs.pop())

        return DatasetMetadata.model_validate(merged_metadata)

    def update_metadata(self, metadata_config: dict, cached_statistics_path: Path | str | None = None) -> None:
        if cached_statistics_path is not None:
            try:
                cached_stats = self.load_merged_statistics(cached_statistics_path)
                self.apply_cached_statistics(cached_stats)
                return
            except (FileNotFoundError, KeyError, ValidationError) as e:
                logger.warning(f'Failed to load cached statistics: {e}')

        self.tag = EmbodimentTag.NEW_EMBODIMENT.value
        self.merged_metadata: dict[str, DatasetMetadata] = {}

        all_metadatas: dict[str, list[DatasetMetadata]] = {}
        for dataset in self.datasets:
            if dataset.tag not in all_metadatas:
                all_metadatas[dataset.tag] = []
            all_metadatas[dataset.tag].append(dataset.metadata)
        for tag, metadatas in all_metadatas.items():
            self.merged_metadata[tag] = self.merge_metadata(
                metadatas=metadatas,
                dataset_sampling_weights=self.dataset_sampling_weights.tolist(),
                percentile_mixing_method=metadata_config["percentile_mixing_method"],
            )
        for dataset in self.datasets:
            dataset.set_transforms_metadata(self.merged_metadata[dataset.tag])

    def save_dataset_statistics(self, save_path: Path | str, format: str = "json") -> None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        statistics_data = {}

        all_used_action_keys = []
        all_used_state_keys = []

        for dataset in self.datasets:
            used_action_keys, used_state_keys = get_used_modality_keys(dataset.modality_keys)
            for used_action_key in used_action_keys:
                if used_action_key not in all_used_action_keys:
                    all_used_action_keys.append(used_action_key)
            for used_state_key in used_state_keys:
                if used_state_key not in all_used_state_keys:
                    all_used_state_keys.append(used_state_key)

        for tag, merged_metadata in self.merged_metadata.items():
            tag_stats = {}

            if hasattr(merged_metadata.statistics, 'action') and merged_metadata.statistics.action:
                action_stats = merged_metadata.statistics.action

                non_gripper_keys = []
                gripper_keys = []

                for key in all_used_action_keys:
                    if key in action_stats:
                        non_gripper_keys.append(key)

                reordered_keys = non_gripper_keys + gripper_keys

                filtered_action_stats = {}
                for key in reordered_keys:
                    filtered_action_stats[key] = action_stats[key]

                if filtered_action_stats:
                    combined_action_stats = combine_modality_stats(filtered_action_stats)

                    _action_norm_modes: dict = {}
                    for _ds in self.datasets:
                        if _ds.tag == tag:
                            for _k, _m in _extract_action_normalization_modes(_ds.transforms).items():
                                if _k not in _action_norm_modes or _m == "binary":
                                    _action_norm_modes[_k] = _m
                    mask = generate_action_mask_for_used_keys(
                        merged_metadata.modalities.action, filtered_action_stats.keys(),
                        normalization_modes=_action_norm_modes,
                    )
                    combined_action_stats["mask"] = mask
                    combined_action_stats["action_keys"] = list(filtered_action_stats)
                    combined_action_stats["action_dims"] = [
                        len(stat.min) for stat in filtered_action_stats.values()
                    ]
                    combined_action_stats["normalization_modes"] = [
                        _action_norm_modes.get(key, "none")
                        for key, stat in filtered_action_stats.items()
                        for _ in range(len(stat.min))
                    ]
                    binary_thresholds = {
                        key.removeprefix("action."): transform.binary_threshold
                        for dataset in self.datasets if dataset.tag == tag
                        for transform in dataset.transforms.transforms
                        if isinstance(transform, StateActionTransform)
                        for key in transform.normalization_modes
                    }
                    combined_action_stats["binary_thresholds"] = [
                        binary_thresholds.get(key, 0.5)
                        for key, stat in filtered_action_stats.items()
                        for _ in range(len(stat.min))
                    ]

                    tag_stats["action"] = combined_action_stats

            if hasattr(merged_metadata.statistics, 'state') and merged_metadata.statistics.state:
                state_stats = merged_metadata.statistics.state

                non_gripper_keys = []
                gripper_keys = []

                for key in all_used_state_keys:
                    if key in state_stats:
                        non_gripper_keys.append(key)

                reordered_keys = non_gripper_keys + gripper_keys

                filtered_state_stats = {}
                for key in reordered_keys:
                    filtered_state_stats[key] = state_stats[key]

                if filtered_state_stats:
                    combined_state_stats = combine_modality_stats(filtered_state_stats)
                    tag_stats["state"] = combined_state_stats

            tag_stats.update(self._get_dataset_counts(tag))

            statistics_data[tag] = tag_stats

        if format.lower() == "json":
            if not str(save_path).endswith('.json'):
                save_path = save_path.with_suffix('.json')
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(statistics_data, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}. Currently only 'json' is supported.")

        logger.info(f'Merged dataset statistics saved to: {save_path}')

    def _combine_modality_stats(self, modality_stats: dict) -> dict:
        return combine_modality_stats(modality_stats)

    def _generate_action_mask_for_used_keys(self, action_modalities: dict, used_action_keys_ordered) -> list[bool]:
        return generate_action_mask_for_used_keys(action_modalities, used_action_keys_ordered)

    def _get_dataset_counts(self, tag: str) -> dict:
        num_transitions = 0
        num_trajectories = 0

        for dataset in self.datasets:
            if dataset.tag == tag:
                num_transitions += len(dataset)
                num_trajectories += len(dataset.trajectory_ids)

        return {
            "num_transitions": num_transitions,
            "num_trajectories": num_trajectories
        }

    @classmethod
    def load_merged_statistics(cls, load_path: Path | str) -> dict:
        load_path = Path(load_path)
        if not load_path.exists():
            raise FileNotFoundError(f"Statistics file not found: {load_path}")

        if load_path.suffix.lower() == '.json':
            with open(load_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif load_path.suffix.lower() == '.pkl':
            import pickle
            with open(load_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {load_path.suffix}")

    def apply_cached_statistics(self, cached_statistics: dict) -> None:
        if "metadata" in cached_statistics:
            cached_dataset_names = set(cached_statistics["metadata"]["dataset_names"])
            current_dataset_names = set(dataset.dataset_name for dataset in self.datasets)

            if cached_dataset_names != current_dataset_names:
                logger.warning(
                    "Statistics dataset mismatch: cached=%s current=%s",
                    cached_dataset_names, current_dataset_names,
                )
                return

        self.merged_metadata = {}
        for tag, stats_data in cached_statistics.items():
            if tag == "metadata":
                continue

            metadata_dict = {
                "embodiment_tag": tag,
                "statistics": {
                    "action": {},
                    "state": {}
                },
                "modalities": {}
            }

            if "action" in stats_data:
                action_data = stats_data["action"]

                metadata_dict["statistics"]["action"] = action_data

            if "state" in stats_data:
                state_data = stats_data["state"]
                metadata_dict["statistics"]["state"] = state_data

            self.merged_metadata[tag] = DatasetMetadata.model_validate(metadata_dict)

        for dataset in self.datasets:
            if dataset.tag in self.merged_metadata:
                dataset.set_transforms_metadata(self.merged_metadata[dataset.tag])

        logger.info(f'Applied cached statistics for {len(self.merged_metadata)} embodiment tags.')
