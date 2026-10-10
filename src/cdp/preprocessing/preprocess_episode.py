#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from cdp.data.config import read_episode_instruction

from cdp.perception.instruction_parser import parse_instruction
from cdp.preprocessing.config import add_config_argument, load_preprocessing_config
from cdp.perception.grounding import build_runtime, ground_images
from cdp.preprocessing.outputs import is_complete


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_config_argument(parser)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return load_preprocessing_config(parser.parse_args())


def read_episode_record(dataset_path: Path, episode_index: int) -> dict:
    with (dataset_path / "meta/episodes.jsonl").open() as handle:
        for line in handle:
            record = json.loads(line)
            if int(record["episode_index"]) == episode_index:
                return record
    raise KeyError(f"Episode {episode_index} not found in {dataset_path}")


def process_view(args, runtime, factors, video_path, view_index,
                 masks_store, mask_valid, dino_boxes, dino_scores, sam_scores, mask_status,
                 expected_frames):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open {video_path}")
    try:
        frame_cursor = 0
        while frame_cursor < expected_frames:
            images = []
            start = frame_cursor
            for _ in range(min(args.batch_size, expected_frames - frame_cursor)):
                ok, bgr = capture.read()
                if not ok:
                    raise RuntimeError(f"Unexpected end of video at frame {frame_cursor}")
                images.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
                frame_cursor += 1
            observations = ground_images(images, factors, args, runtime)
            for offset, observation in enumerate(observations):
                frame = start + offset
                dino_boxes[frame, :, view_index] = observation["boxes"]
                dino_scores[frame, :, view_index] = observation["dino_scores"]
                sam_scores[frame, :, view_index] = observation["sam_scores"]
                mask_valid[frame, :, view_index] = observation["mask_valid"]
                mask_status[frame, :, view_index] = observation["mask_valid"]
                for factor_id, mask in enumerate(observation["masks"]):
                    if mask is not None:
                        packed = np.packbits(mask, axis=-1)
                        masks_store[frame, factor_id, view_index, :mask.shape[0], :packed.shape[1]] = packed
    finally:
        capture.release()


def process_episode(args: argparse.Namespace, runtime: dict, episode_index: int) -> dict:
    output_dir = args.output_root / "masks" / args.dataset_name / f"episode_{episode_index:06d}"
    if not args.overwrite and is_complete(args.output_root, args.dataset_name, episode_index, args.keys):
        return {"status": "skipped", "output_dir": str(output_dir), "episode_index": episode_index}
    dataset_path = args.dataset_root / args.dataset_name
    episode_record = read_episode_record(dataset_path, episode_index)
    instruction = read_episode_instruction(dataset_path, episode_index, args.language_key)
    factors = parse_instruction(instruction)
    unique_entities = list(dict.fromkeys(factor.factor_text for factor in factors))
    factor_query = ". ".join(unique_entities) + "."
    frame_count = int(episode_record["length"])
    info = json.loads((dataset_path / "meta/info.json").read_text())
    episode_chunk = episode_index // int(info["chunks_size"])
    video_metadata = json.loads((dataset_path / "meta/modality.json").read_text())["video"]
    video_paths = []
    for key in args.keys:
        original_key = video_metadata[key].get("original_key")
        if original_key is None:
            original_key = key
        video_paths.append(dataset_path / info["video_path"].format(
            episode_chunk=episode_chunk,
            episode_index=episode_index,
            video_key=original_key,
        ))
    temp_dir = output_dir.with_name(f"{output_dir.name}.tmp-{os.getpid()}")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)

    heights, widths = [], []
    for video_path in video_paths:
        capture = cv2.VideoCapture(str(video_path))
        try:
            if not capture.isOpened():
                raise RuntimeError(f"Unable to open {video_path}")
            heights.append(int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            widths.append(int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
            video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if video_frames != frame_count:
                raise ValueError(f"Frame mismatch for {video_path}: video={video_frames}, metadata={frame_count}")
        finally:
            capture.release()

    factor_count = len(factors)
    view_count = len(args.keys)
    packed_width = (max(widths) + 7) // 8
    masks_store = np.lib.format.open_memmap(
        temp_dir / "masks_packbits.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(frame_count, factor_count, view_count, max(heights), packed_width),
    )
    mask_valid = np.zeros((frame_count, factor_count, view_count), dtype=bool)
    dino_boxes = np.zeros((frame_count, factor_count, view_count, 4), dtype=np.float16)
    dino_scores = np.zeros((frame_count, factor_count, view_count), dtype=np.float16)
    sam_scores = np.zeros((frame_count, factor_count, view_count), dtype=np.float16)
    mask_status = np.zeros((frame_count, factor_count, view_count), dtype=np.uint8)

    start = time.perf_counter()
    for view_index, video_path in enumerate(video_paths):
        process_view(
            args, runtime, factors, video_path, view_index,
            masks_store, mask_valid, dino_boxes, dino_scores, sam_scores, mask_status,
            frame_count,
        )
    masks_store.flush()
    elapsed = time.perf_counter() - start
    factor_valid = mask_valid.any(axis=2)
    grounded_frame_valid = factor_valid.any(axis=1)
    use_global_fallback = ~grounded_frame_valid
    valid_frame_indices = np.arange(frame_count, dtype=np.int32)
    np.savez(
        temp_dir / "meta.npz",
        factor_ids=np.arange(factor_count, dtype=np.int32),
        frame_indices=np.arange(frame_count, dtype=np.int32),
        mask_valid=mask_valid,
        dino_boxes=dino_boxes,
        dino_scores=dino_scores,
        sam_scores=sam_scores,
        mask_status=mask_status,
        use_global_fallback=use_global_fallback,
        valid_frame_indices=valid_frame_indices,
        image_height=np.asarray(heights, dtype=np.int32),
        image_width=np.asarray(widths, dtype=np.int32),
    )
    episode_metadata = {
        "schema_version": 3,
        "dataset_name": args.dataset_name,
        "episode_index": episode_index,
        "trajectory_id": episode_record.get("trajectory_id"),
        "instruction": instruction,
        "factor_query": factor_query,
        "factors": [factor.to_dict() for factor in factors],
        "views": args.keys,
        "duplicate_entity_assignment": "text_occurrence_to_top_score_then_left_to_right",
        "num_frames": frame_count,
        "valid_frames": int(len(valid_frame_indices)),
        "grounded_frames": int(grounded_frame_valid.sum()),
        "global_fallback_frames": int(use_global_fallback.sum()),
        "global_fallback_ratio": float(use_global_fallback.mean()),
        "valid_factor_ratio": float(factor_valid.mean()),
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "sam_checkpoint": str(args.sam_checkpoint),
        "grounding_model": str(args.grounding_model),
        "elapsed_seconds": elapsed,
        "pipeline_fps": frame_count / elapsed,
    }
    with (temp_dir / "episode.json").open("w") as handle:
        json.dump(episode_metadata, handle, indent=2)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    os.replace(temp_dir, output_dir)
    return {"status": "complete", "output_dir": str(output_dir), **episode_metadata}


def main() -> None:
    args = parse_args()
    runtime = build_runtime(args)
    print(json.dumps(process_episode(args, runtime, args.episode_index)))


if __name__ == "__main__":
    main()
