#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from types import SimpleNamespace

from cdp.preprocessing.config import add_config_argument, load_preprocessing_config
from cdp.perception.grounding import build_runtime
from cdp.preprocessing.preprocess_episode import process_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_config_argument(parser)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--episode-indices", required=True)
    parser.add_argument("--status-log", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return load_preprocessing_config(parser.parse_args())


def append_status(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(result, sort_keys=True) + "\n")
        handle.flush()


def main() -> None:
    args = parse_args()
    episode_indices = [int(value) for value in args.episode_indices.split(",") if value]
    runtime_args = SimpleNamespace(**vars(args))
    runtime = build_runtime(runtime_args)
    failures = 0
    for episode_index in episode_indices:
        try:
            result = process_episode(runtime_args, runtime, episode_index)
        except Exception as exc:
            failures += 1
            result = {
                "status": "error",
                "dataset_name": args.dataset_name,
                "episode_index": episode_index,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        append_status(args.status_log, result)
        print(json.dumps(result, sort_keys=True), flush=True)
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
