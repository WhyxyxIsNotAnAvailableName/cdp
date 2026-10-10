#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from tqdm import tqdm

from cdp.preprocessing.config import add_config_argument, load_preprocessing_config
from cdp.preprocessing.outputs import is_complete


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_config_argument(parser)
    parser.add_argument("--gpus", default="2,3")
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--start-task", type=int, default=1, help="1-based inclusive task index")
    parser.add_argument("--end-task", type=int, default=None, help="1-based inclusive task index")
    parser.add_argument("--max-episodes-per-task", type=int, default=None)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--dry-run", action="store_true")
    return load_preprocessing_config(parser.parse_args())


def read_episode_indices(dataset_path: Path) -> list[int]:
    values = []
    with (dataset_path / "meta/episodes.jsonl").open() as handle:
        for line in handle:
            values.append(int(json.loads(line)["episode_index"]))
    return sorted(values)


def terminate_all(processes: list[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    deadline = time.time() + 10
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=max(0.1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                process.kill()


def raise_system_exit() -> None:
    raise SystemExit(143)


def main() -> None:
    args = parse_args()
    source_root = Path(__file__).resolve().parents[2]
    python = Path(sys.executable)
    gpu_ids = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpu_ids or args.workers_per_gpu <= 0:
        raise ValueError("At least one GPU and a positive workers-per-gpu are required")
    gpu_slots = [gpu for gpu in gpu_ids for _ in range(args.workers_per_gpu)]

    tasks = list(args.dataset_names)
    end_task = args.end_task or len(tasks)
    tasks = tasks[args.start_task - 1 : end_task]
    if not tasks:
        raise ValueError("No tasks selected")

    print(
        json.dumps(
            {
                "selected_tasks": len(tasks),
                "gpu_slots": gpu_slots,
                "batch_size": args.batch_size,
                "output_root": str(args.output_root),
            },
            indent=2,
        ),
        flush=True,
    )

    for local_task_index, dataset_name in enumerate(tasks, start=1):
        dataset_path = args.dataset_root / dataset_name
        absolute_task_index = args.start_task + local_task_index - 1
        all_indices = read_episode_indices(dataset_path)
        if args.max_episodes_per_task is not None:
            all_indices = all_indices[: args.max_episodes_per_task]
        completed = {
            idx for idx in all_indices if is_complete(args.output_root, dataset_name, idx, args.keys)
        }
        missing = [idx for idx in all_indices if idx not in completed]
        description = f"task {absolute_task_index}/{len(tasks) + args.start_task - 1} {dataset_name}"
        progress = tqdm(total=len(all_indices), initial=len(completed), desc=description, unit="episode")
        if args.dry_run or not missing:
            progress.close()
            if args.dry_run:
                print(f"DRY RUN: {len(missing)} missing episodes", flush=True)
            continue

        shards = [missing[offset:: len(gpu_slots)] for offset in range(len(gpu_slots))]
        log_dir = args.output_root / "logs" / dataset_name
        log_dir.mkdir(parents=True, exist_ok=True)
        processes: list[subprocess.Popen] = []
        log_handles = []
        try:
            for worker_id, (gpu_id, episode_indices) in enumerate(zip(gpu_slots, shards)):
                if not episode_indices:
                    continue
                status_log = log_dir / f"worker_{worker_id}.status.jsonl"
                process_log = log_dir / f"worker_{worker_id}.log"
                handle = process_log.open("a")
                log_handles.append(handle)
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = gpu_id
                env["PYTHONPATH"] = str(source_root) + os.pathsep + env.get("PYTHONPATH", "")
                env["PYTHONWARNINGS"] = "ignore"
                command = [
                    str(python),
                    "-m",
                    "cdp.preprocessing.preprocess_task_worker",
                    "--config_yaml",
                    str(args.config_yaml),
                    "--dataset-name",
                    dataset_name,
                    "--episode-indices",
                    ",".join(map(str, episode_indices)),
                    "--status-log",
                    str(status_log),
                    "--batch-size",
                    str(args.batch_size),
                ]
                processes.append(
                    subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT)
                )

            previous_completed = len(completed)
            while any(process.poll() is None for process in processes):
                time.sleep(args.poll_seconds)
                current_completed = sum(
                    is_complete(args.output_root, dataset_name, idx, args.keys) for idx in all_indices
                )
                if current_completed > previous_completed:
                    progress.update(current_completed - previous_completed)
                    previous_completed = current_completed
            current_completed = sum(
                is_complete(args.output_root, dataset_name, idx, args.keys) for idx in all_indices
            )
            if current_completed > previous_completed:
                progress.update(current_completed - previous_completed)
            return_codes = [process.wait() for process in processes]
            if current_completed != len(all_indices) or any(code != 0 for code in return_codes):
                raise RuntimeError(
                    f"Task incomplete: {current_completed}/{len(all_indices)}, "
                    f"worker return codes={return_codes}. See {log_dir}"
                )
        except KeyboardInterrupt:
            tqdm.write("Interrupted: terminating active workers. Completed episodes are resumable.")
            terminate_all(processes)
            raise
        except Exception:
            terminate_all(processes)
            raise
        finally:
            progress.close()
            for handle in log_handles:
                handle.close()

    print("All selected tasks completed.", flush=True)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: raise_system_exit())
    main()
