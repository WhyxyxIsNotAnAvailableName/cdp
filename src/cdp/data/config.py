import json
from pathlib import Path


def dataset_specs(data_cfg):
    root = Path(data_cfg["data_root_dir"])
    return [
        (str(entry["name"]), root / entry["name"], float(entry.get("weight", 1.0)))
        for entry in data_cfg["datasets"]
    ]


def infer_action_dim(data_cfg):
    dimensions = []
    for _, path, _ in dataset_specs(data_cfg):
        metadata = json.loads((path / "meta/modality.json").read_text())
        dimensions.append(sum(
            metadata["action"][key]["end"] - metadata["action"][key]["start"]
            for key in data_cfg["action_keys"]
        ))
    if not dimensions or len(set(dimensions)) != 1 or dimensions[0] < 1:
        raise ValueError("Selected datasets must have the same positive action dimension")
    return dimensions[0]


def read_episode_instruction(dataset_path: Path, episode_index: int, language_key: str) -> str:
    import pandas as pd

    info = json.loads((dataset_path / "meta/info.json").read_text())
    modality = json.loads((dataset_path / "meta/modality.json").read_text())
    original_key = modality["annotation"][language_key].get("original_key")
    if original_key is None:
        original_key = f"annotation.{language_key}"
    data_path = dataset_path / info["data_path"].format(
        episode_chunk=episode_index // int(info["chunks_size"]),
        episode_index=episode_index,
    )
    task_index = int(pd.read_parquet(data_path, columns=[original_key])[original_key].iloc[0])
    with (dataset_path / "meta/tasks.jsonl").open() as handle:
        for line in handle:
            task = json.loads(line)
            if int(task["task_index"]) == task_index:
                return task["task"]
    raise KeyError(task_index)
