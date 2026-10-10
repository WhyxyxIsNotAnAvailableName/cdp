import json
from pathlib import Path


def is_complete(output_root: Path, dataset_name: str, episode_index: int, keys: list[str]) -> bool:
    episode_dir = output_root / "masks" / dataset_name / f"episode_{episode_index:06d}"
    required = (
        episode_dir / "episode.json",
        episode_dir / "meta.npz",
        episode_dir / "masks_packbits.npy",
    )
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        return False
    try:
        metadata = json.loads(required[0].read_text())
        return int(metadata.get("schema_version", -1)) == 3 and metadata.get("views") == keys
    except (OSError, ValueError, TypeError):
        return False
