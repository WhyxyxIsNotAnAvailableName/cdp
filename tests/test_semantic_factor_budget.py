import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cdp.data.semantic_masks import SemanticSidecarReader


def test_budget_selects_factor_text_and_corresponding_masks():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        episode = root / "masks/test_dataset/episode_000000"
        episode.mkdir(parents=True)
        factors = [{"factor_id": index, "factor_text": f"entity {index}", "entity_text": f"entity {index}"}
                   for index in range(5)]
        (episode / "episode.json").write_text(json.dumps({
            "schema_version": 3, "views": ["ego"], "factors": factors[::-1]
        }))
        masks = np.zeros((1, 5, 1, 2, 8), dtype=np.uint8)
        for index in range(5):
            masks[0, index, 0, 0, index] = 1
        np.save(episode / "masks_packbits.npy", np.packbits(masks, axis=-1))
        np.savez(episode / "meta.npz", image_width=8,
                 mask_valid=np.ones((1, 5, 1), dtype=bool),
                 dino_scores=np.ones((1, 5, 1)), sam_scores=np.ones((1, 5, 1)),
                 use_global_fallback=np.zeros(1, dtype=bool))
        reader = SemanticSidecarReader(root, "test_dataset", factor_budget=4)
        frame = reader.get_frame(0, 0)
        assert [factor["factor_id"] for factor in frame["factors"]] == [0, 1, 2]
        for index, factor in enumerate(frame["factors"]):
            assert factor["factor_text"] == f"entity {index}"
            np.testing.assert_array_equal(factor["view_masks"][0], masks[0, index, 0])
        assert len(reader._load_episode(0)["metadata"]["factors"]) == 5
        assert not frame["all_semantic_masks_invalid"]
        global_only = SemanticSidecarReader(root, "test_dataset", factor_budget=1).get_frame(0, 0)
        assert global_only["factors"] == []
        assert global_only["all_semantic_masks_invalid"]


if __name__ == "__main__":
    test_budget_selects_factor_text_and_corresponding_masks()
    print("test_semantic_factor_budget: PASS")
