import json
from pathlib import Path

from omegaconf import OmegaConf


def dict_to_namespace(d):
    return OmegaConf.create(d)


def populate_layerwise_dit_cfg(cfg, *, dit_hidden_dim: int, num_dit_layers: int):
    dit_cfg = cfg.framework.action_model.diffusion_model_cfg
    head_dim = dit_cfg.get("attention_head_dim", None) or 64
    dit_cfg.attention_head_dim = head_dim
    dit_cfg.num_layers = int(num_dit_layers)
    dit_cfg.input_embedding_dim = int(dit_hidden_dim)
    dit_cfg.cross_attention_dim = int(dit_hidden_dim)
    dit_cfg.num_attention_heads = int(dit_hidden_dim) // int(head_dim)
    return dit_cfg


def resolve_checkpoint_path(checkpoint):
    checkpoint = Path(checkpoint)
    if checkpoint.is_file():
        return checkpoint
    for name in ("model.safetensors", "pytorch_model.pt", "pytorch_model.bin"):
        candidate = checkpoint / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(checkpoint)


def read_model_config(pretrained_checkpoint):
    checkpoint = resolve_checkpoint_path(pretrained_checkpoint)
    run_dir = checkpoint.parents[1]
    if checkpoint.parent.name.startswith("state_steps_"):
        run_dir = checkpoint.parents[2]
    config_path = run_dir / "config.full.yaml"
    if not config_path.is_file():
        config_path = run_dir / "config.yaml"
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    with (run_dir / "dataset_statistics.json").open() as handle:
        norm_stats = json.load(handle)
    return config, norm_stats
