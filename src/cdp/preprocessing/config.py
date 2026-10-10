from argparse import ArgumentParser, Namespace
from pathlib import Path

from omegaconf import OmegaConf
from cdp.data.config import dataset_specs


def add_config_argument(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--config_yaml",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "configs/train.yaml",
    )


def load_preprocessing_config(args: Namespace) -> Namespace:
    args.config_yaml = args.config_yaml.resolve()
    config = OmegaConf.load(args.config_yaml)
    data = config.datasets.vla_data
    perception = config.perception
    args.dataset_root = Path(data.data_root_dir)
    args.dataset_names = [name for name, _, _ in dataset_specs(data)]
    args.language_key = str(data.language_key)
    args.output_root = Path(data.semantic_sidecar_root)
    args.keys = list(data["keys"])
    if not args.keys or len(args.keys) != len(set(args.keys)):
        raise ValueError("keys must contain distinct camera names")
    args.grounding_model = Path(perception.grounding_model)
    args.sam_checkpoint = Path(perception.sam_checkpoint)
    args.sam_config = str(perception.sam_config)
    if getattr(args, "batch_size", None) is None:
        args.batch_size = int(perception.batch_size)
    args.box_threshold = float(perception.box_threshold)
    args.text_threshold = float(perception.text_threshold)
    return args
