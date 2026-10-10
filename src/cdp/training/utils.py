import re
import torch
import torch.distributed as dist

from accelerate.logging import get_logger

logger = get_logger(__name__)


def _dist_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def normalize_dotlist_args(args):
    normalized = []
    skip = False
    for i in range(len(args)):
        if skip:
            skip = False
            continue

        arg = args[i]
        if arg.startswith("--"):
            key = arg.lstrip("-")
            if "=" in key:
                normalized.append(key)
            elif i + 1 < len(args) and not args[i + 1].startswith("--"):
                normalized.append(f"{key}={args[i + 1]}")
                skip = True
            else:
                normalized.append(f"{key}=true")
        else:
            pass
    return normalized


def build_param_lr_groups(model, cfg):
    lr_cfg = cfg.trainer.learning_rate
    base_lr = lr_cfg.get("base", 1e-4)

    freeze_modules = cfg.trainer.get("freeze_modules", "")
    if not isinstance(freeze_modules, str):
        freeze_modules = ""
    freeze_patterns = [p.strip() for p in freeze_modules.split(",") if p.strip()]

    used_params = set()
    frozen_params = set()
    param_groups = []

    for freeze_path in freeze_patterns:
        module = model
        try:
            for attr in freeze_path.split("."):
                module = getattr(module, attr)
            frozen_params.update(id(p) for p in module.parameters())
        except AttributeError:
            logger.warning("Freeze module not found: %s", freeze_path)
            continue

    for module_name, lr in lr_cfg.items():
        if module_name == "base":
            continue

        module = model
        try:
            for attr in module_name.split("."):
                module = getattr(module, attr)

            params = [p for p in module.parameters() if id(p) not in frozen_params]
            if params:
                param_groups.append({"params": params, "lr": lr, "name": module_name})
                used_params.update(id(p) for p in params)
        except AttributeError:
            logger.warning("Learning-rate module not found: %s", module_name)

    other_params = [p for p in model.parameters() if id(p) not in used_params and id(p) not in frozen_params]
    if other_params:
        param_groups.append({"params": other_params, "lr": base_lr, "name": "base"})

    return param_groups


import torch.distributed as dist


from PIL import Image


def resize_images(images, target_size=(224, 224)):
    if isinstance(images, Image.Image):
        return images.resize(target_size)
    elif isinstance(images, list):
        return [resize_images(img, target_size) for img in images]
    else:
        raise ValueError("Unsupported image type or structure.")


import torch.distributed as dist


class TrainerUtils:
    @staticmethod
    def freeze_backbones(model, freeze_modules=""):
        frozen = []
        if freeze_modules and type(freeze_modules) == str:

            patterns = [p.strip() for p in freeze_modules.split(",") if p.strip()] if freeze_modules else []

            for path in patterns:

                attrs = path.split(".")
                module = model
                try:
                    for attr in attrs:
                        module = getattr(module, attr)

                    for param in module.parameters():
                        param.requires_grad = False
                    frozen.append(path)
                except AttributeError:

                    logger.warning("Freeze module not found: %s", path)
                    continue

        if _dist_rank() == 0:
            logger.info("Frozen modules: %s", frozen)
        return model

    @staticmethod
    def print_trainable_parameters(model):
        if _dist_rank() != 0:
            return
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("Parameters: total=%d trainable=%d", num_params, num_trainable_params)
        return num_params, num_trainable_params

    @staticmethod
    def load_pretrained_backbones(model, checkpoint_path=None, reload_modules=None):
        if not checkpoint_path:
            return []
        try:
            if _is_safetensors_path(checkpoint_path):
                from safetensors.torch import load_file

                checkpoint = load_file(checkpoint_path)
            else:
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
        except Exception as e:
            raise RuntimeError(f"Could not load checkpoint {checkpoint_path}: {e}") from e

        loaded_modules = []

        if reload_modules:
            module_paths = [p.strip() for p in reload_modules.split(",") if p.strip()]
            for path in module_paths:
                reload_modules = path.split(".")
                module = model
                try:
                    for module_name in reload_modules:
                        module = getattr(module, module_name)
                    prefix = path + "."
                    sub_state_dict = {k[len(prefix) :]: v for k, v in checkpoint.items() if k.startswith(prefix)}
                    if sub_state_dict:
                        module.load_state_dict(sub_state_dict, strict=True)
                        loaded_modules.append(path)
                    else:
                        logger.warning("Checkpoint has no parameters for module: %s", path)
                except AttributeError:
                    logger.warning("Module not found: %s", path)
        else:
            try:
                model.load_state_dict(checkpoint, strict=False)
                loaded_modules = ["<full_model>"]
            except Exception as e:
                raise RuntimeError(f"Could not apply checkpoint {checkpoint_path}: {e}") from e
        if _dist_rank() == 0:
            logger.info("Checkpoint loaded: %s; modules=%s", checkpoint_path, loaded_modules)
        return model

    @staticmethod
    def setup_distributed_training(accelerator, *components):
        prepared_components = accelerator.prepare(*components)
        return prepared_components

    @staticmethod
    def set_dataloader_epoch(dataloader, epoch):
        if callable(getattr(dataloader, "set_epoch", None)):
            dataloader.set_epoch(epoch)
        elif callable(getattr(dataloader.sampler, "set_epoch", None)):
            dataloader.sampler.set_epoch(epoch)
        if callable(getattr(dataloader.dataset, "set_epoch", None)):
            dataloader.dataset.set_epoch(epoch)

    @staticmethod
    def _reset_dataloader(dataloader, epoch_counter):
        epoch_counter += 1
        TrainerUtils.set_dataloader_epoch(dataloader, epoch_counter)
        return iter(dataloader), epoch_counter

    def _get_latest_checkpoint(self, checkpoint_dir):
        if not os.path.exists(checkpoint_dir):
            logger.debug("Checkpoint directory not found: %s", checkpoint_dir)
            return None, 0

        state_directories = []
        for name in os.listdir(checkpoint_dir):
            match = re.fullmatch(r"state_steps_(\d+)", name)
            if match and os.path.isdir(os.path.join(checkpoint_dir, name)):
                state_directories.append((name, int(match.group(1))))
        if state_directories:
            latest_checkpoint, completed_steps = max(
                state_directories, key=lambda item: item[1]
            )
            latest_checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
            logger.debug("Latest checkpoint: %s", latest_checkpoint_path)
            return latest_checkpoint_path, completed_steps

        checkpoints = [
            f for f in os.listdir(checkpoint_dir)
            if re.match(r"steps_(\d+)_(?:pytorch_model\.pt|model\.safetensors)$", f)
            and os.path.isfile(os.path.join(checkpoint_dir, f))
        ]

        if not checkpoints:
            logger.debug("No checkpoints in %s", checkpoint_dir)
            return None, 0

        try:
            checkpoints_with_steps = [
                (ckpt, int(re.search(r"steps_(\d+)_(?:pytorch_model\.pt|model\.safetensors)$", ckpt).group(1)))
                for ckpt in checkpoints
            ]
        except AttributeError as e:
            logger.warning("Invalid checkpoint filename: %s", e)
            return None, 0

        checkpoints_with_steps.sort(key=lambda x: x[1])
        latest_checkpoint, completed_steps = checkpoints_with_steps[-1]

        latest_checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        logger.debug("Latest checkpoint: %s", latest_checkpoint_path)
        return latest_checkpoint_path, completed_steps

import os


def is_main_process():
    rank = int(os.environ.get("RANK", 0))
    return rank == 0


def _is_safetensors_path(path):
    return str(path).endswith(".safetensors")
