# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].


import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Tuple


import numpy as np
import torch
import torch.distributed as dist

import wandb
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_scheduler


from cdp.data import build_dataloader
from cdp.models.base import build_policy
from cdp.training.config_tracker import AccessTrackedConfig, wrap_config
from cdp.training.utils import TrainerUtils, build_param_lr_groups, normalize_dotlist_args


os.environ["TOKENIZERS_PARALLELISM"] = "false"


logger = get_logger(__name__)


def setup_directories(cfg) -> Path:
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

    return output_dir


def prepare_data(cfg, accelerator, output_dir) -> DataLoader:
    logger.info("Creating training datasets")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)

    accelerator.dataloader_config.dispatch_batches = False
    if dist.is_initialized():
        dist.barrier()
    return vla_train_dataloader


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
        fused=bool(cfg.trainer.get("fused_optimizer", True)),
    )

    if dist.is_initialized() and dist.get_rank() == 0:
        for group in optimizer.param_groups:
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    sched_kwargs = {k: v for k, v in cfg.trainer.scheduler_specific_kwargs.items()}
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.get(
            "lr_scheduler_total_steps", cfg.trainer.max_train_steps
        ),
        scheduler_specific_kwargs=sched_kwargs,
    )

    return optimizer, lr_scheduler


class VLATrainer(TrainerUtils):
    def __init__(self, cfg, model, vla_train_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        self._save_initial_configs()

        self._init_checkpointing()
        if self._resume_state_path is None:
            self._adjust_lr_scheduler_for_resume()

        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)
        self.print_trainable_parameters(self.model)

        self.model, self.optimizer, self.vla_train_dataloader = self.setup_distributed_training(
            self.accelerator,
            self.model,
            self.optimizer,
            self.vla_train_dataloader,
        )

        if self._resume_state_path is not None:
            self._load_checkpoint(self._resume_state_path)
            logger.info(
                f"Restored full training state from {self._resume_state_path} "
                f"at optimizer step {self.completed_steps}"
            )

        self._init_wandb()

    def _calculate_total_batch_size(self):
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_wandb(self):
        self._wandb_enabled = False
        if os.environ.get("WANDB_MODE") == "disabled" or os.environ.get("WANDB_DISABLED", "").lower() in {
            "1",
            "true",
            "yes",
        }:
            self.accelerator.wait_for_everyone()
            return
        if self.accelerator.is_main_process:
            try:
                wandb.init(
                    name=self.config.run_id,
                    dir=os.path.join(self.config.output_dir, "wandb"),
                    project=self.config.wandb_project,
                    entity=self.config.wandb_entity,
                    group="vla-train",
                )
                self._wandb_enabled = True
            except Exception as exc:
                logger.warning(f"W&B init failed; continuing without W&B: {exc}")
                self._wandb_enabled = False

        self.accelerator.wait_for_everyone()

    def _save_initial_configs(self):
        if not self.accelerator.is_main_process:
            return

        output_dir = Path(self.config.output_dir)

        if isinstance(self.config, AccessTrackedConfig):
            full_cfg = self.config.unwrap()
        else:
            full_cfg = self.config
        full_yaml_path = output_dir / "config.full.yaml"
        OmegaConf.save(full_cfg, full_yaml_path, resolve=True)
        logger.info("Configuration saved: %s", full_yaml_path)

        if isinstance(self.config, AccessTrackedConfig):
            self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)

    def _init_checkpointing(self):
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)
        self.resume_from_checkpoint = pretrained_checkpoint
        self._resume_state_path = None

        if is_resume:
            resume_from_checkpoint, self.completed_steps = self._get_latest_checkpoint(self.checkpoint_dir)
            if resume_from_checkpoint:
                self.resume_from_checkpoint = resume_from_checkpoint
                if os.path.isdir(resume_from_checkpoint):
                    self._resume_state_path = resume_from_checkpoint
                else:
                    self.model = self.load_pretrained_backbones(
                        self.model, self.resume_from_checkpoint, reload_modules=None
                    )
                    logger.warning(
                        f"Resuming from legacy model-only checkpoint {resume_from_checkpoint}; "
                        "optimizer and RNG state will restart."
                    )
                return

            logger.info("No checkpoint in %s", self.checkpoint_dir)
            self.completed_steps = 0

        if pretrained_checkpoint:
            reload_modules = getattr(self.config.trainer, "reload_modules", None)
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)
            self.completed_steps = 0
            self.resume_from_checkpoint = pretrained_checkpoint
            logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}, steps: {self.completed_steps}")
        else:
            logger.debug("No pretrained checkpoint")
            self.completed_steps = 0

    def _adjust_lr_scheduler_for_resume(self):
        if self.completed_steps > 0:
            logger.info(f"Adjusting LR scheduler for resume from step {self.completed_steps}")
            for _ in range(self.completed_steps):
                self.lr_scheduler.step()
            logger.info(
                f"LR scheduler adjusted to step {self.completed_steps}, current LR: {self.lr_scheduler.get_last_lr()}"
            )

    def _load_checkpoint(self, checkpoint_path):
        self.accelerator.load_state(checkpoint_path)
        scheduler_path = Path(checkpoint_path) / "scheduler.bin"
        if scheduler_path.is_file():
            scheduler_state = torch.load(scheduler_path, map_location="cpu")
            self.lr_scheduler.load_state_dict(scheduler_state)
            logger.info(f"Restored LR scheduler state from {scheduler_path}")
        else:
            logger.warning(
                f"Checkpoint {checkpoint_path} predates scheduler-state saving; "
                f"reconstructing scheduler deterministically to step {self.completed_steps}."
            )
            self._adjust_lr_scheduler_for_resume()

    def _save_checkpoint(self):
        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"state_steps_{self.completed_steps}"
        )

        self.accelerator.save_state(checkpoint_path, safe_serialization=False)
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            torch.save(
                self.lr_scheduler.state_dict(),
                Path(checkpoint_path) / "scheduler.bin",
            )
            summary_data = {"steps": self.completed_steps}
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            logger.info("Checkpoint saved: %s", checkpoint_path)

            if isinstance(self.config, AccessTrackedConfig):
                output_dir = Path(self.config.output_dir)
                self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)

            self._prune_full_state_checkpoints()

        self.accelerator.wait_for_everyone()

    def _prune_full_state_checkpoints(self):
        keep = int(self.config.trainer.get("max_checkpoints_to_keep", 0))
        if keep <= 0:
            return
        checkpoint_root = Path(self.checkpoint_dir).resolve()
        candidates = []
        for path in checkpoint_root.iterdir():
            match = re.fullmatch(r"state_steps_(\d+)", path.name)
            if match and path.is_dir():
                candidates.append((int(match.group(1)), path.resolve()))
        candidates.sort(key=lambda item: item[0])
        for step, path in candidates[:-keep]:
            if path.parent != checkpoint_root:
                raise RuntimeError(f"Refusing to prune checkpoint outside {checkpoint_root}: {path}")
            shutil.rmtree(path)
            logger.info(f"Pruned old full-state checkpoint at step {step}: {path}")

    def _log_metrics(self, metrics):
        rank = dist.get_rank() if dist.is_initialized() else 0
        if self.completed_steps % self.config.trainer.logging_frequency != 0:
            return

        if dist.is_initialized():
            numeric_keys = [
                key for key, value in metrics.items() if isinstance(value, (int, float))
            ]
            if numeric_keys:
                values = torch.tensor(
                    [float(metrics[key]) for key in numeric_keys],
                    device=self.accelerator.device,
                    dtype=torch.float64,
                )
                dist.all_reduce(values, op=dist.ReduceOp.SUM)
                values /= dist.get_world_size()
                memory_indices = [
                    index for index, key in enumerate(numeric_keys) if key.startswith("memory/")
                ]
                if memory_indices:
                    memory_values = torch.tensor(
                        [float(metrics[numeric_keys[index]]) for index in memory_indices],
                        device=self.accelerator.device,
                        dtype=torch.float64,
                    )
                    dist.all_reduce(memory_values, op=dist.ReduceOp.MAX)
                    for index, value in zip(memory_indices, memory_values.tolist(), strict=True):
                        values[index] = value
                for key, value in zip(numeric_keys, values.tolist(), strict=True):
                    metrics[key] = value
        if rank == 0:
            last_lrs = self.lr_scheduler.get_last_lr()
            for i, group in enumerate(self.optimizer.param_groups):
                group_name = group.get("name", str(i))
                metrics[f"learning_rate/{group_name}"] = last_lrs[i] if i < len(last_lrs) else last_lrs[-1]
            metrics["epoch"] = round(self.completed_steps / len(self.vla_train_dataloader), 2)
            if getattr(self, "_wandb_enabled", False):
                try:
                    wandb.log(metrics, step=self.completed_steps)
                except Exception as exc:
                    self._wandb_enabled = False
                    logger.warning(f"W&B log failed; disabling W&B: {exc}")
            metrics_record = {"step": self.completed_steps, **metrics}
            with open(Path(self.config.output_dir) / "metrics.jsonl", "a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics_record, sort_keys=True) + "\n")
            logger.info("step=%d loss=%.6f", self.completed_steps, metrics["action_dit_loss"])

    def _create_data_iterators(self):
        if self.completed_steps <= 0:
            self.vla_epoch_count = 0
            self.set_dataloader_epoch(self.vla_train_dataloader, 0)
            self.vla_iter = iter(self.vla_train_dataloader)
            return

        consumed_batches = (
            self.completed_steps * self.accelerator.gradient_accumulation_steps
        )
        batches_per_epoch = len(self.vla_train_dataloader)
        self.vla_epoch_count, batch_offset = divmod(
            consumed_batches, batches_per_epoch
        )
        self.set_dataloader_epoch(self.vla_train_dataloader, self.vla_epoch_count)

        resumed_dataloader = self.accelerator.skip_first_batches(
            self.vla_train_dataloader, num_batches=batch_offset
        )
        self.set_dataloader_epoch(resumed_dataloader, self.vla_epoch_count)
        self.vla_iter = iter(resumed_dataloader)
        logger.info(
            f"Resuming dataloader at epoch={self.vla_epoch_count}, "
            f"batch_offset={batch_offset}/{batches_per_epoch}"
        )

    def _get_next_batch(self):
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)

        return batch_vla

    def train(self):
        self._log_training_config()
        self._create_data_iterators()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        progress_bar = tqdm(
            total=self.config.trainer.max_train_steps,
            initial=self.completed_steps,
            disable=not self.accelerator.is_local_main_process,
        )

        while self.completed_steps < self.config.trainer.max_train_steps:
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()

            synchronize_timing = bool(
                self.config.trainer.get("synchronize_cuda_timing", False)
            )
            if synchronize_timing and torch.cuda.is_available():
                torch.cuda.synchronize()
            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            if synchronize_timing and torch.cuda.is_available():
                torch.cuda.synchronize()
            t_end_model = time.perf_counter()

            optimizer_stepped = step_metrics.pop(
                "_optimizer_step", self.accelerator.sync_gradients
            )
            if optimizer_stepped:
                progress_bar.update(1)
                self.completed_steps += 1

            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                    {
                        "data_times": f"{t_end_data - t_start_data:.3f}",
                        "model_times": f"{t_end_model - t_start_model:.3f}",
                    }
                )

            step_metrics["timing/data"] = t_end_data - t_start_data
            step_metrics["timing/model"] = t_end_model - t_start_model
            if optimizer_stepped:
                self._log_metrics(step_metrics)

            if (
                optimizer_stepped
                and self.completed_steps % self.config.trainer.save_interval == 0
                and self.completed_steps > 0
            ):
                self._save_checkpoint()

            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        self._finalize_training()

    def _log_training_config(self):
        if self.accelerator.is_main_process:
            logger.info(
                "steps=%d batch/device=%d accumulation=%d batch/global=%d",
                self.config.trainer.max_train_steps,
                self.config.datasets.vla_data.per_device_batch_size,
                self.accelerator.gradient_accumulation_steps,
                self.total_batch_size,
            )

    def _train_step(self, batch_vla):
        with self.accelerator.accumulate(self.model):
            with self.accelerator.autocast():
                output_dict = self.model.forward(batch_vla)
                action_loss = output_dict["action_loss"]
            self.accelerator.backward(action_loss)
            if self.accelerator.sync_gradients and self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(), self.config.trainer.gradient_clipping
                )
            self.optimizer.step()
            optimizer_stepped = bool(self.accelerator.sync_gradients)
            if optimizer_stepped:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()

        metrics = {"action_dit_loss": action_loss.item()}
        scalar_keys = (
            "flow_matching_loss",
            "magnitude_loss",
            "magnitude_active_fraction",
            "background_residual_norm",
        )
        for key in scalar_keys:
            if key in output_dict:
                metrics[key] = float(torch.as_tensor(output_dict[key]).float().mean())

        branch_names = self.accelerator.unwrap_model(self.model).branch_names if "branch_norm" in output_dict else ()
        if "branch_norm" in output_dict:
            branch_norm = torch.as_tensor(output_dict["branch_norm"]).float()
            for branch_id, branch_name in enumerate(branch_names):
                metrics[f"branch_norm/{branch_name}"] = float(branch_norm[..., branch_id].mean())
        if "composed_norm" in output_dict:
            metrics["composed_norm"] = float(
                torch.as_tensor(output_dict["composed_norm"]).float().mean()
            )
        if "mask_valid_by_factor" in output_dict:
            mask_valid = torch.as_tensor(output_dict["mask_valid_by_factor"]).float()
            for factor_id, factor_name in enumerate(branch_names[1:]):
                metrics[f"mask_valid/{factor_name}"] = float(mask_valid[factor_id])
                metrics[f"original_fallback/{factor_name}"] = 1.0 - float(mask_valid[factor_id])
        if "branch_cosine" in output_dict:
            cosine = torch.as_tensor(output_dict["branch_cosine"]).float()
            for left in range(len(branch_names)):
                for right in range(left + 1, len(branch_names)):
                    metrics[f"branch_cosine/{branch_names[left]}_{branch_names[right]}"] = float(
                        cosine[..., left, right].mean()
                    )
        if torch.cuda.is_available():
            gib = 1024**3
            metrics.update(
                {
                    "memory/allocated_gib": torch.cuda.memory_allocated() / gib,
                    "memory/reserved_gib": torch.cuda.memory_reserved() / gib,
                    "memory/max_allocated_gib": torch.cuda.max_memory_allocated() / gib,
                    "memory/max_reserved_gib": torch.cuda.max_memory_reserved() / gib,
                }
            )
        metrics["_optimizer_step"] = optimizer_stepped
        return metrics

    def _finalize_training(self):
        if bool(self.config.trainer.get("skip_final_checkpoint", False)):
            logger.info("Skipping final checkpoint as requested by trainer.skip_final_checkpoint")
            if self.accelerator.is_main_process and getattr(self, "_wandb_enabled", False):
                try:
                    wandb.finish()
                except Exception:
                    pass
            self.accelerator.wait_for_everyone()
            return
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, os.path.join(final_checkpoint, "model.safetensors"))
            elif save_format == "pt":
                torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")

        if self.accelerator.is_main_process and getattr(self, "_wandb_enabled", False):
            try:
                wandb.finish()
            except Exception:
                pass

        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    cfg = wrap_config(cfg)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.trainer.gradient_accumulation_steps,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    logger.info(
        "processes=%d device=%s precision=%s",
        accelerator.num_processes, accelerator.device, accelerator.mixed_precision,
    )
    if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        logger.info("Waiting for debugger on port 10092")
        debugpy.wait_for_client()


    output_dir = setup_directories(cfg=cfg)
    vla = build_policy(cfg)
    vla_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    trainer = VLATrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )

    trainer.prepare_training()
    trainer.train()

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default=str(Path(__file__).resolve().parents[3] / "configs/train.yaml"),
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    cfg.config_yaml = args.config_yaml

    main(cfg)
