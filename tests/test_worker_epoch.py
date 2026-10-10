import multiprocessing as mp
import os
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cdp.data.lerobot.datasets import LeRobotMixtureDataset
from cdp.training.train import VLATrainer
from cdp.training.utils import TrainerUtils


class ToyDataset:
    dataset_name = "test"
    trajectory_ids = np.arange(12)
    trajectory_lengths = np.arange(30, 42)

    def __len__(self):
        return 32


class SamplingDataset(LeRobotMixtureDataset):
    def __init__(self):
        super().__init__([(ToyDataset(), 1.0)], "train", False, False, seed=17)

    def update_metadata(self, metadata_config):
        pass

    def __getitem__(self, index):
        _, trajectory, step = self.sample_step(index)
        return dict(index=index, trajectory=int(trajectory), step=int(step), epoch=self.epoch, pid=os.getpid())


def records(iterator):
    result = []
    for batch in iterator:
        result.extend(zip(batch["index"].tolist(), batch["trajectory"].tolist(),
                          batch["step"].tolist(), batch["epoch"].tolist(), batch["pid"].tolist()))
    return result


def close(loader):
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None:
        iterator._shutdown_workers()


def test_persistent_workers_receive_epoch(context):
    dataset = SamplingDataset()
    loader = DataLoader(dataset, batch_size=8, num_workers=2, persistent_workers=True,
                        multiprocessing_context=context)
    try:
        first = records(iter(loader))
        iterator, epoch = TrainerUtils._reset_dataloader(loader, 0)
        second = records(iterator)
        assert epoch == 1 and {row[3] for row in second} == {1}
        assert {row[4] for row in first} == {row[4] for row in second}
        assert len({row[4] for row in second}) == 2
        assert any(a[1:3] != b[1:3] for a, b in zip(first, second))
        for index, trajectory, step, _, _ in second:
            _, expected_trajectory, expected_step = dataset.sample_step(index)
            assert (trajectory, step) == (expected_trajectory, expected_step)
        TrainerUtils.set_dataloader_epoch(loader, 0)
        assert records(iter(loader)) == first
    finally:
        close(loader)


def test_resume_and_next_epoch():
    accelerator = Accelerator(cpu=True, mixed_precision="no", gradient_accumulation_steps=2)
    dataset = SamplingDataset()
    loader = DataLoader(dataset, batch_size=8, num_workers=2, persistent_workers=True)
    wrapped = accelerator.prepare_data_loader(loader)
    trainer = object.__new__(VLATrainer)
    trainer.accelerator = accelerator
    trainer.vla_train_dataloader = wrapped
    resumed_loaders = []
    original_skip = accelerator.skip_first_batches

    def skip(dataloader, num_batches):
        resumed = original_skip(dataloader, num_batches)
        resumed_loaders.append(resumed)
        return resumed

    try:
        dataset.set_epoch(9)
        trainer.completed_steps = 0
        trainer._create_data_iterators()
        assert {row[3] for row in records(trainer.vla_iter)} == {0}
        trainer.completed_steps = 7
        with mock.patch.object(accelerator, "skip_first_batches", side_effect=skip):
            trainer._create_data_iterators()
        resumed = records(trainer.vla_iter)
        assert trainer.vla_epoch_count == 3
        assert [row[0] for row in resumed] == list(range(16, 32))
        assert {row[3] for row in resumed} == {3}
        batch = trainer._get_next_batch()
        assert trainer.vla_epoch_count == 4
        assert batch["epoch"].tolist() == [4] * 8
        for index, trajectory, step in zip(batch["index"], batch["trajectory"], batch["step"]):
            _, expected_trajectory, expected_step = dataset.sample_step(int(index))
            assert (int(trajectory), int(step)) == (expected_trajectory, expected_step)
    finally:
        for resumed in resumed_loaders:
            close(resumed)
        close(wrapped)
        close(loader)


if __name__ == "__main__":
    for context in ("fork", "spawn"):
        if context in mp.get_all_start_methods():
            test_persistent_workers_receive_epoch(context)
    test_resume_and_next_epoch()
    assert not torch.cuda.is_initialized()
    print("test_worker_epoch: PASS")
