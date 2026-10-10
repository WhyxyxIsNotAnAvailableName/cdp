import math
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cdp.models.flow_matching import (
    LayerwiseFlowmatchingActionHead,
)


class DummySharedHead(nn.Module):
    forward_compositional = LayerwiseFlowmatchingActionHead.forward_compositional
    predict_action_compositional = LayerwiseFlowmatchingActionHead.predict_action_compositional

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.25))
        self.num_timestep_buckets = 100
        self.num_inference_timesteps = 3
        self.action_horizon = 2
        self.action_dim = 1
        self.last_noisy = None
        self.last_timesteps = None

    def sample_time(self, batch_size, device, dtype):
        return torch.linspace(0.1, 0.9, batch_size, device=device, dtype=dtype)

    def predict_velocity(
        self,
        vl_embs_list,
        noisy_actions,
        timesteps,
        state=None,
        encoder_attention_mask=None,
    ):
        self.last_noisy = noisy_actions.detach().clone()
        self.last_timesteps = timesteps.detach().clone()
        condition = vl_embs_list[0][:, :1, :1].reshape(-1, 1, 1)
        return self.scale * condition.expand_as(noisy_actions)


def test_compositional_training_shares_flow_state_across_branches():
    torch.manual_seed(0)
    head = DummySharedHead()
    batch_size, repeats, branches = 2, 3, 4
    actions = torch.zeros(batch_size, 2, 1)
    condition_ids = torch.arange(batch_size * branches, dtype=torch.float32).reshape(-1, 1, 1)

    output = head.forward_compositional(
        [condition_ids],
        actions,
        branch_count=branches,
        repeated_diffusion_steps=repeats,
        branch_weights=[0.25] * 4,
        magnitude_alpha=0.4,
    )
    assert output["branch_velocity"].shape == (batch_size, repeats, branches, 2, 1)
    noisy = head.last_noisy.reshape(batch_size, repeats, branches, 2, 1)
    timesteps = head.last_timesteps.reshape(batch_size, repeats, branches)
    assert torch.equal(noisy, noisy[:, :, :1].expand_as(noisy))
    assert torch.equal(timesteps, timesteps[:, :, :1].expand_as(timesteps))
    assert output["magnitude_loss"].item() >= 0

    output["loss"].backward()
    assert head.scale.grad is not None
    assert torch.isfinite(head.scale.grad)


def test_compositional_sampler_has_source_batch_shape():
    torch.manual_seed(0)
    head = DummySharedHead()
    batch_size, branches = 2, 4
    condition_ids = torch.arange(batch_size * branches, dtype=torch.float32).reshape(-1, 1, 1)
    actions = head.predict_action_compositional(
        [condition_ids],
        branch_count=branches,
        branch_weights=[0.25] * 4,
    )
    assert actions.shape == (batch_size, head.action_horizon, head.action_dim)
    assert torch.isfinite(actions).all()


def test_magnitude_loss_sums_four_branch_penalties():
    torch.manual_seed(0)
    head = DummySharedHead()
    conditions = torch.zeros(8, 1, 1)
    actions = torch.full((2, 2, 1), 2.0)
    output = head.forward_compositional([conditions], actions, repeated_diffusion_steps=3)
    target_squared_norm = output["target_velocity"].float().square().sum(dim=(-2, -1))
    expected = 4 * 0.4**2 * target_squared_norm.mean()
    assert expected > 0
    torch.testing.assert_close(output["magnitude_loss"], expected)


def test_flow_matching_loss_sums_trajectory_dimensions():
    torch.manual_seed(0)
    head = DummySharedHead()
    output = head.forward_compositional(
        [torch.zeros(8, 1, 1)], torch.ones(2, 2, 1), repeated_diffusion_steps=3
    )
    expected = output["target_velocity"].square().sum(dim=(-2, -1)).mean()
    torch.testing.assert_close(output["flow_matching_loss"], expected)


def test_inactive_branches_do_not_affect_composition_or_loss():
    head = DummySharedHead()
    conditions = torch.tensor([1., 3., 100., 200., 4., 100., 200., 300.]).reshape(8, 1, 1)
    branch_mask = torch.tensor([[True, True, False, False], [True, False, False, False]])
    torch.manual_seed(0)
    output = head.forward_compositional(
        [conditions], torch.full((2, 2, 1), 2.0), repeated_diffusion_steps=3,
        branch_mask=branch_mask,
    )
    expected = torch.tensor([0.5, 1.0]).reshape(2, 1, 1, 1).expand(2, 3, 2, 1)
    torch.testing.assert_close(output["composed_velocity"], expected)
    deficit = torch.relu(0.4 * output["target_velocity"].float().norm(dim=(-2, -1))[:, :, None]
                         - output["branch_velocity"].float().norm(dim=(-2, -1)))
    expected_magnitude = (deficit.square() * branch_mask[:, None]).sum(dim=2).mean()
    torch.testing.assert_close(output["magnitude_loss"], expected_magnitude)
    output["loss"].backward()
    assert head.scale.grad is not None
    torch.manual_seed(1)
    sampled = head.predict_action_compositional([conditions], branch_mask=branch_mask)
    torch.manual_seed(1)
    initial = torch.randn(2, 2, 1)
    torch.testing.assert_close(sampled, initial + torch.tensor([0.5, 1.0]).reshape(2, 1, 1))


def test_inactive_branches_receive_no_velocity_gradient():
    head = DummySharedHead()
    conditions = torch.zeros(4, 1, 1, requires_grad=True)
    output = head.forward_compositional(
        [conditions], torch.ones(1, 2, 1), repeated_diffusion_steps=1,
        branch_mask=torch.tensor([[True, False, False, False]]),
    )
    output["loss"].backward()
    assert conditions.grad[0].abs().sum() > 0
    assert torch.count_nonzero(conditions.grad[1:]) == 0


def test_cosine_sampler_uses_noise_levels_and_variable_step_sizes():
    class TimeDependentHead(DummySharedHead):
        def __init__(self):
            super().__init__()
            self.num_inference_timesteps = 4
            self.num_timestep_buckets = 1000
            self.recorded_times = []

        def predict_velocity(self, vl_embs_list, noisy_actions, timesteps, **kwargs):
            self.recorded_times.append(int(timesteps[0]))
            return (timesteps.float() / self.num_timestep_buckets)[:, None, None].expand_as(noisy_actions)

    head = TimeDependentHead()
    conditions = torch.zeros(4, 1, 1)
    torch.manual_seed(10)
    sampled = head.predict_action_compositional([conditions], noise_schedule="cosine")
    torch.manual_seed(10)
    initial = torch.randn(1, 2, 1)
    assert head.recorded_times == [0, 76, 292, 617]
    levels = [math.cos(math.pi * step / 8) for step in range(5)]
    expected_update = sum(
        (levels[step] - levels[step + 1]) * timestep / 1000
        for step, timestep in enumerate(head.recorded_times)
    )
    torch.testing.assert_close(sampled, initial + expected_update)


if __name__ == "__main__":
    test_compositional_training_shares_flow_state_across_branches()
    test_compositional_sampler_has_source_batch_shape()
    test_magnitude_loss_sums_four_branch_penalties()
    test_flow_matching_loss_sums_trajectory_dimensions()
    test_inactive_branches_do_not_affect_composition_or_loss()
    test_inactive_branches_receive_no_velocity_gradient()
    test_cosine_sampler_uses_noise_levels_and_variable_step_sizes()
    print("test_compositional_flow: PASS")
