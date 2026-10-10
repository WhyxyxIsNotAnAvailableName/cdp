# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modified by [Jinhui YE/ HKUST] in [2026].

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from cdp.models.action_encoder import SinusoidalPositionalEncoding, swish
from cdp.models.dit import DiT

def _flow_time_schedule(num_steps, device, noise_schedule):
    progress = torch.linspace(0, 1, num_steps + 1, device=device)
    if noise_schedule == 'cosine':
        noise_levels = torch.cos(progress * (torch.pi / 2))
        (noise_levels[0], noise_levels[-1]) = (1.0, 0.0)
        return 1 - noise_levels
    if noise_schedule == 'linear':
        return progress
    raise ValueError(f'Unknown noise schedule: {noise_schedule}')

def _composition_weights(batch_size, branch_count, device, dtype, branch_weights, branch_mask):
    if branch_weights is None:
        weights = torch.ones(branch_count, device=device, dtype=dtype)
    else:
        weights = torch.as_tensor(branch_weights, device=device, dtype=dtype)
        if weights.shape != (branch_count,) or torch.any(weights < 0):
            raise ValueError(f'Invalid branch_weights shape/values: {weights}')
    active = torch.ones(batch_size, branch_count, device=device, dtype=torch.bool) if branch_mask is None else torch.as_tensor(branch_mask, device=device, dtype=torch.bool)
    weights = weights[None] * active
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(torch.finfo(dtype).eps)
    return (weights, active)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, output_dim=2048):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer2(F.relu(self.layer1(x)))

class ActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size=1024):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.layer1 = nn.Linear(action_dim, hidden_size)
        self.layer2 = nn.Linear(2 * hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps):
        (B, T, _) = actions.shape
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError('Expected `timesteps` to have shape (B,) so we can replicate across T.')
        a_emb = self.layer1(actions)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.layer2(x))
        x = self.layer3(x)
        return x
DiTConfig = {'num_layers': 36, 'input_embedding_dim': 2048, 'attention_head_dim': 64, 'num_attention_heads': 32}

class LayerwiseFlowmatchingActionHead(nn.Module):
    def __init__(self, global_config, **kwargs):
        super().__init__()
        action_config = global_config.framework.action_model
        diffusion_model_cfg = action_config.diffusion_model_cfg
        for (k, v) in DiTConfig.items():
            if diffusion_model_cfg.get(k, None) is None:
                diffusion_model_cfg[k] = v
        _DIT_NON_KWARGS = {'action_dit_hidden_dim'}
        diffusion_model_cfg_kwargs = {k: v for (k, v) in diffusion_model_cfg.items() if k not in _DIT_NON_KWARGS}
        self.input_embedding_dim = diffusion_model_cfg_kwargs['input_embedding_dim']
        self.model = DiT(**diffusion_model_cfg_kwargs)
        self.dit_out_hidden_size = self.input_embedding_dim
        self.action_dim = action_config.action_dim
        self.action_horizon = int(action_config.action_horizon)
        self.num_inference_timesteps = action_config.num_inference_timesteps
        self.action_encoder = ActionEncoder(action_dim=action_config.action_dim, hidden_size=self.input_embedding_dim)
        self.action_decoder = MLP(input_dim=self.input_embedding_dim, hidden_dim=1024, output_dim=self.action_dim)
        self.future_tokens = nn.Embedding(action_config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)
        if action_config.add_pos_embed:
            self.position_embedding = nn.Embedding(action_config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        self.beta_dist = Beta(action_config.noise_beta_alpha, action_config.noise_beta_beta)
        self.num_timestep_buckets = action_config.num_timestep_buckets
        self.config = action_config

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return ((self.config.noise_s - sample) / self.config.noise_s).clamp(0, 1)

    def predict_velocity(self, vl_embs_list: list, noisy_actions: torch.Tensor, timesteps: torch.Tensor, encoder_attention_mask=None) -> torch.Tensor:
        batch_size = noisy_actions.shape[0]
        device = noisy_actions.device
        action_features = self.action_encoder(noisy_actions, timesteps)
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)
        sa_embs = torch.cat((future_tokens, action_features), dim=1)
        model_output = self.model(hidden_states=sa_embs, encoder_hidden_states=vl_embs_list, timestep=timesteps, encoder_attention_mask=encoder_attention_mask, return_pre_output=True)
        pred = self.action_decoder(model_output)
        return pred[:, -noisy_actions.shape[1]:]

    def forward_compositional(self, vl_embs_list: list, actions: torch.Tensor, branch_count: int=4, repeated_diffusion_steps: int=8, branch_weights: torch.Tensor | list | tuple | None=None, magnitude_alpha: float=0.4, magnitude_loss_weight: float=1.0, encoder_attention_mask=None, branch_mask: torch.Tensor | None=None) -> dict:
        if branch_count <= 0 or repeated_diffusion_steps <= 0:
            raise ValueError('branch_count and repeated_diffusion_steps must be positive')
        condition_batch = vl_embs_list[0].shape[0]
        if condition_batch % branch_count:
            raise ValueError(f'Condition batch {condition_batch} is not divisible by branch_count={branch_count}')
        batch_size = condition_batch // branch_count
        if actions.shape[0] != batch_size:
            raise ValueError(f'Actions batch {actions.shape[0]} != inferred source batch {batch_size}')
        (device, dtype) = (actions.device, actions.dtype)
        repeats = repeated_diffusion_steps
        action_targets = actions[:, None].expand(-1, repeats, -1, -1)
        noise = torch.randn_like(action_targets)
        t = self.sample_time(batch_size * repeats, device=device, dtype=dtype).reshape(batch_size, repeats)
        t_broadcast = t[..., None, None]
        noisy = (1 - t_broadcast) * noise + t_broadcast * action_targets
        target_velocity = action_targets - noise
        timesteps = (t * self.num_timestep_buckets).long()
        noisy_branches = noisy[:, :, None].expand(-1, -1, branch_count, -1, -1).reshape(batch_size * repeats * branch_count, *actions.shape[1:])
        timestep_branches = timesteps[:, :, None].expand(-1, -1, branch_count).reshape(-1)
        expanded_vl = []
        for hidden in vl_embs_list:
            hidden = hidden.reshape(batch_size, branch_count, *hidden.shape[1:])
            hidden = hidden[:, None].expand(-1, repeats, -1, *[-1] * (hidden.ndim - 2))
            expanded_vl.append(hidden.reshape(batch_size * repeats * branch_count, *hidden.shape[3:]))
        expanded_mask = None
        if encoder_attention_mask is not None:
            mask = encoder_attention_mask.reshape(batch_size, branch_count, -1)
            expanded_mask = mask[:, None].expand(-1, repeats, -1, -1).reshape(batch_size * repeats * branch_count, -1)
        branch_velocity = self.predict_velocity(expanded_vl, noisy_branches, timestep_branches, encoder_attention_mask=expanded_mask).reshape(batch_size, repeats, branch_count, *actions.shape[1:])
        (weights, active) = _composition_weights(batch_size, branch_count, device, dtype, branch_weights, branch_mask)
        composed_velocity = (branch_velocity * weights[:, None, :, None, None]).sum(dim=2)
        flow_matching_loss = (composed_velocity - target_velocity).square().sum(dim=(-2, -1)).mean()
        target_norm = torch.linalg.vector_norm(target_velocity.float(), dim=(-2, -1))
        branch_norm = torch.linalg.vector_norm(branch_velocity.float(), dim=(-2, -1))
        magnitude_floor = float(magnitude_alpha) * target_norm[:, :, None]
        magnitude_deficit = torch.relu(magnitude_floor - branch_norm) * active[:, None]
        magnitude_loss = magnitude_deficit.square().sum(dim=2).mean()
        flat_velocity = branch_velocity.float().flatten(start_dim=-2)
        normalized_velocity = torch.nn.functional.normalize(flat_velocity, dim=-1)
        branch_cosine = torch.einsum('brkd,brld->brkl', normalized_velocity, normalized_velocity)
        total_loss = flow_matching_loss + float(magnitude_loss_weight) * magnitude_loss
        return {'loss': total_loss, 'flow_matching_loss': flow_matching_loss, 'magnitude_loss': magnitude_loss, 'branch_velocity': branch_velocity, 'composed_velocity': composed_velocity, 'target_velocity': target_velocity, 'timesteps': timesteps, 'branch_norm': branch_norm * active[:, None], 'composed_norm': torch.linalg.vector_norm(composed_velocity.float(), dim=(-2, -1)), 'branch_cosine': branch_cosine, 'magnitude_active_fraction': (magnitude_deficit > 0).float().sum() / (active.sum() * repeats)}

    @torch.no_grad()
    def predict_action_compositional(self, vl_embs_list: list, branch_count: int=4, branch_weights: torch.Tensor | list | tuple | None=None, encoder_attention_mask=None, branch_mask: torch.Tensor | None=None, noise_schedule: str='cosine') -> torch.Tensor:
        condition_batch = vl_embs_list[0].shape[0]
        if condition_batch % branch_count:
            raise ValueError(f'Condition batch {condition_batch} is not divisible by branch_count={branch_count}')
        batch_size = condition_batch // branch_count
        (device, dtype) = (vl_embs_list[0].device, vl_embs_list[0].dtype)
        (weights, _) = _composition_weights(batch_size, branch_count, device, dtype, branch_weights, branch_mask)
        actions = torch.randn(batch_size, self.action_horizon, self.action_dim, device=device, dtype=dtype)
        num_steps = self.num_inference_timesteps
        times = _flow_time_schedule(num_steps, device, noise_schedule)
        for step in range(num_steps):
            timestep = int(times[step].item() * self.num_timestep_buckets)
            dt = (times[step + 1] - times[step]).to(dtype=actions.dtype)
            timestep_branches = torch.full((batch_size * branch_count,), timestep, device=device, dtype=torch.long)
            noisy_branches = actions[:, None].expand(-1, branch_count, -1, -1).reshape(batch_size * branch_count, self.action_horizon, self.action_dim)
            velocity = self.predict_velocity(vl_embs_list, noisy_branches, timestep_branches, encoder_attention_mask=encoder_attention_mask).reshape(batch_size, branch_count, self.action_horizon, self.action_dim)
            actions = actions + dt * (velocity * weights[:, :, None, None]).sum(dim=1)
        return actions

def get_action_model(config=None):
    return LayerwiseFlowmatchingActionHead(global_config=config)
