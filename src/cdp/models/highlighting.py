from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class LearnablePackedBackground(nn.Module):
    def __init__(
        self,
        patch_size: int,
        temporal_patch_size: int = 2,
        max_views: int = 3,
        background_size: tuple[int, int] = (224, 224),
        low_frequency_grid: tuple[int, int] = (8, 8),
        merge_size: int = 2,
        image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
        image_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
    ):
        super().__init__()
        self.patch_size = int(patch_size)
        self.temporal_patch_size = int(temporal_patch_size)
        self.patch_dim = 3 * self.temporal_patch_size * self.patch_size**2
        self.max_views = int(max_views)
        self.background_size = tuple(int(value) for value in background_size)
        self.low_frequency_grid = tuple(int(value) for value in low_frequency_grid)
        self.merge_size = int(merge_size)
        self.background_templates = nn.Parameter(
            torch.full((self.max_views, 3, *self.background_size), 0.5)
        )
        self.low_frequency_residual = nn.Parameter(
            torch.zeros(1, 3, *self.low_frequency_grid)
        )
        self.register_buffer("image_mean", torch.tensor(image_mean).reshape(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor(image_std).reshape(1, 3, 1, 1))

    def _pack_grid(self, grid: torch.Tensor):
        grid_t, grid_h, grid_w, patch_dim = grid.shape
        merge = self.merge_size
        return (
            grid.reshape(
                grid_t,
                grid_h // merge,
                merge,
                grid_w // merge,
                merge,
                patch_dim,
            )
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(grid_t * grid_h * grid_w, patch_dim)
        )

    def _background(self, rows: torch.Tensor, grid_t: int, grid_h: int, grid_w: int, view_id: int):
        if view_id < 0 or view_id >= self.max_views:
            raise ValueError(f"view_id={view_id} outside configured max_views={self.max_views}")
        size = (grid_h * self.patch_size, grid_w * self.patch_size)
        template = F.interpolate(
            self.background_templates[view_id : view_id + 1],
            size=size,
            mode="bilinear",
            align_corners=False,
        )
        residual = F.interpolate(
            self.low_frequency_residual,
            size=size,
            mode="bilinear",
            align_corners=False,
        )
        background = ((template + residual - self.image_mean) / self.image_std).to(dtype=rows.dtype)
        patches = background[:, None].expand(grid_t, self.temporal_patch_size, -1, -1, -1)
        patches = patches.reshape(
            grid_t, self.temporal_patch_size, 3,
            grid_h, self.patch_size, grid_w, self.patch_size,
        ).permute(0, 3, 5, 2, 1, 4, 6)
        grid = patches.reshape(grid_t, grid_h, grid_w, self.patch_dim)
        return self._pack_grid(grid)

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        packed_masks: torch.Tensor,
        mask_grid_thw: torch.Tensor,
        apply_background: list[bool],
        view_ids: list[int],
    ) -> torch.Tensor:
        if pixel_values.shape != packed_masks.shape:
            raise ValueError(
                f"Packed image/mask shape mismatch: {pixel_values.shape} vs {packed_masks.shape}"
            )
        if not torch.equal(image_grid_thw.cpu(), mask_grid_thw.cpu()):
            raise ValueError("Image and semantic mask processor grids differ")
        image_count = image_grid_thw.shape[0]
        if len(apply_background) != image_count or len(view_ids) != image_count:
            raise ValueError(
                f"Expected {image_count} flags/view ids, got {len(apply_background)}/{len(view_ids)}"
            )

        output_segments = []
        offset = 0
        for image_id, grid in enumerate(image_grid_thw.tolist()):
            grid_t, grid_h, grid_w = (int(value) for value in grid)
            row_count = grid_t * grid_h * grid_w
            rows = pixel_values[offset : offset + row_count]
            if apply_background[image_id]:
                mask = packed_masks[offset : offset + row_count].to(
                    device=rows.device, dtype=rows.dtype
                ).clamp_(0, 1)
                background = self._background(
                    rows, grid_t, grid_h, grid_w, int(view_ids[image_id])
                )
                rows = mask * rows + (1 - mask) * background
            output_segments.append(rows)
            offset += row_count
        if offset != pixel_values.shape[0]:
            raise ValueError(f"Packed grid consumes {offset} rows, pixel_values has {pixel_values.shape[0]}")
        return torch.cat(output_segments, dim=0)
