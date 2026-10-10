import sys
from pathlib import Path

import torch
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cdp.models.highlighting import LearnablePackedBackground


def test_learned_background_preserves_unmasked_and_receives_gradient():
    torch.manual_seed(0)
    module = LearnablePackedBackground(
        patch_size=2,
        max_views=1,
        background_size=(8, 8),
        low_frequency_grid=(2, 2),
        merge_size=2,
    )
    grid = torch.tensor([[1, 4, 4], [1, 4, 4]])
    pixels = torch.randn(32, module.patch_dim)
    masks = torch.cat([torch.ones(16, module.patch_dim), torch.zeros(16, module.patch_dim)])
    output = module(
        pixels,
        grid,
        masks,
        grid.clone(),
        apply_background=[False, True],
        view_ids=[0, 0],
    )
    assert torch.equal(output[:16], pixels[:16])
    assert not torch.equal(output[16:], pixels[16:])
    output[16:].sum().backward()
    for parameter in (module.background_templates, module.low_frequency_residual):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_background_matches_qwen_pixel_packing():
    module = LearnablePackedBackground(patch_size=2, max_views=2, background_size=(8, 8))
    with torch.no_grad():
        module.background_templates[0].copy_(torch.linspace(0, 1, 192).reshape(3, 8, 8))
        module.background_templates[1].copy_(module.background_templates[0])
        module.low_frequency_residual.fill_(0.05)
    processor = Qwen2VLImageProcessorFast(
        patch_size=2,
        temporal_patch_size=2,
        merge_size=2,
        image_mean=[0.5] * 3,
        image_std=[0.5] * 3,
    )
    rgb = (module.background_templates[0] + module.low_frequency_residual[0]).detach()
    processed = processor(
        images=rgb,
        do_resize=False,
        do_rescale=False,
        return_tensors="pt",
    )
    rows = torch.randn(16, module.patch_dim)
    packed = module._background(rows, 1, 4, 4, 0)
    torch.testing.assert_close(packed, processed["pixel_values"])
    torch.testing.assert_close(packed, module._background(rows + 1, 1, 4, 4, 1))


def test_highlighting_blends_mask_and_background():
    module = LearnablePackedBackground(patch_size=2, max_views=1, background_size=(8, 8))
    with torch.no_grad():
        module.background_templates.fill_(0.75)
        module.low_frequency_residual.fill_(0.1)
    pixels = torch.full((16, module.patch_dim), 0.2)
    masks = torch.full_like(pixels, 0.25)
    grid = torch.tensor([[1, 4, 4]])
    output = module(pixels, grid, masks, grid, apply_background=[True], view_ids=[0])
    expected = 0.25 * 0.2 + 0.75 * ((0.75 + 0.1 - 0.5) / 0.5)
    torch.testing.assert_close(output, torch.full_like(output, expected))


if __name__ == "__main__":
    test_learned_background_preserves_unmasked_and_receives_gradient()
    test_background_matches_qwen_pixel_packing()
    test_highlighting_blends_mask_and_background()
    print("test_learnable_packed_background: PASS")
