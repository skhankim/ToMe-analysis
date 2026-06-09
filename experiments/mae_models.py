"""
MAE ViT model definitions (from facebookresearch/mae models_vit.py).

These subclass timm's VisionTransformer with a global_pool option and fc_norm,
matching the official MAE finetuned checkpoints. tome.patch.mae overrides
forward_features, so we only need the structure + builders.
"""

from functools import partial

import timm.models.vision_transformer
import torch
import torch.nn as nn


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ViT with support for global average pooling (MAE finetune style)."""

    def __init__(self, global_pool=False, **kwargs):
        super().__init__(**kwargs)
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs["norm_layer"]
            embed_dim = kwargs["embed_dim"]
            self.fc_norm = norm_layer(embed_dim)
            del self.norm  # remove the original norm


def vit_base_patch16(**kwargs):
    return VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs,
    )


def vit_large_patch16(**kwargs):
    return VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs,
    )


def vit_huge_patch14(**kwargs):
    return VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs,
    )


MAE_BUILDERS = {
    "base": vit_base_patch16,
    "large": vit_large_patch16,
    "huge": vit_huge_patch14,
}

# Default finetuned checkpoint filenames (in checkpoints/)
MAE_CKPTS = {
    "base": "mae_finetuned_vit_base.pth",
    "large": "mae_finetuned_vit_large.pth",
    "huge": "mae_finetuned_vit_huge.pth",
}


def build_mae_model(size: str, ckpt_path: str, num_classes: int = 1000,
                    global_pool: bool = True) -> VisionTransformer:
    """Build MAE ViT of given size and load finetuned weights."""
    model = MAE_BUILDERS[size](num_classes=num_classes, global_pool=global_pool)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[mae_models] missing keys: {missing}")
    if unexpected:
        print(f"[mae_models] unexpected keys: {unexpected}")
    return model
