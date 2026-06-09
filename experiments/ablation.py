"""
Ablation patch for E4 (Table 1 reproduction).

Provides a knob-able ToMe patch over timm/MAE ViT with configurable:
  - feature:   x_pre | x | k | q | v        (default k)
  - distance:  cosine | euclidean | dot | softmax   (default cosine)
  - head_agg:  mean | concat                (default mean)
  - combine:   wavg | avg | max | keep_one  (default wavg)
  - partition: alternating | sequential | random   (default alternating)
  - prop_attn: bool                         (default False for MAE)

Does NOT modify the installed tome package. Builds on tome's MAE forward_features
(global pooling proportional to size) by reusing the MAE ToMeVisionTransformer.
"""

import math
from typing import Callable, Tuple

import torch
from timm.models.vision_transformer import Attention, Block, VisionTransformer

from tome.utils import parse_r
from tome.merge import merge_wavg, merge_source
from tome.patch.mae import make_tome_class as make_mae_tome_class
from tome.patch.timm import make_tome_class as make_timm_tome_class


DEFAULT_CFG = {
    "feature": "k",
    "distance": "cosine",
    "head_agg": "mean",
    "combine": "wavg",
    "partition": "alternating",
    "prop_attn": False,
}


# ---------------------------------------------------------------------------
# Matching with configurable distance + partition
# ---------------------------------------------------------------------------

def _compute_scores(a: torch.Tensor, b: torch.Tensor, distance: str) -> torch.Tensor:
    """Higher score = more similar (will be merged)."""
    if distance == "cosine":
        a = a / a.norm(dim=-1, keepdim=True)
        b = b / b.norm(dim=-1, keepdim=True)
        return a @ b.transpose(-1, -2)
    if distance == "dot":
        return a @ b.transpose(-1, -2)
    if distance == "softmax":
        return (a @ b.transpose(-1, -2)).softmax(dim=-1)
    if distance == "euclidean":
        # negative L2 distance: closer -> higher score
        d = torch.cdist(a, b, p=2)
        return -d
    raise ValueError(f"unknown distance {distance}")


def ablation_matching(
    metric: torch.Tensor,
    r: int,
    distance: str,
    partition: str,
    class_token: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Returns (a_idx, b_idx, src_idx, dst_idx_in_b, unm_idx, r) needed to merge.
    Generalizes bipartite_soft_matching to arbitrary partition + distance.
    """
    B, N, _ = metric.shape
    device = metric.device

    if partition == "alternating":
        a_idx = torch.arange(0, N, 2, device=device)
        b_idx = torch.arange(1, N, 2, device=device)
    elif partition == "sequential":
        half = N // 2
        a_idx = torch.arange(0, half, device=device)
        b_idx = torch.arange(half, N, device=device)
    elif partition == "random":
        perm = torch.randperm(N, device=device)
        a_idx = perm[0::2]
        b_idx = perm[1::2]
    else:
        raise ValueError(f"unknown partition {partition}")

    # Align lengths so each A token has a candidate in B
    na = a_idx.shape[0]
    a_idx_e = a_idx.view(1, -1, 1).expand(B, -1, metric.shape[-1])
    b_idx_e = b_idx.view(1, -1, 1).expand(B, -1, metric.shape[-1])
    a = metric.gather(1, a_idx_e)
    b = metric.gather(1, b_idx_e)

    r = min(r, na)  # cannot merge more than |A| tokens
    if r <= 0:
        return None

    with torch.no_grad():
        scores = _compute_scores(a, b, distance)  # [B, na, nb]

        # protect class token (assumed at flat index 0, which lands in A for
        # alternating/sequential; for random we find it)
        if class_token:
            cls_in_a = (a_idx == 0).nonzero(as_tuple=True)[0]
            if cls_in_a.numel() > 0:
                scores[:, cls_in_a[0], :] = -math.inf

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)

        merge_a_local = edge_idx[:, :r]      # which A tokens get merged (local idx)
        unm_a_local = edge_idx[:, r:]        # which A tokens stay
        dst_local = node_idx.gather(1, merge_a_local)  # their B targets (local idx)

        # Keep cls token at output index 0: MAE global pool assumes cls at [:,0].
        # cls is protected (-inf score) so it is always in unm_a_local; move it first.
        if class_token:
            cls_in_a = (a_idx == 0).nonzero(as_tuple=True)[0]
            if cls_in_a.numel() > 0:
                cls_local = cls_in_a[0].item()
                mask = unm_a_local[0] != cls_local
                rest = unm_a_local[:, mask]
                cls_col = torch.full((unm_a_local.shape[0], 1), cls_local,
                                     device=unm_a_local.device, dtype=unm_a_local.dtype)
                unm_a_local = torch.cat([cls_col, rest], dim=1)

    return {
        "a_idx": a_idx, "b_idx": b_idx,
        "merge_a_local": merge_a_local, "unm_a_local": unm_a_local,
        "dst_local": dst_local, "r": r,
    }


def apply_merge(x: torch.Tensor, m: dict, combine: str,
                size: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Merge tokens of x ([B,N,C]) given match dict m and combine method.
    Returns (merged_x, new_size).
    Output token order: [unmerged_A ; all_B(with merges applied)].
    """
    B, N, C = x.shape
    device = x.device
    a_idx, b_idx = m["a_idx"], m["b_idx"]
    merge_a_local, unm_a_local, dst_local = m["merge_a_local"], m["unm_a_local"], m["dst_local"]
    r = m["r"]
    nb = b_idx.shape[0]

    if size is None:
        size = torch.ones(B, N, 1, device=device, dtype=x.dtype)

    xa = x.gather(1, a_idx.view(1, -1, 1).expand(B, -1, C))
    xb = x.gather(1, b_idx.view(1, -1, 1).expand(B, -1, C))
    sa = size.gather(1, a_idx.view(1, -1, 1).expand(B, -1, 1))
    sb = size.gather(1, b_idx.view(1, -1, 1).expand(B, -1, 1))

    src = xa.gather(1, merge_a_local.unsqueeze(-1).expand(B, r, C))
    src_size = sa.gather(1, merge_a_local.unsqueeze(-1).expand(B, r, 1))
    unm = xa.gather(1, unm_a_local.unsqueeze(-1).expand(B, unm_a_local.shape[1], C))
    unm_size = sa.gather(1, unm_a_local.unsqueeze(-1).expand(B, unm_a_local.shape[1], 1))

    dst_idx = dst_local.unsqueeze(-1)  # [B, r, 1]

    if combine == "keep_one":
        # drop merged src entirely; keep B as is
        merged_b = xb
        merged_b_size = sb
    elif combine == "wavg":
        # size-weighted sum then divide; start from (xb*sb) so dst's own size counts
        dst = (xb * sb).scatter_reduce(1, dst_idx.expand(B, r, C), src * src_size,
                                       reduce="sum", include_self=True)
        dst_s = sb.scatter_reduce(1, dst_idx.expand(B, r, 1), src_size,
                                  reduce="sum", include_self=True)
        merged_b = dst / dst_s
        merged_b_size = dst_s
    elif combine == "avg":
        # unweighted mean (scatter_reduce mean)
        merged_b = xb.scatter_reduce(1, dst_idx.expand(B, r, C), src, reduce="mean",
                                     include_self=True)
        merged_b_size = sb.scatter_reduce(1, dst_idx.expand(B, r, 1), src_size, reduce="sum")
    elif combine == "max":
        merged_b = xb.scatter_reduce(1, dst_idx.expand(B, r, C), src, reduce="amax",
                                     include_self=True)
        merged_b_size = sb.scatter_reduce(1, dst_idx.expand(B, r, 1), src_size, reduce="sum")
    else:
        raise ValueError(f"unknown combine {combine}")

    out = torch.cat([unm, merged_b], dim=1)
    out_size = torch.cat([unm_size, merged_b_size], dim=1)
    return out, out_size


# ---------------------------------------------------------------------------
# Attention that exposes x_pre, x, q, k, v
# ---------------------------------------------------------------------------

class AblationAttention(Attention):
    def forward(self, x, size=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, N, d]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if size is not None:
            attn = attn + size.log()[:, None, None, :, 0]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        self._qkv_cache = {"q": q, "k": k, "v": v}  # [B,H,N,d]
        return out, k.mean(1)


def _agg_heads(t: torch.Tensor, head_agg: str) -> torch.Tensor:
    """t: [B, H, N, d] -> [B, N, D]."""
    if head_agg == "mean":
        return t.mean(1)
    if head_agg == "concat":
        B, H, N, d = t.shape
        return t.permute(0, 2, 1, 3).reshape(B, N, H * d)
    raise ValueError(f"unknown head_agg {head_agg}")


class AblationBlock(Block):
    def _dp1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _dp2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x):
        cfg = self._tome_info["cfg"]
        attn_size = self._tome_info["size"] if cfg["prop_attn"] else None

        x_pre = x
        x_attn, _ = self.attn(self.norm1(x), attn_size)
        x = x + self._dp1(x_attn)

        r = self._tome_info["r"].pop(0)
        if r > 0:
            feat = cfg["feature"]
            if feat == "x_pre":
                metric = x_pre
            elif feat == "x":
                metric = x
            else:  # q/k/v
                metric = _agg_heads(self.attn._qkv_cache[feat], cfg["head_agg"])

            m = ablation_matching(metric, r, cfg["distance"], cfg["partition"],
                                  class_token=self._tome_info["class_token"])
            if m is not None:
                x, self._tome_info["size"] = apply_merge(
                    x, m, cfg["combine"], self._tome_info["size"])

        x = x + self._dp2(self.mlp(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# Patch entry
# ---------------------------------------------------------------------------

def apply_ablation_patch(model: VisionTransformer, cfg: dict, is_mae: bool = True):
    """Patch model with ablation-configurable ToMe. cfg overrides DEFAULT_CFG."""
    full_cfg = dict(DEFAULT_CFG)
    full_cfg.update(cfg)

    if is_mae:
        ToMeVT = make_mae_tome_class(model.__class__)
    else:
        ToMeVT = make_timm_tome_class(model.__class__)

    model.__class__ = ToMeVT
    model.r = 0
    model._tome_info = {
        "r": model.r,
        "size": None,
        "source": None,
        "trace_source": False,
        "prop_attn": full_cfg["prop_attn"],
        "class_token": model.cls_token is not None,
        "distill_token": False,
        "cfg": full_cfg,
    }

    for module in model.modules():
        if isinstance(module, Block):
            module.__class__ = AblationBlock
            module._tome_info = model._tome_info
        elif isinstance(module, Attention):
            module.__class__ = AblationAttention
    return model
