"""
E5: Matching algorithm comparison (Table 2).

Setup: ViT-L/16 MAE off-the-shelf, r=8, ImageNet-1k val 50K.
6 algorithms compared on accuracy and throughput.

Run:
  nohup python experiments/E5_matching.py > logs/E5_matching.stdout 2>&1 &
"""

import os
from pathlib import Path
import sys

import torch
from timm.models.vision_transformer import Attention, Block, VisionTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ToMe-main"))

from tome.utils import parse_r
from tome.patch.mae import make_tome_class as make_mae_tome_class

from common import (
    CKPT_DIR,
    build_val_loader,
    evaluate,
    get_device,
    measure_throughput_auto,
    save_results,
    set_seed,
    setup_logging,
)
from mae_models import MAE_CKPTS, build_mae_model
from matching import ALGOS


R_FIXED = 8
INPUT = 224
BS_EVAL = 64


# ---------------------------------------------------------------------------
# Custom attention exposing K and attention map
# ---------------------------------------------------------------------------

class MatchingAttention(Attention):
    def forward(self, x, size=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if size is not None:
            attn = attn + size.log()[:, None, None, :, 0]
        attn = attn.softmax(dim=-1)
        # attention received per token: sum over query dim, mean over heads
        attn_received = attn.mean(dim=1).sum(dim=1)  # [B,N]
        attn_d = self.attn_drop(attn)
        out = (attn_d @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out, k.mean(1), attn_received


class MatchingBlock(Block):
    def _dp1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _dp2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x):
        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
        x_attn, metric, attn_received = self.attn(self.norm1(x), attn_size)
        x = x + self._dp1(x_attn)

        r = self._tome_info["r"].pop(0)
        if r > 0:
            if self._tome_info["size"] is None:
                self._tome_info["size"] = torch.ones(x.shape[0], x.shape[1], 1,
                                                    device=x.device, dtype=x.dtype)
            algo_fn = self._tome_info["algo_fn"]
            x, self._tome_info["size"] = algo_fn(
                x, self._tome_info["size"], metric, attn_received, r,
                class_token=self._tome_info["class_token"],
            )

        x = x + self._dp2(self.mlp(self.norm2(x)))
        return x


def apply_matching_patch(model: VisionTransformer, algo_name: str, prop_attn=False):
    algo_fn, _ = ALGOS[algo_name]
    ToMeVT = make_mae_tome_class(model.__class__)
    model.__class__ = ToMeVT
    model.r = 0
    model._tome_info = {
        "r": model.r,
        "size": None,
        "source": None,
        "trace_source": False,
        "prop_attn": prop_attn,
        "class_token": model.cls_token is not None,
        "distill_token": False,
        "algo_fn": algo_fn,
    }
    for module in model.modules():
        if isinstance(module, Block):
            module.__class__ = MatchingBlock
            module._tome_info = model._tome_info
        elif isinstance(module, Attention):
            module.__class__ = MatchingAttention
    return model


def main():
    seed = int(os.environ.get("SEED", "0"))
    set_seed(seed)
    logger = setup_logging("E5_matching")
    device = get_device()
    logger.info(f"device={device}, ViT-L/16 MAE r={R_FIXED}, prop_attn=False, seed={seed}")

    loader = build_val_loader(input_size=INPUT, batch_size=BS_EVAL,
                              num_workers=8, crop_pct=0.875)

    all_results = []
    for algo_name in ["prune_random", "prune_attn", "kmeans2", "kmeans5",
                      "greedy", "bipartite"]:
        logger.info(f"\n--- {algo_name} ---")
        ckpt = CKPT_DIR / MAE_CKPTS["large"]
        model = build_mae_model("large", str(ckpt), global_pool=True)
        apply_matching_patch(model, algo_name, prop_attn=False)
        model.r = R_FIXED
        model = model.to(device).eval()

        acc = evaluate(model, loader, device=device)
        thr, bs_thr = measure_throughput_auto(
            model, input_size=(3, INPUT, INPUT), device=device,
            use_fp16=False, logger=logger,
        )

        row = {
            "algo": algo_name,
            "style": ALGOS[algo_name][1],
            "r": R_FIXED,
            "acc": round(acc, 3),
            "throughput": round(thr, 2),
            "throughput_batch": bs_thr,
        }
        all_results.append(row)
        logger.info(f"  acc={acc:.3f}  throughput={thr:.1f} im/s (bs={bs_thr})")

        del model
        torch.cuda.empty_cache() if device == "cuda" else None
        save_results("E5_matching", all_results)

    save_results("E5_matching", all_results)
    logger.info(f"\nE5 done. Results -> results/E5_matching.json")


if __name__ == "__main__":
    main()
