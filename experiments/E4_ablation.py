"""
E4: Ablation reproduction (Table 1).

Default: ViT-L/16 MAE off-the-shelf, r=8, ImageNet-1k val 50K (paper Table 1).
22 configs: 6 sub-tables × design choices.

Env vars:
  MODEL_SIZE=base|large   default "large" (paper). "base" for r=16 stress test.
                          ViT-L/16 = 24 layers → r_max=8 (paper at max already).
                          ViT-B/16 = 12 layers → r_max=16, lets us stress at r=16.
  R_VALUE=N               default 8. r per layer.
  ONLY_AUGREG=1           only Table 1f augreg rows (B.2 selective run).

Output naming:
  default (large, r=8) → results/E4_ablation.json (+ E4_ablation_augreg.json)
  otherwise            → results/E4_ablation_<size>_r<N>.json

Run examples:
  # paper default (already done)
  nohup python experiments/E4_ablation.py > logs/E4_ablation.stdout 2>&1 &

  # stress test: ViT-B MAE r=16 (max for B/16, exposes design-choice gaps)
  MODEL_SIZE=base R_VALUE=16 nohup python experiments/E4_ablation.py \\
      > logs/E4_ablation_base_r16.stdout 2>&1 &
"""

import os
from pathlib import Path
import sys

import timm
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ToMe-main"))

from ablation import DEFAULT_CFG, apply_ablation_patch
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


R_FIXED = int(os.environ.get("R_VALUE", "8"))
MODEL_SIZE = os.environ.get("MODEL_SIZE", "large")  # "base" | "large"
INPUT = 224
BS_EVAL = 64

# AugReg backbone (Table 1f) must match MAE size (same #layers/tokens)
AUGREG_NAME = {"base": "vit_base_patch16_224", "large": "vit_large_patch16_224"}[MODEL_SIZE]

# 22 configs: (sub_table, label, cfg_overrides, backbone)
# backbone: "mae" (default) or "augreg" (only Table 1f)
CONFIGS = [
    # (a) feature
    ("a", "x_pre",   {"feature": "x_pre"}, "mae"),
    ("a", "x",       {"feature": "x"},     "mae"),
    ("a", "k",       {"feature": "k"},     "mae"),  # default
    ("a", "q",       {"feature": "q"},     "mae"),
    ("a", "v",       {"feature": "v"},     "mae"),
    # (b) distance
    ("b", "euclidean", {"distance": "euclidean"}, "mae"),
    ("b", "cosine",    {"distance": "cosine"},    "mae"),  # default
    ("b", "dot",       {"distance": "dot"},       "mae"),
    ("b", "softmax",   {"distance": "softmax"},   "mae"),
    # (c) head aggregation
    ("c", "concat",  {"head_agg": "concat"}, "mae"),
    ("c", "mean",    {"head_agg": "mean"},   "mae"),  # default
    # (d) combine
    ("d", "keep_one", {"combine": "keep_one"}, "mae"),
    ("d", "max",      {"combine": "max"},      "mae"),
    ("d", "avg",      {"combine": "avg"},      "mae"),
    ("d", "wavg",     {"combine": "wavg"},     "mae"),  # default
    # (e) partition
    ("e", "sequential",  {"partition": "sequential"},  "mae"),
    ("e", "alternating", {"partition": "alternating"}, "mae"),  # default
    ("e", "random",      {"partition": "random"},      "mae"),
    # (f) proportional attention
    ("f", "mae_no_prop",  {"prop_attn": False}, "mae"),     # default for MAE
    ("f", "mae_prop",     {"prop_attn": True},  "mae"),
    ("f", "augreg_no_prop", {"prop_attn": False}, "augreg"),
    ("f", "augreg_prop",    {"prop_attn": True},  "augreg"),  # default for AugReg
]


def load_backbone(kind: str, device: str):
    if kind == "mae":
        ckpt = CKPT_DIR / MAE_CKPTS[MODEL_SIZE]
        m = build_mae_model(MODEL_SIZE, str(ckpt), global_pool=True)
        is_mae = True
    elif kind == "augreg":
        m = timm.create_model(AUGREG_NAME, pretrained=True)
        is_mae = False
    else:
        raise ValueError(kind)
    return m.to(device).eval(), is_mae


def main():
    seed = int(os.environ.get("SEED", "0"))
    set_seed(seed)
    logger = setup_logging(f"E4_ablation_{MODEL_SIZE}_r{R_FIXED}")
    device = get_device()
    only_augreg = os.environ.get("ONLY_AUGREG", "0") == "1"
    # output naming: default L/r=8 keeps old names; otherwise tagged
    default_run = (MODEL_SIZE == "large" and R_FIXED == 8)
    tag = "" if default_run else f"_{MODEL_SIZE}_r{R_FIXED}"
    out_name = f"E4_ablation{tag}" + ("_augreg" if only_augreg else "")
    logger.info(f"device={device}, ViT-{MODEL_SIZE[0].upper()}/16, r={R_FIXED}, "
                f"augreg_name={AUGREG_NAME}, 50K val, only_augreg={only_augreg}, "
                f"out={out_name}.json, seed={seed}")

    # MAE backbone: ImageNet norm, crop_pct=0.875
    loader_mae = build_val_loader(input_size=INPUT, batch_size=BS_EVAL,
                                  num_workers=8, crop_pct=0.875)
    # AugReg backbone (Table 1f): mean/std=(0.5,)*3, crop_pct=0.9 from default_cfg.
    # Built lazily on first augreg config.
    loader_augreg = None

    all_results = []
    for sub, label, overrides, backbone in CONFIGS:
        if only_augreg and backbone != "augreg":
            continue
        logger.info(f"\n--- Table 1{sub} | {label} ({backbone}) ---")
        if backbone == "augreg":
            if loader_augreg is None:
                ref = timm.create_model(AUGREG_NAME, pretrained=True)
                c = ref.default_cfg
                del ref
                logger.info(f"  augreg loader: mean={c['mean']} std={c['std']} "
                            f"crop_pct={c['crop_pct']}")
                loader_augreg = build_val_loader(
                    input_size=INPUT, batch_size=BS_EVAL, num_workers=8,
                    mean=list(c["mean"]), std=list(c["std"]), crop_pct=c["crop_pct"])
            loader = loader_augreg
        else:
            loader = loader_mae
        cfg = dict(DEFAULT_CFG)
        # AugReg default has prop_attn=True; only override if explicit
        if backbone == "augreg" and "prop_attn" not in overrides:
            cfg["prop_attn"] = True
        cfg.update(overrides)
        logger.info(f"  cfg={cfg}")

        model, is_mae = load_backbone(backbone, device)
        apply_ablation_patch(model, cfg, is_mae=is_mae)
        model.r = R_FIXED

        acc = evaluate(model, loader, device=device)
        thr, bs_thr = measure_throughput_auto(
            model, input_size=(3, INPUT, INPUT), device=device,
            use_fp16=False, logger=logger,
        )

        row = {
            "sub_table": sub,
            "label": label,
            "backbone": backbone,
            "cfg": {k: cfg[k] for k in ["feature", "distance", "head_agg",
                                         "combine", "partition", "prop_attn"]},
            "r": R_FIXED,
            "acc": round(acc, 3),
            "throughput": round(thr, 2),
            "throughput_batch": bs_thr,
        }
        all_results.append(row)
        logger.info(f"  acc={acc:.3f}  throughput={thr:.1f} im/s (bs={bs_thr})")

        del model
        torch.cuda.empty_cache() if device == "cuda" else None
        save_results(out_name, all_results)

    save_results(out_name, all_results)
    logger.info(f"\nE4 done. {len(all_results)} configs. Results -> results/{out_name}.json")


if __name__ == "__main__":
    main()
