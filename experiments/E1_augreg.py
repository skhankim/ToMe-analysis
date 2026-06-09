"""
E1: AugReg sweep (Figure 3a / Table 8 reproduction).

Models (from timm, AugReg pretrained):
  - vit_tiny_patch16_224       r in 0..16 (17)
  - vit_small_patch16_224      r in 0..16 (17)
  - vit_base_patch16_224       r in 0..16 (17)
  - vit_large_patch16_224      r in 0..8  (9)
  - vit_large_patch16_384      r in [0,5,10,15,20,23] (6)

Total: 66 evaluations.

Run:
  nohup python experiments/E1_augreg.py > logs/E1_augreg.stdout 2>&1 &

Result -> results/E1_augreg.json, log -> logs/E1_augreg.log
"""

from pathlib import Path

import timm
import torch

import os
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ToMe-main"))

import tome
from common import (
    build_val_loader,
    evaluate,
    get_device,
    measure_throughput_auto,
    measure_throughput_fixed,
    save_results,
    set_seed,
    setup_logging,
)


CONFIGS = [
    {"name": "vit_tiny_patch16_224",  "input": 224, "rs": list(range(17)),       "bs_eval": 256},
    {"name": "vit_small_patch16_224", "input": 224, "rs": list(range(17)),       "bs_eval": 256},
    {"name": "vit_base_patch16_224",  "input": 224, "rs": list(range(17)),       "bs_eval": 128},
    {"name": "vit_large_patch16_224", "input": 224, "rs": list(range(9)),        "bs_eval": 64},
    {"name": "vit_large_patch16_384", "input": 384, "rs": [0, 5, 10, 15, 20, 23],"bs_eval": 32},
]


def main():
    seed = int(os.environ.get("SEED", "0"))
    set_seed(seed)
    logger = setup_logging("E1_augreg")
    device = get_device()
    logger.info(f"device={device}, fp32, prop_attn=True, seed={seed}")

    all_results = []

    for cfg in CONFIGS:
        name = cfg["name"]
        input_size = cfg["input"]
        rs = cfg["rs"]
        bs_eval = cfg["bs_eval"]

        # AugReg models use mean/std=(0.5,0.5,0.5) and crop_pct from default_cfg
        # (NOT ImageNet norm / 256-224 ratio). Read it directly to be faithful.
        ref = timm.create_model(name, pretrained=True)
        cfg = ref.default_cfg
        mean = list(cfg["mean"])
        std = list(cfg["std"])
        crop_pct = cfg["crop_pct"]
        del ref
        logger.info(f"\n=== {name} @ {input_size} | r in {rs} | "
                    f"mean={mean} std={std} crop_pct={crop_pct} ===")
        loader = build_val_loader(input_size=input_size, batch_size=bs_eval,
                                  num_workers=8, mean=mean, std=std, crop_pct=crop_pct)

        fixed_bs = None  # determined at r=0, reused for all r (paper-faithful)

        for r in rs:
            logger.info(f"--- {name} r={r} ---")
            model = timm.create_model(name, pretrained=True)
            tome.patch.timm(model, prop_attn=True)
            model.r = r
            model = model.to(device).eval()

            acc = evaluate(model, loader, device=device)
            if fixed_bs is None:
                logger.info("  [throughput] auto-searching batch at r=0...")
                thr, fixed_bs = measure_throughput_auto(
                    model,
                    input_size=(3, input_size, input_size),
                    device=device,
                    use_fp16=False,
                    logger=logger,
                )
            else:
                thr, bs_used = measure_throughput_fixed(
                    model,
                    batch_size=fixed_bs,
                    input_size=(3, input_size, input_size),
                    device=device,
                    use_fp16=False,
                    logger=logger,
                )
            bs_thr = fixed_bs

            row = {
                "model": name,
                "input": input_size,
                "r": r,
                "acc": round(acc, 3),
                "throughput": round(thr, 2),
                "throughput_batch": bs_thr,
            }
            all_results.append(row)
            logger.info(f"  acc={acc:.3f}  throughput={thr:.1f} im/s (bs={bs_thr})")

            del model
            torch.cuda.empty_cache() if device == "cuda" else None

            # Save incrementally so partial results survive crashes
            save_results("E1_augreg", all_results)

    save_results("E1_augreg", all_results)
    logger.info(f"\nE1 done. {len(all_results)} evals. Results -> results/E1_augreg.json")


if __name__ == "__main__":
    main()
