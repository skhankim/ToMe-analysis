"""
E2: MAE off-the-shelf sweep (Figure 3c off-the-shelf / Table 10a reproduction).

MAE finetuned models (all @224, global_pool=True), patched off-the-shelf
with prop_attn=False (Table 1f: MAE does NOT need proportional attention).

  - base   r in 0..16 (17)
  - large  r in 0..8  (9)
  - huge   r in 0..7  (8)   [patch14 -> more tokens, r max ~7]

Total: 34 evaluations.

Requires checkpoints/mae_finetuned_vit_{base,large,huge}.pth

Run:
  nohup python experiments/E2_mae.py > logs/E2_mae.stdout 2>&1 &
"""

import os
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ToMe-main"))

import tome
from common import (
    CKPT_DIR,
    build_val_loader,
    evaluate,
    get_device,
    measure_throughput_auto,
    measure_throughput_fixed,
    save_results,
    set_seed,
    setup_logging,
)
from mae_models import MAE_CKPTS, build_mae_model


CONFIGS = [
    {"size": "base",  "input": 224, "rs": list(range(17)), "bs_eval": 128},
    {"size": "large", "input": 224, "rs": list(range(9)),  "bs_eval": 64},
    {"size": "huge",  "input": 224, "rs": list(range(8)),  "bs_eval": 32},
]


def main():
    seed = int(os.environ.get("SEED", "0"))
    set_seed(seed)
    logger = setup_logging("E2_mae")
    device = get_device()
    logger.info(f"device={device}, fp32, prop_attn=False (MAE off-the-shelf), seed={seed}")

    all_results = []

    for cfg in CONFIGS:
        size = cfg["size"]
        input_size = cfg["input"]
        rs = cfg["rs"]
        bs_eval = cfg["bs_eval"]

        ckpt_path = CKPT_DIR / MAE_CKPTS[size]
        if not ckpt_path.exists():
            logger.warning(f"missing ckpt {ckpt_path}, skipping {size}")
            continue

        # MAE finetuned: ImageNet mean/std, crop_pct=0.875 (standard timm val)
        logger.info(f"\n=== MAE-{size} @ {input_size} | r in {rs} ===")
        loader = build_val_loader(input_size=input_size, batch_size=bs_eval,
                                  num_workers=8, crop_pct=0.875)

        fixed_bs = None  # determined at r=0, reused for all r (paper-faithful)

        for r in rs:
            logger.info(f"--- MAE-{size} r={r} ---")
            model = build_mae_model(size, str(ckpt_path), global_pool=True)
            tome.patch.mae(model, prop_attn=False)
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
                thr, _ = measure_throughput_fixed(
                    model,
                    batch_size=fixed_bs,
                    input_size=(3, input_size, input_size),
                    device=device,
                    use_fp16=False,
                    logger=logger,
                )
            bs_thr = fixed_bs

            row = {
                "model": f"mae_vit_{size}",
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
            save_results("E2_mae", all_results)

    save_results("E2_mae", all_results)
    logger.info(f"\nE2 done. {len(all_results)} evals. Results -> results/E2_mae.json")


if __name__ == "__main__":
    main()
