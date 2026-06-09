"""
E3: SWAG sweep (Figure 3b / Table 9 reproduction).

SWAG IN1K fine-tuned models from torch.hub:
  - vit_b16_in1k @ 384   r in [0,5,10,...,45]   (10)
  - vit_l16_in1k @ 512   r in [0,5,10,...,40]   (9)
  - vit_h14_in1k @ 518   r in [0,5,10,...,40]   (9)   [SLOW: opt-in via INCLUDE_H=1]

Total: 28 evals (B+L only, default) / 28+9=37 if H included.

SWAG transform: Resize(input) + CenterCrop(input), no 256/224 scaling.
prop_attn=True (supervised pretraining).

Run:
  nohup python experiments/E3_swag.py > logs/E3_swag.stdout 2>&1 &
  INCLUDE_H=1 nohup python experiments/E3_swag.py > logs/E3_swag.stdout 2>&1 &
"""

import os
from pathlib import Path
import sys

import torch

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
    {"hub_name": "vit_b16_in1k", "tag": "swag_vit_b16",
     "input": 384, "rs": list(range(0, 50, 5))[:10],  # 0,5,..,45
     "bs_eval": 32},
    {"hub_name": "vit_l16_in1k", "tag": "swag_vit_l16",
     "input": 512, "rs": list(range(0, 45, 5))[:9],   # 0,5,..,40
     "bs_eval": 4},
    {"hub_name": "vit_h14_in1k", "tag": "swag_vit_h14",
     "input": 518, "rs": list(range(0, 45, 5))[:9],   # 0,5,..,40
     "bs_eval": 1, "huge": True},
]


def main():
    seed = int(os.environ.get("SEED", "0"))
    set_seed(seed)
    logger = setup_logging("E3_swag")
    device = get_device()
    include_huge = os.environ.get("INCLUDE_H", "0") == "1"
    only_huge = os.environ.get("ONLY_H", "0") == "1"
    if only_huge:
        include_huge = True  # ONLY_H implies INCLUDE_H
    logger.info(f"device={device}, fp32, prop_attn=True, "
                f"include_huge={include_huge}, only_huge={only_huge}, seed={seed}")

    # ONLY_H writes to a separate file so it never clobbers B+L results
    out_name = "E3_swag_h" if only_huge else "E3_swag"

    all_results = []

    for cfg in CONFIGS:
        is_huge = cfg.get("huge", False)
        if is_huge and not include_huge:
            logger.info(f"skipping {cfg['tag']} (set INCLUDE_H=1 to enable)")
            continue
        if only_huge and not is_huge:
            logger.info(f"skipping {cfg['tag']} (ONLY_H=1 set, huge only)")
            continue

        hub_name = cfg["hub_name"]
        tag = cfg["tag"]
        input_size = cfg["input"]
        rs = cfg["rs"]
        bs_eval = cfg["bs_eval"]

        # SWAG: ImageNet mean/std, Resize(input)+CenterCrop(input) == crop_pct=1.0
        logger.info(f"\n=== {tag} @ {input_size} | r in {rs} ===")
        loader = build_val_loader(input_size=input_size, batch_size=bs_eval,
                                  num_workers=4, crop_pct=1.0)

        fixed_bs = None  # determined at r=0, reused for all r (paper-faithful)

        for r in rs:
            logger.info(f"--- {tag} r={r} ---")
            model = torch.hub.load("facebookresearch/swag", model=hub_name)
            tome.patch.swag(model, prop_attn=True)
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
                    start_batch=1 if cfg.get("huge") else 4,
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
                "model": tag,
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
            save_results(out_name, all_results)

    save_results(out_name, all_results)
    logger.info(f"\nE3 done. {len(all_results)} evals. Results -> results/{out_name}.json")


if __name__ == "__main__":
    main()
