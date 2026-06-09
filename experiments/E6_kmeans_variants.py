"""
E6: kmeans baseline strawman test (critique of paper Table 2).

Paper §4.1 claim:
  "kmeans... only slightly better than pruning. While it may minimize
  reconstruction error, kmeans allows a large number of tokens to match to
  the same cluster, which increases the probability of dissimilar tokens
  being merged."

Paper Tab 2: kmeans2 = 80.19, kmeans5 = 80.29, bipartite = 84.25 (~4pp gap).
Paper attributes this 4pp gap to fundamental "cluster collapse" failure of
kmeans. Paper's exact kmeans setting (init, metric, iters) is not disclosed.

Our existing kmeans2/5 (linspace init, cosine on K, 2-5 iters) already scores
82.3 (paper +2pp). Suggests paper used a worse setting.

This experiment tests two new variants vs paper claim, all on ViT-L MAE r=8:

  kmeans_random5    — random token-position init, 5 iter.
                      Closest to paper's likely strawman.
                      Hypothesis: matches paper 80.2 → paper number is init artifact.

  kmeans_kpp_full   — greedy max-min cosine init (kmeans++ approx), full
                      convergence (max 20 iter, stops on stable assignment).
                      Best-effort kmeans baseline.
                      Hypothesis: approaches bipartite 84.25 → paper "kmeans
                      fundamentally fails" claim is a strawman.

Run:
  nohup python experiments/E6_kmeans_variants.py > logs/E6_kmeans_variants.stdout 2>&1 &

Optional env vars:
  ALGOS=kmeans_random5,kmeans_kpp_full   override algo list (comma-sep)
  OUT_NAME=E6_kmeans_clsfix              override output filename
                                         (default: E6_kmeans_variants)

Re-run only the cls-fixed random init variant (preserves existing results):
  ALGOS=kmeans_random5_clsfix OUT_NAME=E6_kmeans_clsfix \\
      nohup python experiments/E6_kmeans_variants.py > logs/E6_kmeans_clsfix.stdout 2>&1 &
"""

import os
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ToMe-main"))

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
from E5_matching import apply_matching_patch
from mae_models import MAE_CKPTS, build_mae_model
from matching import ALGOS


R_FIXED = 8
INPUT = 224
BS_EVAL = 64

DEFAULT_VARIANTS = ["kmeans_random5", "kmeans_kpp_full"]


def main():
    seed = int(os.environ.get("SEED", "0"))
    set_seed(seed)
    out_name = os.environ.get("OUT_NAME", "E6_kmeans_variants")
    logger = setup_logging(out_name)
    device = get_device()
    algos_env = os.environ.get("ALGOS", "")
    variants = [a.strip() for a in algos_env.split(",") if a.strip()] or DEFAULT_VARIANTS
    for a in variants:
        if a not in ALGOS:
            raise ValueError(f"unknown algo: {a}. available: {list(ALGOS)}")
    logger.info(f"device={device}, ViT-L/16 MAE r={R_FIXED}, prop_attn=False, "
                f"variants={variants}, out={out_name}.json, seed={seed}")

    loader = build_val_loader(input_size=INPUT, batch_size=BS_EVAL,
                              num_workers=8, crop_pct=0.875)

    all_results = []
    for algo_name in variants:
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
        save_results(out_name, all_results)

    save_results(out_name, all_results)
    logger.info(f"\nE6 done. {len(all_results)} variants. "
                f"Results -> results/{out_name}.json")


if __name__ == "__main__":
    main()
