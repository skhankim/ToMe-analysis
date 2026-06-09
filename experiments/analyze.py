"""
Analysis: turn results/E*.json into paper-style tables + figures.

Outputs:
  results/figures/tables.md         — all tables (markdown), measured vs paper
  results/figures/fig3a_augreg.png  — AugReg throughput-vs-acc curves (Fig 3a)
  results/figures/fig3c_mae.png     — MAE throughput-vs-acc curves (Fig 3c)
  results/figures/fig3b_swag.png    — SWAG curves (only if E3 results present)

Run:
  python experiments/analyze.py
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = RES / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def load(name):
    p = RES / f"{name}.json"
    return json.load(open(p)) if p.exists() else None


# Paper reference values (acc) for r=0 baselines and ablation/matching cells.
PAPER_E1_R0 = {"vit_tiny_patch16_224": 75.5, "vit_small_patch16_224": 81.4,
               "vit_base_patch16_224": 84.5, "vit_large_patch16_224": 85.8,
               "vit_large_patch16_384": 86.6}
PAPER_E2_R0 = {"mae_vit_base": 83.7, "mae_vit_large": 85.96, "mae_vit_huge": 86.9}
# Paper Table 8 (AugReg Appendix A.1.1) — full per-r acc.
PAPER_E1 = {
    "vit_tiny_patch16_224": {0:75.50, 1:75.39, 2:75.40, 3:75.34, 4:75.27, 5:75.18,
        6:75.06, 7:74.88, 8:74.76, 9:74.50, 10:74.26, 11:73.76, 12:73.53,
        13:73.04, 14:72.65, 15:71.80, 16:70.79},
    "vit_small_patch16_224": {0:81.41, 1:81.37, 2:81.35, 3:81.30, 4:81.24, 5:81.12,
        6:81.02, 7:80.94, 8:80.78, 9:80.53, 10:80.33, 11:80.06, 12:79.60,
        13:79.30, 14:78.89, 15:78.14, 16:77.01},
    "vit_base_patch16_224": {0:84.57, 1:84.61, 2:84.52, 3:84.39, 4:84.39, 5:84.24,
        6:84.17, 7:83.99, 8:83.94, 9:83.73, 10:83.37, 11:83.28, 12:82.86,
        13:82.60, 14:82.04, 15:81.39, 16:80.38},
    "vit_large_patch16_224": {0:85.82, 1:85.80, 2:85.70, 3:85.58, 4:85.37, 5:85.17,
        6:84.71, 7:84.26, 8:83.55},
    "vit_large_patch16_384": {0:86.92, 5:86.87, 10:86.85, 15:86.75, 20:86.53, 23:86.14},
}
# Paper Table 10a (MAE off-the-shelf Appendix A.1.3) — full per-r acc.
PAPER_E2 = {
    "mae_vit_base": {0:83.62, 1:83.55, 2:83.50, 3:83.44, 4:83.39, 5:83.36,
        6:83.22, 7:83.01, 8:82.93, 9:82.69, 10:82.52, 11:82.18, 12:81.92,
        13:81.41, 14:80.85, 15:80.01, 16:78.75},
    "mae_vit_large": {0:85.66, 1:85.63, 2:85.59, 3:85.51, 4:85.39, 5:85.26,
        6:85.03, 7:84.55, 8:83.92},
    "mae_vit_huge": {0:86.88, 1:86.86, 2:86.82, 3:86.80, 4:86.69, 5:86.52,
        6:86.31, 7:85.94},
}
# Paper Table 9 (SWAG, Appendix A.1.2) — full per-r acc.
PAPER_E3 = {
    "swag_vit_b16": {0: 85.30, 5: 85.27, 10: 85.21, 15: 85.18, 20: 85.09,
                     25: 85.03, 30: 84.98, 35: 84.90, 40: 84.89, 45: 84.59},
    "swag_vit_l16": {0: 88.06, 5: 88.02, 10: 87.98, 15: 87.95, 20: 87.96,
                     25: 87.87, 30: 87.89, 35: 87.82, 40: 87.80},
    "swag_vit_h14": {0: 88.55, 5: 88.53, 10: 88.44, 15: 88.49, 20: 88.46,
                     25: 88.39, 30: 88.34, 35: 88.35, 40: 88.25},
}
PAPER_T1 = {
    ("a", "x_pre"): (83.02, 186.8), ("a", "x"): (83.70, 182.8), ("a", "k"): (84.25, 182.9),
    ("a", "q"): (84.04, 182.8), ("a", "v"): (83.80, 182.9),
    ("b", "euclidean"): (84.26, 182.5), ("b", "cosine"): (84.25, 182.9),
    ("b", "dot"): (82.78, 183.0), ("b", "softmax"): (82.00, 183.0),
    ("c", "concat"): (84.32, 180.3), ("c", "mean"): (84.25, 182.9),
    ("d", "keep_one"): (81.01, 185.4), ("d", "max"): (83.50, 184.6),
    ("d", "avg"): (83.57, 183.8), ("d", "wavg"): (84.25, 182.9),
    ("e", "sequential"): (81.07, 183.0), ("e", "alternating"): (84.25, 182.9),
    ("e", "random"): (83.80, 181.7),
    ("f", "mae_no_prop"): (84.25, 182.9), ("f", "mae_prop"): (83.84, 180.9),
    ("f", "augreg_no_prop"): (82.15, 182.8), ("f", "augreg_prop"): (83.51, 180.8),
}
PAPER_T2 = {
    "prune_random": (79.22, 184.4), "prune_attn": (79.48, 183.8),
    "kmeans2": (80.19, 169.7), "kmeans5": (80.29, 147.5),
    "greedy": (84.36, 179.4), "bipartite": (84.25, 182.9),
}


def group_by_model(rows):
    g = defaultdict(list)
    for r in rows:
        g[r["model"]].append(r)
    for m in g:
        g[m].sort(key=lambda x: x["r"])
    return g


# label -> paper display name, per sub-table (PDF Table 1 wording)
T1_LABEL_COL = {"a": "feature", "b": "function", "c": "aggregate",
                "d": "method", "e": "order", "f": "src prop"}
T1_NAME = {
    "x_pre": "X_pre", "x": "X", "k": "K", "q": "Q", "v": "V",
    "euclidean": "eucl", "cosine": "cosine", "dot": "dot", "softmax": "softmax",
    "concat": "concat", "mean": "mean",
    "keep_one": "keep one", "max": "max pool", "avg": "avg pool", "wavg": "weighted avg",
    "sequential": "sequential", "alternating": "alternating", "random": "random",
}
# (f) src/prop split: label -> (src, prop_check)
T1_F = {"mae_no_prop": ("mae", ""), "mae_prop": ("mae", "yes"),
        "augreg_no_prop": ("augreg", ""), "augreg_prop": ("augreg", "yes")}
# defaults (purple in paper)
T1_DEFAULT = {("a", "k"), ("b", "cosine"), ("c", "mean"), ("d", "wavg"),
              ("e", "alternating"), ("f", "mae_no_prop"), ("f", "augreg_prop")}
T1_CAPTION = {"a": "Feature Choice", "b": "Distance Function", "c": "Head Aggregation",
              "d": "Combining Method", "e": "Partition Style", "f": "Proportional Attn"}
T1_ORDER = {
    "a": ["x_pre", "x", "k", "q", "v"],
    "b": ["euclidean", "cosine", "dot", "softmax"],
    "c": ["concat", "mean"],
    "d": ["keep_one", "max", "avg", "wavg"],
    "e": ["sequential", "alternating", "random"],
    "f": ["mae_no_prop", "mae_prop", "augreg_no_prop", "augreg_prop"],
}
T2_STYLE = {"prune_random": ("prune", "random"), "prune_attn": ("prune", "attn-based"),
            "kmeans2": ("merge", "kmeans (2 iter)"), "kmeans5": ("merge", "kmeans (5 iter)"),
            "greedy": ("merge", "greedy matching"), "bipartite": ("merge", "bipartite matching")}


def md_sweep_table(title, rows, paper_ref):
    """Paper Appendix layout: model | r | acc | drop | im/s | speed | paper acc | Δ.
    paper_ref: either {model: r0_acc} (scalar) → only r=0 paper shown,
               or     {model: {r: acc}}        → per-r paper acc + Δ shown."""
    lines = [f"### {title}", "",
             "| model | r | acc | drop | im/s | speed | paper acc | Δ |",
             "|---|---|---|---|---|---|---|---|"]
    g = group_by_model(rows)
    for m, rs in g.items():
        a0 = rs[0]["acc"]
        t0 = rs[0]["throughput"]
        ref = paper_ref.get(m)
        per_r = isinstance(ref, dict)
        for i, r in enumerate(rs):
            drop = r["acc"] - a0
            sp = r["throughput"] / t0 if t0 else 0
            if per_r:
                p = ref.get(r["r"])
                pstr = f"{p:.2f}" if p is not None else ""
                dstr = f"{r['acc']-p:+.2f}" if p is not None else ""
            else:
                pstr = f"{ref:.2f}" if (i == 0 and ref is not None) else ""
                dstr = f"{r['acc']-ref:+.2f}" if (i == 0 and ref is not None) else ""
            lines.append(f"| {m} | {r['r']} | {r['acc']:.2f} | {drop:+.2f} | "
                         f"{r['throughput']:.1f} | {sp:.2f} | {pstr} | {dstr} |")
    lines.append("")
    return "\n".join(lines)


def md_table1(mae_rows, aug_rows):
    idx = {(r["sub_table"], r["label"]): r
           for r in ([x for x in mae_rows if x["backbone"] != "augreg"] + (aug_rows or []))}
    out = ["### Table 1 — Ablation (ViT-L/16 MAE off-the-shelf, r=8)", "",
           "Baseline (ToMe 미적용): 85.96 acc. 기본값은 **(def)** 표시. "
           "`im/s`는 로컬 측정 (fp32), `paper acc`는 논문 값.", ""]
    for sub in ["a", "b", "c", "d", "e", "f"]:
        out.append(f"**(1{sub}) {T1_CAPTION[sub]}**")
        out.append("")
        if sub == "f":
            out.append("| src | prop | acc | im/s | paper acc |")
            out.append("|---|---|---|---|---|")
        else:
            out.append(f"| {T1_LABEL_COL[sub]} | acc | im/s | paper acc |")
            out.append("|---|---|---|---|")
        for label in T1_ORDER[sub]:
            r = idx.get((sub, label))
            if r is None:
                continue
            p = PAPER_T1.get((sub, label))
            pacc = f"{p[0]:.2f}" if p else "?"
            mark = " **(def)**" if (sub, label) in T1_DEFAULT else ""
            if sub == "f":
                src, prop = T1_F[label]
                out.append(f"| {src}{mark} | {prop} | {r['acc']:.2f} | "
                           f"{r['throughput']:.1f} | {pacc} |")
            else:
                name = T1_NAME.get(label, label) + mark
                out.append(f"| {name} | {r['acc']:.2f} | {r['throughput']:.1f} | {pacc} |")
        out.append("")
    return "\n".join(out)


TABLE2_NOTES = """#### Table 2 — 우리 reimpl vs paper 차이 원인

`bipartite` (84.20 vs 84.25), `greedy` (84.18 vs 84.36) — paper와 ±0.2 일치.
ToMe 핵심 method 정확히 재현. 재구현 baseline 3개에서 차이:

- **prune_attn**: 76.07 vs paper 79.48 (-3.41). Importance 정의 차이. 우리는
  attention received를 모든 query에 대해 합산; paper는 cls-attention (EViT 스타일)
  추정. Baseline 정의 선택일 뿐 ToMe 결함 아님.
- **kmeans2/5**: 82.3 vs paper 80.2 (+2.1). 우리 setting (evenly-spaced init,
  cosine on K, n_out = N-r)이 collapse 회피. Paper의 정확한 kmeans setting 미공개.
- **prune_random**: +0.29. Random seed noise.

Paper baseline code 미공개. 정확한 prune_attn / kmeans setting 재현 불가능.
"""


def plot_pareto(rows, title, fname, label_map=None, mark_dominated=False):
    """Overlay: all baselines (square markers) + ToMe sweep curves.
    If mark_dominated=True, ToMe points dominated by smaller-model baselines
    are marked with a red X (critique analysis). Default False (clean overlay)."""
    import math
    import matplotlib.ticker as mticker
    g = group_by_model(rows)
    dom_keys = set()
    if mark_dominated:
        dom = pareto_dominated(rows)
        dom_keys = {(t["model"], t["r"]) for t, _ in dom}
    fig, ax = plt.subplots(figsize=(8, 5.5))
    all_thr = []
    colors = plt.cm.tab10.colors
    for i, (m, rs) in enumerate(g.items()):
        xs = [r["throughput"] for r in rs]
        ys = [r["acc"] for r in rs]
        all_thr.extend(xs)
        lab = (label_map or {}).get(m, m)
        ax.plot(xs, ys, "-", color=colors[i % 10], lw=1.2, alpha=0.7, label=lab)
        ax.plot(xs[0], ys[0], "s", color=colors[i % 10], ms=11,
                markeredgecolor="black", markeredgewidth=1.0, zorder=5)
        if mark_dominated:
            for r in rs[1:]:
                if (r["model"], r["r"]) in dom_keys:
                    ax.plot(r["throughput"], r["acc"], "x", color="red", ms=12,
                            mew=2.5, zorder=6)
    ax.set_xscale("log")
    lo = math.floor(math.log2(min(all_thr)))
    hi = math.ceil(math.log2(max(all_thr)))
    xticks = [2 ** k for k in range(lo, hi + 1)]
    ax.set_xticks(xticks)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlim(xticks[0], xticks[-1])
    ax.set_xlabel("throughput (im/s, fp32)  [log scale]")
    ax.set_ylabel("ImageNet-1k top-1 acc (%)")
    suffix = "  (■ = baseline, × = dominated)" if mark_dominated else "  (■ = baseline)"
    ax.set_title(title + suffix)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.12)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=150)
    plt.close(fig)
    return fname


def pareto_dominated(rows):
    """Return list of (tome_row, dominator_baseline_row) for every ToMe r>0 point
    that is strictly dominated by some baseline of a different (smaller) model."""
    baselines = [r for r in rows if r["r"] == 0]
    tome = [r for r in rows if r["r"] > 0]
    out = []
    for t in tome:
        for b in baselines:
            if b["model"] == t["model"]:
                continue
            if b["throughput"] > t["throughput"] and b["acc"] > t["acc"]:
                out.append((t, b))
                break
    return out


def md_pareto(rows, family):
    dom = pareto_dominated(rows)
    lines = [f"### Compute trade-off — {family}",
             "",
             "작은 모델 baseline이 큰 모델+ToMe를 throughput과 accuracy 둘 다에서 "
             "이기는 지점. Paper Fig 3a에 이미 시각화돼 있음 — 사소한 운영 관찰용 표.",
             "",
             f"Dominated 큰모델+ToMe 지점: **{len(dom)}** / "
             f"{len([r for r in rows if r['r']>0])} ToMe 구성.",
             ""]
    if not dom:
        lines.append("_Domination 없음 — 이 family에서는 ToMe가 Pareto-optimal._\n")
        return "\n".join(lines)
    lines += ["| 큰모델+ToMe | im/s | acc | dominated by (baseline) | im/s | acc | Δim/s | Δacc |",
              "|---|---|---|---|---|---|---|---|"]
    for t, b in dom:
        d_thr = b["throughput"] / t["throughput"]
        d_acc = b["acc"] - t["acc"]
        lines.append(f"| {t['model']} r={t['r']} | {t['throughput']:.1f} | {t['acc']:.2f} | "
                     f"**{b['model']}** | {b['throughput']:.1f} | {b['acc']:.2f} | "
                     f"{d_thr:.2f}x | +{d_acc:.2f} |")
    lines.append("")
    return "\n".join(lines)


PARETO_NOTES = """#### Compute trade-off — 사소한 관찰 (비판 아님)

**Disclaimer**: 비판 아님. Paper Fig 3a는 AugReg 5개 모델을 한 axes에 plot —
cross-model 비교가 paper에서 이미 가능. 아래는 그 정보를 dominated 표 형태로
재정리한 것뿐. 운영 의사결정용 참고.

- **AugReg**: 4건 dominated. ViT-B+ToMe (r=15,16) ← ViT-S baseline,
  ViT-L+ToMe (r=7,8) ← ViT-B baseline. 작은 모델이 ~1.7배 빠르고 acc도 높음.
- **MAE / SWAG**: 0건. r_max가 짧아 (MAE L/H r≤8, SWAG r≤45) curve가
  smaller-baseline 영역까지 도달 못 함. 우리 실험으로 검증 불가.

자세한 framing은 [`critique.md`](critique.md) §1.
"""


def md_table2(rows):
    lines = ["### Table 2 — Matching Algorithm (ViT-L/16 MAE, r=8)", "",
             "| style | algorithm | acc | im/s | paper acc | paper im/s |",
             "|---|---|---|---|---|---|"]
    order = ["prune_random", "prune_attn", "kmeans2", "kmeans5", "greedy", "bipartite"]
    idx = {r["algo"]: r for r in rows}
    for a in order:
        if a not in idx:
            continue
        r = idx[a]
        style, algo = T2_STYLE[a]
        p = PAPER_T2.get(a)
        pacc = f"{p[0]:.2f}" if p else "?"
        pim = f"{p[1]:.1f}" if p else "?"
        lines.append(f"| {style} | {algo} | {r['acc']:.2f} | {r['throughput']:.1f} | "
                     f"{pacc} | {pim} |")
    lines.append("")
    return "\n".join(lines)


KMEANS_VARIANT_DESC = {
    "kmeans_random5": "random init (cls 미보호)",
    "kmeans_random5_clsfix": "random init (cluster 0 = cls)",
    "kmeans_kpp_full": "kmeans++ init + full convergence",
}


def md_kmeans_variants(e5_rows, e6_rows, e6b_rows):
    """E6 (kmeans variant strawman test) vs E5 baseline. Critique §3 evidence."""
    out = ["### E6 — kmeans variant 테스트 (ViT-L/16 MAE, r=8)",
           "",
           "Paper §4.1은 kmeans가 \"only slightly better than pruning\", "
           "\"allows large number of tokens to one cluster\"라 주장. "
           "Paper Tab 2: kmeans2/5 = 80.19/80.29, bipartite = 84.25.",
           "",
           "Paper의 정확한 kmeans setting (init, metric, cls 처리, iter) 미공개. "
           "우리 reimpl로 init/수렴 변경하며 paper 결과 재현 시도.",
           "",
           "| variant | setting | acc | im/s | vs bipartite | vs paper kmeans2 |",
           "|---|---|---|---|---|---|"]
    e5_idx = {r["algo"]: r for r in e5_rows}
    e6_idx = {r["algo"]: r for r in (e6_rows or [])}
    e6b_idx = {r["algo"]: r for r in (e6b_rows or [])}
    bp = e5_idx.get("bipartite")
    bp_acc = bp["acc"] if bp else None
    paper_km = 80.19
    rows = [
        ("bipartite (E5)", "reference", e5_idx.get("bipartite")),
        ("kmeans2 (E5)", "linspace init, 2 iter", e5_idx.get("kmeans2")),
        ("kmeans5 (E5)", "linspace init, 5 iter", e5_idx.get("kmeans5")),
        ("kmeans_kpp_full (E6)", "kmeans++ init, full converge", e6_idx.get("kmeans_kpp_full")),
        ("kmeans_random5_clsfix (E6)", "random init, cluster 0 = cls", e6b_idx.get("kmeans_random5_clsfix")),
    ]
    for label, setting, r in rows:
        if r is None:
            continue
        vs_bp = f"{r['acc'] - bp_acc:+.2f}" if bp_acc else "?"
        vs_pk = f"{r['acc'] - paper_km:+.2f}"
        out.append(f"| {label} | {setting} | {r['acc']:.2f} | {r['throughput']:.1f} | "
                   f"{vs_bp} | {vs_pk} |")
    out.append("")
    return "\n".join(out)


KMEANS_NOTES = """#### E6 — 핵심 결과

**Paper 표현 (§4.1)**: "kmeans... only slightly better than pruning",
"allows large number of tokens to one cluster". Paper는 "fundamentally fails"
라고 표현 안 함.

**용어 — CLS 토큰 처리**: ViT 입력은 패치 토큰 196개 + 맨 앞 CLS 토큰 1개
(총 N=197). CLS는 classification에 쓰이므로 ToMe와 모든 kmeans variant는 절대
merge 안 함 (보호). kmeans에서 보호는 두 결정으로 구현:
(1) cluster 0의 centroid를 CLS의 K로 init하는가, (2) Lloyd iteration에서 CLS를
cluster 0에 강제 배정하는가. linspace/kmeans++는 (1) yes, random은 (1) no.
모든 variant가 (2) yes. (1)이 no인데 (2)가 yes면 CLS가 random centroid 가진
cluster와 섞여 정보 손상. clsfix variant는 (1)을 명시해서 회피.

**관찰**:
1. **paper kmeans 80.2 우리 어떤 setting으로도 재현 불가** — 모든 sensible
   variant ≥ 82. Paper의 정확한 setting 미공개.
2. **kmeans_kpp_full = 83.98 ≈ bipartite 84.20** (-0.22pp, noise 수준).
   적절한 init + 수렴으로 kmeans가 bipartite 품질 도달.
3. **Init은 sensible 범위에서 영향 작음** — linspace = random+clsfix = 82.32.

**유효 critique** (자세한 분석은 [`critique.md`](critique.md) §2-3):
- Paper의 "slightly better than pruning" empirical 표현이 우리 setting에서 안 맞음
  (kpp_full vs prune_random = +4.47pp, "slightly" 아님)
- Paper Tab 2의 4pp acc 격차가 bipartite의 실제 contribution
  (structural guarantee + ~7% 속도)을 과대 강조
- Paper kmeans setting 미공개 → 후속 인용 시 reproducibility 우려

**한계** — 우리 결과로 반박 못 하는 것:
- Paper의 structural argument (bipartite의 guaranteed balance) — 한 setting
  결과로는 불가능
- Bipartite 불필요성 — guarantee 없는 kmeans는 여전히 collapse risk
"""


def plot_sweep(rows, title, fname, label_map=None, xticks=None):
    """throughput (x, log) vs acc (y), one line per model. Paper Fig 3 style.
    xticks: explicit log-2 tick list. If None → auto power-of-2 ticks covering data."""
    import math
    import matplotlib.ticker as mticker
    g = group_by_model(rows)
    fig, ax = plt.subplots(figsize=(7, 5))
    all_thr = []
    for m, rs in g.items():
        xs = [r["throughput"] for r in rs]
        ys = [r["acc"] for r in rs]
        all_thr.extend(xs)
        lab = (label_map or {}).get(m, m)
        ax.plot(xs, ys, marker="o", ms=4, lw=1.5, label=lab)
        ax.annotate(f"r={rs[0]['r']}", (xs[0], ys[0]), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
        ax.annotate(f"r={rs[-1]['r']}", (xs[-1], ys[-1]), fontsize=7,
                    xytext=(3, -8), textcoords="offset points")
    ax.set_xscale("log")
    if xticks is None:
        lo = math.floor(math.log2(min(all_thr)))
        hi = math.ceil(math.log2(max(all_thr)))
        xticks = [2 ** k for k in range(lo, hi + 1)]
    ax.set_xticks(xticks)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlim(xticks[0], xticks[-1])
    ax.set_xlabel("throughput (im/s, fp32)  [log scale]")
    ax.set_ylabel("ImageNet-1k top-1 acc (%)")
    ax.set_title(title)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.12)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=150)
    plt.close(fig)
    return fname


def main():
    md = ["# ToMe 재현 — Tables & Figures", "",
          "측정 환경: NVIDIA RTX A4000 (16GB GDDR6, ~448 GB/s), "
          "PyTorch 1.12.1 + CUDA 11.3, vanilla attention, fp32, "
          "batch size per-config auto-tuned. ImageNet-1k val 50K 전체.",
          "",
          "Paper throughput은 V100 기준 (~900 GB/s, 2배). "
          "절대 im/s는 다르지만 accuracy는 직접 비교 가능. 비판 정리는 "
          "[`critique.md`](critique.md) 참고.", ""]

    e1 = load("E1_augreg")
    e2 = load("E2_mae")
    e3 = load("E3_swag")
    e4 = load("E4_ablation")
    e4a = load("E4_ablation_augreg")
    e5 = load("E5_matching")
    e6 = load("E6_kmeans_variants")
    e6b = load("E6_kmeans_clsfix")

    figs = []

    if e1:
        md.append(md_sweep_table("Table 8 — AugReg sweep (Fig 3a)", e1, PAPER_E1))
        figs.append(plot_sweep(e1, "Fig 3a — AugReg ViT off-the-shelf", "fig3a_augreg.png",
                               {"vit_tiny_patch16_224": "ViT-Ti/16",
                                "vit_small_patch16_224": "ViT-S/16",
                                "vit_base_patch16_224": "ViT-B/16",
                                "vit_large_patch16_224": "ViT-L/16",
                                "vit_large_patch16_384": "ViT-L/16@384"}))
    if e2:
        md.append(md_sweep_table("Table 10a — MAE off-the-shelf sweep (Fig 3c)", e2, PAPER_E2))
        figs.append(plot_sweep(e2, "Fig 3c — MAE ViT off-the-shelf", "fig3c_mae.png",
                               {"mae_vit_base": "ViT-B/16",
                                "mae_vit_large": "ViT-L/16",
                                "mae_vit_huge": "ViT-H/14"}))
    if e3:
        md.append(md_sweep_table("Table 9 — SWAG sweep (Fig 3b)", e3, PAPER_E3))
        figs.append(plot_sweep(e3, "Fig 3b — SWAG ViT off-the-shelf", "fig3b_swag.png"))
    else:
        md.append("### Table 9 / Fig 3b — SWAG\n\n_E3 미실행._\n")

    if e4:
        md.append(md_table1(e4, e4a))
    if e5:
        md.append(md_table2(e5))
        md.append(TABLE2_NOTES)

    if e5 and (e6 or e6b):
        md.append(md_kmeans_variants(e5, e6, e6b))
        md.append(KMEANS_NOTES)

    # Compute trade-off (Pareto) analysis
    if e1 or e2 or e3:
        md.append("## Compute trade-off (Pareto frontier 분석)\n")
        if e1:
            md.append(md_pareto(e1, "AugReg"))
            figs.append(plot_pareto(e1, "Pareto — AugReg", "pareto_augreg.png",
                                    {"vit_tiny_patch16_224": "ViT-Ti/16",
                                     "vit_small_patch16_224": "ViT-S/16",
                                     "vit_base_patch16_224": "ViT-B/16",
                                     "vit_large_patch16_224": "ViT-L/16",
                                     "vit_large_patch16_384": "ViT-L/16@384"}))
        if e2:
            md.append(md_pareto(e2, "MAE"))
            figs.append(plot_pareto(e2, "Pareto — MAE", "pareto_mae.png",
                                    {"mae_vit_base": "ViT-B/16",
                                     "mae_vit_large": "ViT-L/16",
                                     "mae_vit_huge": "ViT-H/14"}))
        if e3:
            md.append(md_pareto(e3, "SWAG"))
            figs.append(plot_pareto(e3, "Pareto — SWAG", "pareto_swag.png"))
        md.append(PARETO_NOTES)

    md.append("## 그림 (Figures)\n")
    for f in figs:
        md.append(f"![{f}]({f})")
    md.append("")

    (OUT / "tables.md").write_text("\n".join(md))
    print(f"wrote {OUT/'tables.md'}")
    for f in figs:
        print(f"wrote {OUT/f}")


if __name__ == "__main__":
    main()
