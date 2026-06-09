# ToMe Reproduction Runbook

End-to-end ops guide for E1-E5 on A4000 16GB

---

## 0. Prereqs (local)

- Code, ckpt, data ready at `/mnt/f/kimauve/ToMe/`

---

## 1. Build Docker image (one-time per server)

---

## 2. Start persistent container

```bash
./docker_run.sh up
```

---

## 3. Launch experiments (background, persistent)

```bash
cd /workspace/ToMe
# E1 (~6.1h)
nohup python experiments/E1_augreg.py > logs/E1_augreg.stdout 2>&1 &
# E2 (~5.9h)
nohup python experiments/E2_mae.py > logs/E2_mae.stdout 2>&1 &

# E3 (~23.5h)
INCLUDE_H=1 nohup python experiments/E3_swag.py > logs/E3_swag.stdout 2>&1 &

# E4 ablation, ViT-L MAE r=8 (paper default, ~3.5h)
nohup python experiments/E4_ablation.py > logs/E4_ablation.stdout 2>&1 &

# E4 Table 1f AugReg rows only (selective, ~0.5h)
ONLY_AUGREG=1 nohup python experiments/E4_ablation.py > logs/E4_ablation_augreg.stdout 2>&1 &

# E4 stress: ViT-B MAE r=16 ablation (논문 범위 외, ~3h)
MODEL_SIZE=base R_VALUE=16 nohup python experiments/E4_ablation.py > logs/E4_ablation_base_r16.stdout 2>&1 &

# E5 matching algorithm reproduction (~1.5h)
nohup python experiments/E5_matching.py > logs/E5_matching.stdout 2>&1 &

# E6 kmeans variant tests (논문 범위 외, ~1.5h)
nohup python experiments/E6_kmeans_variants.py > logs/E6_kmeans_variants.stdout 2>&1 &

# E6 random init clsfix variant (별도 output, ~15min)
ALGOS=kmeans_random5_clsfix OUT_NAME=E6_kmeans_clsfix \
    nohup python experiments/E6_kmeans_variants.py > logs/E6_kmeans_clsfix.stdout 2>&1 &
```

---

## 4. Files written per experiment

| Script (env)                                                | results/                                | logs/                                  |
| ----------------------------------------------------------- | --------------------------------------- | -------------------------------------- |
| E1_augreg.py                                                | `E1_augreg.json`                        | `E1_augreg.log` + `.stdout`            |
| E2_mae.py                                                   | `E2_mae.json`                           | `E2_mae.log` + `.stdout`               |
| E3_swag.py                                                  | `E3_swag.json` (B+L; +H if INCLUDE_H=1) | `E3_swag.log` + `.stdout`              |
| E3_swag.py `ONLY_H=1`                                       | `E3_swag_h.json` (H only)               | `E3_swag.log` + `.stdout`              |
| E4_ablation.py                                              | `E4_ablation.json`                      | `E4_ablation.log` + `.stdout`          |
| E4_ablation.py `ONLY_AUGREG=1`                              | `E4_ablation_augreg.json`               | `E4_ablation_augreg.log` + `.stdout`   |
| E4_ablation.py `MODEL_SIZE=base R_VALUE=16`                 | `E4_ablation_base_r16.json`             | `E4_ablation_base_r16.log` + `.stdout` |
| E5_matching.py                                              | `E5_matching.json`                      | `E5_matching.log` + `.stdout`          |
| E6_kmeans_variants.py                                       | `E6_kmeans_variants.json`               | `E6_kmeans_variants.log` + `.stdout`   |
| E6_kmeans_variants.py `ALGOS=... OUT_NAME=E6_kmeans_clsfix` | `E6_kmeans_clsfix.json`                 | `E6_kmeans_clsfix.log` + `.stdout`     |

Each JSON row: `{model, input, r (or config), acc, throughput, throughput_batch}`.

## Env var reference

| Script                | Env var                  | Default                          | Effect                                                                              |
| --------------------- | ------------------------ | -------------------------------- | ----------------------------------------------------------------------------------- |
| E3_swag.py            | `INCLUDE_H=1`            | 0                                | ViT-H/14@518 포함 (~16h 추가)                                                       |
| E3_swag.py            | `ONLY_H=1`               | 0                                | ViT-H/14@518만 (B+L 결과 보존, `E3_swag_h.json`에 저장)                             |
| E4_ablation.py        | `MODEL_SIZE=base\|large` | large                            | MAE backbone 크기. ViT-B = 12 layer, r_max=16                                       |
| E4_ablation.py        | `R_VALUE=N`              | 8                                | layer당 token merge 수. ViT-L r_max=8, ViT-B r_max=16                               |
| E4_ablation.py        | `ONLY_AUGREG=1`          | 0                                | Table 1f의 AugReg 4개 config만 (paper baseline 비교)                                |
| E6_kmeans_variants.py | `ALGOS=a,b,c`            | `kmeans_random5,kmeans_kpp_full` | comma-separated. 가용: `kmeans_random5`, `kmeans_random5_clsfix`, `kmeans_kpp_full` |
| E6_kmeans_variants.py | `OUT_NAME=name`          | `E6_kmeans_variants`             | output JSON 파일명 (기존 결과 보존용)                                               |

기본값 명령은 paper 재현. 환경변수 override는 논문 범위 외 추가 실험 (E4 stress / E6 variant).

---
