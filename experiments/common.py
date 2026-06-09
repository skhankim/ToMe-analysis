"""
Common utilities for ToMe reproduction experiments.

- ImageNet val ImageFolder loader with configurable input size
- Top-1 accuracy evaluation
- Throughput measurement with auto batch-size search (paper "optimal batch size" semantics)
- Result/log helpers
"""

import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def set_seed(seed: int = 0) -> None:
    """Fix RNG seeds for reproducibility. Affects torch / torch.cuda / numpy /
    Python random. Random init algorithms in matching.py (prune_random,
    kmeans_random5*) become deterministic across runs."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

# Ensure tome is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ToMe-main"))

# Standard ImageNet stats
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DATA_ROOT = os.environ.get("IMAGENET_VAL", str(ROOT / "data"))
RESULTS_DIR = ROOT / "results"
LOGS_DIR = ROOT / "logs"
CKPT_DIR = ROOT / "checkpoints"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(name: str) -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"{name}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    logger.info(f"Logging to {log_path}")
    return logger


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_transform(
    input_size: int,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
    crop_pct: float = 0.875,
) -> transforms.Compose:
    """
    Standard timm-style val transform:
      resize_size = int(input_size / crop_pct)
      Resize(resize_size, bicubic) -> CenterCrop(input_size) -> Normalize(mean,std)

    Defaults (crop_pct=0.875, ImageNet mean/std): MAE/SWAG @224, MAE @standard.
    For AugReg: pass model.default_cfg's mean/std (0.5,0.5,0.5) and crop_pct (0.9 or 1.0).
    For SWAG: crop_pct=1.0 reproduces Resize(input)+CenterCrop(input).
    """
    resize_to = int(round(input_size / crop_pct))
    return transforms.Compose([
        transforms.Resize(resize_to, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def build_val_loader(
    input_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 8,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
    crop_pct: float = 0.875,
    data_root: Optional[str] = None,
    subset: Optional[int] = None,
) -> DataLoader:
    root = data_root or DATA_ROOT
    transform = build_transform(input_size, mean=mean, std=std, crop_pct=crop_pct)
    dataset = datasets.ImageFolder(root, transform=transform)
    if subset is not None and subset < len(dataset):
        # Deterministic subset (stride sampling, balanced across classes)
        indices = list(range(0, len(dataset), len(dataset) // subset))[:subset]
        dataset = torch.utils.data.Subset(dataset, indices)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: str = "cuda") -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


# ---------------------------------------------------------------------------
# Throughput
# ---------------------------------------------------------------------------

def _try_batch(model, batch_size: int, input_size: Tuple[int, int, int],
               device: str, runs: int, use_fp16: bool) -> Optional[float]:
    """Returns throughput at given batch_size, or None on OOM."""
    try:
        from tome.utils import benchmark
        torch.cuda.empty_cache()
        thr = benchmark(
            model,
            device=device,
            input_size=input_size,
            batch_size=batch_size,
            runs=runs,
            use_fp16=use_fp16,
            verbose=False,
        )
        torch.cuda.empty_cache()
        return thr
    except RuntimeError as e:
        msg = str(e).lower()
        if "out of memory" in msg or "cuda" in msg:
            torch.cuda.empty_cache()
            return None
        raise


def measure_throughput_auto(
    model: torch.nn.Module,
    input_size: Tuple[int, int, int] = (3, 224, 224),
    device: str = "cuda",
    use_fp16: bool = False,
    runs: int = 40,
    start_batch: int = 32,
    max_batch: int = 1024,
    logger: Optional[logging.Logger] = None,
) -> Tuple[float, int]:
    """
    Find largest batch (power of 2) that fits, return (throughput im/s, batch_size).
    Matches paper "optimal batch size" semantics.

    NOTE: For paper-faithful r sweeps, only call this once at r=0 (max memory)
    per model, then reuse the discovered batch with measure_throughput_fixed
    for all r values. Author's notebook uses fixed batch across r.
    """
    best_thr = None
    best_bs = None
    bs = start_batch
    while bs <= max_batch:
        thr = _try_batch(model, bs, input_size, device, runs, use_fp16)
        if thr is None:
            if logger:
                logger.info(f"  batch={bs} OOM, stopping search")
            break
        if logger:
            logger.info(f"  batch={bs} -> {thr:.1f} im/s")
        if best_thr is None or thr > best_thr:
            best_thr = thr
            best_bs = bs
        bs *= 2
    if best_thr is None:
        # Even start_batch failed; try batch=1 as last resort
        thr = _try_batch(model, 1, input_size, device, runs, use_fp16)
        return (thr or 0.0), 1
    if logger:
        logger.info(f"  -> best batch={best_bs} @ {best_thr:.1f} im/s")
    return best_thr, best_bs


def measure_throughput_fixed(
    model: torch.nn.Module,
    batch_size: int,
    input_size: Tuple[int, int, int] = (3, 224, 224),
    device: str = "cuda",
    use_fp16: bool = False,
    runs: int = 40,
    logger: Optional[logging.Logger] = None,
) -> Tuple[float, int]:
    """
    Measure throughput at a fixed batch_size (paper-faithful for r sweeps).
    Returns (throughput im/s, batch_size). If OOM at given batch, halves
    until it fits, returning the actually-used batch.
    """
    bs = batch_size
    while bs >= 1:
        thr = _try_batch(model, bs, input_size, device, runs, use_fp16)
        if thr is not None:
            if logger:
                logger.info(f"  batch={bs} (fixed) -> {thr:.1f} im/s")
            return thr, bs
        if logger:
            logger.info(f"  batch={bs} OOM, halving")
        bs //= 2
    return 0.0, 0


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def save_results(name: str, data) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
