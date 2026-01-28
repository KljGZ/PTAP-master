# Precision Strike (PBADT) evaluation utilities (clean acc + ASR).
import os
import sys
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
for p in (
    _REPO_ROOT,
    os.path.join(_REPO_ROOT, "attack", "eval_bd", "train_models", "precision_strike"),
):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

from models.resnet import resnet18
from attack.eval_bd.train_models.precision_strike.utils import (
    PoisonedDataset,
    TriggerGenerator,
    eval_clean_accuracy,
    generate_poison_images,
    get_cifar10_raw,
    get_test_transform,
    set_seed,
)


def _resolve_ckpt_path(args) -> str:
    ckpt_arg = getattr(args, "precision_ckpt", "")
    if ckpt_arg and os.path.isfile(ckpt_arg):
        return ckpt_arg
    candidates = [
        os.path.join(ckpt_arg, "precision_strike_resnet18_cifar10.pth") if ckpt_arg else "",
        os.path.join("outputs", "precision_strike", "precision_strike_resnet18_cifar10.pth"),
    ]
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    raise FileNotFoundError(f"Precision-Strike checkpoint not found. Tried: {candidates}")


def _resolve_generator_cache(ckpt_path: str) -> str:
    base_dir = os.path.dirname(os.path.abspath(ckpt_path))
    candidate = os.path.join(base_dir, "poison_cache.pt")
    return candidate if os.path.isfile(candidate) else ""


def _resolve_feature_ckpt(args, ckpt_path: str) -> str:
    if getattr(args, "precision_feature_ckpt", None) and os.path.isfile(args.precision_feature_ckpt):
        return args.precision_feature_ckpt
    base_dir = os.path.dirname(os.path.abspath(ckpt_path))
    candidate = os.path.join(base_dir, "clean_resnet18.pth")
    if os.path.isfile(candidate):
        return candidate
    raise FileNotFoundError("Precision-Strike feature model checkpoint not found; pass --precision-feature-ckpt.")


def build_precision_bd_dataset(
    dataset_name: str,
    target_label: int,
    ckpt_path: str,
    device: torch.device,
    generator_cache: str = "",
    feature_ckpt: str = "",
    data_root: str = "./data",
    alpha: float = 0.6,
    patch_size: int = 5,
    batch_size: int = 64,
    max_samples: int = 0,
    seed: int = 0,
):
    """
    Build a poisoned test dataset for Precision Strike (labels already set to target_label).
    Requires a cached generator (poison_cache.pt) with generator weights.
    """
    if dataset_name != "cifar10":
        raise ValueError("Precision-Strike build only supports cifar10 in this repo")

    set_seed(seed)

    if not generator_cache:
        generator_cache = _resolve_generator_cache(ckpt_path)
    if not generator_cache or not os.path.isfile(generator_cache):
        raise FileNotFoundError("poison_cache.pt with generator weights not found; train precision_strike first.")

    if not feature_ckpt:
        feature_ckpt = _resolve_feature_ckpt(args=None, ckpt_path=ckpt_path) if feature_ckpt == "" else feature_ckpt
    feature_model = resnet18(num_classes=10).to(device)
    feature_state = torch.load(feature_ckpt, map_location="cpu")
    if isinstance(feature_state, dict) and "state_dict" in feature_state:
        feature_state = feature_state["state_dict"]
    feature_model.load_state_dict(feature_state)
    feature_model.eval().requires_grad_(False)

    cache = torch.load(generator_cache, map_location="cpu")
    poison_images = cache.get("poison_images", {})
    gen_state = cache.get("generator", None)
    if gen_state is None:
        raise ValueError("Generator weights not found in poison_cache.pt")
    generator = TriggerGenerator(patch_size=patch_size).to(device)
    generator.load_state_dict(gen_state)
    generator.eval()

    test_raw = get_cifar10_raw(data_root, train=False)
    indices = list(range(len(test_raw)))
    if max_samples and 0 < max_samples < len(indices):
        rng = np.random.RandomState(seed)
        indices = rng.choice(indices, size=max_samples, replace=False).tolist()

    missing = [idx for idx in indices if idx not in poison_images]
    if missing:
        # generate missing poison images using the trained generator
        new_images = generate_poison_images(
            generator=generator,
            feature_model=feature_model,
            raw_dataset=test_raw,
            indices=missing,
            device=device,
            alpha=alpha,
            batch_size=batch_size,
        )
        poison_images.update(new_images)

    poison_set = PoisonedDataset(test_raw, poison_images, target_label, transform=get_test_transform())
    return Subset(poison_set, indices)


def eval_precision(
    victim: torch.nn.Module,
    bd_loader: DataLoader,
    clean_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    ba = eval_clean_accuracy(victim, clean_loader, device)
    asr = eval_clean_accuracy(victim, bd_loader, device)
    return ba, asr
