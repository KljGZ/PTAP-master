# SGBA evaluation utilities (clean acc + ASR).
import os
import sys
from typing import Tuple

import torch
import numpy as np
from torch.utils.data import DataLoader

# Ensure repo root and SGBA utils on path.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
for p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "attack", "eval_bd", "train_models", "SGBA")):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

from models.resnet import resnet18, resnet50
from attack.eval_bd.train_models.SGBA.utils import (
    PoisonedDataset,
    build_poison_set,
    compute_subspace,
    eval_asr,
    eval_clean_accuracy,
    get_cifar10_raw,
    get_test_transform,
    set_seed,
)


def network_loader(args):
    if args.network == "resnet18":
        return resnet18(num_classes=args.num_classes)
    if args.network == "resnet50":
        return resnet50(num_classes=args.num_classes)
    raise ValueError(f"Unsupported network: {args.network}")


def _resolve_ckpt_path(args) -> str:
    ckpt_arg = getattr(args, "checkpoints", "")
    if ckpt_arg and os.path.isfile(ckpt_arg):
        return ckpt_arg
    candidates = [
        os.path.join(ckpt_arg, "sgba_resnet18_cifar10.pth") if ckpt_arg else "",
        os.path.join("outputs", "sgba", "sgba_resnet18_cifar10.pth"),
    ]
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    raise FileNotFoundError(f"SGBA checkpoint not found. Tried: {candidates}")


def _resolve_feature_ckpt(args, ckpt_path: str) -> str:
    if args.feature_ckpt and os.path.isfile(args.feature_ckpt):
        return args.feature_ckpt
    base_dir = os.path.dirname(os.path.abspath(ckpt_path))
    candidate = os.path.join(base_dir, "clean_resnet18.pth")
    if os.path.isfile(candidate):
        return candidate
    raise FileNotFoundError("Feature model checkpoint not found. Pass --feature_ckpt explicitly.")


def _resolve_subspace(args, ckpt_path: str) -> str:
    if args.subspace_cache and os.path.isfile(args.subspace_cache):
        return args.subspace_cache
    base_dir = os.path.dirname(os.path.abspath(ckpt_path))
    candidate = os.path.join(base_dir, "subspace.pt")
    return candidate if os.path.isfile(candidate) else ""


def eval_sgba(
    victim: torch.nn.Module,
    feature_model: torch.nn.Module,
    subspace: torch.Tensor,
    subspace_mean: torch.Tensor,
    test_loader: DataLoader,
    test_raw,
    args,
    device: torch.device,
) -> Tuple[float, float]:
    ba = eval_clean_accuracy(victim, test_loader, device)
    asr = eval_asr(
        victim_model=victim,
        feature_model=feature_model,
        raw_dataset=test_raw,
        subspace=subspace,
        subspace_mean=subspace_mean,
        subspace_dim=args.subspace_dim,
        target_label=args.target_label,
        device=device,
        trigger_steps=args.trigger_steps,
        trigger_lr=args.trigger_lr,
        lambda_ce=args.lambda_ce,
        lambda_reg=args.lambda_reg,
        init_delta_std=args.init_delta_std,
        batch_size=args.trigger_batch_size,
        max_samples=args.eval_asr_samples,
    )
    return ba, asr


def build_sgba_bd_dataset(
    dataset_name: str,
    target_label: int,
    ckpt_path: str,
    device: torch.device,
    feature_ckpt: str = "",
    subspace_cache: str = "",
    data_root: str = "./data",
    subspace_samples: int = 500,
    subspace_dim: int = 20,
    trigger_steps: int = 200,
    trigger_lr: float = 0.01,
    lambda_ce: float = 1.0,
    lambda_reg: float = 1e-3,
    init_delta_std: float = 1e-3,
    trigger_batch_size: int = 16,
    max_samples: int = 0,
    seed: int = 0,
):
    """
    Build a poisoned test dataset for SGBA (labels already set to target_label).
    """
    if dataset_name != "cifar10":
        raise ValueError("SGBA build only supports cifar10 in this repo")

    set_seed(seed)

    # feature model
    if not feature_ckpt:
        base_dir = os.path.dirname(os.path.abspath(ckpt_path))
        feature_ckpt = os.path.join(base_dir, "clean_resnet18.pth")
    if not os.path.isfile(feature_ckpt):
        raise FileNotFoundError(f"Feature model checkpoint not found: {feature_ckpt}")

    feature_model = resnet18(num_classes=10).to(device)
    feature_state = torch.load(feature_ckpt, map_location="cpu")
    if isinstance(feature_state, dict) and "state_dict" in feature_state:
        feature_state = feature_state["state_dict"]
    feature_model.load_state_dict(feature_state)
    feature_model.eval().requires_grad_(False)

    # subspace
    if not subspace_cache:
        base_dir = os.path.dirname(os.path.abspath(ckpt_path))
        subspace_cache = os.path.join(base_dir, "subspace.pt")
    if subspace_cache and os.path.isfile(subspace_cache):
        cache = torch.load(subspace_cache, map_location="cpu")
        subspace = cache["subspace"]
        subspace_mean = cache["mean"]
    else:
        train_raw = get_cifar10_raw(data_root, train=True)
        subspace, subspace_mean = compute_subspace(
            model=feature_model,
            raw_dataset=train_raw,
            target_label=target_label,
            k=subspace_samples,
            device=device,
            batch_size=trigger_batch_size,
        )
        subspace = subspace.cpu()
        subspace_mean = subspace_mean.cpu()

    test_raw = get_cifar10_raw(data_root, train=False)
    indices = list(range(len(test_raw)))
    if max_samples and max_samples > 0 and max_samples < len(indices):
        rng = np.random.RandomState(seed)
        indices = rng.choice(indices, size=max_samples, replace=False).tolist()

    poison_images = build_poison_set(
        model=feature_model,
        raw_dataset=test_raw,
        poison_indices=indices,
        target_label=target_label,
        subspace=subspace,
        subspace_mean=subspace_mean,
        subspace_dim=subspace_dim,
        device=device,
        trigger_steps=trigger_steps,
        trigger_lr=trigger_lr,
        lambda_ce=lambda_ce,
        lambda_reg=lambda_reg,
        init_delta_std=init_delta_std,
        batch_size=trigger_batch_size,
        log_interval=0,
    )

    poison_set = PoisonedDataset(test_raw, poison_images, target_label, transform=get_test_transform())
    from torch.utils.data import Subset

    return Subset(poison_set, indices)
