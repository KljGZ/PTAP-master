# ITI evaluation utilities (build poisoned test dataset).
import os
import sys
from typing import List

import numpy as np
import torch
from torchvision import models as tv_models

# Ensure repo root and ITI utils on path.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
for p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "attack", "eval_bd", "train_models", "ITI")):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

from attack.eval_bd.train_models.ITI.utils import (  # noqa: E402
    PoisonedDataset,
    build_poison_set,
    get_cifar10_raw,
    get_test_transform,
    set_seed,
)


def _load_vgg19(device: torch.device) -> torch.nn.Module:
    try:
        vgg = tv_models.vgg19(weights=tv_models.VGG19_Weights.IMAGENET1K_V1)
    except Exception:
        vgg = tv_models.vgg19(pretrained=True)
    vgg.to(device)
    vgg.eval().requires_grad_(False)
    return vgg


def build_iti_bd_dataset(
    dataset_name: str,
    target_label: int,
    ckpt_path: str,
    device: torch.device,
    data_root: str = "./data",
    trigger_steps: int = 800,
    trigger_lr: float = 0.01,
    alpha: float = 5.0,
    beta: float = 20.0,
    trigger_weights: str = "1,0.8,0.5,0.3,0.1",
    content_layer: str = "conv2_2",
    ssim_thresh: float = 0.99,
    trigger_choice: str = "target",
    max_samples: int = 256,
    seed: int = 0,
    poison_batch_size: int = 8,
):
    """
    Build a poisoned test dataset for ITI (labels are set to target_label).

    Notes:
    - ITI trigger generation is iterative and expensive; keep max_samples small.
    - ckpt_path is accepted for CLI symmetry but not used here.
    """
    if dataset_name != "cifar10":
        raise ValueError("ITI build only supports cifar10 in this repo")

    set_seed(seed)

    vgg = _load_vgg19(device)
    test_raw = get_cifar10_raw(data_root, train=False)
    indices: List[int] = list(range(len(test_raw)))
    if max_samples and max_samples > 0 and max_samples < len(indices):
        rng = np.random.RandomState(seed)
        indices = rng.choice(indices, size=max_samples, replace=False).tolist()

    tw = [float(x) for x in trigger_weights.split(",")]
    trigger_layers = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]

    poison_images = build_poison_set(
        raw_dataset=test_raw,
        poison_indices=indices,
        target_label=target_label,
        vgg=vgg,
        device=device,
        trigger_steps=trigger_steps,
        trigger_lr=trigger_lr,
        alpha=alpha,
        beta=beta,
        trigger_weights=tw,
        content_layer=content_layer,
        trigger_layers=trigger_layers,
        ssim_thresh=ssim_thresh,
        trigger_choice=trigger_choice,
        batch_size=poison_batch_size,
        seed=seed,
    )

    poison_set = PoisonedDataset(test_raw, poison_images, target_label, transform=get_test_transform())
    from torch.utils.data import Subset

    return Subset(poison_set, indices)

