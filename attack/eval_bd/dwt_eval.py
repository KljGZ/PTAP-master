# DWT dynamic frequency-domain trigger evaluation (clean acc + ASR).
import os
import sys
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
for p in (
    _REPO_ROOT,
    os.path.join(_REPO_ROOT, "attack", "eval_bd", "train_models", "DWT"),
):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

from models.resnet import resnet18  # for feature extraction if needed in future
from attack.eval_bd.train_models.DWT.utils import (
    HaarDWT,
    HaarIDWT,
    PoisonedDataset,
    TriggerGenerator,
    eval_clean_accuracy,
    get_raw_cifar10,
    get_test_transform,
    normalize_tensor,
    set_seed,
)


def _resolve_ckpt_path(args=None) -> str:
    ckpt_arg = getattr(args, "dwt_ckpt", "") if args is not None else ""
    if ckpt_arg and os.path.isfile(ckpt_arg):
        return ckpt_arg
    candidates = [
        os.path.join(ckpt_arg, "dwt_resnet18_cifar10.pth") if ckpt_arg else "",
        os.path.join("outputs", "DWT", "dwt_resnet18_cifar10.pth"),
    ]
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    raise FileNotFoundError(f"DWT victim checkpoint not found. Tried: {candidates}")


def _resolve_generator_ckpt(args, ckpt_path: str) -> str:
    candidate_arg = getattr(args, "dwt_generator_ckpt", None) if args is not None else None
    if candidate_arg and os.path.isfile(candidate_arg):
        return candidate_arg
    base_dir = os.path.dirname(os.path.abspath(ckpt_path))
    candidate = os.path.join(base_dir, "dwt_generator.pth")
    return candidate if os.path.isfile(candidate) else ""


def _resolve_poison_cache(args, ckpt_path: str) -> str:
    candidate_arg = getattr(args, "dwt_poison_cache", None) if args is not None else None
    if candidate_arg and os.path.isfile(candidate_arg):
        return candidate_arg
    base_dir = os.path.dirname(os.path.abspath(ckpt_path))
    candidate = os.path.join(base_dir, "poison_cache.pt")
    return candidate if os.path.isfile(candidate) else ""


def _generate_poison_batch(
    generator: TriggerGenerator,
    dwt: HaarDWT,
    idwt: HaarIDWT,
    images: torch.Tensor,
    secret_bits: int,
) -> torch.Tensor:
    b = images.size(0)
    device = images.device
    secrets = torch.randint(0, 2, (b, secret_bits), device=device, dtype=images.dtype)
    ll, lh, hl, hh = dwt(images)
    secret_map = secrets.unsqueeze(-1).unsqueeze(-1).expand(-1, secret_bits, ll.size(2), ll.size(3))
    gen_input = torch.cat([ll, lh, hl, hh, secret_map], dim=1)
    pd = generator(gen_input)
    pd_split = torch.chunk(pd, 4, dim=1)
    poisoned = torch.clamp(idwt(pd_split), 0.0, 1.0)
    return poisoned


def build_dwt_bd_dataset(
    dataset_name: str,
    target_label: int,
    ckpt_path: str,
    device: torch.device,
    generator_ckpt: str = "",
    poison_cache: str = "",
    data_root: str = "./data",
    secret_bits: int = 3,
    max_samples: int = 0,
    batch_size: int = 64,
    seed: int = 0,
):
    """
    Build a poisoned test dataset for DWT backdoor (labels already set to target_label).
    Prefers cached poison images; otherwise generates on the fly with the trained generator.
    """
    if dataset_name != "cifar10":
        raise ValueError("DWT build only supports cifar10 in this repo.")

    set_seed(seed)

    # resolve paths
    if not ckpt_path:
        ckpt_path = _resolve_ckpt_path(args=None)
    if not generator_ckpt:
        generator_ckpt = _resolve_generator_ckpt(args=None, ckpt_path=ckpt_path)
    if not poison_cache:
        poison_cache = _resolve_poison_cache(args=None, ckpt_path=ckpt_path)

    # load generator
    gen_state = torch.load(generator_ckpt, map_location="cpu") if generator_ckpt and os.path.isfile(generator_ckpt) else {}
    if isinstance(gen_state, dict) and "generator" in gen_state:
        secret_bits = gen_state.get("secret_bits", secret_bits)
        gen_weights = gen_state["generator"]
    else:
        raise FileNotFoundError("DWT generator checkpoint not found; train DWT first or pass --dwt-generator-ckpt.")
    in_channels = 12 + secret_bits
    generator = TriggerGenerator(in_channels=in_channels).to(device)
    generator.load_state_dict(gen_weights)
    generator.eval()

    dwt = HaarDWT().to(device)
    idwt = HaarIDWT().to(device)

    # load dataset
    test_raw = get_raw_cifar10(data_root, train=False)
    indices = list(range(len(test_raw)))
    if max_samples and 0 < max_samples < len(indices):
        rng = np.random.RandomState(seed)
        indices = rng.choice(indices, size=max_samples, replace=False).tolist()

    # load cache if available
    poison_images = {}
    if poison_cache and os.path.isfile(poison_cache):
        cache = torch.load(poison_cache, map_location="cpu")
        poison_images = cache.get("poison_images", {})

    # generate missing
    missing = [idx for idx in indices if idx not in poison_images]
    if missing:
        # process in batches to avoid OOM
        for start in range(0, len(missing), batch_size):
            chunk_ids = missing[start : start + batch_size]
            imgs = []
            for idx in chunk_ids:
                img, _ = test_raw[idx]
                if not isinstance(img, torch.Tensor):
                    img = torch.tensor(img)
                imgs.append(img)
            batch = torch.stack(imgs, dim=0).to(device)
            with torch.no_grad():
                poisoned = _generate_poison_batch(generator, dwt, idwt, batch, secret_bits).cpu()
            for idx_local, idx_global in enumerate(chunk_ids):
                poison_images[idx_global] = poisoned[idx_local]

    poison_set = PoisonedDataset(test_raw, poison_images, target_label, transform=get_test_transform())
    return Subset(poison_set, indices)


def eval_dwt(
    victim: torch.nn.Module,
    bd_loader: DataLoader,
    clean_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    ba = eval_clean_accuracy(victim, clean_loader, device)
    asr = eval_clean_accuracy(victim, bd_loader, device)
    return ba, asr
