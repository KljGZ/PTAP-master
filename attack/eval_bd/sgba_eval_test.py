#!/usr/bin/env python3
"""
SGBA evaluation helper: load SGBA victim checkpoint, build subspace, and print
clean accuracy (BA) and backdoor accuracy (ASR).
"""
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
for p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "attack", "eval_bd")):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

import torch

from sgba_eval import (
    _resolve_ckpt_path,
    _resolve_feature_ckpt,
    _resolve_subspace,
    eval_sgba,
    network_loader,
)
from attack.eval_bd.train_models.SGBA.config import get_arguments
from attack.eval_bd.train_models.SGBA.utils import (
    PoisonedDataset,
    compute_subspace,
    get_cifar10_raw,
    get_test_transform,
    set_seed,
)
from torch.utils.data import DataLoader


def main():
    args = get_arguments().parse_args()
    if args.dataset != "cifar10":
        raise ValueError("SGBA eval only supports cifar10 in this repo")

    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    args.device = str(device)
    set_seed(args.seed)

    ckpt_path = _resolve_ckpt_path(args)
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state_dict = state["state_dict"]
    else:
        state_dict = state

    victim = network_loader(args).to(device)
    victim.load_state_dict(state_dict)
    victim.eval().requires_grad_(False)

    feature_ckpt = _resolve_feature_ckpt(args, ckpt_path)
    feature_state = torch.load(feature_ckpt, map_location="cpu")
    if isinstance(feature_state, dict) and "state_dict" in feature_state:
        feature_state = feature_state["state_dict"]
    feature_model = network_loader(args).to(device)
    feature_model.load_state_dict(feature_state)
    feature_model.eval().requires_grad_(False)

    subspace_path = _resolve_subspace(args, ckpt_path)
    if subspace_path and os.path.isfile(subspace_path):
        cache = torch.load(subspace_path, map_location="cpu")
        subspace = cache["subspace"]
        subspace_mean = cache["mean"]
    else:
        train_raw = get_cifar10_raw(args.data_root, train=True)
        subspace, subspace_mean = compute_subspace(
            model=feature_model,
            raw_dataset=train_raw,
            target_label=args.target_label,
            k=args.subspace_samples,
            device=device,
            batch_size=args.batch_size,
        )
        subspace = subspace.cpu()
        subspace_mean = subspace_mean.cpu()

    test_raw = get_cifar10_raw(args.data_root, train=False)
    test_set = PoisonedDataset(test_raw, {}, args.target_label, transform=get_test_transform())
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    ba, asr = eval_sgba(
        victim=victim,
        feature_model=feature_model,
        subspace=subspace,
        subspace_mean=subspace_mean,
        test_loader=test_loader,
        test_raw=test_raw,
        args=args,
        device=device,
    )

    print(f"[SGBA Eval] ckpt={ckpt_path}")
    print(f"[SGBA Eval] clean_acc={ba:.4f}, bd_acc={asr:.4f}")


if __name__ == "__main__":
    main()
