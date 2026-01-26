#!/usr/bin/env python3
"""
Evaluate DFST models (clean acc + ASR using saved poison data or on-the-fly poisoning).
"""

import os
import sys
from typing import Optional

import torch
from torch.utils.data import DataLoader

# Ensure DFST package on sys.path (prefer local DFST models/backdoors)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (
    _THIS_DIR,
    os.path.join(_THIS_DIR, "train_models"),
    os.path.join(_THIS_DIR, "train_models", "DFST"),
):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

from train_models.DFST.utils import get_model, get_dataset, get_norm  # type: ignore
from train_models.DFST.attack import Attack  # type: ignore
from train_models.DFST.utils import CustomDataset  # type: ignore
from train_models.DFST.utils import get_backdoor  # type: ignore


@torch.no_grad()
def eval_top1(model, loader, preprocess, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        preds = model(preprocess(images)).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
    return correct / max(total, 1)


def parse_args():
    import argparse

    p = argparse.ArgumentParser(description="Evaluate DFST BA/ASR")
    p.add_argument("--checkpoint", type=str, required=True, help="Trained model.pt (clean or dfst).")
    p.add_argument("--dataset", type=str, default="cifar10")
    p.add_argument("--network", type=str, default="resnet18", choices=["resnet18", "resnet34", "vgg11", "vgg13"])
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--attack", type=str, default="dfst", choices=["clean", "dfst", "badnet"])
    p.add_argument("--target", type=int, default=0)
    p.add_argument("--poison-rate", type=float, default=0.05, help="Used when generating poison on-the-fly for evaluation.")
    p.add_argument("--alpha", type=float, default=0.6, help="Style transparency for DFST poison when generating on-the-fly.")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")

    # Build datasets
    clean_test = get_dataset(args.dataset, datadir=args.data_root, train=False, augment=False, download=False)
    clean_loader = DataLoader(clean_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    preprocess, _ = get_norm(args.dataset)

    # Load model
    model = torch.load(args.checkpoint, map_location="cpu")
    # If checkpoint saved only state_dict, rebuild
    if isinstance(model, dict):
        net = get_model(args.dataset, args.network)
        net.load_state_dict(model)
        model = net
    model.to(device).eval()

    # Poisoned loader
    poison_loader: Optional[DataLoader] = None
    if args.attack == "dfst":
        # Try to reuse saved poison data if alongside checkpoint
        ckpt_dir = os.path.dirname(args.checkpoint)
        poison_path = os.path.join(ckpt_dir, "poison_data.pt")
        if os.path.isfile(poison_path):
            poison_data = torch.load(poison_path, map_location="cpu")
            poison_x_test = poison_data["test"]
            poison_y_test = torch.full((poison_x_test.size(0),), args.target)
            poison_set = CustomDataset(poison_x_test, poison_y_test)
            poison_loader = DataLoader(poison_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        else:
            # On-the-fly poisoning using Attack helper
            config = {
                "dataset": args.dataset,
                "network": args.network,
                "attack": "dfst",
                "target": args.target,
                "poison_rate": args.poison_rate,
                "alpha": args.alpha,
                "data_root": args.data_root,
            }
            backdoor = None  # Attack will internally construct DFST backdoor via config
            attack = Attack(config, backdoor, save_folder=ckpt_dir)
            poison_loader = DataLoader(attack.poison_test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    ba = eval_top1(model, clean_loader, preprocess, device)
    if poison_loader is not None:
        asr = eval_top1(model, poison_loader, preprocess, device)
        print(f"[DFST Eval] BA(clean)={ba:.4f}, ASR(poison)={asr:.4f}")
    else:
        print(f"[DFST Eval] BA(clean)={ba:.4f} (no poison loader for attack={args.attack})")


def build_dfst_bd_dataset(
    dataset_name: str,
    target_label: int,
    ckpt_path: str,
    device: torch.device,
    network: str = "resnet18",
    attack: str = "dfst",
    data_root: str = "./data",
    poison_rate: float = 0.05,
    alpha: float = 0.6,
):
    """
    Given a clean dataset name (e.g., cifar10) and a DFST/BadNets checkpoint path,
    return a Dataset of poisoned test samples (labels already set to target).

    - If <ckpt_dir>/poison_data.pt exists, reuse it.
    - Otherwise, generate on-the-fly via Attack helper (will also save poison_data.pt).
    """
    ckpt_dir = os.path.dirname(os.path.abspath(ckpt_path))
    preprocess, _ = get_norm(dataset_name)

    # Load or generate poison test set
    poison_path = os.path.join(ckpt_dir, "poison_data.pt")
    if os.path.isfile(poison_path):
        poison_data = torch.load(poison_path, map_location="cpu")
        poison_x_test = poison_data["test"]
        poison_y_test = torch.full((poison_x_test.size(0),), target_label)
        poison_set = CustomDataset(poison_x_test, poison_y_test)
    else:
        config = {
            "dataset": dataset_name,
            "network": network,
            "attack": attack,
            "target": target_label,
            "poison_rate": poison_rate,
            "alpha": alpha,
            "data_root": data_root,
            "batch_size": 128,
        }
        backdoor = get_backdoor(config, device)
        atk = Attack(config, backdoor, save_folder=ckpt_dir)
        poison_set = atk.poison_test_set

    # Wrap dataset to include preprocessing
    class DFSTBDS(torch.utils.data.Dataset):
        def __len__(self):
            return len(poison_set)
        def __getitem__(self, idx):
            img, label = poison_set[idx]
            # poison_set returns tensor already; apply preprocess if needed
            img = preprocess(img) if preprocess is not None else img
            return img, label

    return DFSTBDS()


if __name__ == "__main__":
    main()
