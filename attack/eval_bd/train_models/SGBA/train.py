import os
import sys
from typing import Dict

import torch
import numpy as np
from torch.utils.data import DataLoader

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from models.resnet import resnet18
from attack.eval_bd.train_models.SGBA.config import get_arguments
from attack.eval_bd.train_models.SGBA.utils import (
    PoisonedDataset,
    build_poison_set,
    compute_subspace,
    eval_asr,
    eval_clean_accuracy,
    get_cifar10_raw,
    get_test_transform,
    get_train_transform,
    set_seed,
)


def train_clean_model(args, device: torch.device, save_path: str) -> torch.nn.Module:
    model = resnet18(num_classes=args.num_classes).to(device)
    model.train()

    train_transform = get_train_transform()
    test_transform = get_test_transform()
    train_raw = get_cifar10_raw(args.data_root, train=True)
    test_raw = get_cifar10_raw(args.data_root, train=False)

    train_set = PoisonedDataset(train_raw, {}, args.target_label, transform=train_transform)
    test_set = PoisonedDataset(test_raw, {}, args.target_label, transform=test_transform)
    train_loader = DataLoader(train_set, batch_size=args.clean_batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.clean_batch_size, shuffle=False, num_workers=args.num_workers)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.clean_lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    criterion = torch.nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(args.clean_epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        scheduler.step()
        acc = eval_clean_accuracy(model, test_loader, device)
        if acc > best_acc:
            best_acc = acc
            torch.save({"state_dict": model.state_dict()}, save_path)
        print(f"[clean] epoch {epoch+1}/{args.clean_epochs} acc={acc:.4f} best={best_acc:.4f}")

    ckpt = torch.load(save_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def load_feature_model(args, device: torch.device) -> torch.nn.Module:
    save_dir = os.path.abspath(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    clean_ckpt_path = args.feature_ckpt or os.path.join(save_dir, "clean_resnet18.pth")
    if os.path.isfile(clean_ckpt_path):
        model = resnet18(num_classes=args.num_classes).to(device)
        ckpt = torch.load(clean_ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        print(f"[clean] loaded feature model from {clean_ckpt_path}")
        return model

    print("[clean] no checkpoint found, training clean feature model...")
    return train_clean_model(args, device, clean_ckpt_path)


def train_sgba(args):
    if args.dataset != "cifar10":
        raise ValueError("SGBA implementation only supports cifar10 in this repo")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    save_dir = os.path.abspath(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # prepare datasets
    train_raw = get_cifar10_raw(args.data_root, train=True)
    test_raw = get_cifar10_raw(args.data_root, train=False)

    # feature model and subspace
    feature_model = load_feature_model(args, device)
    if args.subspace_cache and os.path.isfile(args.subspace_cache):
        cache = torch.load(args.subspace_cache, map_location="cpu")
        subspace = cache["subspace"]
        subspace_mean = cache["mean"]
        print(f"[sgba] loaded subspace from {args.subspace_cache}")
    else:
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
        subspace_path = os.path.join(save_dir, "subspace.pt")
        torch.save({"subspace": subspace, "mean": subspace_mean}, subspace_path)
        print(f"[sgba] saved subspace to {subspace_path}")

    # poison selection
    num_poison = int(len(train_raw) * args.poison_rate)
    rng = np.random.RandomState(args.seed)
    poison_indices = rng.choice(len(train_raw), size=num_poison, replace=False).tolist()

    poison_images: Dict[int, torch.Tensor]
    if args.poison_cache and os.path.isfile(args.poison_cache):
        cache = torch.load(args.poison_cache, map_location="cpu")
        poison_images = cache["poison_images"]
        print(f"[sgba] loaded poison cache from {args.poison_cache}")
    else:
        poison_images = build_poison_set(
            model=feature_model,
            raw_dataset=train_raw,
            poison_indices=poison_indices,
            target_label=args.target_label,
            subspace=subspace,
            subspace_mean=subspace_mean,
            subspace_dim=args.subspace_dim,
            device=device,
            trigger_steps=args.trigger_steps,
            trigger_lr=args.trigger_lr,
            lambda_ce=args.lambda_ce,
            lambda_reg=args.lambda_reg,
            init_delta_std=args.init_delta_std,
            batch_size=args.trigger_batch_size,
        )
        if args.save_poison_cache:
            cache_path = os.path.join(save_dir, "poison_cache.pt")
            torch.save({"poison_images": poison_images}, cache_path)
            print(f"[sgba] saved poison cache to {cache_path}")

    # build poisoned training set
    train_transform = get_train_transform()
    test_transform = get_test_transform()

    train_set = PoisonedDataset(train_raw, poison_images, args.target_label, transform=train_transform)
    test_set = PoisonedDataset(test_raw, {}, args.target_label, transform=test_transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # victim model
    victim = resnet18(num_classes=args.num_classes).to(device)
    optimizer = torch.optim.SGD(
        victim.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    criterion = torch.nn.CrossEntropyLoss()

    best_score = -1.0
    best_path = os.path.join(save_dir, "sgba_resnet18_cifar10.pth")

    for epoch in range(args.epochs):
        victim.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = victim(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        ba = eval_clean_accuracy(victim, test_loader, device)
        asr = 0.0
        if args.eval_asr:
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
        score = ba + asr
        if score > best_score:
            best_score = score
            torch.save(
                {"state_dict": victim.state_dict(), "ba": ba, "asr": asr, "epoch": epoch + 1},
                best_path,
            )
        print(
            f"[sgba] epoch {epoch+1}/{args.epochs} BA={ba:.4f} ASR={asr:.4f} best_score={best_score:.4f}"
        )

    print(f"[sgba] saved best checkpoint to {best_path}")


if __name__ == "__main__":
    args = get_arguments().parse_args()
    train_sgba(args)
