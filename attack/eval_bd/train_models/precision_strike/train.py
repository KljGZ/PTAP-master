import os
import sys
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from models.resnet import resnet18
from attack.eval_bd.train_models.precision_strike.config import get_arguments
from attack.eval_bd.train_models.precision_strike.utils import (
    PoisonedDataset,
    build_poison_set,
    compute_forgetting_counts,
    eval_asr,
    eval_clean_accuracy,
    get_cifar10_raw,
    get_test_transform,
    get_train_transform,
    select_poison_indices,
    set_seed,
)


def train_clean_model(args, device: torch.device, save_path: str) -> torch.nn.Module:
    model = resnet18(num_classes=args.num_classes).to(device)
    model.train()

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        ]
    )
    from torchvision import datasets

    train_set = datasets.CIFAR10(root=args.data_root, train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10(root=args.data_root, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    criterion = torch.nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(args.epochs):
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
        print(f"[precision-strike][clean] epoch {epoch+1}/{args.epochs} acc={acc:.4f} best={best_acc:.4f}")

    ckpt = torch.load(save_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def train_precision_strike(args):
    if args.num_classes != 10:
        raise ValueError("This reproduction currently supports CIFAR-10 only (num_classes=10).")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    save_dir = os.path.abspath(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # raw datasets (no normalization) for poison generation
    train_raw = get_cifar10_raw(args.data_root, train=True)
    test_raw = get_cifar10_raw(args.data_root, train=False)

    # 1) Train model while tracking forgetting events
    print("[precision-strike] computing forgetting events for sample selection...")
    selection_model = resnet18(num_classes=args.num_classes).to(device)
    forgetting_counts = compute_forgetting_counts(
        model=selection_model,
        train_dataset=train_raw,
        device=device,
        epochs=args.selection_epochs,
        batch_size=args.selection_batch_size,
        lr=args.selection_lr,
        momentum=args.selection_momentum,
    )
    poison_indices = select_poison_indices(
        train_dataset=train_raw,
        target_label=args.target_label,
        poison_rate=args.poison_rate,
        counts=forgetting_counts,
        seed=args.seed,
    )
    print(f"[precision-strike] selected {len(poison_indices)} poison samples (rate={args.poison_rate})")

    # 2) Train a clean feature model (used for CAM + trigger optimization)
    clean_ckpt = os.path.join(save_dir, "clean_resnet18.pth")
    feature_model = train_clean_model(args, device, clean_ckpt)

    # 3) Train trigger generator and build poison images
    poison_cache_path = os.path.join(save_dir, "poison_cache.pt")
    if os.path.isfile(poison_cache_path):
        cache = torch.load(poison_cache_path, map_location="cpu")
        poison_images: Dict[int, torch.Tensor] = cache["poison_images"]
        print(f"[precision-strike] loaded poison cache from {poison_cache_path}")
    else:
        poison_images, generator = build_poison_set(
            raw_dataset=train_raw,
            poison_indices=poison_indices,
            target_label=args.target_label,
            feature_model=feature_model,
            device=device,
            patch_size=args.patch_size,
            alpha=args.trigger_alpha,
            trigger_epochs=args.trigger_epochs,
            trigger_lr=args.trigger_lr,
            trigger_batch_size=args.trigger_batch_size,
            lambda_lpips=args.lambda_lpips,
            seed=args.seed,
        )
        torch.save({"poison_images": poison_images, "generator": generator.state_dict()}, poison_cache_path)
        print(f"[precision-strike] saved poison cache to {poison_cache_path}")
    if "generator" not in locals():
        # reload generator from cache if needed
        from attack.eval_bd.train_models.precision_strike.utils import TriggerGenerator

        generator = TriggerGenerator(patch_size=args.patch_size).to(device)
        gen_state = cache.get("generator")
        if gen_state is None:
            raise ValueError("Cached poison file does not contain generator weights.")
        generator.load_state_dict(gen_state)
        generator.to(device)
        generator.eval()

    # 4) Build poisoned training set
    train_transform = get_train_transform()
    test_transform = get_test_transform()
    train_set = PoisonedDataset(train_raw, poison_images, args.target_label, transform=train_transform)
    test_set = PoisonedDataset(test_raw, {}, args.target_label, transform=test_transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 5) Train victim model
    victim = resnet18(num_classes=args.num_classes).to(device)
    optimizer = torch.optim.SGD(
        victim.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    criterion = torch.nn.CrossEntropyLoss()

    best_score = -1.0
    best_path = os.path.join(save_dir, "precision_strike_resnet18_cifar10.pth")

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
                generator=generator,
                feature_model=feature_model,
                raw_dataset=test_raw,
                target_label=args.target_label,
                device=device,
                patch_size=args.patch_size,
                alpha=args.trigger_alpha,
                batch_size=args.batch_size,
                max_samples=args.eval_asr_samples,
            )
        score = ba + asr
        if score > best_score:
            best_score = score
            torch.save({"state_dict": victim.state_dict(), "ba": ba, "asr": asr, "epoch": epoch + 1}, best_path)
        print(
            f"[precision-strike] epoch {epoch+1}/{args.epochs} BA={ba:.4f} ASR={asr:.4f} best_score={best_score:.4f}"
        )

    print(f"[precision-strike] saved best checkpoint to {best_path}")


if __name__ == "__main__":
    args = get_arguments().parse_args()
    train_precision_strike(args)
