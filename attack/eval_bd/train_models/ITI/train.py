import os
import sys
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import models as tv_models

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from models.resnet import resnet18
from attack.eval_bd.train_models.ITI.config import get_arguments
from attack.eval_bd.train_models.ITI.utils import (
    PoisonedDataset,
    build_poison_set,
    eval_asr,
    eval_clean_accuracy,
    get_cifar10_raw,
    get_test_transform,
    get_train_transform,
    get_poison_train_transform,
    normalize_cifar,
    set_seed,
)


def load_vgg19(device: torch.device) -> torch.nn.Module:
    # VGG19 pretrained on ImageNet, used only as feature extractor
    try:
        vgg = tv_models.vgg19(weights=tv_models.VGG19_Weights.IMAGENET1K_V1)
    except Exception:
        vgg = tv_models.vgg19(pretrained=True)
    vgg.to(device)
    vgg.eval()
    return vgg


def train_victim(args):
    if args.dataset != "cifar10":
        raise ValueError("ITI reproduction only supports cifar10 in this repo")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    save_dir = os.path.abspath(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    trigger_weights: List[float] = [float(x) for x in args.trigger_weights.split(",")]
    trigger_layers = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]

    # data
    train_raw = get_cifar10_raw(args.data_root, train=True)
    test_raw = get_cifar10_raw(args.data_root, train=False)

    # feature extractor
    vgg = load_vgg19(device)

    # select poison indices
    num_poison = int(len(train_raw) * args.poison_rate)
    rng = np.random.RandomState(args.seed)
    poison_indices = rng.choice(len(train_raw), size=num_poison, replace=False).tolist()

    # build or load poison set
    poison_images: Dict[int, torch.Tensor]
    if args.poison_cache and os.path.isfile(args.poison_cache):
        cache = torch.load(args.poison_cache, map_location="cpu")
        poison_images = cache["poison_images"]
        print(f"[iti] loaded poison cache from {args.poison_cache}")
    else:
        poison_images = build_poison_set(
            raw_dataset=train_raw,
            poison_indices=poison_indices,
            target_label=args.target_label,
            vgg=vgg,
            device=device,
            trigger_steps=args.trigger_steps,
            trigger_lr=args.trigger_lr,
            alpha=args.alpha,
            beta=args.beta,
            trigger_weights=trigger_weights,
            content_layer=args.content_layer,
            trigger_layers=trigger_layers,
            ssim_thresh=args.ssim_thresh,
            trigger_choice=args.trigger_choice,
            batch_size=8,
            seed=args.seed,
        )
        if args.save_poison_cache:
            cache_path = os.path.join(save_dir, "poison_cache.pt")
            torch.save({"poison_images": poison_images}, cache_path)
            print(f"[iti] saved poison cache to {cache_path}")

    # dataloaders
    train_transform = get_poison_train_transform()
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
    best_path = os.path.join(save_dir, "iti_resnet18_cifar10.pth")

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
                raw_dataset=test_raw,
                vgg=vgg,
                device=device,
                target_label=args.target_label,
                trigger_steps=args.trigger_steps,
                trigger_lr=args.trigger_lr,
                alpha=args.alpha,
                beta=args.beta,
                trigger_weights=trigger_weights,
                content_layer=args.content_layer,
                trigger_layers=trigger_layers,
                ssim_thresh=args.ssim_thresh,
                trigger_choice=args.trigger_choice,
                batch_size=8,
                max_samples=args.eval_asr_samples,
                seed=args.seed + epoch,
            )
        score = ba + asr
        if score > best_score:
            best_score = score
            torch.save(
                {"state_dict": victim.state_dict(), "ba": ba, "asr": asr, "epoch": epoch + 1},
                best_path,
            )
        print(f"[iti] epoch {epoch+1}/{args.epochs} BA={ba:.4f} ASR={asr:.4f} best_score={best_score:.4f}")

    print(f"[iti] saved best checkpoint to {best_path}")


if __name__ == "__main__":
    args = get_arguments().parse_args()
    train_victim(args)
