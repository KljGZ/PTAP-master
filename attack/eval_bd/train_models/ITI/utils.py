import math
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms


# CIFAR-10 statistics
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2023, 0.1994, 0.2010]

# ImageNet statistics (for VGG19)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# -----------------------
# utils
# -----------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_cifar10_raw(data_root: str, train: bool) -> datasets.CIFAR10:
    return datasets.CIFAR10(root=data_root, train=train, download=True, transform=transforms.ToTensor())


def get_train_transform():
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def get_poison_train_transform():
    # For poisoned images, avoid spatial augmentation to keep trigger pattern intact
    return transforms.Compose([transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])


def get_test_transform():
    return transforms.Compose([transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])


def normalize_cifar(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    mean = torch.tensor(CIFAR10_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=device).view(1, 3, 1, 1)
    return (x - mean) / std


def normalize_imagenet(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return (x - mean) / std


def resize_to_224(img: torch.Tensor, device: torch.device) -> torch.Tensor:
    # img: (C,H,W) in [0,1], return (1,C,224,224)
    img = img.unsqueeze(0).to(device)
    return torch.nn.functional.interpolate(img, size=(224, 224), mode="bilinear", align_corners=False)


class IndexedDataset(Dataset):
    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        return img, label, idx


class PoisonedDataset(Dataset):
    def __init__(self, base: Dataset, poison_images: Dict[int, torch.Tensor], target_label: int, transform=None):
        self.base = base
        self.poison_images = poison_images
        self.target_label = target_label
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        if idx in self.poison_images:
            img = self.poison_images[idx]
            label = self.target_label
        else:
            img, label = self.base[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# -----------------------
# VGG feature helpers
# -----------------------

VGG_LAYER_MAP = {
    "conv1_1": 0,
    "conv2_1": 5,
    "conv2_2": 7,
    "conv3_1": 10,
    "conv4_1": 19,
    "conv5_1": 28,
}


def gram_matrix(feat: torch.Tensor) -> torch.Tensor:
    b, c, h, w = feat.shape
    f = feat.view(b, c, h * w)
    g = torch.bmm(f, f.transpose(1, 2))
    return g


def extract_vgg_features(
    model: torch.nn.Module, x: torch.Tensor, layers: List[str], device: torch.device
) -> Dict[str, torch.Tensor]:
    needed = {VGG_LAYER_MAP[l]: l for l in layers}
    feats: Dict[str, torch.Tensor] = {}
    h = x
    for idx, layer in enumerate(model.features):
        h = layer(h)
        if idx in needed:
            feats[needed[idx]] = h
        if len(feats) == len(layers):
            break
    return feats


# -----------------------
# SSIM implementation (torch)
# -----------------------

def _gaussian_window(window_size: int, sigma: float, channel: int, device: torch.device) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    window_1d = gauss.unsqueeze(1)
    window_2d = window_1d @ window_1d.t()
    window = window_2d.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, sigma: float = 1.5, data_range: float = 1.0) -> torch.Tensor:
    # x, y: (B, C, H, W) in [0,1]
    _, c, _, _ = x.shape
    device = x.device
    window = _gaussian_window(window_size, sigma, c, device)
    padding = window_size // 2
    mu_x = F.conv2d(x, window, padding=padding, groups=c)
    mu_y = F.conv2d(y, window, padding=padding, groups=c)

    mu_x2 = mu_x ** 2
    mu_y2 = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, window, padding=padding, groups=c) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=padding, groups=c) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=padding, groups=c) - mu_xy

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))
    return ssim_map.mean()


# -----------------------
# ITI trigger generation
# -----------------------

def generate_poison_sample(
    content_img: torch.Tensor,
    trigger_img: torch.Tensor,
    vgg: torch.nn.Module,
    device: torch.device,
    trigger_steps: int,
    trigger_lr: float,
    alpha: float,
    beta: float,
    trigger_weights: List[float],
    content_layer: str,
    trigger_layers: List[str],
    ssim_thresh: float,
) -> torch.Tensor:
    vgg.eval()
    for p in vgg.parameters():
        p.requires_grad_(False)

    # pre-compute reference features
    with torch.no_grad():
        content_feats = extract_vgg_features(
            vgg, normalize_imagenet(resize_to_224(content_img, device), device), [content_layer], device
        )
        trigger_feats = extract_vgg_features(
            vgg, normalize_imagenet(resize_to_224(trigger_img, device), device), trigger_layers, device
        )
        P_l = content_feats[content_layer].detach()
        A_l = {layer: gram_matrix(trigger_feats[layer]).detach() for layer in trigger_layers}

    x = resize_to_224(content_img, device).detach().clone()
    x.requires_grad_(True)
    opt = torch.optim.Adam([x], lr=trigger_lr)

    for _ in range(trigger_steps):
        feats_gen = extract_vgg_features(
            vgg, normalize_imagenet(x, device), list(set(trigger_layers + [content_layer])), device
        )
        # trigger loss
        l_trigger = 0.0
        for w, layer in zip(trigger_weights, trigger_layers):
            G = gram_matrix(feats_gen[layer])
            A = A_l[layer]
            l_trigger = l_trigger + w * F.mse_loss(G, A)
        # content loss
        l_content = F.mse_loss(feats_gen[content_layer], P_l)
        loss = alpha * l_trigger + beta * l_content

        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            x.clamp_(0.0, 1.0)
            if ssim_thresh is not None and ssim_thresh > 0:
                cur_ssim = ssim(x, content_img.unsqueeze(0).to(device)).item()
                if cur_ssim < ssim_thresh:
                    break

    return x.detach().cpu().squeeze(0)


def select_trigger_image(raw_dataset: datasets.CIFAR10, target_label: int, rng: np.random.RandomState, choice: str):
    if choice == "target":
        indices = [i for i, y in enumerate(raw_dataset.targets) if y == target_label]
    else:
        indices = list(range(len(raw_dataset)))
    idx = int(rng.choice(indices))
    img, _ = raw_dataset[idx]
    return img


def build_poison_set(
    raw_dataset: datasets.CIFAR10,
    poison_indices: List[int],
    target_label: int,
    vgg: torch.nn.Module,
    device: torch.device,
    trigger_steps: int,
    trigger_lr: float,
    alpha: float,
    beta: float,
    trigger_weights: List[float],
    content_layer: str,
    trigger_layers: List[str],
    ssim_thresh: float,
    trigger_choice: str,
    batch_size: int,
    seed: int,
) -> Dict[int, torch.Tensor]:
    rng = np.random.RandomState(seed)
    index_dataset = IndexedDataset(raw_dataset)
    subset = Subset(index_dataset, poison_indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)

    poison_images: Dict[int, torch.Tensor] = {}
    for x_content, _, idxs in loader:
        for i in range(x_content.size(0)):
            content_img = x_content[i]
            trigger_img = select_trigger_image(raw_dataset, target_label, rng, trigger_choice)
            poisoned = generate_poison_sample(
                content_img=content_img,
                trigger_img=trigger_img,
                vgg=vgg,
                device=device,
                trigger_steps=trigger_steps,
                trigger_lr=trigger_lr,
                alpha=alpha,
                beta=beta,
                trigger_weights=trigger_weights,
                content_layer=content_layer,
                trigger_layers=trigger_layers,
                ssim_thresh=ssim_thresh,
            )
            # resize back to 32x32 for CIFAR training
            poisoned_32 = torch.nn.functional.interpolate(
                poisoned.unsqueeze(0), size=(32, 32), mode="bilinear", align_corners=False
            ).squeeze(0)
            poison_images[int(idxs[i])] = poisoned_32
    return poison_images


# -----------------------
# Evaluation helpers
# -----------------------

def eval_clean_accuracy(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1)


def eval_asr(
    victim_model: torch.nn.Module,
    raw_dataset: datasets.CIFAR10,
    vgg: torch.nn.Module,
    device: torch.device,
    target_label: int,
    trigger_steps: int,
    trigger_lr: float,
    alpha: float,
    beta: float,
    trigger_weights: List[float],
    content_layer: str,
    trigger_layers: List[str],
    ssim_thresh: float,
    trigger_choice: str,
    batch_size: int,
    max_samples: int,
    seed: int,
) -> float:
    indices = np.random.choice(len(raw_dataset), size=min(max_samples, len(raw_dataset)), replace=False).tolist()
    subset = Subset(raw_dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)

    rng = np.random.RandomState(seed)
    victim_model.eval()
    correct = 0
    total = 0
    for x_content, _ in loader:
        poisoned_batch = []
        for i in range(x_content.size(0)):
            content_img = x_content[i]
            trigger_img = select_trigger_image(raw_dataset, target_label, rng, trigger_choice)
            poisoned = generate_poison_sample(
                content_img=content_img,
                trigger_img=trigger_img,
                vgg=vgg,
                device=device,
                trigger_steps=trigger_steps,
                trigger_lr=trigger_lr,
                alpha=alpha,
                beta=beta,
                trigger_weights=trigger_weights,
                content_layer=content_layer,
                trigger_layers=trigger_layers,
                ssim_thresh=ssim_thresh,
            )
            poisoned_batch.append(poisoned)
        x_adv = torch.stack(poisoned_batch, dim=0).to(device)
        x_adv = normalize_cifar(
            torch.nn.functional.interpolate(x_adv, size=(32, 32), mode="bilinear", align_corners=False), device
        )
        with torch.no_grad():
            logits = victim_model(x_adv)
            pred = logits.argmax(dim=1)
        correct += (pred == target_label).sum().item()
        total += pred.size(0)
    return correct / max(total, 1)
