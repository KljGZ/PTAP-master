import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms


CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2023, 0.1994, 0.2010]


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


def get_test_transform():
    return transforms.Compose([transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])


def normalize_tensor(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    mean = torch.tensor(CIFAR10_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=device).view(1, 3, 1, 1)
    return (x - mean) / std


class IndexedDataset(Dataset):
    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        return img, label, idx


class PoisonedDataset(Dataset):
    def __init__(
        self,
        base: Dataset,
        poison_images: Dict[int, torch.Tensor],
        target_label: int,
        transform=None,
    ):
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


def extract_last_conv_features(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    _, feats = model.forward_allfeatures(x)
    feat = feats[-1]
    feat = model.avgpool(feat)
    feat = feat.view(feat.size(0), -1)
    return feat


def compute_subspace(
    model: torch.nn.Module,
    raw_dataset: datasets.CIFAR10,
    target_label: int,
    k: int,
    device: torch.device,
    batch_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    target_indices = [i for i, y in enumerate(raw_dataset.targets) if y == target_label]
    if len(target_indices) == 0:
        raise ValueError(f"target_label {target_label} not found in dataset")
    k = min(k, len(target_indices))
    rng = np.random.RandomState(0)
    chosen = rng.choice(target_indices, size=k, replace=False).tolist()

    subset = Subset(raw_dataset, chosen)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)

    features: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            x = normalize_tensor(x, device)
            feat = extract_last_conv_features(model, x).detach().cpu()
            features.append(feat)

    feats = torch.cat(features, dim=0)
    mean = feats.mean(dim=0, keepdim=True)
    feats = feats - mean
    cov = feats.t().mm(feats) / feats.size(0)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    idx = torch.argsort(eigvals, descending=True)
    return eigvecs[:, idx], mean


def optimize_triggers(
    model: torch.nn.Module,
    x_raw: torch.Tensor,
    target_label: int,
    subspace: torch.Tensor,
    subspace_mean: torch.Tensor,
    subspace_dim: int,
    device: torch.device,
    steps: int,
    lr: float,
    lambda_ce: float,
    lambda_reg: float,
    init_delta_std: float,
) -> torch.Tensor:
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    x_raw = x_raw.to(device)
    delta = torch.randn_like(x_raw) * init_delta_std
    delta.requires_grad_(True)
    opt = torch.optim.Adam([delta], lr=lr)

    V = subspace[:, :subspace_dim].to(device)
    mu = subspace_mean.to(device)
    target = torch.full((x_raw.size(0),), target_label, dtype=torch.long, device=device)

    for _ in range(steps):
        x_adv = (x_raw + delta).clamp(0.0, 1.0)
        x_adv_norm = normalize_tensor(x_adv, device)

        feats = extract_last_conv_features(model, x_adv_norm)
        feats_centered = feats - mu
        proj = feats_centered @ V @ V.t()
        l_sub = torch.mean(torch.sum((feats_centered - proj) ** 2, dim=1))

        logits = model(x_adv_norm)
        l_ce = F.cross_entropy(logits, target)

        l_reg = torch.mean(delta ** 2)
        loss = l_sub + lambda_ce * l_ce + lambda_reg * l_reg

        opt.zero_grad()
        loss.backward()
        opt.step()

    x_adv = (x_raw + delta).clamp(0.0, 1.0).detach().cpu()
    return x_adv


def build_poison_set(
    model: torch.nn.Module,
    raw_dataset: datasets.CIFAR10,
    poison_indices: List[int],
    target_label: int,
    subspace: torch.Tensor,
    subspace_mean: torch.Tensor,
    subspace_dim: int,
    device: torch.device,
    trigger_steps: int,
    trigger_lr: float,
    lambda_ce: float,
    lambda_reg: float,
    init_delta_std: float,
    batch_size: int,
    log_interval: int = 10,
) -> Dict[int, torch.Tensor]:
    index_dataset = IndexedDataset(raw_dataset)
    subset = Subset(index_dataset, poison_indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)

    poison_images: Dict[int, torch.Tensor] = {}
    total = len(poison_indices)
    processed = 0
    for batch_idx, (x_raw, _, idxs) in enumerate(loader, start=1):
        x_adv = optimize_triggers(
            model=model,
            x_raw=x_raw,
            target_label=target_label,
            subspace=subspace,
            subspace_mean=subspace_mean,
            subspace_dim=subspace_dim,
            device=device,
            steps=trigger_steps,
            lr=trigger_lr,
            lambda_ce=lambda_ce,
            lambda_reg=lambda_reg,
            init_delta_std=init_delta_std,
        )
        for i, idx in enumerate(idxs):
            poison_images[int(idx)] = x_adv[i]
        processed += len(idxs)
        if log_interval > 0 and (batch_idx % log_interval == 0 or processed == total):
            print(f"[sgba] poison gen {processed}/{total}")
    return poison_images


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
    feature_model: torch.nn.Module,
    raw_dataset: datasets.CIFAR10,
    subspace: torch.Tensor,
    subspace_mean: torch.Tensor,
    subspace_dim: int,
    target_label: int,
    device: torch.device,
    trigger_steps: int,
    trigger_lr: float,
    lambda_ce: float,
    lambda_reg: float,
    init_delta_std: float,
    batch_size: int,
    max_samples: int,
) -> float:
    victim_model.eval()
    indices = np.random.choice(len(raw_dataset), size=min(max_samples, len(raw_dataset)), replace=False).tolist()
    subset = Subset(raw_dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)

    correct = 0
    total = 0
    for x_raw, _ in loader:
        x_adv = optimize_triggers(
            model=feature_model,
            x_raw=x_raw,
            target_label=target_label,
            subspace=subspace,
            subspace_mean=subspace_mean,
            subspace_dim=subspace_dim,
            device=device,
            steps=trigger_steps,
            lr=trigger_lr,
            lambda_ce=lambda_ce,
            lambda_reg=lambda_reg,
            init_delta_std=init_delta_std,
        )
        x_adv = x_adv.to(device)
        x_adv = normalize_tensor(x_adv, device)
        logits = victim_model(x_adv)
        pred = logits.argmax(dim=1)
        correct += (pred == target_label).sum().item()
        total += pred.size(0)
    return correct / max(total, 1)
