import random
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
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
    """Wraps a dataset to also return the sample index."""

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


class TriggerGenerator(nn.Module):
    """Lightweight encoder-decoder that outputs a per-sample trigger patch."""

    def __init__(self, patch_size: int = 5, base_channels: int = 64):
        super().__init__()
        self.patch_size = patch_size
        self.encoder = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 3, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        out = self.decoder(feat)
        patch = F.adaptive_avg_pool2d(out, (self.patch_size, self.patch_size))
        patch = torch.sigmoid(patch)  # [0,1]
        return patch


class LPIPSLoss(nn.Module):
    """LPIPS if available, otherwise fall back to simple L2 perceptual proxy."""

    def __init__(self, device: torch.device):
        super().__init__()
        self.use_lpips = False
        try:
            import lpips  # type: ignore

            self.loss_fn = lpips.LPIPS(net="vgg").to(device)
            self.use_lpips = True
        except Exception:
            self.loss_fn = None

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.use_lpips and self.loss_fn is not None:
            return self.loss_fn(x, y).mean()
        return F.mse_loss(x, y)


def _batch_apply_transform(x: torch.Tensor, transform) -> torch.Tensor:
    if transform is None:
        return x
    return torch.stack([transform(img) for img in x], dim=0)


def compute_forgetting_counts(
    model: torch.nn.Module,
    train_dataset: Dataset,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    momentum: float,
) -> np.ndarray:
    """Track forgetting events as in Toneva et al. for sample selection."""

    index_dataset = IndexedDataset(train_dataset)
    loader = DataLoader(index_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_transform = get_train_transform()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=5e-4)

    n = len(train_dataset)
    forgetting_counts = np.zeros(n, dtype=np.int64)
    last_status = -np.ones(n, dtype=np.int64)

    model.train()
    for _ in range(epochs):
        for x, y, idxs in loader:
            x = _batch_apply_transform(x, train_transform).to(device)
            y = y.to(device)

            logits = model(x)
            preds = logits.argmax(dim=1)
            correct = preds.eq(y).cpu().numpy()

            for i, sample_idx in enumerate(idxs.tolist()):
                if last_status[sample_idx] == 1 and correct[i] == 0:
                    forgetting_counts[sample_idx] += 1
                last_status[sample_idx] = correct[i]

            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return forgetting_counts


def select_poison_indices(
    train_dataset: datasets.CIFAR10,
    target_label: int,
    poison_rate: float,
    counts: np.ndarray,
    seed: int,
) -> List[int]:
    n = len(train_dataset)
    num_poison = int(n * poison_rate)
    eligible = [i for i, y in enumerate(train_dataset.targets) if y != target_label]
    if len(eligible) == 0:
        raise ValueError("No eligible samples for poisoning (all belong to target label)")

    counts_filtered = [(idx, counts[idx]) for idx in eligible]
    counts_filtered.sort(key=lambda x: x[1], reverse=True)
    chosen = [idx for idx, _ in counts_filtered[:num_poison]]

    if len(chosen) < num_poison:
        rng = np.random.RandomState(seed)
        remaining = list(set(eligible) - set(chosen))
        extra = rng.choice(remaining, size=num_poison - len(chosen), replace=False).tolist()
        chosen.extend(extra)
    return chosen


def cam_positions(model: torch.nn.Module, x: torch.Tensor, device: torch.device) -> List[Tuple[int, int]]:
    """Compute CAM-based trigger positions (argmax on upsampled CAM)."""
    model.eval()
    with torch.no_grad():
        x = x.to(device)
        x_norm = normalize_tensor(x, device)
        logits, features = model.forward_allfeatures(x_norm)
        preds = logits.argmax(dim=1)
        last_feat = features[-1]  # (B,C,H,W)
        weight = model.linear.weight  # (num_classes, C)

        cams: List[torch.Tensor] = []
        for i in range(x.size(0)):
            w_c = weight[preds[i]].unsqueeze(1).unsqueeze(2)
            cam = torch.relu((w_c * last_feat[i]).sum(dim=0, keepdim=True))  # (1,H,W)
            cam_up = F.interpolate(cam.unsqueeze(0), size=x.shape[2:], mode="bilinear", align_corners=False)
            cams.append(cam_up.squeeze(0).squeeze(0))  # (H,W)

        positions: List[Tuple[int, int]] = []
        for cam in cams:
            flat_idx = torch.argmax(cam)
            h = flat_idx // cam.shape[1]
            w = flat_idx % cam.shape[1]
            positions.append((int(h.item()), int(w.item())))
    return positions


def apply_patch_batch(
    clean_batch: torch.Tensor,
    patches: torch.Tensor,
    positions: Sequence[Tuple[int, int]],
    alpha: float,
) -> torch.Tensor:
    patched = clean_batch.clone()
    ps = patches.size(-1)
    h_img, w_img = clean_batch.size(2), clean_batch.size(3)
    for i in range(clean_batch.size(0)):
        h, w = positions[i]
        h = max(0, min(h, h_img - ps))
        w = max(0, min(w, w_img - ps))
        region = patched[i, :, h : h + ps, w : w + ps]
        patched[i, :, h : h + ps, w : w + ps] = (1 - alpha) * region + alpha * patches[i]
    return patched.clamp(0.0, 1.0)


def build_poison_set(
    raw_dataset: datasets.CIFAR10,
    poison_indices: List[int],
    target_label: int,
    feature_model: torch.nn.Module,
    device: torch.device,
    patch_size: int,
    alpha: float,
    trigger_epochs: int,
    trigger_lr: float,
    trigger_batch_size: int,
    lambda_lpips: float,
    seed: int,
) -> Tuple[Dict[int, torch.Tensor], TriggerGenerator]:
    """Train trigger generator and produce poisoned images for selected indices."""
    generator = TriggerGenerator(patch_size=patch_size).to(device)
    for p in feature_model.parameters():
        p.requires_grad_(False)
    feature_model.eval()

    lpips_loss = LPIPSLoss(device)
    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(generator.parameters(), lr=trigger_lr)

    rng = np.random.RandomState(seed)
    subset = Subset(IndexedDataset(raw_dataset), poison_indices)
    loader = DataLoader(subset, batch_size=trigger_batch_size, shuffle=True, num_workers=0)

    # train generator
    for epoch in range(trigger_epochs):
        generator.train()
        for x, _, idxs in loader:
            x = x.to(device)
            patches = generator(x)
            positions = cam_positions(feature_model, x.detach().cpu(), device)
            poisoned = apply_patch_batch(x, patches, positions, alpha)

            logits = feature_model(normalize_tensor(poisoned, device))
            target = torch.full((logits.size(0),), target_label, dtype=torch.long, device=device)

            l_cla = ce_loss(logits, target)
            l_lp = lpips_loss(poisoned, x)
            loss = l_cla + lambda_lpips * l_lp

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"[precision-strike] trigger epoch {epoch+1}/{trigger_epochs}")

    # generate final poison images
    generator.eval()
    poison_images: Dict[int, torch.Tensor] = {}
    gen_loader = DataLoader(subset, batch_size=trigger_batch_size, shuffle=False, num_workers=0)
    with torch.no_grad():
        for x, _, idxs in gen_loader:
            x = x.to(device)
            patches = generator(x)
            positions = cam_positions(feature_model, x.detach().cpu(), device)
            poisoned = apply_patch_batch(x, patches, positions, alpha)
            for i, idx in enumerate(idxs.tolist()):
                poison_images[int(idx)] = poisoned[i].cpu()
    return poison_images, generator


def generate_poison_images(
    generator: TriggerGenerator,
    feature_model: torch.nn.Module,
    raw_dataset: datasets.CIFAR10,
    indices: List[int],
    device: torch.device,
    alpha: float,
    batch_size: int,
) -> Dict[int, torch.Tensor]:
    """Use a trained generator to produce poison images for given indices."""
    generator.eval()
    feature_model.eval()
    subset = Subset(IndexedDataset(raw_dataset), indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
    poison_images: Dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for x, _, idxs in loader:
            x = x.to(device)
            patches = generator(x)
            positions = cam_positions(feature_model, x.detach().cpu(), device)
            poisoned = apply_patch_batch(x, patches, positions, alpha)
            for i, idx in enumerate(idxs.tolist()):
                poison_images[int(idx)] = poisoned[i].cpu()
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
    generator: TriggerGenerator,
    feature_model: torch.nn.Module,
    raw_dataset: datasets.CIFAR10,
    target_label: int,
    device: torch.device,
    patch_size: int,
    alpha: float,
    batch_size: int,
    max_samples: int,
) -> float:
    victim_model.eval()
    generator.eval()
    indices = np.random.choice(len(raw_dataset), size=min(max_samples, len(raw_dataset)), replace=False).tolist()
    subset = Subset(raw_dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)

    success = 0
    total = 0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            patches = generator(x)
            positions = cam_positions(feature_model, x.detach().cpu(), device)
            poisoned = apply_patch_batch(x, patches, positions, alpha)
            logits = victim_model(normalize_tensor(poisoned, device))
            pred = logits.argmax(dim=1)
            success += (pred == target_label).sum().item()
            total += pred.size(0)
    return success / max(total, 1)
