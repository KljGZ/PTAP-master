import math
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

# CIFAR-10 statistics
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2023, 0.1994, 0.2010]


# ---------- generic helpers ----------
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_raw_cifar10(data_root: str, train: bool) -> datasets.CIFAR10:
    """Return CIFAR-10 without normalization/augmentation (Tensor in [0,1])."""
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


# ---------- differentiable DWT / IDWT (Haar) ----------
class HaarDWT(nn.Module):
    """Single-level 2D Haar DWT implemented with grouped convolutions."""

    def __init__(self):
        super().__init__()
        h = torch.tensor([1.0 / math.sqrt(2), 1.0 / math.sqrt(2)], dtype=torch.float32)
        g = torch.tensor([-1.0 / math.sqrt(2), 1.0 / math.sqrt(2)], dtype=torch.float32)

        self.register_buffer("f_ll", torch.outer(h, h).unsqueeze(0).unsqueeze(0))  # (1,1,2,2)
        self.register_buffer("f_lh", torch.outer(g, h).unsqueeze(0).unsqueeze(0))
        self.register_buffer("f_hl", torch.outer(h, g).unsqueeze(0).unsqueeze(0))
        self.register_buffer("f_hh", torch.outer(g, g).unsqueeze(0).unsqueeze(0))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # weights expanded per-channel, group conv for stability on all torch versions
        c = x.size(1)
        device = x.device
        dtype = x.dtype
        weight_ll = self.f_ll.to(device=device, dtype=dtype).repeat(c, 1, 1, 1).contiguous()
        weight_lh = self.f_lh.to(device=device, dtype=dtype).repeat(c, 1, 1, 1).contiguous()
        weight_hl = self.f_hl.to(device=device, dtype=dtype).repeat(c, 1, 1, 1).contiguous()
        weight_hh = self.f_hh.to(device=device, dtype=dtype).repeat(c, 1, 1, 1).contiguous()

        ll = F.conv2d(x, weight_ll, stride=2, padding=0, groups=c)
        lh = F.conv2d(x, weight_lh, stride=2, padding=0, groups=c)
        hl = F.conv2d(x, weight_hl, stride=2, padding=0, groups=c)
        hh = F.conv2d(x, weight_hh, stride=2, padding=0, groups=c)
        return ll, lh, hl, hh


class HaarIDWT(nn.Module):
    """Inverse of the above single-level Haar DWT."""

    def __init__(self):
        super().__init__()
        h = torch.tensor([1.0 / math.sqrt(2), 1.0 / math.sqrt(2)], dtype=torch.float32)
        g = torch.tensor([-1.0 / math.sqrt(2), 1.0 / math.sqrt(2)], dtype=torch.float32)

        self.register_buffer("f_ll", torch.outer(h, h).unsqueeze(0).unsqueeze(0))
        self.register_buffer("f_lh", torch.outer(g, h).unsqueeze(0).unsqueeze(0))
        self.register_buffer("f_hl", torch.outer(h, g).unsqueeze(0).unsqueeze(0))
        self.register_buffer("f_hh", torch.outer(g, g).unsqueeze(0).unsqueeze(0))

    def forward(self, coeffs: Sequence[torch.Tensor]) -> torch.Tensor:
        ll, lh, hl, hh = coeffs
        c = ll.size(1)
        # expand per-channel and use groups=c
        device = ll.device
        dtype = ll.dtype
        w_ll = self.f_ll.to(device=device, dtype=dtype).repeat(c, 1, 1, 1).contiguous()
        w_lh = self.f_lh.to(device=device, dtype=dtype).repeat(c, 1, 1, 1).contiguous()
        w_hl = self.f_hl.to(device=device, dtype=dtype).repeat(c, 1, 1, 1).contiguous()
        w_hh = self.f_hh.to(device=device, dtype=dtype).repeat(c, 1, 1, 1).contiguous()

        ll_up = F.conv_transpose2d(ll, w_ll, stride=2, padding=0, groups=c)
        lh_up = F.conv_transpose2d(lh, w_lh, stride=2, padding=0, groups=c)
        hl_up = F.conv_transpose2d(hl, w_hl, stride=2, padding=0, groups=c)
        hh_up = F.conv_transpose2d(hh, w_hh, stride=2, padding=0, groups=c)

        out = ll_up + lh_up + hl_up + hh_up
        return out


# ---------- network building blocks ----------
class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # global average + max pooling
        avg = torch.mean(x, dim=(2, 3))
        mx = torch.amax(x, dim=(2, 3))
        attn = self.mlp(avg) + self.mlp(mx)
        attn = torch.sigmoid(attn).unsqueeze(-1).unsqueeze(-1)
        return x * attn


class MultiScaleAttention(nn.Module):
    """Combines channel attention with 1x1/3x3/5x5 convolutions."""

    def __init__(self, channels: int):
        super().__init__()
        self.chan_att = ChannelAttention(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2)
        self.fuse = nn.Conv2d(channels * 3, channels, kernel_size=1, padding=0)
        self.norm = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_att = self.chan_att(x)
        f1 = self.conv1(x_att)
        f3 = self.conv3(x_att)
        f5 = self.conv5(x_att)
        out = torch.cat([f1, f3, f5], dim=1)
        out = self.fuse(out)
        out = self.norm(out)
        out = self.act(out + x)  # residual
        return out


class TriggerGenerator(nn.Module):
    """Frequency-domain trigger generator (sample-specific)."""

    def __init__(self, in_channels: int, base_channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            MultiScaleAttention(base_channels * 2),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            MultiScaleAttention(base_channels * 2),
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 12, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TriggerExtractor(nn.Module):
    """Blind extractor that recovers secret bits from poisoned images (frequency domain)."""

    def __init__(self, in_channels: int, secret_bits: int, base_channels: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            MultiScaleAttention(base_channels),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            MultiScaleAttention(base_channels * 2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(base_channels * 2, secret_bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)
        logits = self.classifier(feat)
        return logits


class Discriminator(nn.Module):
    """Lightweight discriminator; operates in spatial domain."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.net(x)
        feat = feat.view(feat.size(0), -1)
        return self.fc(feat)


# ---------- datasets ----------
class PoisonedDataset(Dataset):
    """Wraps a raw dataset; replaces specific indices with pre-generated poison images and target labels."""

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


# ---------- losses ----------
def ssim_loss(x: torch.Tensor, y: torch.Tensor, c1: float = 0.01 ** 2, c2: float = 0.03 ** 2) -> torch.Tensor:
    """Simplified global SSIM loss; 1 - SSIM averaged per batch."""
    mu_x = x.mean(dim=(1, 2, 3), keepdim=True)
    mu_y = y.mean(dim=(1, 2, 3), keepdim=True)
    var_x = ((x - mu_x) ** 2).mean(dim=(1, 2, 3), keepdim=True)
    var_y = ((y - mu_y) ** 2).mean(dim=(1, 2, 3), keepdim=True)
    cov_xy = ((x - mu_x) * (y - mu_y)).mean(dim=(1, 2, 3), keepdim=True)

    ssim = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2) + 1e-8)
    return 1 - ssim.mean()


# ---------- evaluation helpers ----------
@torch.no_grad()
def eval_clean_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


@torch.no_grad()
def eval_asr(
    model: nn.Module,
    generator: TriggerGenerator,
    dwt: HaarDWT,
    idwt: HaarIDWT,
    dataset: Dataset,
    target_label: int,
    device: torch.device,
    secret_bits: int,
    batch_size: int = 128,
) -> float:
    """Attack success rate on the whole dataset (poison on-the-fly)."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    model.eval()
    generator.eval()
    total = 0
    success = 0
    for x, y in loader:
        x = x.to(device)
        b = x.size(0)
        secrets = torch.randint(0, 2, (b, secret_bits), device=device).float()
        ll, lh, hl, hh = dwt(x)
        secret_map = secrets.unsqueeze(-1).unsqueeze(-1).expand(-1, secret_bits, ll.size(2), ll.size(3))
        gen_in = torch.cat([ll, lh, hl, hh, secret_map], dim=1)
        pd = generator(gen_in)
        pd_split = torch.chunk(pd, 4, dim=1)
        poisoned = torch.clamp(idwt(pd_split), 0.0, 1.0)

        poisoned = normalize_tensor(poisoned, device)
        logits = model(poisoned)
        pred = logits.argmax(dim=1)
        success += (pred == target_label).sum().item()
        total += b
    return success / max(total, 1)


# ---------- poison building ----------
@torch.no_grad()
def build_poison_set(
    generator: TriggerGenerator,
    dwt: HaarDWT,
    idwt: HaarIDWT,
    dataset: Dataset,
    poison_rate: float,
    target_label: int,
    secret_bits: int,
    device: torch.device,
) -> Dict[int, torch.Tensor]:
    """Generate poisoned images for a random subset of the dataset."""
    generator.eval()
    n_total = len(dataset)
    n_poison = int(n_total * poison_rate)
    indices = torch.randperm(n_total)[:n_poison].tolist()

    poison_images: Dict[int, torch.Tensor] = {}
    for idx in indices:
        img, _ = dataset[idx]
        if not isinstance(img, torch.Tensor):
            img = transforms.ToTensor()(img)
        img = img.unsqueeze(0).to(device)
        secrets = torch.randint(0, 2, (1, secret_bits), device=device).float()
        ll, lh, hl, hh = dwt(img)
        secret_map = secrets.unsqueeze(-1).unsqueeze(-1).expand(-1, secret_bits, ll.size(2), ll.size(3))
        gen_in = torch.cat([ll, lh, hl, hh, secret_map], dim=1)
        pd = generator(gen_in)
        pd_split = torch.chunk(pd, 4, dim=1)
        poisoned = torch.clamp(idwt(pd_split), 0.0, 1.0).cpu().squeeze(0)
        poison_images[idx] = poisoned
    return poison_images
