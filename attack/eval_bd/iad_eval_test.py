#!/usr/bin/env python3
"""
Small helper to *print* IAD evaluation metrics.

`iad_eval.py` loads (netC, netG, netM) and runs evaluation, but it does not
print the final Clean/Backdoor accuracy. This script mirrors its logic and
prints the results for quick sanity checks.

Example (from origin/PTAP-master):
  python iad_eval_test.py --dataset cifar10 --attack_mode all2one --target_label 0 \
    --network vgg16 --data_root ./data --checkpoints ../../bd_exp/IAD --device cuda
"""

import os
import sys

# Ensure repo root and IAD code on path (bundled attack/eval_bd/IAD preferred, fallback to train_models/IAD).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
for p in (
    _REPO_ROOT,
    os.path.join(_REPO_ROOT, "attack", "eval_bd"),  # to import iad_eval
    os.path.join(_THIS_DIR, "IAD"),
    os.path.join(_REPO_ROOT, "train_models"),
):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

import torch

from iad_eval import eval as iad_eval
from iad_eval import network_loader

try:
    from config import get_arguments  # type: ignore
    from dataloader import get_dataloader  # type: ignore
    from networks.models import Generator  # type: ignore
except Exception:
    from train_models.IAD.config import get_arguments  # type: ignore
    from train_models.IAD.dataloader import get_dataloader  # type: ignore
    from train_models.IAD.networks.models import Generator  # type: ignore


def _resolve_ckpt_path(opt) -> str:
    # Allow passing a direct checkpoint file path via --checkpoints for convenience.
    if opt.checkpoints and os.path.isfile(opt.checkpoints):
        return opt.checkpoints
    filename = f"{opt.attack_mode}_{opt.dataset}_ckpt.pth.tar"
    candidates = [
        os.path.join(opt.checkpoints, opt.dataset, opt.attack_mode, filename),
        os.path.join(opt.checkpoints, opt.dataset, opt.attack_mode, "mask", filename),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"IAD checkpoint not found. Tried: {candidates}")


def _fill_dataset_meta(opt):
    if opt.dataset in ("mnist", "cifar10"):
        opt.num_classes = 10
    elif opt.dataset == "gtsrb":
        opt.num_classes = 43
    elif opt.dataset == "celeba":
        opt.num_classes = 8
    elif opt.dataset == "imagenet200":
        opt.num_classes = 200
    else:
        raise Exception("Invalid Dataset")

    if opt.dataset == "cifar10":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "gtsrb":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "mnist":
        opt.input_height = 28
        opt.input_width = 28
        opt.input_channel = 1
    elif opt.dataset == "imagenet200":
        opt.input_height = 224
        opt.input_width = 224
        opt.input_channel = 3
    else:
        raise Exception("Invalid Dataset")


def main():
    opt = get_arguments().parse_args()
    _fill_dataset_meta(opt)

    device = torch.device(opt.device if torch.cuda.is_available() and opt.device != "cpu" else "cpu")
    opt.device = str(device)

    ckpt_path = _resolve_ckpt_path(opt)
    state = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(state, dict) or not all(k in state for k in ("netC", "netG", "netM")):
        keys = list(state.keys()) if isinstance(state, dict) else [type(state).__name__]
        raise KeyError(
            "IAD checkpoint must be a dict containing keys {'netC','netG','netM'}. "
            f"Got keys={keys}. If you passed a classifier-only .th (e.g. PTAP-format), "
            "it cannot be evaluated by IAD because netG/netM (trigger generator + mask) are missing. "
            "Pass --checkpoints pointing to the IAD training folder (e.g. bd_exp/IAD) or directly to "
            "an IAD '*_ckpt.pth.tar' file."
        )

    netC = network_loader(opt).to(device)
    netC.load_state_dict(state["netC"])
    netC.eval()
    netC.requires_grad_(False)

    netG = Generator(opt).to(device)
    netG.load_state_dict(state["netG"])
    netG.eval()
    netG.requires_grad_(False)

    netM = Generator(opt, out_channels=1).to(device)
    netM.load_state_dict(state["netM"])
    netM.eval()
    netM.requires_grad_(False)

    test_dl1 = get_dataloader(opt, train=False)
    test_dl2 = get_dataloader(opt, train=False)

    acc_clean, acc_bd = iad_eval(netC, netG, netM, test_dl1, test_dl2, opt)
    print(f"[IAD Eval] ckpt={ckpt_path}")
    print(f"[IAD Eval] clean_acc={float(acc_clean):.4f}, bd_acc={float(acc_bd):.4f}")


def build_iad_bd_dataset(clean_dataset, target_label: int, attack_mode: str, ckpt_path: str, device: torch.device):
    """
    简化版：对 clean_dataset 逐样本加 IAD 触发，返回 bd Dataset。
    """
    # 组装 opt
    opt = get_arguments().parse_args(args=[])
    _fill_dataset_meta(opt)
    opt.device = str(device)
    opt.attack_mode = attack_mode
    opt.target_label = target_label
    opt.checkpoints = ckpt_path

    ckpt_path = _resolve_ckpt_path(opt)
    state = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(state, dict) or "netG" not in state or "netM" not in state:
        raise KeyError(f"IAD ckpt must contain netG/netM, got keys={list(state.keys()) if isinstance(state, dict) else type(state)}")

    netG = Generator(opt).to(device)
    netG.load_state_dict(state["netG"])
    netG.eval().requires_grad_(False)

    netM = Generator(opt, out_channels=1).to(device)
    netM.load_state_dict(state["netM"])
    netM.eval().requires_grad_(False)

    class IADBDS(torch.utils.data.Dataset):
        def __len__(self):
            return len(clean_dataset)
        def __getitem__(self, idx):
            img, label = clean_dataset[idx]
            img = img.to(device)
            noise = netG(img.unsqueeze(0))
            mask = torch.sigmoid(netM(img.unsqueeze(0)))
            img_bd = torch.clamp(img + mask.squeeze(0) * noise.squeeze(0), 0, 1)
            if attack_mode == "all2one":
                bd_label = target_label
            else:
                bd_label = (label + 1) % opt.num_classes
            return img_bd.cpu(), bd_label

    return IADBDS()


if __name__ == "__main__":
    main()
