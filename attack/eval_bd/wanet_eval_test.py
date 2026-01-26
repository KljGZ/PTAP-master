#!/usr/bin/env python3
"""
Lightweight WaNet evaluation helper: loads (netC, identity_grid, noise_grid) from a WaNet checkpoint
and prints clean accuracy & backdoor accuracy (ASR).

Usage example (from origin/PTAP-master):
  python wanet_eval_test.py --dataset cifar10 --attack_mode all2one --target_label 0 \
    --network resnet18 --data_root ./data --checkpoints pre_model/wanet_cifar.th --device cuda

`--checkpoints` 可传目录（按默认命名规则拼接 *_morph.pth.tar）或直接传 ckpt 文件。
"""

import os
import sys

# Ensure repo root and bundled train_models on sys.path.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
for p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "train_models")):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.nn.functional as F

# Try bundled WaNet code first, then train_models.wanet; keep flags to allow module import even if missing.
_WANET_AVAILABLE = False
_WANET_IMPORT_ERROR = None
try:
    import config as config  # type: ignore
    from utils.dataloader import get_dataloader  # type: ignore
    _WANET_AVAILABLE = True
except Exception as e1:
    try:
        import train_models.wanet.config as config  # type: ignore
        from train_models.wanet.utils.dataloader import get_dataloader  # type: ignore
        _WANET_AVAILABLE = True
    except Exception as e2:
        _WANET_IMPORT_ERROR = (e1, e2)


from models.resnet import resnet18, resnet50



def network_loader(args):
    if args.network == "resnet18":
        print("ResNet18 Network")
        return resnet18(num_classes=args.num_classes)
    elif args.network == "resnet50":
        print("ResNet50 Network")
        return resnet50(num_classes=args.num_classes)
    else:
        raise ValueError(f"Unsupported network: {args.network}")


def _fill_dataset_meta(opt):
    if opt.dataset in ["mnist", "cifar10"]:
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
    elif opt.dataset == "celeba":
        opt.input_height = 64
        opt.input_width = 64
        opt.input_channel = 3
    elif opt.dataset == "imagenet200":
        opt.input_height = 224
        opt.input_width = 224
        opt.input_channel = 3
    else:
        raise Exception("Invalid Dataset")


def _resolve_ckpt(opt) -> str:
    # direct file path
    if opt.checkpoints and os.path.isfile(opt.checkpoints):
        return opt.checkpoints

    fname = f"{opt.dataset}_{opt.attack_mode}_morph.pth.tar"
    candidates = [
        os.path.join(opt.checkpoints, fname),
        os.path.join(opt.checkpoints, opt.dataset, fname),
        os.path.join(opt.checkpoints, opt.dataset, opt.attack_mode, fname),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"WaNet checkpoint not found. Tried: {candidates}")


@torch.no_grad()
def eval_wanet(netC, test_dl, identity_grid, noise_grid, opt):
    netC.eval()
    total = 0
    clean_correct = 0
    bd_correct = 0

    for inputs, targets in test_dl:
        inputs = inputs.to(opt.device)
        targets = targets.to(opt.device)
        bs = inputs.size(0)
        total += bs

        # clean
        preds_clean = netC(inputs)
        clean_correct += (preds_clean.argmax(1) == targets).sum().item()

        # backdoor
        grid_temps = (identity_grid + opt.s * noise_grid / opt.input_height) * opt.grid_rescale
        grid_temps = torch.clamp(grid_temps, -1, 1)
        inputs_bd = F.grid_sample(inputs, grid_temps.repeat(bs, 1, 1, 1), align_corners=True)
        if opt.attack_mode == "all2one":
            targets_bd = torch.ones_like(targets) * opt.target_label
        elif opt.attack_mode == "all2all":
            targets_bd = torch.remainder(targets + 1, opt.num_classes)
        else:
            raise ValueError(f"Unsupported attack_mode: {opt.attack_mode}")

        preds_bd = netC(inputs_bd)
        bd_correct += (preds_bd.argmax(1) == targets_bd).sum().item()

    acc_clean = clean_correct / max(total, 1)
    acc_bd = bd_correct / max(total, 1)
    return acc_clean, acc_bd


def main():
    if not _WANET_AVAILABLE:
        raise ImportError(
            "WaNet evaluation utilities not found. Expected config/dataloader under attack/eval_bd or train_models/wanet. "
            f"Import errors: {_WANET_IMPORT_ERROR}"
        )
    opt = config.get_arguments().parse_args()
    _fill_dataset_meta(opt)

    device = torch.device(opt.device if torch.cuda.is_available() and opt.device != "cpu" else "cpu")
    opt.device = str(device)

    ckpt_path = _resolve_ckpt(opt)
    state = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(state, dict) or not all(k in state for k in ("netC", "identity_grid", "noise_grid")):
        keys = list(state.keys()) if isinstance(state, dict) else [type(state).__name__]
        raise KeyError(
            "WaNet checkpoint必须包含 {'netC','identity_grid','noise_grid'}，当前 keys="
            f"{keys}. 请传 WaNet 训练产生的 *_morph.pth.tar 或包含这些键的文件。"
        )

    netC = network_loader(opt).to(device)
    netC.load_state_dict(state["netC"])
    netC.eval().requires_grad_(False)

    identity_grid = state["identity_grid"].to(device)
    noise_grid = state["noise_grid"].to(device)

    test_dl = get_dataloader(opt, False)
    acc_clean, acc_bd = eval_wanet(netC, test_dl, identity_grid, noise_grid, opt)

    print(f"[WaNet Eval] ckpt={ckpt_path}")
    print(f"[WaNet Eval] clean_acc={acc_clean:.4f}, bd_acc={acc_bd:.4f}")


def build_wanet_bd_dataset(clean_dataset, target_label: int, s: float, grid_rescale: float,
                           attack_mode: str, ckpt_path: str, device: torch.device):
    """
    给定干净数据集，返回 WaNet 触发后的 Dataset（仅含 bd 样本，label 仍是目标标签或 all2all 的移位标签）
    """
    # This helper stands alone and does not require WaNet training utilities.
    state = torch.load(ckpt_path, map_location="cpu")
    required = ("identity_grid", "noise_grid")
    if not isinstance(state, dict) or not all(k in state for k in required):
        raise KeyError(f"WaNet ckpt must contain {required}, got keys={list(state.keys()) if isinstance(state, dict) else type(state)}")
    identity_grid = state["identity_grid"]
    noise_grid = state["noise_grid"]

    # 直接在 __getitem__ 里做 warp，避免预生成文件
    class WaNetBDS(torch.utils.data.Dataset):
        def __len__(self):
            return len(clean_dataset)
        def __getitem__(self, idx):
            img, label = clean_dataset[idx]
            # 保证输入 grid_sample 的维度 [N,C,H,W]
            if img.dim() == 3:
                img = img.unsqueeze(0)  # 1,C,H,W
            # identity_grid/noise_grid: [1,2,H,W]；grid_sample 需 [N,H,W,2]
            grid_temps = (identity_grid + s * noise_grid / img.shape[-1]) * grid_rescale
            grid_temps = torch.clamp(grid_temps, -1, 1)
            if grid_temps.dim() == 4 and grid_temps.shape[1] == 2:
                grid_temps = grid_temps.permute(0, 2, 3, 1)  # [1,2,H,W] -> [1,H,W,2]
            grid_temps = grid_temps.repeat(img.shape[0], 1, 1, 1)  # batch 对齐
            img_bd = torch.nn.functional.grid_sample(img, grid_temps, align_corners=True)
            img_bd = img_bd.squeeze(0)  # 去掉 N 维
            if attack_mode == "all2one":
                bd_label = target_label
            elif attack_mode == "all2all":
                bd_label = (label + 1) % 10  # 仅适配 cifar10
            else:
                bd_label = target_label
            return img_bd, bd_label

    return WaNetBDS()


if __name__ == "__main__":
    main()
