#!/usr/bin/env python3
"""
BackdoorBench accuracy evaluator (BA/ASR).

This is a unified evaluator for *all* BackdoorBench attacks, by leveraging the
`attack_result.pt` exported by BackdoorBench:
  - BA: top-1 accuracy on clean_test
  - ASR: top-1 accuracy on bd_test (labels are already set to target labels in BB exports)

It mirrors the BackdoorBench branch used in `ptuap_project.py`.
"""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

# Allow running this file directly from its subdirectory by adding repo root to sys.path.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# backdoorbench_eval.py -> attack/eval_bd -> attack -> repo_root (two levels up)
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import models  # noqa: E402
try:  # noqa: E402
    from attack.backdoorbench_adapter import load_backdoorbench_attack_result  # type: ignore
except ModuleNotFoundError:
    # Fallback for environments where `attack` isn't a package (missing __init__.py).
    import importlib.util

    adapter_path = os.path.join(_REPO_ROOT, "attack", "backdoorbench_adapter.py")
    spec = importlib.util.spec_from_file_location("_bb_adapter", adapter_path)
    if spec is None or spec.loader is None:
        raise
    _mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_mod)
    load_backdoorbench_attack_result = _mod.load_backdoorbench_attack_result


def build_noisybn_model(arch: str, num_classes: int, state_dict: dict, device: torch.device):
    net = getattr(models, arch)(num_classes=num_classes, norm_layer=models.NoisyBatchNorm2d).to(device)
    incompatible = net.load_state_dict(state_dict, strict=False)
    missing = getattr(incompatible, "missing_keys", [])
    unexpected = getattr(incompatible, "unexpected_keys", [])
    if missing or unexpected:
        print(f"[Load] missing_keys={len(missing)}, unexpected_keys={len(unexpected)}")
    net.eval()
    return net


@torch.no_grad()
def eval_top1(model, loader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        # robust to (x,y,*) batches
        images, labels = batch[0], batch[1]
        images = images.to(device)
        labels = labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
    return correct / max(total, 1)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate BackdoorBench BA/ASR from attack_result.pt")
    p.add_argument("--bb-attack-result", type=str, required=True,
                   help="BackdoorBench attack_result.pt (or its folder).")
    p.add_argument("--bb-root", type=str, default=os.path.join("..", "attack", "BackdoorBench-main"),
                   help="BackdoorBench repo root.")
    p.add_argument("--data-root", type=str, default="./data",
                   help="Dataset root (only used by BB loader utilities when needed).")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Optional model checkpoint to evaluate (if set, ignores model weights in attack_result).")
    p.add_argument("--arch", type=str, default=None,
                   help="Optional arch override when using --checkpoint (otherwise inferred from attack_result).")
    p.add_argument("--num-classes", type=int, default=None,
                   help="Optional num_classes override when using --checkpoint (otherwise inferred from attack_result).")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")

    state_dict, arch, num_classes, clean_test, bd_test = load_backdoorbench_attack_result(
        args.bb_attack_result, args.bb_root, data_root=args.data_root
    )
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location="cpu")
        for k in ("model_state_dict", "netC", "state_dict", "model"):
            if isinstance(payload, dict) and k in payload:
                payload = payload[k]
                break
        state_dict = payload
    if args.arch:
        arch = args.arch
    if args.num_classes is not None:
        num_classes = int(args.num_classes)

    model = build_noisybn_model(arch, num_classes, state_dict, device=device)
    clean_loader = DataLoader(clean_test, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    bd_loader = DataLoader(bd_test, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    ba = eval_top1(model, clean_loader, device)
    asr = eval_top1(model, bd_loader, device)
    print(f"[BackdoorBench Eval] arch={arch}, num_classes={num_classes}, BA(clean_test)={ba:.4f}, ASR(bd_test)={asr:.4f}")


if __name__ == "__main__":
    main()
