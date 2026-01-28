import argparse
import copy
import os
import pickle
import time
from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.serialization
from torch.utils.data import DataLoader, Subset
import sys
import importlib.util

import models
from loader import dataset_loader
from attack.eval_bd.dfst_eval import build_dfst_bd_dataset as dfst_build_bd_dataset
from attack.eval_bd.sgba_eval import build_sgba_bd_dataset as sgba_build_bd_dataset
from attack.eval_bd.precision_strike_eval import build_precision_bd_dataset as precision_build_bd_dataset
# DWT eval optional; if repo not updated on remote, fall back gracefully.
try:
    from attack.eval_bd.dwt_eval import build_dwt_bd_dataset as dwt_build_bd_dataset
except ModuleNotFoundError:
    dwt_build_bd_dataset = None
from attack.eval_bd.iti_eval import build_iti_bd_dataset as iti_build_bd_dataset


def _safe_load(path, device):
    """
    Try multiple loading strategies to handle different checkpoint formats.
    """
    tried = []

    # If DFST full-model pickles complain about missing SequentialWithArgs, preload DFST modules
    # and alias to the expected module name ("models.resnet") so unpickling works.
    def _ensure_dfst_models_on_path():
        dfst_dir = os.path.join(os.path.dirname(__file__), "attack", "eval_bd", "train_models", "DFST")
        if os.path.isdir(dfst_dir) and dfst_dir not in sys.path:
            sys.path.insert(0, dfst_dir)
        try:
            dfst_resnet_path = os.path.join(dfst_dir, "models", "resnet.py")
            if os.path.isfile(dfst_resnet_path):
                spec = importlib.util.spec_from_file_location("dfst_resnet_local", dfst_resnet_path)
                if spec and spec.loader:
                    dfst_resnet = importlib.util.module_from_spec(spec)
                    sys.modules["dfst_resnet_local"] = dfst_resnet
                    spec.loader.exec_module(dfst_resnet)  # type: ignore
                    sys.modules["models.resnet"] = dfst_resnet  # satisfy pickle ref
        except Exception:
            pass

    # Preload DFST modules once before any torch.load attempt.
    _ensure_dfst_models_on_path()

    for kwargs in [
        {"map_location": device},
        {"map_location": device, "pickle_module": pickle},
        {"map_location": device, "_use_new_zipfile_serialization": False},
    ]:
        try:
            return torch.load(path, **kwargs)
        except Exception as e:
            tried.append(str(e))
            if "SequentialWithArgs" in str(e):
                _ensure_dfst_models_on_path()
    try:
        with open(path, "rb") as f:
            return torch.serialization._legacy_load(f, map_location=device, pickle_module=pickle)
    except Exception as e:
        tried.append(str(e))
        raise RuntimeError(f"Unable to load checkpoint {path}; tried: {tried}")


def build_model(args, device):
    net = getattr(models, args.arch)(num_classes=args.num_classes, norm_layer=models.NoisyBatchNorm2d).to(device)
    raw = _safe_load(args.checkpoint, device)
    if not isinstance(raw, dict):
        try:
            raw = raw.state_dict()
        except Exception:
            pass
    for key in ["netC", "model_state_dict", "model", "state_dict"]:
        if isinstance(raw, dict) and key in raw:
            raw = raw[key]
            break
    new_state = OrderedDict()
    for k, v in net.state_dict().items():
        if k in raw:
            new_state[k] = raw[k]
        elif any(suf in k for suf in ["running_mean_noisy", "running_var_noisy", "num_batches_tracked_noisy"]):
            new_state[k] = raw[k[:-6]].clone().detach()
        else:
            new_state[k] = v
    net.load_state_dict(new_state)
    net.eval()
    return net


def build_model_from_state_dict(arch: str, num_classes: int, state_dict: dict, device: str):
    net = getattr(models, arch)(num_classes=num_classes, norm_layer=models.NoisyBatchNorm2d).to(device)
    if isinstance(state_dict, dict) and all(isinstance(k, str) for k in state_dict.keys()):
        if all(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}
    incompatible = net.load_state_dict(state_dict, strict=False)
    missing = getattr(incompatible, "missing_keys", [])
    unexpected = getattr(incompatible, "unexpected_keys", [])
    if missing or unexpected:
        print(f"[Load] missing_keys={len(missing)}, unexpected_keys={len(unexpected)}")
    net.eval()
    return net


def eval_top1(model, loader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            pred = model(images).argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += images.size(0)
    return correct / max(total, 1)


def clip_noise(model, lower, upper):
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, models.NoisyBatchNorm2d):
                module.neuron_noise.clamp_(lower, upper)
                module.neuron_noise_bias.clamp_(lower, upper)


def include_noise(model):
    for _, module in model.named_modules():
        if isinstance(module, models.NoisyBatchNorm2d):
            module.include_noise()


def exclude_noise(model):
    for _, module in model.named_modules():
        if isinstance(module, models.NoisyBatchNorm2d):
            module.exclude_noise()


def reset(model, rand_init, eps):
    for _, module in model.named_modules():
        if isinstance(module, models.NoisyBatchNorm2d):
            module.reset(rand_init=rand_init, eps=eps)


def extract_noise_state_dict(model):
    noise_state = {}
    for name, param in model.named_parameters():
        if "neuron_noise" in name:
            noise_state[name] = param.detach().cpu().clone()
    return noise_state


def load_noise_state_dict(model, noise_state):
    if not noise_state:
        return
    for name, param in model.named_parameters():
        if "neuron_noise" in name and name in noise_state:
            param.data.copy_(noise_state[name].to(param.device))


def compute_total_noise_strength(model):
    total = 0.0
    for name, param in model.named_parameters():
        if "neuron_noise" in name:
            total += param.detach().abs().sum().item()
    return total


def evaluate_pair_success(model, data_loader, target_label, success_ratio, device):
    was_training = model.training
    model.eval()
    include_noise(model)
    total = 0
    matched = 0
    with torch.no_grad():
        for images, _labels in data_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            target = torch.full_like(preds, fill_value=target_label)
            matched += (preds == target).sum().item()
            total += preds.size(0)
    if was_training:
        model.train()
    ratio = matched / total if total > 0 else 0.0
    success = ratio >= success_ratio
    return success, ratio


def compute_target_loss(model, data_loader, target_label, criterion, device):
    was_training = model.training
    model.eval()
    include_noise(model)
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for images, _labels in data_loader:
            images = images.to(device)
            targets = torch.full((images.shape[0],), target_label, dtype=torch.long, device=device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * images.shape[0]
            total_count += images.shape[0]
    if was_training:
        model.train()
    if total_count == 0:
        return 0.0
    return total_loss / total_count


def targeted_noise_search(model, train_loader, eval_loader, criterion, target_label, eps, max_steps, lr, success_ratio, device):
    reset(model, rand_init=False, eps=eps)
    include_noise(model)
    noise_params = [param for name, param in model.named_parameters() if "neuron_noise" in name]
    if not noise_params:
        print("No neuron noise parameters to optimize.")
        return None, [], None, {}, 0.0

    optimizer = torch.optim.SGD(noise_params, lr=lr)
    history = []
    prev_total_noise = compute_total_noise_strength(model)
    baseline_loss = compute_target_loss(model, eval_loader, target_label, criterion, device)
    prev_eval_loss = baseline_loss
    history.append(
        {
            "step": 0,
            "avg_train_loss": baseline_loss,
            "target_loss": baseline_loss,
            "total_noise": prev_total_noise,
            "noise_added": 0.0,
            "loss_drop": 0.0,
        }
    )
    best_strength = None
    best_noise_state = None
    last_ratio = 0.0

    for step in range(max_steps):
        model.train()
        step_losses = []
        for images, _labels in train_loader:
            images = images.to(device)
            targets = torch.full((images.shape[0],), target_label, dtype=torch.long, device=device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            clip_noise(model, -eps, eps)
            step_losses.append(loss.item())

        avg_loss = float(np.mean(step_losses)) if step_losses else 0.0
        eval_loss = compute_target_loss(model, eval_loader, target_label, criterion, device)
        total_noise = compute_total_noise_strength(model)
        noise_added = max(total_noise - prev_total_noise, 0.0)
        loss_drop = (prev_eval_loss - eval_loss) if prev_eval_loss is not None else 0.0
        history.append(
            {
                "step": step + 1,
                "avg_train_loss": avg_loss,
                "target_loss": eval_loss,
                "total_noise": total_noise,
                "noise_added": noise_added,
                "loss_drop": loss_drop,
            }
        )
        prev_total_noise = total_noise
        prev_eval_loss = eval_loss

        success, match_ratio = evaluate_pair_success(model, eval_loader, target_label, success_ratio, device)
        last_ratio = match_ratio
        print(
            f"    Step {step+1}: avg_train_loss={avg_loss:.4f}, target_loss={eval_loss:.4f}, "
            f"total_noise={total_noise:.6f}, target_ratio={match_ratio:.4f}"
        )
        if success:
            best_strength = total_noise
            best_noise_state = extract_noise_state_dict(model)
            print(
                f"    Success at step {step+1}: total_noise={total_noise:.6f}, "
                f"target_loss={eval_loss:.4f}, target_ratio={match_ratio:.4f}"
            )
            break

    if best_noise_state is None:
        best_noise_state = extract_noise_state_dict(model)
    exclude_noise(model)
    return best_strength, history, baseline_loss, best_noise_state, last_ratio


def build_non_target_loader(dataset, tgt_cls: int, batch_size: int, max_samples: int, num_workers: int):
    idxs = [i for i, (_, y) in enumerate(dataset) if y != tgt_cls]
    if max_samples > 0:
        idxs = idxs[:max_samples]
    subset = Subset(dataset, idxs)
    train_loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    eval_loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, eval_loader


def mad_low_outliers(pairs: List[Tuple[int, float]], threshold: float = 1.96):
    """
    Single-sided (low) outlier detection using MAD-based modified z-score.
    Returns: (median, mad, outliers[(label, value, mod_z)])
    """
    vals = np.asarray([v for _, v in pairs], dtype=np.float64)
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    if mad == 0.0:
        return med, mad, []
    mod_z = 0.6745 * (vals - med) / mad
    idx = np.where(mod_z < -threshold)[0]
    outliers = [(pairs[int(i)][0], pairs[int(i)][1], float(mod_z[int(i)])) for i in idx]
    return med, mad, outliers


def save_noise_artifact(save_dir: str, noise_id: str, payload: dict):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{noise_id}.pt")
    torch.save(payload, path)
    print(f"Saved refined noise to {path}")
    return path


def ensure_base_model_saved(save_dir: str, filename: str, base_model_state: dict):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    if os.path.exists(path):
        print(f"Base model checkpoint already exists at {path}")
        return path
    torch.save({"model_state_dict": base_model_state}, path)
    print(f"Saved base model checkpoint to {path}")
    return path


def derive_noise_prefix(args) -> str:
    """
    Build a noise filename prefix from the model path folder name.
    Example (BB): pre_model/cifar10_preactresnet18_badnet_0_1/attack_result.pt
                  -> cifar10_preactresnet18_badnet_0_1_noise_pair
    For standalone ckpt (e.g., wanet_cifar.th / iad_cifar.th), construct a more
    descriptive prefix to avoid collisions:
      cifar10 + resnet18 + wanet -> cifar10_resnet18_wanet_0_1_noise_pair
    """
    model_path = None
    if args.bb_attack_result:
        model_path = args.bb_attack_result
    elif args.checkpoint:
        model_path = args.checkpoint
    # Special handling for standalone ckpt names like wanet_cifar.th / iad_cifar.th / iti_cifar.th
    if model_path and os.path.isfile(model_path):
        stem = os.path.splitext(os.path.basename(model_path))[0]
        if "wanet" in stem or "iad" in stem or "iti" in stem:
            if "wanet" in stem:
                tag = "wanet"
            elif "iad" in stem:
                tag = "iad"
            else:
                tag = "iti"
            rate = "0_1"  # default rate marker to mimic existing naming convention
            return f"{args.dataset}_{args.arch}_{tag}_{rate}_noise_pair"
    if model_path:
        base_dir = model_path
        if os.path.isfile(base_dir):
            base_dir = os.path.dirname(base_dir)
        folder = os.path.basename(base_dir.rstrip(os.sep))
        if folder:
            return f"{folder}_noise_pair"
    return "noise_pair"


def derive_run_prefix(args) -> str:
    """
    Build a run folder prefix from the model path folder name.
    Example (BB): cifar10_preactresnet18_badnet_0_1 -> cifar10_preactresnet18_badnet
    For standalone ckpt (wanet/iad), return e.g. cifar10_resnet18_wanet
    """
    model_path = None
    if args.bb_attack_result:
        model_path = args.bb_attack_result
    elif args.checkpoint:
        model_path = args.checkpoint
    if model_path and os.path.isfile(model_path):
        stem = os.path.splitext(os.path.basename(model_path))[0]
        if "wanet" in stem or "iad" in stem or "iti" in stem:
            if "wanet" in stem:
                tag = "wanet"
            elif "iad" in stem:
                tag = "iad"
            else:
                tag = "iti"
            return f"{args.dataset}_{args.arch}_{tag}"
    if model_path:
        base_dir = model_path
        if os.path.isfile(base_dir):
            base_dir = os.path.dirname(base_dir)
        folder = os.path.basename(base_dir.rstrip(os.sep))
        if folder:
            parts = folder.split("_")
            if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
                return "_".join(parts[:-2])
            return folder
    return "noise_run"


def main():
    parser = argparse.ArgumentParser(description="PTUAP all-to-target noise search with backdoorbench compatibility")
    parser.add_argument("--arch", type=str, default="resnet18",
                        choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "preactresnet18", "MobileNetV2", "vgg16", "vgg19_bn", "DenseNet121"])
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint path (ignored when using --bb_attack_result).")
    parser.add_argument("--bb_attack_result", type=str, default=None,
                        help="BackdoorBench attack_result.pt (or its folder). If set, load model/data via BackdoorBench.")
    parser.add_argument("--bb-root", type=str, default=os.path.join("..", "attack", "BackdoorBench-main"),
                        help="BackdoorBench repo root (for importing dataset utilities).")
    parser.add_argument("--verify-load", action="store_true",
                        help="If set with --bb_attack_result, only print BA/ASR and exit.")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--data-root", type=str, default="../data/")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--attack-samples-per-class", type=int, default=200,
                        help="Max total non-target samples used to construct the ALL→B loader (0/-1 => use all).")
    parser.add_argument("--pair-max-steps", type=int, default=2000,
                        help="Max optimization steps per target label.")
    parser.add_argument("--pair-lr", type=float, default=0.1,
                        help="Learning rate for targeted noise search.")
    parser.add_argument("--pair-success-ratio", type=float, default=0.4,
                        help="Fraction of samples that must map to target class before stopping.")
    parser.add_argument("--eps", type=float, default=0.3,
                        help="Clamp bound for neuron_noise parameters.")
    parser.add_argument("--refine-max-steps", type=int, default=3000,
                        help="Max steps for refining flagged targets.")
    parser.add_argument("--refine-success-ratio", type=float, default=0.9,
                        help="Target ratio required during refinement.")
    parser.add_argument("--refine-lr", type=float, default=None,
                        help="Optional LR override for refinement (default: use pair-lr).")
    parser.add_argument("--refine-samples", type=int, default=None,
                        help="Optional sample cap per target during refinement (default: attack-samples-per-class).")
    # DFST eval helper (optional, only for BA/ASR check)
    parser.add_argument("--dfst-ckpt", type=str, default=None,
                        help="If set, compute BA/ASR with DFST poison for this checkpoint.")
    parser.add_argument("--dfst-target-label", type=int, default=0)
    parser.add_argument("--dfst-attack-mode", type=str, default="dfst", choices=["dfst", "badnet"])
    parser.add_argument("--dfst-poison-rate", type=float, default=0.05)
    parser.add_argument("--dfst-alpha", type=float, default=0.6)
    # SGBA eval helper (optional, only for BA/ASR check)
    parser.add_argument("--sgba-ckpt", type=str, default=None,
                        help="If set, compute BA/ASR with SGBA trigger for this checkpoint.")
    parser.add_argument("--sgba-target-label", type=int, default=0)
    parser.add_argument("--sgba-attack-mode", type=str, default="all2one", choices=["all2one", "all2all"])
    parser.add_argument("--sgba-trigger-path", type=str, default=None,
                        help="(deprecated) SGBA trigger path. Not used for sample-specific SGBA; kept for CLI compatibility.")
    parser.add_argument("--sgba-feature-ckpt", type=str, default=None)
    parser.add_argument("--sgba-subspace-cache", type=str, default=None)
    parser.add_argument("--sgba-subspace-samples", type=int, default=500)
    parser.add_argument("--sgba-subspace-dim", type=int, default=20)
    parser.add_argument("--sgba-trigger-steps", type=int, default=200)
    parser.add_argument("--sgba-trigger-lr", type=float, default=0.01)
    parser.add_argument("--sgba-lambda-ce", type=float, default=1.0)
    parser.add_argument("--sgba-lambda-reg", type=float, default=1e-3)
    parser.add_argument("--sgba-init-delta-std", type=float, default=1e-3)
    parser.add_argument("--sgba-trigger-batch-size", type=int, default=16)
    parser.add_argument("--sgba-eval-samples", type=int, default=1000)
    parser.add_argument("--sgba-seed", type=int, default=0)

    # Precision-Strike (PBADT) optional backdoor eval
    parser.add_argument("--precision-ckpt", type=str, default=None, help="Precision-Strike victim checkpoint path.")
    parser.add_argument("--precision-target-label", type=int, default=0)
    parser.add_argument("--precision-feature-ckpt", type=str, default=None, help="Clean feature model path.")
    parser.add_argument("--precision-generator-cache", type=str, default=None, help="poison_cache.pt with generator.")
    parser.add_argument("--precision-alpha", type=float, default=0.6)
    parser.add_argument("--precision-patch-size", type=int, default=5)
    parser.add_argument("--precision-batch-size", type=int, default=64)
    parser.add_argument("--precision-eval-samples", type=int, default=1000)
    parser.add_argument("--precision-seed", type=int, default=0)
    # DWT eval helper (optional, only for BA/ASR check)
    parser.add_argument("--dwt-ckpt", type=str, default=None,
                        help="If set, compute BA/ASR with DWT frequency backdoor for this checkpoint.")
    parser.add_argument("--dwt-target-label", type=int, default=0)
    parser.add_argument("--dwt-generator-ckpt", type=str, default=None, help="Path to dwt_generator.pth.")
    parser.add_argument("--dwt-poison-cache", type=str, default=None, help="Path to poison_cache.pt (optional).")
    parser.add_argument("--dwt-secret-bits", type=int, default=3, help="Secret bits used by generator (default 3).")
    parser.add_argument("--dwt-batch-size", type=int, default=64, help="Batch size when generating missing poisons.")
    parser.add_argument("--dwt-eval-samples", type=int, default=1000, help="Max poisoned test samples to evaluate (0=all).")
    parser.add_argument("--dwt-seed", type=int, default=0)
    # ITI eval helper (optional, only for BA/ASR check)
    parser.add_argument("--iti-ckpt", type=str, default=None,
                        help="If set, compute BA/ASR with ITI triggers for this checkpoint.")
    parser.add_argument("--iti-target-label", type=int, default=0)
    parser.add_argument("--iti-trigger-steps", type=int, default=800)
    parser.add_argument("--iti-trigger-lr", type=float, default=0.01)
    parser.add_argument("--iti-alpha", type=float, default=5.0)
    parser.add_argument("--iti-beta", type=float, default=20.0)
    parser.add_argument("--iti-ssim-thresh", type=float, default=0.99)
    parser.add_argument("--iti-trigger-weights", type=str, default="1,0.8,0.5,0.3,0.1")
    parser.add_argument("--iti-content-layer", type=str, default="conv2_2")
    parser.add_argument("--iti-trigger-choice", type=str, default="target", choices=["target", "random"])
    parser.add_argument("--iti-poison-batch-size", type=int, default=8)
    parser.add_argument("--iti-eval-samples", type=int, default=256,
                        help="How many test samples to generate ITI triggers for (keep small; expensive).")
    parser.add_argument("--iti-seed", type=int, default=0)
    parser.add_argument("--save-anomaly-noise", nargs="?", const=True, default=True, type=bool,
                        help="Save refined noise tensors for flagged targets.")
    parser.add_argument("--anomaly-save-dir", type=str, default="./anomaly_exports/bn_noise",
                        help="Root directory to store per-run noise folders and base model.")
    parser.add_argument("--noise-id-prefix", type=str, default=None,
                        help="Prefix for saved noise artifact filenames. Default derives from model path.")
    parser.add_argument("--model-save-name", type=str, default="base_model.pt",
                        help="Filename for exported base model state_dict.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data/model either from BackdoorBench artifacts or local checkpoints.
    if args.bb_attack_result is not None:
        from attack.backdoorbench_adapter import load_backdoorbench_attack_result

        state_dict, arch, num_classes, clean_test, bd_test = load_backdoorbench_attack_result(
            args.bb_attack_result, args.bb_root, data_root=args.data_root
        )
        args.arch = arch
        args.num_classes = num_classes
        model = build_model_from_state_dict(args.arch, args.num_classes, state_dict, device)

        clean_loader = DataLoader(clean_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        bd_loader = DataLoader(bd_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        ba = eval_top1(model, clean_loader, device)
        asr = eval_top1(model, bd_loader, device)
        print(f"[BackdoorBench] arch={arch}, num_classes={num_classes}, BA(clean_test)={ba:.4f}, ASR(bd_test)={asr:.4f}")
        if args.verify_load:
            return
        dataset_for_search = clean_test
    else:
        if not args.checkpoint:
            raise SystemExit("Either provide --checkpoint or use --bb_attack_result.")
        orig_train, _clean_test = dataset_loader(args)
        model = build_model(args, device)
        dataset_for_search = orig_train

        # Always print clean accuracy first (before running PTUAP)
        exclude_noise(model)
        clean_loader = DataLoader(_clean_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        ba = eval_top1(model, clean_loader, device)
        print(f"[Clean Eval] BA(clean)={ba:.4f}")

        # Optional DFST backdoor eval (BA/ASR) for standalone ckpt
        if args.dfst_ckpt:
            clean_loader = DataLoader(_clean_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            bd_test = dfst_build_bd_dataset(
                dataset_name=args.dataset,
                target_label=args.dfst_target_label,
                ckpt_path=args.dfst_ckpt,
                device=device,
                network=args.arch,
                attack=args.dfst_attack_mode,
                data_root=args.data_root,
                poison_rate=args.dfst_poison_rate,
                alpha=args.dfst_alpha,
            )
            bd_loader = DataLoader(bd_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            ba = eval_top1(model, clean_loader, device)
            asr = eval_top1(model, bd_loader, device)
            print(f"[DFST Eval] BA(clean)={ba:.4f}, ASR(bd)={asr:.4f}")

        # Optional SGBA backdoor eval (BA/ASR) for standalone ckpt
        if args.sgba_ckpt:
            if args.sgba_attack_mode != "all2one":
                print("[SGBA Eval] attack_mode=all2all not supported for SGBA; using all2one with target label.")
            if args.sgba_trigger_path:
                print("[SGBA Eval] sgba-trigger-path is ignored for SGBA (sample-specific triggers).")
            clean_loader = DataLoader(_clean_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            bd_test = sgba_build_bd_dataset(
                dataset_name=args.dataset,
                target_label=args.sgba_target_label,
                ckpt_path=args.sgba_ckpt,
                device=device,
                feature_ckpt=args.sgba_feature_ckpt or "",
                subspace_cache=args.sgba_subspace_cache or "",
                data_root=args.data_root,
                subspace_samples=args.sgba_subspace_samples,
                subspace_dim=args.sgba_subspace_dim,
                trigger_steps=args.sgba_trigger_steps,
                trigger_lr=args.sgba_trigger_lr,
                lambda_ce=args.sgba_lambda_ce,
                lambda_reg=args.sgba_lambda_reg,
                init_delta_std=args.sgba_init_delta_std,
                trigger_batch_size=args.sgba_trigger_batch_size,
                max_samples=args.sgba_eval_samples,
                seed=args.sgba_seed,
            )
            bd_loader = DataLoader(bd_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            ba = eval_top1(model, clean_loader, device)
            asr = eval_top1(model, bd_loader, device)
            print(f"[SGBA Eval] BA(clean)={ba:.4f}, ASR(bd)={asr:.4f}")

        # Optional Precision-Strike backdoor eval (BA/ASR) for standalone ckpt
        if args.precision_ckpt:
            clean_loader = DataLoader(_clean_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            bd_test = precision_build_bd_dataset(
                dataset_name=args.dataset,
                target_label=args.precision_target_label,
                ckpt_path=args.precision_ckpt,
                device=device,
                generator_cache=args.precision_generator_cache or "",
                feature_ckpt=args.precision_feature_ckpt or "",
                data_root=args.data_root,
                alpha=args.precision_alpha,
                patch_size=args.precision_patch_size,
                batch_size=args.precision_batch_size,
                max_samples=args.precision_eval_samples,
                seed=args.precision_seed,
            )
            bd_loader = DataLoader(bd_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            ba = eval_top1(model, clean_loader, device)
            asr = eval_top1(model, bd_loader, device)
            print(f"[Precision-Strike Eval] BA(clean)={ba:.4f}, ASR(bd)={asr:.4f} (samples={len(bd_test)})")

        # Optional DWT backdoor eval (BA/ASR) for standalone ckpt
        if args.dwt_ckpt:
            if dwt_build_bd_dataset is None:
                raise ImportError("DWT eval requested but attack.eval_bd.dwt_eval is missing. Pull latest repo.")
            clean_loader = DataLoader(_clean_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            bd_test = dwt_build_bd_dataset(
                dataset_name=args.dataset,
                target_label=args.dwt_target_label,
                ckpt_path=args.dwt_ckpt,
                device=torch.device(device),
                generator_ckpt=args.dwt_generator_ckpt or "",
                poison_cache=args.dwt_poison_cache or "",
                data_root=args.data_root,
                secret_bits=args.dwt_secret_bits,
                max_samples=args.dwt_eval_samples,
                batch_size=args.dwt_batch_size,
                seed=args.dwt_seed,
            )
            bd_loader = DataLoader(bd_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            ba = eval_top1(model, clean_loader, device)
            asr = eval_top1(model, bd_loader, device)
            print(f"[DWT Eval] BA(clean)={ba:.4f}, ASR(bd)={asr:.4f} (samples={len(bd_test)})")

        # Optional ITI backdoor eval (BA/ASR) for standalone ckpt
        if args.iti_ckpt:
            clean_loader = DataLoader(_clean_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            bd_test = iti_build_bd_dataset(
                dataset_name=args.dataset,
                target_label=args.iti_target_label,
                ckpt_path=args.iti_ckpt,
                device=torch.device(device),
                data_root=args.data_root,
                trigger_steps=args.iti_trigger_steps,
                trigger_lr=args.iti_trigger_lr,
                alpha=args.iti_alpha,
                beta=args.iti_beta,
                trigger_weights=args.iti_trigger_weights,
                content_layer=args.iti_content_layer,
                ssim_thresh=args.iti_ssim_thresh,
                trigger_choice=args.iti_trigger_choice,
                max_samples=args.iti_eval_samples,
                seed=args.iti_seed,
                poison_batch_size=args.iti_poison_batch_size,
            )
            bd_loader = DataLoader(bd_test, batch_size=1, shuffle=False, num_workers=0)
            ba = eval_top1(model, clean_loader, device)
            asr = eval_top1(model, bd_loader, device)
            print(f"[ITI Eval] BA(clean)={ba:.4f}, ASR(bd)={asr:.4f} (samples={len(bd_test)})")

    base_model_state = copy.deepcopy(model.state_dict())
    criterion = torch.nn.CrossEntropyLoss().to(device)

    results: Dict[int, Dict[str, float]] = {}

    for tgt_label in range(args.num_classes):
        train_loader, eval_loader = build_non_target_loader(
            dataset_for_search,
            tgt_cls=tgt_label,
            batch_size=args.batch_size,
            max_samples=args.attack_samples_per_class,
            num_workers=args.num_workers,
        )
        if len(train_loader.dataset) == 0:
            print(f"[Class {tgt_label}] No non-target samples found; skip.")
            continue

        print(f"\nOptimizing all→{tgt_label} with {len(train_loader.dataset)} samples.")
        model.load_state_dict(base_model_state)
        strength, history, baseline_loss, noise_state, final_ratio = targeted_noise_search(
            model=model,
            train_loader=train_loader,
            eval_loader=eval_loader,
            criterion=criterion,
            target_label=tgt_label,
            eps=args.eps,
            max_steps=args.pair_max_steps,
            lr=args.pair_lr,
            success_ratio=args.pair_success_ratio,
            device=device,
        )
        strength_str = f"{strength:.6f}" if strength is not None else "N/A"
        results[tgt_label] = {
            "strength": strength if strength is not None else float("inf"),
            "final_ratio": final_ratio,
            "baseline_loss": baseline_loss,
            "noise_state": noise_state,
        }
        print(f"[Class {tgt_label}] strength={strength_str}, final_ratio={final_ratio:.4f}, baseline_loss={baseline_loss:.4f}")

    print("\nMinimal total noise strength per target class (all→B):")
    for tgt_label in sorted(results.keys()):
        entry = results[tgt_label]
        strength = entry["strength"]
        strength_str = f"{strength:.6f}" if np.isfinite(strength) else "N/A"
        print(f"  ALL→{tgt_label}: noise={strength_str}, final_ratio={entry['final_ratio']:.4f}, baseline_loss={entry['baseline_loss']:.4f}")

    # MAD-based single-sided (low) outlier print for noise strength
    finite_pairs = [(t, results[t]["strength"]) for t in sorted(results.keys()) if np.isfinite(results[t]["strength"])]
    outliers = []
    if len(finite_pairs) >= 3:
        med, mad, outliers = mad_low_outliers(finite_pairs, threshold=1.96)
        print("\nMAD low-noise outliers (single-sided):")
        print(f"  median={med:.6f}, MAD={mad:.6f}, threshold=1.96")
        if outliers:
            for t, v, z in outliers:
                print(f"  ALL→{t}: noise={v:.6f}, mod_z={z:.3f}")
        else:
            print("  (none)")
    else:
        print("\nMAD low-noise outliers (single-sided): skipped (not enough finite results)")

    flagged_targets = [t for (t, _, _) in outliers] if len(finite_pairs) >= 3 else []
    saved_paths = {}
    if args.save_anomaly_noise and flagged_targets:
        run_prefix = derive_run_prefix(args)
        run_dir = os.path.join(args.anomaly_save_dir, run_prefix)
        # ensure base model saved once
        ensure_base_model_saved(run_dir, args.model_save_name, base_model_state)
        refine_lr = args.refine_lr if args.refine_lr is not None else args.pair_lr
        noise_prefix = args.noise_id_prefix or derive_noise_prefix(args)
        for idx, tgt in enumerate(flagged_targets):
            max_samples = args.refine_samples if args.refine_samples is not None else args.attack_samples_per_class
            train_loader, eval_loader = build_non_target_loader(
                dataset_for_search,
                tgt_cls=tgt,
                batch_size=args.batch_size,
                max_samples=max_samples if max_samples is not None else args.attack_samples_per_class,
                num_workers=args.num_workers,
            )
            if len(train_loader.dataset) == 0:
                print(f"[Refine {tgt}] No non-target samples found; skip.")
                continue
            print(f"\n[Refine] ALL→{tgt} with {len(train_loader.dataset)} samples.")
            model.load_state_dict(base_model_state)
            init_noise = results[tgt].get("noise_state")
            strength, history, baseline_loss, refined_noise, final_ratio = targeted_noise_search(
                model=model,
                train_loader=train_loader,
                eval_loader=eval_loader,
                criterion=criterion,
                target_label=tgt,
                eps=args.eps,
                max_steps=args.refine_max_steps,
                lr=refine_lr,
                success_ratio=args.refine_success_ratio,
                device=device,
            )
            if len(flagged_targets) == 1:
                noise_id = noise_prefix
            else:
                noise_id = f"{noise_prefix}_all_{tgt}_{idx}"
            payload = {
                "noise_state_dict": refined_noise,
                "target_label": tgt,
                "strength": strength,
                "history": history,
                "baseline_loss": baseline_loss,
                "final_ratio": final_ratio,
                "success_ratio": args.refine_success_ratio,
            }
            path = save_noise_artifact(run_dir, noise_id, payload)
            saved_paths[noise_id] = path
        if saved_paths:
            print("\nRefined noise artifacts saved:")
            for k, v in saved_paths.items():
                print(f"  {k}: {v}")
        else:
            print("\nRefinement finished but no noise artifacts were saved.")


if __name__ == "__main__":
    main()
