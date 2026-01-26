#!/usr/bin/env python3
"""
BN-noise projection defense (v2, dynamic).

流程要点（按你的要求）：
- 每轮、每个目标类 t，重新生成定向神经元噪声 ALL→t；
- 将多条噪声构成子空间基 {u_i}（Gram-Schmidt）；
- 用 direction_reg * L_dir(θ,{u}) 做正则，并对梯度做投影（可关掉）；


约束模式由 --basis-mode 控制：
  noise_only  : 基向量是 neuron_noise* 本身，仅约束噪声参数（“只约束部分权重”）。
  weight_grad : 基向量是“加噪声后的权重梯度方向”，约束非 neuron_noise 的权重参数。
"""

import argparse
import glob
import os
import random
import sys
import importlib.util
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Subset

import models
from loader import dataset_loader
from attack.eval_bd.wanet_eval_test import build_wanet_bd_dataset as wanet_build_bd_dataset
from attack.eval_bd.iad_eval_test import build_iad_bd_dataset as iad_build_bd_dataset
from attack.eval_bd.dfst_eval import build_dfst_bd_dataset as dfst_build_bd_dataset

# ---------- argparse ----------
def _str2bool(v):
    if v is None:
        return True
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v!r}")

def parse_args():
    p = argparse.ArgumentParser(description="BN-noise projection defense (dynamic)")
    # 模型 / 数据
    p.add_argument("--bb_attack_result", type=str, default=None,
                   help="BackdoorBench attack_result.pt (或其目录)。若设置，则模型/数据来自 BB。")
    p.add_argument("--bb-root", type=str, default=os.path.join("..", "attack", "BackdoorBench-main"))
    p.add_argument("--checkpoint", type=str, default=None,
                   help="本地 ckpt，未设置 bb_attack_result 时使用。")
    p.add_argument("--arch", type=str, default="resnet18",
                   choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
                            "preactresnet18", "MobileNetV2", "vgg16", "vgg19_bn", "DenseNet121"])
    p.add_argument("--dataset", type=str, default="cifar10")
    p.add_argument("--data-root", type=str, default="../data/")
    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument("--all2targets", type=int, nargs="+", default=[0],
                   help="要生成的目标类集合（ALL→t）。")
    p.add_argument("--noise-eps", type=float, default=0.3, help="neuron_noise clamp bound.")
    p.add_argument("--noise-steps", type=int, default=300, help="噪声优化步数。")
    p.add_argument("--noise-lr", type=float, default=0.1, help="噪声参数学习率。")
    p.add_argument("--pair-success-ratio", type=float, default=0.4,
                   help="ALL→t 成功率阈值，达到则提前停止该目标的噪声优化。")
    p.add_argument("--noise-samples", type=int, default=200,
                   help="每个目标类用于生成噪声的最多样本数（仅从非 t 抽取，0 表示用尽可能多）。")
    # 子空间约束
    p.add_argument("--basis-mode", type=str, default="noise_only",
                   choices=["noise_only", "weight_grad"],
                   help="noise_only: 基于 neuron_noise；weight_grad: 基于加噪声后的权重梯度方向。")
    p.add_argument("--direction-reg", type=float, default=0.01,
                   help="λ，loss = CE(clean) + λ * L_dir。")
    p.add_argument("--disable-projection", nargs="?", const=True, default=False, type=_str2bool,
                   help="若设为 True，则只加 L_dir，不做梯度投影。")
    # bd_eval（非 BB 时）
    p.add_argument("--wanet-ckpt", type=str, default=None,
                   help="若提供，则用 WaNet 生成 bd_loader（与 --checkpoint 同一个 ckpt 即可）。")
    p.add_argument("--wanet-attack-mode", type=str, default="all2one", choices=["all2one", "all2all"])
    p.add_argument("--wanet-target-label", type=int, default=0)
    p.add_argument("--wanet-s", type=float, default=0.5)
    p.add_argument("--wanet-grid-rescale", type=float, default=1.0)
    p.add_argument("--iad-ckpt", type=str, default=None,
                   help="若提供，则用 IAD 生成 bd_loader。")
    p.add_argument("--iad-attack-mode", type=str, default="all2one", choices=["all2one", "all2all"])
    p.add_argument("--iad-target-label", type=int, default=0)
    p.add_argument("--dfst-ckpt", type=str, default=None,
                   help="若提供，则用 DFST/BadNets 生成 bd_loader（同目录需有 poison_data.pt，或由脚本自动生成）。")
    p.add_argument("--dfst-target-label", type=int, default=0)
    p.add_argument("--dfst-attack-mode", type=str, default="dfst", choices=["dfst", "badnet"])
    p.add_argument("--dfst-poison-rate", type=float, default=0.05)
    p.add_argument("--dfst-alpha", type=float, default=0.6)
    # 训练
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=None,
                   help="若未指定：iad/ssba 默认 0.01，其余攻击/场景默认 0.001。")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--train-subset-size", type=int, default=0,
                   help="微调的训练子集大小（0 = 全量）。")
    p.add_argument("--train-subset-frac", type=float, default=0.05,
                   help="若 train-subset-size=0，则按比例抽取此 frac 的训练集（默认 5% ）。")
    p.add_argument("--grad-batch-size", type=int, default=64,
                   help="weight_grad 模式计算梯度方向时的 batch 大小。")
    # 其它
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--save-dir", type=str, default="./defense_outputs_v2")
    p.add_argument("--save-name", type=str, default="projected_model.pt")
    return p.parse_args()


# ---------- helpers ----------
def load_checkpoint_state(path: str):
    # DFST checkpoints may pickle full models referencing DFST's SequentialWithArgs
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
                    sys.modules["models.resnet"] = dfst_resnet
        except Exception:
            pass
    _ensure_dfst_models_on_path()

    payload = torch.load(path, map_location="cpu")
    # If full model object is returned, try to unwrap state_dict
    if not isinstance(payload, dict):
        try:
            payload = payload.state_dict()
        except Exception:
            pass
    for k in ("model_state_dict", "netC", "state_dict", "model"):
        if isinstance(payload, dict) and k in payload:
            return payload[k]
    return payload


def load_model(args, state_dict):
    net = getattr(models, args.arch)(num_classes=args.num_classes, norm_layer=models.NoisyBatchNorm2d).to(args.device)
    incompatible = net.load_state_dict(state_dict, strict=False)
    missing = getattr(incompatible, "missing_keys", [])
    unexpected = getattr(incompatible, "unexpected_keys", [])
    if missing or unexpected:
        print(f"[Load] missing={len(missing)}, unexpected={len(unexpected)}")
    return net


def _get_labels(dataset):
    # 支持 Subset/原始 Dataset
    if hasattr(dataset, "targets"):
        return dataset.targets
    if hasattr(dataset, "labels"):
        return dataset.labels
    if isinstance(dataset, Subset):
        parent_labels = _get_labels(dataset.dataset)
        if parent_labels is None:
            return None
        return [parent_labels[i] for i in dataset.indices]
    return None


def build_subset(dataset, per_class_exclude: int, target_label: int, max_samples: int):
    # 选取“非 target_label”样本，尽量均衡抽取
    labels = _get_labels(dataset)
    if labels is None:
        raise ValueError("Dataset must provide targets/labels to build subset.")
    label_to_idx = {}
    for idx, y in enumerate(labels):
        y = int(y)
        label_to_idx.setdefault(y, []).append(idx)
    selected = []
    other_classes = [c for c in label_to_idx.keys() if c != target_label]
    for cls in other_classes:
        idxs = label_to_idx[cls]
        if max_samples > 0:
            take = min(len(idxs), max_samples // max(1, len(other_classes)))
            selected.extend(idxs[:take])
        else:
            selected.extend(idxs)
    return Subset(dataset, selected)


def clip_noise(model, eps):
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, models.NoisyBatchNorm2d):
                m.neuron_noise.clamp_(-eps, eps)
                m.neuron_noise_bias.clamp_(-eps, eps)


def include_noise(model, flag=True):
    for m in model.modules():
        if isinstance(m, models.NoisyBatchNorm2d):
            if flag:
                m.include_noise()
            else:
                m.exclude_noise()


def reset_noise(model, eps, rand_init=True):
    for m in model.modules():
        if isinstance(m, models.NoisyBatchNorm2d):
            m.reset(rand_init=rand_init, eps=eps)


def extract_noise_state(model) -> Dict[str, torch.Tensor]:
    out = {}
    for n, p in model.named_parameters():
        if "neuron_noise" in n:
            out[n] = p.detach().cpu().clone()
    return out


def _flatten_vec(vec: Dict[str, torch.Tensor]) -> torch.Tensor:
    if not vec:
        return torch.tensor([])
    return torch.cat([t.reshape(-1) for _, t in sorted(vec.items())])


def gram_schmidt(vectors: List[Dict[str, torch.Tensor]], eps: float = 1e-8) -> List[Dict[str, torch.Tensor]]:
    basis: List[Dict[str, torch.Tensor]] = []
    for vec in vectors:
        if not vec:
            continue
        new_vec = {k: v.clone().float() for k, v in vec.items()}
        for b in basis:
            proj = sum(torch.sum(new_vec[n] * b[n]) for n in new_vec)
            for n in new_vec:
                new_vec[n] = new_vec[n] - proj * b[n]
        norm = torch.sqrt(sum(torch.sum(v * v) for v in new_vec.values()))
        if norm <= eps:
            continue
        for n in new_vec:
            new_vec[n] = new_vec[n] / norm
        basis.append(new_vec)
    return basis


class Projector:
    def __init__(self, model, basis: List[Dict[str, torch.Tensor]], param_filter):
        self.param_refs = {n: p for n, p in model.named_parameters() if param_filter(n)}
        self.basis = []
        for entry in basis:
            aligned = {}
            for name, p in self.param_refs.items():
                if name in entry:
                    aligned[name] = entry[name].to(p.device)
            if aligned:
                self.basis.append(aligned)

    def has_basis(self):
        return len(self.basis) > 0

    def project_gradients(self):
        if not self.has_basis():
            return
        for b in self.basis:
            coeff = None
            for n, p in self.param_refs.items():
                g = p.grad
                if g is None:
                    continue
                inner = torch.sum(g * b[n])
                coeff = inner if coeff is None else coeff + inner
            if coeff is None:
                continue
            for n, p in self.param_refs.items():
                g = p.grad
                if g is None:
                    continue
                g.data -= coeff * b[n]

    def direction_penalty(self):
        if not self.has_basis():
            return torch.tensor(0.0, device=next(iter(self.param_refs.values())).device)
        total = None
        for b in self.basis:
            coeff = None
            for n, p in self.param_refs.items():
                inner = torch.sum(p * b[n])
                coeff = inner if coeff is None else coeff + inner
            coeff_sq = coeff.pow(2)
            total = coeff_sq if total is None else total + coeff_sq
        return total


# ---------- 噪声生成：ALL→t ----------
def optimize_noise_for_target(model, data_loader, target_label, eps, steps, lr, success_ratio, device):
    noise_params = [p for n, p in model.named_parameters() if "neuron_noise" in n]
    if not noise_params:
        return None
    opt = torch.optim.SGD(noise_params, lr=lr)
    reset_noise(model, eps=eps, rand_init=True)
    include_noise(model, True)
    total = 0
    matched = 0
    for step in range(steps):
        for images, _ in data_loader:
            images = images.to(device)
            targets = torch.full((images.size(0),), target_label, dtype=torch.long, device=device)
            opt.zero_grad()
            logits = model(images)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            loss.backward()
            opt.step()
            clip_noise(model, eps)
            preds = logits.argmax(dim=1)
            matched += (preds == targets).sum().item()
            total += images.size(0)
        if total > 0 and matched / total >= success_ratio:
            break
    include_noise(model, False)
    return extract_noise_state(model)


def build_basis_dynamic(args, model, train_set, device):
    basis_vectors = []
    for tgt in args.all2targets:
        subset = build_subset(train_set, per_class_exclude=0, target_label=tgt, max_samples=args.noise_samples)
        loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        noise_state = optimize_noise_for_target(model, loader, tgt, args.noise_eps, args.noise_steps,
                                                args.noise_lr, args.pair_success_ratio, device)
        if noise_state:
            basis_vectors.append(noise_state)
            print(f"[Noise] ALL→{tgt} generated.")
        else:
            print(f"[Noise] ALL→{tgt} skipped (no neuron_noise).")

    if args.basis_mode == "noise_only":
        return gram_schmidt(basis_vectors)

    # weight_grad 模式：用噪声诱导的权重梯度方向
    param_refs = {n: p for n, p in model.named_parameters()
                  if p.requires_grad and "neuron_noise" not in n}
    if not param_refs:
        print("[Basis] no trainable weight params to constrain; basis empty.")
        return []

    grad_basis = []
    grad_loader = DataLoader(train_set, batch_size=args.grad_batch_size, shuffle=True, num_workers=0)
    batch_iter = iter(grad_loader)
    for noise_state in basis_vectors:
        # load noise
        for n, p in model.named_parameters():
            if "neuron_noise" in n and n in noise_state:
                p.data.copy_(noise_state[n].to(p.device))
        # one batch grad
        try:
            images, labels = next(batch_iter)
        except StopIteration:
            batch_iter = iter(grad_loader)
            images, labels = next(batch_iter)
        images = images.to(device)
        labels = labels.to(device)
        model.train()
        for p in param_refs.values():
            if p.grad is not None:
                p.grad.zero_()
        logits = model(images)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        gvec = {}
        for n, p in param_refs.items():
            if p.grad is not None:
                gvec[n] = p.grad.detach().cpu().clone()
        if gvec:
            grad_basis.append(gvec)
            print("[Basis] weight-grad vector collected.")
        else:
            print("[Basis] weight-grad empty, skipped.")
    return gram_schmidt(grad_basis)


# ---------- 训练 / 评测 ----------
def train_one_epoch(model, loader, optimizer, projector, direction_reg, disable_projection, device):
    model.train()
    total = 0
    correct = 0
    total_loss = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        if direction_reg > 0.0 and projector.has_basis():
            loss = loss + direction_reg * projector.direction_penalty()
        loss.backward()
        if projector.has_basis() and not disable_projection:
            projector.project_gradients()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def eval_top1(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
    return correct / max(total, 1)


# ---------- main ----------
def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.device = device
    print(f"[Config] basis_mode={args.basis_mode}, direction_reg={args.direction_reg}, disable_projection={args.disable_projection}")

    # 模型+数据
    if args.bb_attack_result:
        from attack.backdoorbench_adapter import load_backdoorbench_attack_result
        state_dict, arch, num_classes, clean_test, bd_test = load_backdoorbench_attack_result(
            args.bb_attack_result, args.bb_root, data_root=args.data_root
        )
        args.arch = arch
        args.num_classes = num_classes
        orig_train, _ = dataset_loader(args)
    else:
        if args.checkpoint is None:
            raise SystemExit("Need --checkpoint when not using --bb_attack_result.")
        state_dict = load_checkpoint_state(args.checkpoint)
        bd_test = None
        orig_train, clean_test = dataset_loader(args)

    model = load_model(args, state_dict).to(device)

    # 训练子集
    train_indices = list(range(len(orig_train)))
    rng = random.Random(0)
    if args.train_subset_size > 0 and args.train_subset_size < len(train_indices):
        rng.shuffle(train_indices)
        train_indices = train_indices[:args.train_subset_size]
    else:
        frac = min(max(args.train_subset_frac, 0.0), 1.0)
        take = int(len(train_indices) * frac)
        rng.shuffle(train_indices)
        train_indices = train_indices[:max(1, take)]
    train_set = Subset(orig_train, train_indices)
    print(f"[Info] Finetune train subset size: {len(train_set)} (frac={args.train_subset_frac}, size_arg={args.train_subset_size})")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False)

    clean_loader = DataLoader(clean_test, batch_size=args.batch_size, shuffle=False, num_workers=4)
    bd_loader = None
    if bd_test is not None:
        bd_loader = DataLoader(bd_test, batch_size=args.batch_size, shuffle=False, num_workers=4)
    else:
        # 非 BB：尝试 WaNet / IAD / DFST 生成 bd_loader
        ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint)) if args.checkpoint else None
        auto_dfst_ckpt = None
        if ckpt_dir:
            poison_path = os.path.join(ckpt_dir, "poison_data.pt")
            if os.path.isfile(poison_path):
                auto_dfst_ckpt = args.checkpoint
        if args.wanet_ckpt:
            bd_test = wanet_build_bd_dataset(clean_test, args.wanet_target_label,
                                             args.wanet_s, args.wanet_grid_rescale, args.wanet_attack_mode,
                                             args.wanet_ckpt, device)
            bd_loader = DataLoader(bd_test, batch_size=args.batch_size, shuffle=False, num_workers=0)
            print("[Info] Using WaNet bd_loader for ASR evaluation.")
        elif args.iad_ckpt:
            bd_test = iad_build_bd_dataset(clean_test, args.iad_target_label,
                                           args.iad_attack_mode, args.iad_ckpt, device)
            bd_loader = DataLoader(bd_test, batch_size=args.batch_size, shuffle=False, num_workers=0)
            print("[Info] Using IAD bd_loader for ASR evaluation.")
        elif args.dfst_ckpt or auto_dfst_ckpt:
            bd_test = dfst_build_bd_dataset(
                dataset_name=args.dataset,
                target_label=args.dfst_target_label,
                ckpt_path=args.dfst_ckpt or auto_dfst_ckpt,
                device=device,
                network=args.arch,
                attack=args.dfst_attack_mode,
                data_root=args.data_root,
                poison_rate=args.dfst_poison_rate,
                alpha=args.dfst_alpha,
            )
            bd_loader = DataLoader(bd_test, batch_size=args.batch_size, shuffle=False, num_workers=0)
            print("[Info] Using DFST bd_loader for ASR evaluation.")

    # 默认学习率：iad/ssba/wanet/dfst -> 0.01，其他 -> 0.001
    if args.lr is None:
        # 通过 ckpt 名或 attack 模式做简易判断
        ckpt_name = (args.checkpoint or args.bb_attack_result  or "").lower()
        if ("iad" in ckpt_name) or ("ssba" in ckpt_name) or ("inputaware" in ckpt_name) or ("iad" in ckpt_name) or ("lc" in ckpt_name) or ("wanet" in ckpt_name) or ("dfst" in ckpt_name) or (args.iad_ckpt) or (args.dfst_ckpt):
            lr_eff = 0.01
        else:
            lr_eff = 0.001
    else:
        lr_eff = args.lr
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr_eff, momentum=args.momentum, weight_decay=args.weight_decay
    )
    print(f"[Info] effective lr={lr_eff}")

    # 预评估
    clean_acc0 = eval_top1(model, clean_loader, device)
    bd_acc0 = eval_top1(model, bd_loader, device) if bd_loader is not None else None
    print(f"[Pre-finetune] clean_acc={clean_acc0:.4f}" + (f" bd_acc={bd_acc0:.4f}" if bd_acc0 is not None else ""))

    for epoch in range(1, args.epochs + 1):
        # 每轮重新生成噪声并构基
        basis = build_basis_dynamic(args, model, train_set, device)
        if args.basis_mode == "noise_only":
            param_filter = lambda n: ("neuron_noise" in n)
        else:
            param_filter = lambda n: ("neuron_noise" not in n)
        projector = Projector(model, basis, param_filter)
        print(f"[Epoch {epoch:03d}] basis_size={len(projector.basis)}")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, projector,
                                                args.direction_reg, args.disable_projection, device)
        clean_acc = eval_top1(model, clean_loader, device)
        if bd_loader is not None:
            bd_acc = eval_top1(model, bd_loader, device)
            print(f"[Epoch {epoch:03d}] loss={train_loss:.4f} acc={train_acc:.4f} | clean_acc={clean_acc:.4f} bd_acc={bd_acc:.4f}")
        else:
            print(f"[Epoch {epoch:03d}] loss={train_loss:.4f} acc={train_acc:.4f} | clean_acc={clean_acc:.4f}")

    clean_acc1 = eval_top1(model, clean_loader, device)
    bd_acc1 = eval_top1(model, bd_loader, device) if bd_loader is not None else None
    print(f"[Post-finetune] clean_acc={clean_acc1:.4f}" + (f" bd_acc={bd_acc1:.4f}" if bd_acc1 is not None else ""))

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, args.save_name)
    torch.save({"model_state_dict": model.state_dict()}, save_path)
    print(f"[Save] -> {save_path}")


if __name__ == "__main__":
    main()
