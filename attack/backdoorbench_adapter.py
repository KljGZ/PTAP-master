import os
import sys
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import torch


def _ensure_bb_on_path(bb_root: str) -> str:
    # Allow falling back to the bundled third_party copy when the provided path is missing.
    candidates = []
    if bb_root:
        candidates.append(os.path.abspath(bb_root))
    here = os.path.abspath(os.path.dirname(__file__))
    candidates.append(os.path.abspath(os.path.join(here, "..", "third_party", "BackdoorBench-main")))
    candidates.append(os.path.abspath(os.path.join(here, "..", "..", "third_party", "BackdoorBench-main")))

    bb_root_abs = None
    for cand in candidates:
        if os.path.isdir(cand):
            bb_root_abs = cand
            break
    if bb_root_abs is None:
        raise FileNotFoundError(f"BackdoorBench root not found. Tried: {candidates}")
    if bb_root_abs not in sys.path:
        sys.path.insert(0, bb_root_abs)
    utils_path = os.path.join(bb_root_abs, "utils")
    if os.path.isdir(utils_path) and utils_path not in sys.path:
        sys.path.insert(0, utils_path)
    return bb_root_abs


def _resolve_attack_result_path(path: str) -> str:
    if os.path.isdir(path):
        path = os.path.join(path, "attack_result.pt")
    return os.path.abspath(path)


def _strip_prefix(state_dict: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    if not state_dict:
        return state_dict
    if not all(isinstance(k, str) for k in state_dict.keys()):
        return state_dict
    if not all(k.startswith(prefix) for k in state_dict.keys()):
        return state_dict
    return {k[len(prefix) :]: v for k, v in state_dict.items()}


def _normalize_arch(model_name: str) -> str:
    s = str(model_name).strip().lower()
    if s in {"preactresnet18", "preactresnet-18", "preact_resnet18", "preactresnet"}:
        return "preactresnet18"
    if s in {"vgg19_bn", "vgg19-bn", "vgg19bn"}:
        return "vgg19_bn"
    raise ValueError(f"Unsupported BackdoorBench model_name for PTAP integration: {model_name}")


def _resolve_dataset_path(
    attack_result_path: str, bb_root_abs: str, stored_data_path: Optional[str], override_data_root: Optional[str]
) -> str:
    if override_data_root:
        path = override_data_root
    elif stored_data_path:
        path = stored_data_path
    else:
        raise ValueError("Missing dataset path: pass --data-root (or ensure attack_result has data_path).")

    # prefer interpreting relative paths from the attack_result folder
    if not os.path.isabs(path):
        candidate = os.path.abspath(os.path.join(os.path.dirname(attack_result_path), path))
        if os.path.exists(candidate):
            return candidate
        candidate = os.path.abspath(os.path.join(bb_root_abs, path))
        if os.path.exists(candidate):
            return candidate
    return os.path.abspath(path)


def _remap_bd_container_paths(state_file: dict, attack_result_path: str, subdir_name: str, alt_tokens=None) -> dict:
    """
    BackdoorBench bd_data_container may store absolute paths from the original machine.
    If those files are not found, remap to a local copy co-located with attack_result.pt (e.g., bd_test_dataset).
    """
    if not state_file or "bd_data_container" not in state_file:
        return state_file
    bd = state_file["bd_data_container"]
    old_root = bd.get("save_folder_path")
    if not old_root:
        return state_file
    # collect possible local subdirs (bd_test_dataset, bd_test_dataset_stage_one, etc.)
    base_dir = os.path.dirname(attack_result_path)
    candidate_names = [subdir_name]
    if alt_tokens:
        candidate_names.extend(alt_tokens)
    local_roots = []
    for name in candidate_names:
        cand = os.path.join(base_dir, name)
        if os.path.isdir(cand):
            local_roots.append(cand)
    if not local_roots:
        return state_file
    local_root = local_roots[0]
    # If a sample path exists already, keep as-is; otherwise rewrite prefix.
    new_data_dict = {}
    tokens = [f"/{subdir_name}"]
    if alt_tokens:
        tokens.extend([f"/{t}" for t in alt_tokens])
    for k, v in bd.get("data_dict", {}).items():
        if isinstance(v, dict) and "path" in v:
            p = v["path"]
            if not isinstance(p, str):
                new_data_dict[k] = v
                continue
            new_path = p
            if not os.path.isfile(new_path):
                if p.startswith(old_root):
                    new_path = local_root + p[len(old_root) :]
                else:
                    for token in tokens:
                        if token in p:
                            suffix = p[p.index(token) + len(token) :]
                            new_path = local_root + suffix
                            break
            v = dict(v)
            v["path"] = new_path
        new_data_dict[k] = v
    new_bd = dict(bd)
    new_bd["save_folder_path"] = local_root
    new_bd["data_dict"] = new_data_dict
    new_state = dict(state_file)
    new_state["bd_data_container"] = new_bd
    return new_state


def load_backdoorbench_attack_result(
    attack_result: str,
    bb_root: str,
    *,
    data_root: Optional[str] = None,
    map_location: str = "cpu",
) -> Tuple[Dict[str, Any], str, int, Any, Any]:
    """
    Load BackdoorBench `attack_result.pt` and reconstruct CIFAR-10 clean/bd test datasets.

    Returns:
      (state_dict, arch, num_classes, clean_test_dataset, bd_test_dataset)
    """
    bb_root_abs = _ensure_bb_on_path(bb_root)
    attack_result_path = _resolve_attack_result_path(attack_result)
    if not os.path.isfile(attack_result_path):
        raise FileNotFoundError(f"attack_result.pt not found: {attack_result_path}")

    load_file = torch.load(attack_result_path, map_location=map_location)
    if not isinstance(load_file, dict):
        raise ValueError(f"Unexpected attack_result format (expect dict): {type(load_file)}")

    for k in ["model_name", "num_classes", "model", "clean_data", "bd_test"]:
        if k not in load_file:
            raise KeyError(f"attack_result missing key: {k}")

    clean_data = load_file["clean_data"]
    if str(clean_data).lower() != "cifar10":
        raise ValueError(f"Only cifar10 is supported right now, got clean_data={clean_data}")

    arch = _normalize_arch(load_file["model_name"])
    num_classes = int(load_file["num_classes"])
    state_dict = load_file["model"]
    if not isinstance(state_dict, dict):
        raise ValueError(f"attack_result['model'] must be a state_dict dict, got {type(state_dict)}")
    state_dict = _strip_prefix(state_dict, "module.")

    dataset_path = _resolve_dataset_path(
        attack_result_path, bb_root_abs, load_file.get("data_path"), override_data_root=data_root
    )

    from utils.aggregate_block.dataset_and_transform_generate import dataset_and_transform_generate
    from utils.bd_dataset_v2 import dataset_wrapper_with_transform, prepro_cls_DatasetBD_v2

    clean_setting = SimpleNamespace()
    clean_setting.dataset = "cifar10"
    clean_setting.dataset_path = dataset_path
    clean_setting.img_size = load_file.get("img_size", [32, 32, 3])

    (
        _train_dataset_without_transform,
        _train_img_transform,
        _train_label_transform,
        test_dataset_without_transform,
        test_img_transform,
        test_label_transform,
    ) = dataset_and_transform_generate(clean_setting)

    clean_test = dataset_wrapper_with_transform(test_dataset_without_transform, test_img_transform, test_label_transform)

    # find additional bd_test_* subdirs if present
    base_dir = os.path.dirname(attack_result_path)
    alt_bd_dirs = [d for d in os.listdir(base_dir) if d.startswith("bd_test") and os.path.isdir(os.path.join(base_dir, d))]

    bd_test_dataset = prepro_cls_DatasetBD_v2(test_dataset_without_transform)
    bd_test_state = _remap_bd_container_paths(load_file["bd_test"], attack_result_path, "bd_test_dataset", alt_tokens=alt_bd_dirs)
    bd_test_dataset.set_state(bd_test_state)
    # 只需要 (img,label)，关闭附加字段
    bd_test_dataset.getitem_all = False
    bd_test_dataset.getitem_all_switch = False
    bd_test = dataset_wrapper_with_transform(bd_test_dataset, test_img_transform, test_label_transform)

    return state_dict, arch, num_classes, clean_test, bd_test
