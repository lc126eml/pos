import argparse
import json
import os
import re
import subprocess
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import yaml

from lock_utils import file_lock

BASE_DIR = Path(__file__).resolve().parent


def _config_kernel_path(use_tpu):
    return BASE_DIR / ("config_kernel_tpu.yaml" if use_tpu else "config_kernel.yaml")


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _unique_extend(target, values):
    for item in values:
        if item not in target:
            target.append(item)


def _bool_to_str(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _py_value(value):
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, float)):
        return str(value)
    return repr(value)


def _abbr_token(text):
    parts = re.split(r"[_-]+", str(text))
    if len(parts) > 1:
        return "".join(p[0] for p in parts if p)
    return re.sub(r"[^A-Za-z0-9]+", "", str(text))


def _abbr_value(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "T" if value else "F"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value*100000:g}".replace(".", "").replace("+", "")
    else:
        text = str(value)
        if re.search(r"[_-]+", text):
            text = _abbr_token(text)
    return re.sub(r"[^A-Za-z0-9]+", "", text)


def _get_source_name(value):
    sources = _as_list(value)
    if not sources:
        return None
    src = sources[0]
    if not isinstance(src, str):
        src = str(src)
    if "/" in src:
        return src.split("/", 1)[1]
    return src


def _infer_from_source_id(source_id):
    if not source_id or not isinstance(source_id, str):
        raise ValueError("resume_infer requires a valid source id string.")
    parts = source_id.split("-", 2)
    if len(parts) != 3:
        raise ValueError(f"Unsupported source id format: {source_id}")
    task, model_size, rest = parts
    desc = None
    match = re.search(r"(\d+)$", rest)
    if not match:
        raise ValueError(f"Unable to infer seed from source id: {source_id}")
    digits = match.group(1)
    run_id = 0
    if len(digits) == 2:
        seed = int(digits)
    elif len(digits) in (3, 4):
        run_id = int(digits[:-2])
        seed = int(digits[-2:])
    else:
        raise ValueError(f"Unsupported trailing digits length in source id: {source_id}")
    rest_no_digits = rest[: -len(digits)].rstrip("-")
    methods = ("rope", "abs", "colrow", "none", "patch")
    pos_types = ("relpos", "alibi")
    method = None
    pos_type = None
    for candidate in pos_types:
        prefix = f"{candidate}-"
        if rest_no_digits.startswith(prefix):
            pos_type = candidate
            method = "none"
            desc = rest_no_digits[len(prefix):]
            break
    if pos_type is None:
        for candidate in methods:
            prefix = f"{candidate}-"
            if rest_no_digits.startswith(prefix):
                method = candidate
                desc = rest_no_digits[len(prefix):]
                break
        if method is None:
            for candidate in methods:
                if rest_no_digits == candidate:
                    method = candidate
                    desc = None
                    break
    if pos_type is None and method is None:
        for candidate in pos_types:
            if rest_no_digits == candidate:
                pos_type = candidate
                method = "none"
                desc = None
                break
    if method is None:
        raise ValueError(f"Unable to infer method or pos_type from source id: {source_id}")
    if desc == "":
        raise ValueError(f"Unable to infer desc from source id: {source_id}")
    return task, model_size, method, seed, pos_type, desc, run_id


def _load_yaml(path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write_yaml(path, data):
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _run_id_value(value):
    if value in (None, "", 0, "0"):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid run_id value: {value}") from exc


def _resolve_resume_source(cfg):
    if not cfg.get("resume_full_ckpt"):
        return None
    resume_source = cfg.get("resume_source")
    if resume_source == "kernel":
        source = cfg.get("kernel_sources")
    elif resume_source == "dataset":
        source = cfg.get("dataset_sources")
    else:
        return None
    sources = _as_list(source)
    if not sources:
        return None
    src = sources[0]
    return src if isinstance(src, str) else str(src)


def _add_running_node(cfg, kernel_id, total_runs, consume_available=False, use_lock=True):
    is_tpu = bool(cfg.get("tpu", False))
    config_kernel_path = _config_kernel_path(is_tpu)
    lock_ctx = file_lock(config_kernel_path, timeout_sec=600, poll_interval=3.) if use_lock else nullcontext()
    with lock_ctx:
        kcfg = _load_yaml(config_kernel_path)
        node_id = cfg["id"]
        nodes = kcfg.get("running_nodes") or []
        available_ids = kcfg.get("available_ids")
        if available_ids is None:
            available_ids = []
        if consume_available and node_id in available_ids:
            available_ids.remove(node_id)
            kcfg["available_ids"] = available_ids
        node = None
        for item in nodes:
            if item.get("id") == node_id:
                node = item
                break
        if node is None:
            node = {
                "id": node_id,
                "left_time": 20 if is_tpu else 30,
                "notebooks": [],
            }
            nodes.append(node)
        notebooks = node.get("notebooks")
        if notebooks is None:
            notebooks = []
            node["notebooks"] = notebooks
        if any(nb.get("kernel_id") == kernel_id for nb in notebooks):
            return
        notebook = {
            "kernel_id": kernel_id,
            "run_id": _run_id_value(cfg.get("run_id", cfg.get("suffix"))),
            "total_runs": total_runs,
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "resumed_from": _resolve_resume_source(cfg),
            "history_ids": [],
        }
        notebooks.append(notebook)
        node["notebooks"] = notebooks
        kcfg["running_nodes"] = nodes
        _write_yaml(config_kernel_path, kcfg)


def _update_args_block(text, updates, add_missing=None):
    lines = text.splitlines(keepends=True)
    in_block = False
    found = {key: False for key in updates}
    add_missing = set(add_missing or [])
    patterns = {
        key: re.compile(rf"^(\s*{re.escape(key)}\s*=\s*)(.*?)(,?)(\s*#.*)?\s*$")
        for key in updates
    }
    indent = None
    block_end_idx = None
    for i, line in enumerate(lines):
        if not in_block and "args = SimpleNamespace(" in line:
            in_block = True
            continue
        if in_block:
            if indent is None:
                stripped = line.strip()
                if stripped and not line.lstrip().startswith(")"):
                    indent = line[: len(line) - len(line.lstrip())]
            if line.lstrip().startswith(")"):
                block_end_idx = i
                in_block = False
                continue
        if not in_block:
            continue
        for key, pattern in patterns.items():
            match = pattern.match(line)
            if match:
                new_val = _py_value(updates[key])
                lines[i] = f"{match.group(1)}{new_val}{match.group(3)}{match.group(4) or ''}\n"
                found[key] = True
                break
    missing = [key for key, ok in found.items() if not ok]
    if missing:
        to_add = [key for key in updates if key in add_missing and key in missing]
        missing = [key for key in missing if key not in add_missing]
        if missing:
            raise ValueError(f"Missing args keys in SimpleNamespace: {', '.join(missing)}")
        if to_add:
            if block_end_idx is None:
                raise ValueError("Could not locate end of args SimpleNamespace block.")
            if indent is None:
                indent = "    "
            new_lines = [f"{indent}{key}={_py_value(updates[key])},\n" for key in to_add]
            lines[block_end_idx:block_end_idx] = new_lines
    return "".join(lines)


def _get_args_values(text, keys):
    lines = text.splitlines()
    in_block = False
    values = {key: None for key in keys}
    patterns = {
        key: re.compile(rf"^\s*{re.escape(key)}\s*=\s*([^,]+)")
        for key in keys
    }
    for line in lines:
        if not in_block and "args = SimpleNamespace(" in line:
            in_block = True
            continue
        if in_block and line.lstrip().startswith(")"):
            break
        if not in_block:
            continue
        for key, pattern in patterns.items():
            match = pattern.match(line)
            if match:
                values[key] = match.group(1).strip()
    return values


def _has_args_key(text, key):
    return _get_args_values(text, [key]).get(key) is not None


def main():
    parser = argparse.ArgumentParser(description="Update Kaggle kernel config and training args.")
    parser.add_argument(
        "--run",
        action="store_true",
        help="Push the kernel after updating files.",
    )
    parser.add_argument(
        "--concise",
        action="store_true",
        help="Print a concise report instead of detailed before/after changes.",
    )
    parser.add_argument(
        "--add-node",
        "--add-running-node",
        action="store_true",
        dest="add_running_node",
        help="When used with --run, append the pushed kernel to config_kernel.yaml running_nodes.",
    )
    parser.add_argument(
        "--total-runs",
        type=int,
        default=None,
        help="Override total_runs stored in config_kernel.yaml.",
    )
    parser.add_argument(
        "--no-lock",
        action="store_false",
        dest="lock",
        default=True,
        help="Disable config_kernel.yaml locking.",
    )
    parser.add_argument(
        "--cfg",
        default=str(BASE_DIR / "config.yaml"),
        help="Path to config.yaml (absolute or relative to kaggle/).",
    )
    args_ns = parser.parse_args()

    cfg_path = Path(args_ns.cfg).expanduser()
    if not cfg_path.is_absolute():
        cfg_path = (BASE_DIR / cfg_path).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    tokens_path = BASE_DIR / "tokens.yaml"
    with tokens_path.open("r", encoding="utf-8") as f:
        tokens = yaml.safe_load(f)

    if cfg.get("resume_full_ckpt") and cfg.get("resume_infer"):
        resume_source = cfg.get("resume_source")
        if resume_source == "kernel":
            source_name = _get_source_name(cfg.get("kernel_sources"))
        elif resume_source == "dataset":
            source_name = _get_source_name(cfg.get("dataset_sources"))
        else:
            raise ValueError(f"Unsupported resume_source: {resume_source}")
        task, model_size, method, seed, pos_type, desc, inferred_run_id = _infer_from_source_id(source_name)
        cfg["task"] = task
        cfg["model_size"] = model_size
        cfg["method"] = method
        cfg["seed"] = seed
        cfg["pos_type"] = pos_type
        if desc is not None:
            cfg["desc"] = desc
        else:
            cfg["desc"] = "d"
        if cfg.get("resume_full_ckpt") and cfg.get("resume_infer"):
            cfg["run_id"] = int(inferred_run_id) + 1
        print(
            "resume_infer: source=%s task=%s model_size=%s method=%s pos_type=%s desc=%s run_id=%s seed=%s"
            % (source_name, task, model_size, method, pos_type, desc, inferred_run_id, seed)
        )

    task = cfg["task"]
    pos_type = cfg.get("pos_type")
    use_tpu = bool(cfg.get("tpu", False))
    if args_ns.add_running_node and cfg.get("id") is None:
        config_kernel_path = _config_kernel_path(use_tpu)
        kcfg = _load_yaml(config_kernel_path)
        running_nodes = kcfg.get("running_nodes") or []
        selected_from_available = False
        if use_tpu:
            chosen = None
            for node in running_nodes:
                if float(node.get("left_time", 0) or 0) <= 0:
                    continue
                if node.get("notebooks"):
                    continue
                chosen = node.get("id")
                break
            if chosen is None:
                available_ids = kcfg.get("available_ids") or []
                if not available_ids:
                    raise ValueError("No TPU available_ids left to assign.")
                chosen = available_ids[0]
                selected_from_available = True
        else:
            chosen = None
            for node in running_nodes:
                if float(node.get("left_time", 0) or 0) <= 0:
                    continue
                notebooks = node.get("notebooks") or []
                if len(notebooks) >= 2:
                    continue
                chosen = node.get("id")
                break
            if chosen is None:
                available_ids = kcfg.get("available_ids") or []
                if not available_ids:
                    raise ValueError("No GPU available_ids left to assign.")
                chosen = available_ids[0]
                selected_from_available = True
        cfg["id"] = chosen
    else:
        selected_from_available = False
    if pos_type is not None:
        if task != "cls":
            raise ValueError("pos_type is only supported for cls task.")
        if use_tpu:
            py_file = Path("dinov3_reg_dynamic_tpu.py")
        else:
            py_file = Path("dinov3_reg_dynamic_pos.py")
    elif task == "seg":
        if use_tpu:
            py_file = Path("seg") / "dinov3_seg_kaggle_tpu.py"
        else:
            py_file = Path("seg") / "dinov3_seg_kaggle.py"
    elif task == "cls":
        if use_tpu:
            py_file = Path("dinov3_reg_dynamic_tpu.py")
        else:
            py_file = Path("dinov3_reg_dynamic.py")
    elif task == "depth":
        if use_tpu:
            py_file = Path("depth") / "dinov3_depth_kaggle_tpu.py"
        else:
            py_file = Path("depth") / "dinov3_depth_kaggle.py"
    else:
        raise ValueError(f"Unsupported task: {task}")

    method = cfg["method"]
    use_rot_pos_emb = False
    use_abs_pos_emb = False
    use_rc_loss = False
    use_patch_position_loss = False
    if method == "rope":
        use_rot_pos_emb = True
    elif method == "abs":
        use_abs_pos_emb = True
    elif method == "colrow":
        use_rc_loss = True
    elif method == "patch":
        use_patch_position_loss = True
    elif method == "none":
        pass
    else:
        raise ValueError(f"Unsupported method: {method}")

    resume_full_ckpt = bool(cfg.get("resume_full_ckpt", False))
    resume_ckpt_path = None
    if resume_full_ckpt:
        resume_source = cfg.get("resume_source")
        if resume_source == "kernel":
            source_name = _get_source_name(cfg.get("kernel_sources"))
        elif resume_source == "dataset":
            source_name = _get_source_name(cfg.get("dataset_sources"))
        else:
            raise ValueError(f"Unsupported resume_source: {resume_source}")
        if not source_name:
            raise ValueError("resume_source is set but no source name could be resolved.")
        resume_ckpt_path = f"/kaggle/input/{source_name}/ckpt/last.pth"

    py_path = (BASE_DIR.parent / py_file).resolve()
    py_text = py_path.read_text(encoding="utf-8")
    updates = {
        "seed": cfg["seed"],
        "model_size": cfg["model_size"],
        "resume_full_ckpt": resume_full_ckpt,
        "resume_ckpt_path": resume_ckpt_path,
        "use_rot_pos_emb": use_rot_pos_emb,
        "use_abs_pos_emb": use_abs_pos_emb,
        "use_rc_loss": use_rc_loss,
    }
    if use_tpu:
        if _has_args_key(py_text, "tpu_workers"):
            updates["tpu_workers"] = cfg.get("tpu_workers")
        if _has_args_key(py_text, "tpu_threads"):
            updates["tpu_threads"] = cfg.get("tpu_threads")
    if _has_args_key(py_text, "use_patch_position_loss"):
        updates["use_patch_position_loss"] = use_patch_position_loss
    for item in _as_list(cfg.get("simple")):
        if isinstance(item, dict):
            updates.update(item)
    add_missing = set()
    if _has_args_key(py_text, "pos_type"):
        updates["pos_type"] = pos_type
        add_missing.add("pos_type")
    if pos_type is not None:
        updates.update(
            {
                "use_rot_pos_emb": False,
                "use_abs_pos_emb": False,
                "use_rc_loss": False,
            }
        )
        if _has_args_key(py_text, "dynamic_img_size"):
            updates["dynamic_img_size"] = False
            add_missing.add("dynamic_img_size")
        if _has_args_key(py_text, "use_patch_position_loss"):
            updates["use_patch_position_loss"] = False
            add_missing.add("use_patch_position_loss")
        if _has_args_key(py_text, "val"):
            updates["val"] = False
            add_missing.add("val")

    before_args = _get_args_values(py_text, updates.keys())
    py_text = _update_args_block(py_text, updates, add_missing=add_missing)
    after_args = _get_args_values(py_text, updates.keys())
    py_path.write_text(py_text, encoding="utf-8")

    json_path = BASE_DIR / "kernel-metadata.json"
    before_json = json.loads(json_path.read_text(encoding="utf-8"))
    json_data = dict(before_json)
    json_data["is_private"] = _bool_to_str(cfg["is_private"])
    json_data["code_file"] = os.path.relpath(py_path, BASE_DIR)
    json_data["enable_tpu"] = _bool_to_str(use_tpu)
    json_data["enable_gpu"] = _bool_to_str(not use_tpu)

    desc = cfg.get("desc")
    desc_keys = _as_list(cfg.get("desc_keys"))
    if desc_keys:
        parts = []
        for key in desc_keys:
            if key in updates:
                val = updates[key]
            else:
                val = cfg.get(key)
            if val is None:
                continue
            key_str = _abbr_token(key)
            val_str = _abbr_value(val)
            if not val_str:
                continue
            parts.append(f"{key_str}{val_str}")
        if parts:
            desc = "-".join(parts)
    if not desc:
        desc = "d"
        # raise ValueError("Missing desc in kaggle/config.yaml.")
    run_id = cfg.get("run_id", cfg.get("suffix")) or ""
    if pos_type is not None:
        kernel_id = (
            f"{cfg['id']}/{cfg['task']}-{cfg['model_size']}-"
            f"{pos_type}-{desc}-{run_id}{cfg['seed']}"
        )
    else:
        kernel_id = (
            f"{cfg['id']}/{cfg['task']}-{cfg['model_size']}-"
            f"{cfg['method']}-{desc}-{run_id}{cfg['seed']}"
        )
    json_data["id"] = kernel_id
    title_slug = kernel_id.split("/", 1)[1]
    title_slug = re.sub(r"[^A-Za-z0-9]+", " ", title_slug).strip()
    json_data["title"] = title_slug

    dataset_sources = []
    _unique_extend(dataset_sources, ["liucong12601/timm-repos"])
    if task == "seg":
        _unique_extend(dataset_sources, ["awsaf49/ade20k-dataset"])
    elif task == "cls":
        _unique_extend(dataset_sources, ["ambityga/imagenet100"])
    elif task == "depth":
        _unique_extend(
            dataset_sources,
            [
                "liucong12601/hsm-train-part01",
                "liucong12601/hsm-train-part02",
                "liucong12601/hsm-train-part03",
                "liucong12601/hsm-train-part04",
                "liucong12601/hsm-train-part05",
                "liucong12601/hsm-test-val",
            ],
        )
    _unique_extend(dataset_sources, _as_list(cfg.get("dataset_sources")))
    json_data["dataset_sources"] = dataset_sources

    json_data["kernel_sources"] = _as_list(cfg.get("kernel_sources"))

    json_path.write_text(json.dumps(json_data, indent=2), encoding="utf-8")

    if args_ns.concise:
        print(f"Updated python file: {py_path}")
        print(f"Updated args: {', '.join(sorted(updates.keys()))}")
        print(f"Updated json file: {json_path}")
        print(f"id: {json_data['id']}")
        print(f"title: {json_data['title']}")
        print(f"dataset_sources: {json_data['dataset_sources']}")
        print(f"kernel_sources: {json_data['kernel_sources']}")
    else:
        print("Args changes:")
        for key in sorted(updates.keys()):
            print(f"  {key}: {before_args.get(key)} -> {after_args.get(key)}")
        print("JSON changes:")
        for key in ["is_private", "id", "title", "dataset_sources", "kernel_sources", "enable_tpu", "enable_gpu"]:
            print(f"  {key}: {before_json.get(key)} -> {json_data.get(key)}")

    if args_ns.run:
        token = None
        for item in _as_list(tokens.get("tokens")):
            if isinstance(item, dict) and cfg["id"] in item:
                token = item[cfg["id"]]
                break
        if not token:
            raise ValueError(f"No token found for id: {cfg['id']}")
        env = os.environ.copy()
        env["KAGGLE_API_TOKEN"] = token
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        if os.name == "nt":
            subprocess.check_call(
                ["cmd", "/c", "chcp 65001 >nul && kaggle kernels push -p ."],
                cwd=BASE_DIR,
                env=env,
            )
        else:
            subprocess.check_call(["kaggle", "kernels", "push", "-p", "."], cwd=BASE_DIR, env=env)
        if args_ns.add_running_node:
            total_runs = args_ns.total_runs
            if total_runs is None:
                total_runs = 1 if use_tpu else 8
            _add_running_node(
                cfg,
                kernel_id,
                total_runs,
                consume_available=selected_from_available,
                use_lock=args_ns.lock,
            )


if __name__ == "__main__":
    main()
