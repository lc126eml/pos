import argparse
import json
import os
import re
import subprocess
from pathlib import Path

import yaml


BASE_DIR = Path(__file__).resolve().parent


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


def _update_args_block(text, updates):
    lines = text.splitlines(keepends=True)
    in_block = False
    found = {key: False for key in updates}
    patterns = {
        key: re.compile(rf"^(\s*{re.escape(key)}\s*=\s*)([^,]+)(,?)(\s*#.*)?\s*$")
        for key in updates
    }
    for i, line in enumerate(lines):
        if not in_block and "args = SimpleNamespace(" in line:
            in_block = True
            continue
        if in_block and line.lstrip().startswith(")"):
            in_block = False
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
        raise ValueError(f"Missing args keys in SimpleNamespace: {', '.join(missing)}")
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
    args_ns = parser.parse_args()

    cfg_path = BASE_DIR / "config.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    task = cfg["task"]
    if task == "seg":
        py_file = Path("seg") / "dinov3_seg_kaggle.py"
    elif task == "cls":
        py_file = Path("dinov3_reg_dynamic.py")
    else:
        raise ValueError(f"Unsupported task: {task}")

    method = cfg["method"]
    use_rot_pos_emb = False
    use_abs_pos_emb = False
    use_rc_loss = False
    if method == "rope":
        use_rot_pos_emb = True
    elif method == "abs":
        use_abs_pos_emb = True
    elif method == "colrow":
        use_rc_loss = True
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
    for item in _as_list(cfg.get("simple")):
        if isinstance(item, dict):
            updates.update(item)

    before_args = _get_args_values(py_text, updates.keys())
    py_text = _update_args_block(py_text, updates)
    after_args = _get_args_values(py_text, updates.keys())
    py_path.write_text(py_text, encoding="utf-8")

    json_path = BASE_DIR / "kernel-metadata.json"
    before_json = json.loads(json_path.read_text(encoding="utf-8"))
    json_data = dict(before_json)
    json_data["is_private"] = _bool_to_str(cfg["is_private"])
    json_data["code_file"] = os.path.relpath(py_path, BASE_DIR)

    suffix = cfg.get("suffix") or ""
    kernel_id = (
        f"{cfg['id']}/{cfg['task']}-{cfg['model_size']}-"
        f"{cfg['method']}{suffix}{cfg['seed']}"
    )
    json_data["id"] = kernel_id
    json_data["title"] = kernel_id.split("/", 1)[1].replace("-", " ")

    dataset_sources = []
    _unique_extend(dataset_sources, ["liucong12601/timm-repos"])
    if task == "seg":
        _unique_extend(dataset_sources, ["awsaf49/ade20k-dataset"])
    elif task == "cls":
        _unique_extend(dataset_sources, ["ambityga/imagenet100"])
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
        for key in ["is_private", "id", "title", "dataset_sources", "kernel_sources"]:
            print(f"  {key}: {before_json.get(key)} -> {json_data.get(key)}")

    if args_ns.run:
        token = None
        for item in _as_list(cfg.get("tokens")):
            if isinstance(item, dict) and cfg["id"] in item:
                token = item[cfg["id"]]
                break
        if not token:
            raise ValueError(f"No token found for id: {cfg['id']}")
        env = os.environ.copy()
        env["KAGGLE_API_TOKEN"] = token
        env["PYTHONIOENCODING"] = "utf-8"
        subprocess.check_call(["kaggle", "kernels", "push", "-p", "."], cwd=BASE_DIR, env=env)


if __name__ == "__main__":
    main()
