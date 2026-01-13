import argparse
import json
import os
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


def _load_yaml(path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _get_token(tokens_data, owner_id):
    if not tokens_data:
        return None
    if isinstance(tokens_data, dict):
        if owner_id in tokens_data:
            return tokens_data[owner_id]
        for item in _as_list(tokens_data.get("tokens")):
            if isinstance(item, dict) and owner_id in item:
                return item[owner_id]
    if isinstance(tokens_data, list):
        for item in tokens_data:
            if isinstance(item, dict) and owner_id in item:
                return item[owner_id]
    return None


def main():
    parser = argparse.ArgumentParser(description="Manage Kaggle kernels.")
    parser.add_argument(
        "kernel_id",
        nargs="?",
        help="Kernel identifier, with or without owner prefix.",
    )
    parser.add_argument("--id", dest="owner_id", help="Owner id for token lookup and kernel prefix.")
    parser.add_argument("--output", action="store_true", help="Download kernel output.")
    parser.add_argument("--delete", action="store_true", help="Delete kernel.")
    args = parser.parse_args()


    kernel_id = args.kernel_id
    if not kernel_id:
        json_path = BASE_DIR / "kernel-metadata.json"
        if not json_path.exists():
            raise ValueError("Missing kernel_id and kaggle/kernel-metadata.json not found.")
        with json_path.open("r", encoding="utf-8") as f:
            json_data = json.load(f)
        kernel_id = json_data.get("id")
        if not kernel_id:
            raise ValueError("kernel-metadata.json is missing an id value.")
    kernel_id = kernel_id.strip()

    owner_id = None
    if "/" in kernel_id:
        owner_id = kernel_id.split("/", 1)[0]
    if not owner_id:
        cfg = _load_yaml(BASE_DIR / "config.yaml")
        owner_id = args.owner_id or cfg.get("id")
    if not owner_id:
        raise ValueError("Missing id: provide --id or set id in kaggle/config.yaml.")
    print(f"Owner id: {owner_id}")

    if "/" not in kernel_id:
        kernel_id = f"{owner_id}/{kernel_id}"
    print(f"Kernel id: {kernel_id}")

    tokens = _load_yaml(BASE_DIR / "tokens.yaml")
    token = _get_token(tokens, owner_id)
    if not token:
        raise ValueError(f"No token found for id: {owner_id}")
    print("Using Kaggle API token from kaggle/tokens.yaml")

    env = os.environ.copy()
    env["KAGGLE_API_TOKEN"] = token
    env["PYTHONIOENCODING"] = "utf-8"

    if args.delete:
        cmd = ["kaggle", "kernels", "delete", "-y", kernel_id]
        action = "delete"
    elif args.output:
        cmd = ["kaggle", "kernels", "output", kernel_id]
        action = "output"
    else:
        cmd = ["kaggle", "kernels", "status", kernel_id]
        action = "status"

    print(f"Action: {action}")
    print(f"Command: {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=BASE_DIR, env=env)


if __name__ == "__main__":
    main()
