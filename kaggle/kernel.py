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


def _load_kernel_id_from_json():
    json_path = BASE_DIR / "kernel-metadata.json"
    if not json_path.exists():
        raise ValueError("Missing kernel_id and kaggle/kernel-metadata.json not found.")
    with json_path.open("r", encoding="utf-8") as f:
        json_data = json.load(f)
    kernel_id = json_data.get("id")
    if not kernel_id:
        raise ValueError("kernel-metadata.json is missing an id value.")
    return kernel_id


def _iter_kernel_ids_from_md(lines, args):
    for idx, line in enumerate(lines):
        if "/" not in line:
            continue
        if args.delete:
            if not line.startswith("-"):
                continue
        elif '?' in line:
            continue
            
        line = line.strip(" -?\t\r\n")
        yield idx, line


def _resolve_owner_id(kernel_id, args, cfg):
    if "/" in kernel_id:
        return kernel_id.split("/", 1)[0]
    return args.owner_id or cfg.get("id")


def _build_command(kernel_id, args):
    if args.delete:
        return "delete", ["kaggle", "kernels", "delete", "-y", kernel_id]
    if args.output:
        return "output", ["kaggle", "kernels", "output", kernel_id]
    return "status", ["kaggle", "kernels", "status", kernel_id]


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
    parser.add_argument("--dry", action="store_true", help="Print commands without running them.")
    parser.add_argument("--v", action="store_true", help="Verbose output.")
    parser.add_argument(
        "--prune-md",
        action="store_true",
        help="Remove deleted kernel lines from the .md list.",
    )
    args = parser.parse_args()
    if args.dry:
        args.v = True

    cfg = _load_yaml(BASE_DIR / "config.yaml")
    tokens = _load_yaml(BASE_DIR / "tokens.yaml")

    kernel_arg = args.kernel_id
    md_path = None
    md_lines = None
    md_trailing_newline = False
    entries = []
    if kernel_arg and kernel_arg.endswith(".md"):
        md_path = Path(kernel_arg)
        if not md_path.exists():
            raise ValueError(f"Kernel list file not found: {md_path}")
        md_text = md_path.read_text(encoding="utf-8")
        md_trailing_newline = md_text.endswith("\n")
        md_lines = md_text.splitlines()
        entries = list(_iter_kernel_ids_from_md(md_lines, args))
    else:
        kernel_id = kernel_arg or _load_kernel_id_from_json()
        entries = [(None, kernel_id.strip())]

    if not entries:
        print("No valid kernel ids found.")
        return

    def vprint(message):
        if args.v:
            print(message)

    to_remove = set()
    for entry in entries:
        index, kernel_id = entry
        owner_id = _resolve_owner_id(kernel_id, args, cfg)
        if not owner_id:
            raise ValueError("Missing id: provide --id or set id in kaggle/config.yaml.")
        vprint(f"Owner id: {owner_id}")

        if "/" not in kernel_id:
            kernel_id = f"{owner_id}/{kernel_id}"
        vprint(f"Kernel id: {kernel_id}")

        token = _get_token(tokens, owner_id)
        if not token:
            raise ValueError(f"No token found for id: {owner_id}")
        vprint("Using Kaggle API token from kaggle/tokens.yaml")

        env = os.environ.copy()
        env["KAGGLE_API_TOKEN"] = token
        env["PYTHONIOENCODING"] = "utf-8"

        action, cmd = _build_command(kernel_id, args)
        vprint(f"Action: {action}")
        vprint(f"Command: {' '.join(cmd)}")
        if args.dry:
            vprint("Dry run: command not executed.")
            continue
        result = subprocess.run(cmd, cwd=BASE_DIR, env=env, check=False)
        if result.returncode != 0:
            continue
        if args.delete and args.prune_md and md_path and index is not None:
            to_remove.add(index)

    if args.delete and args.prune_md and md_path and to_remove:
        new_lines = [
            line for idx, line in enumerate(md_lines or []) if idx not in to_remove
        ]
        new_text = "\n".join(new_lines)
        if md_trailing_newline:
            new_text += "\n"
        md_path.write_text(new_text, encoding="utf-8")
        vprint(f"Updated {md_path}, removed {len(to_remove)} line(s).")


if __name__ == "__main__":
    main()
