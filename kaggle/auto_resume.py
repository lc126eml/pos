import argparse
import copy
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml
import logging

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR.parent))

import process_kaggle  # noqa: E402
import kernel  # noqa: E402


USAGE = """\
Expected config_kernel.yaml fields:
- sleep_time_hr: float
- poll_interval_minutes: float
- available_ids: [owner_id, ...]
- exhausted_ids: [owner_id, ...]
- tpu:
  - available_ids: [owner_id, ...]
  - exhausted_ids: [owner_id, ...]
- running_nodes:
  - id: owner_id
    tpu_node: bool (optional; default false)
    left_time: float
    notebooks:
      - kernel_id: owner/task-model-tag-desc-suffixseed
        total_runs: int
        run_id: int
        start_time: ISO8601 string or unix timestamp
        resumed_from: kernel_id or null
        history_ids: [kernel_id, ...]
- finished_notebooks: [notebook, ...]
"""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def _load_yaml(path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write_yaml(path, data):
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _now_naive():
    return datetime.now()


def _parse_time(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def _format_time(dt):
    if dt is None:
        return None
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _owner_from_kernel_id(kernel_id):
    if "/" in kernel_id:
        return kernel_id.split("/", 1)[0]
    return None


def _parse_status(output):
    text = (output or "").lower()
    if "running" in text:
        return "running"
    if "complete" in text or "finished" in text or "success" in text:
        return "finished"
    if "error" in text or "failed" in text:
        return "error"
    return "unknown"


def _kernel_status(kernel_id, tokens):
    owner_id = _owner_from_kernel_id(kernel_id)
    if not owner_id:
        return "error", "Missing owner id in kernel id."
    token = kernel._get_token(tokens, owner_id)
    if not token:
        return "error", f"No token found for id: {owner_id}"
    env = os.environ.copy()
    env["KAGGLE_API_TOKEN"] = token
    env["PYTHONIOENCODING"] = "utf-8"
    cmd = ["kaggle", "kernels", "status", kernel_id]
    result = subprocess.run(cmd, cwd=BASE_DIR, env=env, check=False, capture_output=True, text=True)
    output = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0:
        return "error", output.strip()
    return _parse_status(output), output.strip()


def _build_kernel_id(cfg):
    suffix = cfg.get("suffix") or ""
    desc = cfg.get("desc") or "desc"
    if cfg.get("pos_type") is not None:
        tag = cfg["pos_type"]
    else:
        tag = cfg["method"]
    return f"{cfg['id']}/{cfg['task']}-{cfg['model_size']}-{tag}-{desc}-{suffix}{cfg['seed']}"


def _prepare_cfg_from_resume(base_cfg, resumed_from, run_id, target_id):
    cfg = copy.deepcopy(base_cfg)
    source_name = process_kaggle._get_source_name(resumed_from)
    task, model_size, method, seed, pos_type, desc, _ = process_kaggle._infer_from_source_id(
        source_name
    )
    cfg["task"] = task
    cfg["model_size"] = model_size
    cfg["method"] = method
    cfg["seed"] = seed
    cfg["pos_type"] = pos_type
    cfg["desc"] = desc or cfg.get("desc") or "desc"
    cfg["suffix"] = run_id
    cfg["id"] = target_id
    cfg["resume_full_ckpt"] = True
    cfg["resume_source"] = "kernel"
    cfg["kernel_sources"] = resumed_from
    cfg["resume_infer"] = False
    return cfg


def _push_kernel(cfg):
    cfg_path = BASE_DIR / "config.yaml"
    _write_yaml(cfg_path, cfg)
    result = subprocess.run(
        ["python", str(BASE_DIR / "process_kaggle.py"), "--run", "--concise"],
        cwd=BASE_DIR,
        check=False,
        capture_output=True,
        text=True,
    )
    output = (result.stdout or "") + (result.stderr or "")
    return result.returncode == 0, output.strip()


def _move_finished(notebook, finished_notebooks):
    finished = dict(notebook)
    for key in ("run_id", "resumed_from"):
        finished.pop(key, None)
    finished_notebooks.append(finished)


def _move_to_new_node(node, notebook, running_nodes, available_ids, exhausted_ids, is_tpu):
    node_notebooks = node.get("notebooks") or []
    for candidate in running_nodes:
        if candidate is node:
            continue
        if bool(candidate.get("tpu_node", False)) != is_tpu:
            continue
        if float(candidate.get("left_time", 0)) <= 0:
            continue
        candidate_notebooks = candidate.get("notebooks") or []
        if len(candidate_notebooks) < 2:
            candidate_notebooks.append(notebook)
            candidate["notebooks"] = candidate_notebooks
            if notebook in node_notebooks:
                node_notebooks.remove(notebook)
                node["notebooks"] = node_notebooks
            if not node_notebooks:
                running_nodes.remove(node)
                exhausted_ids.append(node.get("id"))
            return candidate

    if not available_ids:
        logging.warning("No available ids left to resume.")
        return None
    new_id = available_ids.pop(0)
    target_node = {"id": new_id, "tpu_node": is_tpu, "notebooks": []}
    target_node["left_time"] = 20 if is_tpu else 30
    target_node["notebooks"].append(notebook)
    running_nodes.append(target_node)
    if notebook in node_notebooks:
        node_notebooks.remove(notebook)
        node["notebooks"] = node_notebooks
    if not node_notebooks:
        running_nodes.remove(node)
        exhausted_ids.append(node.get("id"))
    return target_node


def main():
    parser = argparse.ArgumentParser(description="Auto resume Kaggle kernels.", epilog=USAGE)
    parser.add_argument(
        "--config",
        default=str(BASE_DIR / "config_kernel.yaml"),
        help="Path to config_kernel.yaml.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress poll status details.",
    )
    parser.add_argument(
        "--dry",
        action="store_true",
        help="Do not run kaggle commands; only report them.",
    )
    args = parser.parse_args()
    verbose = not args.quiet

    config_path = Path(args.config)
    kcfg = _load_yaml(config_path)

    sleep_time_hr = float(kcfg.get("sleep_time_hr", 0))

    if sleep_time_hr > 0:
        time.sleep(sleep_time_hr * 3600)

    while True:
        kcfg = _load_yaml(config_path)
        tokens = _load_yaml(BASE_DIR / "tokens.yaml")
        base_cfg = _load_yaml(BASE_DIR / "config.yaml")
        poll_interval_minutes = float(kcfg.get("poll_interval_minutes", 10))

        now = _now_naive()
        changed = False
        running_nodes = kcfg.get("running_nodes") or []
        available_ids = kcfg.get("available_ids") or []
        exhausted_ids = kcfg.get("exhausted_ids") or []
        tpu_cfg = kcfg.get("tpu") or {}
        tpu_available_ids = tpu_cfg.get("available_ids") or []
        tpu_exhausted_ids = tpu_cfg.get("exhausted_ids") or []
        finished_notebooks = kcfg.get("finished_notebooks") or []

        for node in list(running_nodes):
            notebooks = node.get("notebooks") or []
            is_tpu = bool(node.get("tpu_node", False))
            if "left_time" not in node:
                node["left_time"] = 20 if is_tpu else 30
            for notebook in list(notebooks):
                kernel_id = notebook.get("kernel_id")
                if not kernel_id:
                    continue
                if args.dry:
                    if verbose:
                        logging.info("Dry run: kaggle kernels status %s", kernel_id)
                    status, detail = "unknown", ""
                else:
                    status, detail = _kernel_status(kernel_id, tokens)
                if verbose:
                    logging.info("%s: %s", kernel_id, status)
                if status == "running":
                    continue
                if status == "error":
                    logging.error(detail)
                    continue

                if status == "finished":
                    notebook["run_id"] = int(notebook.get("run_id", 0)) + 1
                    resumed_from = notebook.get("resumed_from")
                    history_ids = list(notebook.get("history_ids") or [])
                    if resumed_from:
                        history_ids.append(resumed_from)
                    notebook["history_ids"] = history_ids
                    notebook["resumed_from"] = kernel_id

                    start_time = _parse_time(notebook.get("start_time"))
                    before_left = float(node.get("left_time", 20 if is_tpu else 30))
                    if start_time is None:
                        if verbose:
                            logging.warning(
                                "Missing start_time for %s; skipping left_time update",
                                notebook.get("kernel_id"),
                            )
                        left_time = before_left
                    else:
                        elapsed_hr = (now - start_time).total_seconds() / 3600
                        left_time = before_left - elapsed_hr
                        node["left_time"] = left_time
                        if verbose:
                            logging.info(
                                "Updated left_time for %s: %.2f -> %.2f (elapsed %.2f h)",
                                node.get("id"),
                                before_left,
                                left_time,
                                elapsed_hr,
                            )

                    total_runs = int(notebook.get("total_runs", 0))
                    if notebook["run_id"] >= total_runs:
                        _move_finished(notebook, finished_notebooks)
                        notebooks.remove(notebook)
                        changed = True
                        continue

                    target_node = node
                    if left_time <= 0:
                        if is_tpu:
                            target_node = _move_to_new_node(
                                node,
                                notebook,
                                running_nodes,
                                tpu_available_ids,
                                tpu_exhausted_ids,
                                True,
                            )
                        else:
                            target_node = _move_to_new_node(
                                node,
                                notebook,
                                running_nodes,
                                available_ids,
                                exhausted_ids,
                                False,
                            )
                        if target_node is None:
                            continue
                        if verbose:
                            logging.info(
                                "Moved notebook to node: %s -> %s",
                                node.get("id"),
                                target_node.get("id"),
                            )

                    resumed_from_id = notebook.get("resumed_from")
                    if not resumed_from_id:
                        logging.warning("Missing resumed_from for notebook.")
                        continue

                    cfg = _prepare_cfg_from_resume(
                        base_cfg,
                        resumed_from_id,
                        notebook["run_id"],
                        target_node.get("id"),
                    )
                    cfg["tpu"] = bool(target_node.get("tpu_node", False))
                    new_kernel_id = _build_kernel_id(cfg)
                    notebook["kernel_id"] = new_kernel_id
                    notebook["start_time"] = _format_time(now)
                    if verbose:
                        logging.info(
                            "Updated kernel_id for notebook: %s",
                            new_kernel_id,
                        )

                    if verbose:
                        logging.info("Submitting kernel: %s", new_kernel_id)
                    if args.dry:
                        logging.info(
                            "Dry run: python %s --run --concise",
                            str(BASE_DIR / "process_kaggle.py"),
                        )
                        ok, output = True, ""
                    else:
                        ok, output = _push_kernel(cfg)
                    if not ok:
                        quota_msg = "Maximum weekly GPU quota of 30.00 hours reached"
                        if quota_msg in output:
                            if is_tpu:
                                target_node = _move_to_new_node(
                                    target_node,
                                    notebook,
                                    running_nodes,
                                    tpu_available_ids,
                                    tpu_exhausted_ids,
                                    True,
                                )
                            else:
                                target_node = _move_to_new_node(
                                    target_node,
                                    notebook,
                                    running_nodes,
                                    available_ids,
                                    exhausted_ids,
                                    False,
                                )
                            if target_node is None:
                                continue
                            if verbose:
                                logging.info(
                                    "Quota reached. Moved notebook to node: %s",
                                    target_node.get("id"),
                                )
                            cfg["id"] = target_node.get("id")
                            cfg["tpu"] = bool(target_node.get("tpu_node", False))
                            new_kernel_id = _build_kernel_id(cfg)
                            notebook["kernel_id"] = new_kernel_id
                            notebook["start_time"] = _format_time(now)
                            if verbose:
                                logging.info(
                                    "Updated kernel_id for notebook: %s",
                                    new_kernel_id,
                                )
                            if verbose:
                                logging.info("Submitting kernel: %s", new_kernel_id)
                            if args.dry:
                                logging.info(
                                    "Dry run: python %s --run --concise",
                                    str(BASE_DIR / "process_kaggle.py"),
                                )
                                ok, output = True, ""
                            else:
                                ok, output = _push_kernel(cfg)
                        if not ok:
                            logging.error(output)
                            continue
                    changed = True

            node["notebooks"] = notebooks

        kcfg["running_nodes"] = running_nodes
        kcfg["available_ids"] = available_ids
        kcfg["exhausted_ids"] = exhausted_ids
        if tpu_cfg:
            tpu_cfg["available_ids"] = tpu_available_ids
            tpu_cfg["exhausted_ids"] = tpu_exhausted_ids
            kcfg["tpu"] = tpu_cfg
        kcfg["finished_notebooks"] = finished_notebooks

        if changed:
            _write_yaml(config_path, kcfg)

        del kcfg
        logging.info("sleeping ...")
        time.sleep(poll_interval_minutes * 60)


if __name__ == "__main__":
    main()
