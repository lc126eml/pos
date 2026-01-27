import glob
import math
import os
import sys
import subprocess
import gc
import time
import logging
import random
import shutil
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from PIL import Image
from contextlib import nullcontext
from types import SimpleNamespace
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


SCRIPT_REV = "tpu-debug-20260125-1"


def _tpu_worker(index):
    os.environ["RUN_TPU_WORKER"] = "1"
    return main()

def _spawn_tpu(main_fn):
    print(f"SCRIPT_REV={SCRIPT_REV}", flush=True)
    if os.environ.get("RUN_TPU_WORKER") == "1":
        main_fn()
        return
    os.environ.setdefault("PJRT_DEVICE", "TPU")
    if os.path.exists("/kaggle/working"):
        os.environ.pop("TPU_PROCESS_ADDRESSES", None)
        os.environ.pop("TPU_PROCESS_COUNT", None)
        print("Cleared TPU_PROCESS_ADDRESSES/TPU_PROCESS_COUNT for PJRT auto-config.", flush=True)
        print(f"TPU_PROCESS_ADDRESSES={os.environ.get('TPU_PROCESS_ADDRESSES')}", flush=True)
        print(f"TPU_PROCESS_COUNT={os.environ.get('TPU_PROCESS_COUNT')}", flush=True)
        print(f"TPU_RUNTIME_METRICS_PORTS={os.environ.get('TPU_RUNTIME_METRICS_PORTS')}", flush=True)
    if os.environ.get("TPU_ENV_DUMPED") != "1":
        tpu_env = {k: v for k, v in os.environ.items() if k.startswith("TPU") or k.startswith("XLA")}
        print("TPU_ENV:", tpu_env, flush=True)
        os.environ["TPU_ENV_DUMPED"] = "1"
    if os.environ.get("TPU_UNINSTALL_TIMM_DONE") != "1":
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "timm"])
            os.environ["TPU_UNINSTALL_TIMM_DONE"] = "1"
        except Exception as exc:
            print(f"WARNING: timm uninstall failed ({exc}); continuing.")
    if os.environ.get("TPU_FIX_TF_DONE") != "1":
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "tensorflow"])
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "tensorflow-cpu"])
            os.environ["TPU_FIX_TF_DONE"] = "1"
        except Exception as exc:
            print(f"WARNING: tensorflow-cpu setup failed ({exc}); continuing.")
    print("TPU spawn: importing torch_xla", flush=True)
    try:
        import torch_xla.distributed.xla_multiprocessing as _xmp
    except Exception:
        _xmp = None
    if _xmp is None:
        main_fn()
        return
    try:
        print("TPU spawn: launching xmp.spawn", flush=True)
        _xmp.spawn(_tpu_worker, nprocs=None)
    except Exception as exc:
        raise RuntimeError(f"TPU spawn failed: {exc}") from exc

def main():
    xm = None
    pl = None
    txla = None
    xr = None
    _xla_available = False
    _TPU_WORKER = os.environ.get("RUN_TPU_WORKER") == "1"
    _XLA_PROCESS_INDEX = os.environ.get("XLA_PROCESS_INDEX", "0")
    _IS_XLA_MASTER = _XLA_PROCESS_INDEX == "0"
    os.environ.setdefault("PJRT_DEVICE", "TPU")
    train_start_time = time.time()
    if _TPU_WORKER and os.environ.get("TPU_UNINSTALL_TIMM_DONE") != "1":
        print("WARNING: timm uninstall flag not set before TPU worker import.")
    LOCAL_TIMM = "/kaggle/input/timm-repos/pytorch-image-models"
    sys.path.insert(0, LOCAL_TIMM)
    
    import timm
    if args.pos_type is not None:
        try:
            if args.pos_type == "relpos":
                import timm_pe.eva_relpos  # noqa: F401
            elif args.pos_type == "alibi":
                import timm_pe.eva_alibi  # noqa: F401
            elif args.pos_type == "sin":
                import timm_pe.eva_sin  # noqa: F401
            else:
                raise ValueError(f"Unsupported pos_type: {args.pos_type}")
        except Exception as exc:
            raise RuntimeError(
                f"pos_type={args.pos_type} requires timm_pe modules. "
                f"Ensure timm_pe is available in the kernel environment. ({exc})"
            ) from exc
    if _IS_XLA_MASTER:
        print("timm:", timm.__version__, flush=True)
        print("torch:", torch.__version__, flush=True)
    
    _is_kaggle_env = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.path.exists("/kaggle/working"))
    
    is_kaggle = _is_kaggle_env
    if _is_kaggle_env:
        is_kaggle = True
        root_dir = "/kaggle/working"
        BASE_PATH = "/kaggle/input/imagenet100/"
        if _IS_XLA_MASTER:
            print("kaggle", flush=True)
            print(os.listdir("/kaggle/input"), flush=True)
    
    
    else:
        print("not kaggle", flush=True)
    args = SimpleNamespace(
        pos_type = None,
        dynamic_img_size=True,
        model_type= "dinov3",
        use_abs_pos_emb=False,
        use_rot_pos_emb=True,
        model_size='base',
        num_classes=100,
        patch_size = 16,
        batch_size=64,
        img_sizes=[224, 192, 288],
        val_img_sizes=[160, 176, 192, 208,224, 256, 272, 288, 320, 336, 352, 368, 384, 400, 416],
        lr=7e-05,
        lr_aux=1e-5,
        eta_min=0.0,
        weight_decay=0.01,
        epochs=130,
        overlap=0,
        pretrained=None,
        seed=50,
        use_patch_position_loss=False,
        use_rc_loss=False,
        rc_alpha=600.0,
        warmup_steps_for_aux=1,
        workers=5,
        re_prob=0.0,
        train=True,
        val=True,
        tpu_size_schedule="epoch",
        tpu_size_hold_batches=0,
        tpu_workers=0,
        tpu_threads=1,
        ckpt_path=None,
        lock=False,
        save_full_ckpt=True,
        resume_full_ckpt=False,
        resume_ckpt_path=None,
        resume_scheduler=True,
        resume_optimizer=True,
        resume_bs=True,
        composite_lr=True,
        warmup_steps=3000,
        clip_value=1.0,
        log_interval=100,
        csv_interval=1,
        show_peak_gpu_mem=False,
        compile_model=False,
        debug_xla=True,
        log_all_ranks=False,
        total_run_time_hr=12.0,
        root_dir=root_dir,
    )
    resume_ckpt=None
    if args.resume_full_ckpt and args.resume_ckpt_path:
        if not os.path.exists(args.resume_ckpt_path):
            resume_dir = os.path.dirname(args.resume_ckpt_path)
            parts = os.path.normpath(resume_dir).split(os.sep)
            if os.path.isabs(resume_dir):
                prefix_parts = parts[1:4]
                search_root = os.path.join(os.sep, *prefix_parts)
            else:
                prefix_parts = parts[:3]
                search_root = os.path.join(*prefix_parts)
            candidates = sorted(
                glob.glob(os.path.join(search_root, "**", "last.pth"), recursive=True)
            )
            if candidates:
                args.resume_ckpt_path = candidates[0]
        skip_keys = [
            "resume_full_ckpt",
            "resume_ckpt_path",
            "resume_bs",
            "resume_scheduler",
            "resume_optimizer",
            "total_run_time_hr",
            "grad_accum_steps",
        ]
        if not args.resume_scheduler:
            skip_keys.extend([
                "epochs",
                "warmup_steps",
                "eta_min",
                "composite_lr",
            ])
        if not args.resume_bs:
            skip_keys.extend(["batch_size"])
        resume_ckpt = torch.load(args.resume_ckpt_path, map_location="cpu", weights_only=False)
        print(f"Resumed args from '{args.resume_ckpt_path}'")
        ckpt_args = resume_ckpt.get("args", None)
        if ckpt_args is not None:
            for k, v in vars(ckpt_args).items():
                if k not in skip_keys:
                    setattr(args, k, v)
    if args.pos_type is not None:
        args.has_pos = True
        args.overlap = 0
        args.use_rc_loss=False
        args.use_patch_position_loss=False
        args.dynamic_img_size=False
        args.val=False
        args.use_abs_pos_emb = False
        args.use_rot_pos_emb = False
    if args.use_abs_pos_emb or args.use_rot_pos_emb:
        args.overlap = 0
        args.use_patch_position_loss=False
        args.use_rc_loss = False
    if args.model_size == "base":
        args.rc_alpha = 600.0
    else:
        args.rc_alpha = 300.0
    if args.val and isinstance(args.val_img_sizes, (list, tuple)) and len(set(args.val_img_sizes)) > 1:
        args.dynamic_img_size = True
    
    offset = 0
    if args.pos_type is not None:
        pos_str = f"{args.pos_type}_"
    else:
        pos_str = ""
    MODEL_NAME = f"vit_{pos_str}{args.model_size}_patch16_{args.model_type}"
    if is_kaggle:
        output_dir = args.root_dir
        ckpt_output_dir = os.path.join(output_dir, "ckpt")
    else:
        print("not kaggle")
        sys.exit(0)
    last_ckpt_path = os.path.join(ckpt_output_dir, f'last.pth')
    
    def _import_xla():
        nonlocal xm, pl, _xla_available, txla, xr
        if _xla_available:
            return
        try:
            import torch_xla as _txla
            import torch_xla.core.xla_model as _xm
            import torch_xla.distributed.parallel_loader as _pl
            import torch_xla.runtime as _xr
        except Exception:
            _xla_available = False
            return
        txla = _txla
        xm = _xm
        pl = _pl
        xr = _xr
        _xla_available = True
    
    
    def _select_device():
        _import_xla()
        xla_devices = []
        if _xla_available:
            try:
                xla_devices = xm.get_xla_supported_devices()
            except Exception:
                xla_devices = []
        if xla_devices:
            if txla is not None and hasattr(txla, "device"):
                return txla.device(), xla_devices
            return xm.xla_device(), xla_devices
        raise RuntimeError("TPU/XLA is required but not available.")
    
    DEVICE, _xla_devices = _select_device()
    if xr is not None and hasattr(xr, "world_size"):
        WORLD_SIZE = xr.world_size()
        RANK = xr.global_ordinal()
    else:
        WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
        RANK = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))

    print(f"✅ Success! TPU Rank {RANK} initialized. World size: {WORLD_SIZE}", flush=True)
    tpu_mem_warned = False
    tpu_info_checked = False
    tpu_info_available = False
    tpu_info_last_ts = 0.0
    tpu_info_last_vals = None
    tpu_info_peak_mb = None
    def _tpu_info_mem():
        nonlocal tpu_info_checked, tpu_info_available, tpu_info_last_ts, tpu_info_last_vals, tpu_info_peak_mb
        if RANK != 0:
            return None
        if not tpu_info_checked:
            tpu_info_available = shutil.which("tpu-info") is not None
            tpu_info_checked = True
        if not tpu_info_available:
            return None
        now = time.time()
        if tpu_info_last_vals is not None and (now - tpu_info_last_ts) < 10.0:
            return tpu_info_last_vals
        try:
            proc = subprocess.run(
                ["tpu-info"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5,
            )
        except Exception:
            return tpu_info_last_vals
        output = (proc.stdout or "") + "\n" + (proc.stderr or "")
        pattern = re.compile(r"([0-9.]+)\s*([KMGTP]i?B)\s*/\s*([0-9.]+)\s*([KMGTP]i?B)")
        matches = pattern.findall(output)
        if not matches:
            return tpu_info_last_vals
        def _to_mb(val, unit):
            val = float(val)
            unit = unit.upper()
            if unit in ("KIB", "KB"):
                return val / 1024.0
            if unit in ("MIB", "MB"):
                return val
            if unit in ("GIB", "GB"):
                return val * 1024.0
            if unit in ("TIB", "TB"):
                return val * 1024.0 * 1024.0
            return val
        used_mbs = []
        total_mbs = []
        for used_val, used_unit, total_val, total_unit in matches:
            used_mbs.append(_to_mb(used_val, used_unit))
            total_mbs.append(_to_mb(total_val, total_unit))
        if not used_mbs or not total_mbs:
            return tpu_info_last_vals
        used_mb = max(used_mbs)
        total_mb = max(total_mbs)
        if tpu_info_peak_mb is None or used_mb > tpu_info_peak_mb:
            tpu_info_peak_mb = used_mb
        tpu_info_last_vals = (used_mb, total_mb, tpu_info_peak_mb)
        tpu_info_last_ts = now
        return tpu_info_last_vals
    
    use_amp = True
    use_bf16 = True
    autocast_dtype = torch.bfloat16
    
    print(f"Using device: {DEVICE} (xla=True)", use_bf16, autocast_dtype)
    base_global_batch = 128
    global_batch = args.batch_size * WORLD_SIZE
    lr_scale = min(global_batch / base_global_batch, 4.0)
    args.lr *= lr_scale
    args.lr_aux *= lr_scale
    tpu_workers = getattr(args, "tpu_workers", 0)
    if tpu_workers is None:
        tpu_workers = 0
    args.workers = min(args.workers, int(tpu_workers))
    tpu_threads = int(getattr(args, "tpu_threads", 1))
    if tpu_threads < 1:
        tpu_threads = 1
    torch.set_num_threads(tpu_threads)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(tpu_threads)
    seed = args.seed + RANK
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    pos_prefix = ""
    if args.pos_type is not None:
        pos_prefix = f"{args.pos_type}_"
    
    abs_pos = ""
    if args.use_abs_pos_emb:
        abs_pos = "_abs_pos"
    
    rot_pos = ""
    if args.use_rot_pos_emb:
        rot_pos = "_rot_pos"
    
    patch_pos = ""
    if args.use_patch_position_loss:
        patch_pos = "_patch_pos"
    
    subdir_name = (
        f"{pos_prefix}{args.model_size}{abs_pos}{rot_pos}_overlap_{args.overlap}_"
        f"rc_{args.use_rc_loss}{patch_pos}_alpha_{int(args.rc_alpha)}lr{int(args.lr/1e-5)}_s{args.seed}"
    ).replace(',', '_').replace('[', '_').replace(']', '_').replace(' ', '')
    if not is_kaggle:
        output_dir = os.path.join(output_dir, subdir_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ckpt_output_dir, exist_ok=True)
    
    log_file_path = os.path.join(output_dir, f'{subdir_name}.log')
    if xr is not None and hasattr(xr, "global_ordinal"):
        IS_MASTER = xr.global_ordinal() == 0
    elif xm is not None and hasattr(xm, "is_master_ordinal"):
        IS_MASTER = xm.is_master_ordinal()
    else:
        IS_MASTER = RANK == 0
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    log_handlers = [logging.StreamHandler(sys.stdout)]
    if IS_MASTER:
        log_handlers.insert(0, logging.FileHandler(log_file_path))
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=log_handlers,
        force=True,
    )
    logger = logging.getLogger()
    if not IS_MASTER and not getattr(args, "log_all_ranks", False):
        logger.setLevel(logging.WARNING)
    
    logger.info(f"Using device: {DEVICE} (xla=True)")
    precision_label = "bfloat16" if use_bf16 else ("float16" if use_amp else "float32")
    logger.info(f"Using mixed precision: {precision_label}")
    logger.info(args)
    logger.info(output_dir)
    logger.info(subdir_name)
    if IS_MASTER:
        if getattr(args, "debug_xla", False):
            try:
                xm.master_print("DEBUG: XLA debug logging enabled", flush=True)
            except Exception:
                print("DEBUG: XLA debug logging enabled", flush=True)
    if args.compile_model:
        logger.info("compile_model disabled on TPU (XLA best practice).")
        args.compile_model = False
    if IS_MASTER:
        tpu_specs = {
            "TPU_ACCELERATOR_TYPE": os.environ.get("TPU_ACCELERATOR_TYPE"),
            "TPU_CHIPS_PER_HOST_BOUNDS": os.environ.get("TPU_CHIPS_PER_HOST_BOUNDS"),
            "TPU_HOST_BOUNDS": os.environ.get("TPU_HOST_BOUNDS"),
            "TPU_PROCESS_ADDRESSES": os.environ.get("TPU_PROCESS_ADDRESSES"),
            "TPU_PROCESS_COUNT": os.environ.get("TPU_PROCESS_COUNT"),
            "TPU_RUNTIME_METRICS_PORTS": os.environ.get("TPU_RUNTIME_METRICS_PORTS"),
            "XLA_FLAGS": os.environ.get("XLA_FLAGS"),
            "WORLD_SIZE": WORLD_SIZE,
        }
        logger.info(f"TPU specs: {tpu_specs}")
        if _xla_devices:
            logger.info(f"XLA devices: {_xla_devices}")
    
    def _autocast():
        return torch.autocast("xla", dtype=autocast_dtype)

    def _master_print(msg):
        try:
            xm.master_print(msg, flush=True)
            return
        except Exception:
            pass
        print(msg, flush=True)

    def _debug(msg):
        if getattr(args, "debug_xla", False):
            _master_print(msg)

    def _xla_sync():
        if txla is not None and hasattr(txla, "sync"):
            txla.sync()
        else:
            xm.mark_step()
    
    
    logger.info("Cleaning up memory...")
    gc.collect()
    logger.info("Memory cleanup complete.")
    _debug(f"device={DEVICE} xla=True world_size={WORLD_SIZE} rank={RANK}")
    
    TRAIN_PATHS = [
        os.path.join(BASE_PATH, 'train.X1'),
        os.path.join(BASE_PATH, 'train.X2'),
        os.path.join(BASE_PATH, 'train.X3'),
        os.path.join(BASE_PATH, 'train.X4'),
    ]
    
    VALID_PATH = os.path.join(BASE_PATH, 'val.X')
    LABEL_PATH = os.path.join(BASE_PATH, 'Labels.json')
    
    
    
    
    from typing import Literal, Optional, Dict, Tuple
    
    LossType = Literal["mse", "smooth_l1", "l1"]
    
    class PatchRowColRegressionCriterion(nn.Module):
        def __init__(self, feat_dim, grid_h, grid_w, normalize=True, huber_beta=None):
            """
            Predict row and column index of each patch via regression (single resolution).
    
            Args:
                feat_dim (int): Dimension of patch features (D)
                grid_h (int): Number of patch rows (fixed)
                grid_w (int): Number of patch columns (fixed)
                normalize (bool): If True, normalize row/col targets to [0, 1]
            """
            super().__init__()
            self.grid_h = grid_h
            self.grid_w = grid_w
            self.normalize = normalize
    
            self.row_mlp = nn.Sequential(
                nn.Linear(feat_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )
    
            self.col_mlp = nn.Sequential(
                nn.Linear(feat_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )
    
            if huber_beta is None:
                self.loss_fn = nn.SmoothL1Loss()
            else:
                self.loss_fn = nn.SmoothL1Loss(beta=0.5/self.grid_h)
    
            rows_2d = torch.arange(grid_h, dtype=torch.float32).unsqueeze(1).repeat(1, grid_w)
            cols_2d = torch.arange(grid_w, dtype=torch.float32).unsqueeze(0).repeat(grid_h, 1)
    
            if normalize:
                rows_2d = rows_2d / (grid_h - 1)
                cols_2d = cols_2d / (grid_w - 1)
    
            row_targets = rows_2d.flatten()
            col_targets = cols_2d.flatten()
    
            self.register_buffer("row_targets", row_targets, persistent=False)
            self.register_buffer("col_targets", col_targets, persistent=False)
    
        def forward(self, feats):
            """
            Args:
                feats: (B, N, D) patch features, N = grid_h * grid_w
    
            Returns:
                avg_loss: scalar, average of row and column regression losses
            """
            B, N, D = feats.shape
            assert N == self.grid_h * self.grid_w, f"Expected N = grid_h * grid_w = {self.grid_h * self.grid_w}, got N = {N}"
    
            x = feats.reshape(-1, D)
    
            row_targets = self.row_targets.repeat(B)
            col_targets = self.col_targets.repeat(B)
    
            row_pred = self.row_mlp(x).squeeze(-1)
            col_pred = self.col_mlp(x).squeeze(-1)
    
            loss_row = self.loss_fn(row_pred, row_targets)
            loss_col = self.loss_fn(col_pred, col_targets)
    
            return (loss_row + loss_col) / 2.0
    
    class PatchRowColRegressionCriterionDynamic(nn.Module):
        def __init__(self, feat_dim, grid_h, grid_w, normalize=True):
            """
            Predict row and column index of each patch via regression,
            supporting dynamic training resolutions.
    
            Args:
                feat_dim (int): Dimension of patch features (D)
                grid_h (int): Max number of patch rows (upper bound)
                grid_w (int): Max number of patch columns (upper bound)
                normalize (bool): If True, normalize row/col targets to [0, 1]
                                  based on the *current* hp/wp for each batch.
            """
            super().__init__()
            self.grid_h = grid_h
            self.grid_w = grid_w
            self.normalize = normalize
    
            self.row_mlp = nn.Sequential(
                nn.Linear(feat_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )
    
            self.col_mlp = nn.Sequential(
                nn.Linear(feat_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )
    
            self.loss_fn = nn.SmoothL1Loss()
    
            rows = torch.arange(grid_h, dtype=torch.float32).unsqueeze(1).repeat(1, grid_w)
            cols = torch.arange(grid_w, dtype=torch.float32).unsqueeze(0).repeat(grid_h, 1)
    
            self.register_buffer("row_index_full", rows, persistent=False)
            self.register_buffer("col_index_full", cols, persistent=False)
    
        def forward(self, feats, hp=None, wp=None):
            """
            Args:
                feats: (B, N, D) patch features, with N = hp * wp for this batch.
                hp, wp: number of patch rows / columns used for this batch
                        (single scalar each; one resolution per batch).
    
            Returns:
                avg_loss: scalar, average of row and column regression losses.
            """
            B, N, D = feats.shape
    
            if hp is None:
                hp = self.grid_h
            if wp is None:
                wp = self.grid_w
    
            assert N == hp * wp, f"Expected N = hp * wp = {hp * wp}, got N = {N}"
    
            x = feats.reshape(-1, D)
    
            row_idx_2d = self.row_index_full[:hp, :wp]
            col_idx_2d = self.col_index_full[:hp, :wp]
    
            if self.normalize:
                row_idx_2d = row_idx_2d / max(hp - 1, 1)
                col_idx_2d = col_idx_2d / max(wp - 1, 1)
    
            row_targets = row_idx_2d.flatten().repeat(B)
            col_targets = col_idx_2d.flatten().repeat(B)
    
            row_pred = self.row_mlp(x).squeeze(-1)
            col_pred = self.col_mlp(x).squeeze(-1)
    
            loss_row = self.loss_fn(row_pred, row_targets)
            loss_col = self.loss_fn(col_pred, col_targets)
    
            return (loss_row + loss_col) / 2.0
    
    class PatchRowColCriterionDynamic(nn.Module):
        def __init__(self, feat_dim, grid_h, grid_w):
            """
            Predict row and column of each patch independently.
    
            Args:
                feat_dim (int): Dimension of patch features (D)
                grid_h (int): Number of patch rows
                grid_w (int): Number of patch columns
            """
            super().__init__()
            self.grid_h = grid_h
            self.grid_w = grid_w
    
            self.row_mlp = nn.Sequential(
                nn.Linear(feat_dim, 256),
                nn.ReLU(),
                nn.Linear(256, grid_h)
            )
    
            self.col_mlp = nn.Sequential(
                nn.Linear(feat_dim, 256),
                nn.ReLU(),
                nn.Linear(256, grid_w)
            )
    
            self.ce = nn.CrossEntropyLoss()
    
            rows = torch.arange(grid_h).unsqueeze(1).repeat(1, grid_w)
            cols = torch.arange(grid_w).unsqueeze(0).repeat(grid_h, 1)
            self.register_buffer("row_labels", rows)
            self.register_buffer("col_labels", cols)
    
        def forward(self, feats, hp=None, wp=None):
            """
            Args:
                feats: (B, N, D) patch features, N = grid_h * grid_w
                wp, hp: (B,) number of patches in each row/column
            Returns:
                avg_loss: scalar, sum of row and column classification losses
            """
            B, N, D = feats.shape
    
            x = feats.reshape(-1, D)
    
            if hp is None or wp is None:
                hp = self.grid_h
                wp = self.grid_w
            row_labels = self.row_labels[:hp, :wp].flatten().repeat(B)
            col_labels = self.col_labels[:hp, :wp].flatten().repeat(B)
    
            row_logits = self.row_mlp(x)
            col_logits = self.col_mlp(x)
    
            loss_row = self.ce(row_logits, row_labels)
            loss_col = self.ce(col_logits, col_labels)
    
            return (loss_row + loss_col) / 2
    
    class PatchRowColCriterion(nn.Module):
        def __init__(self, feat_dim, grid_h, grid_w):
            """
            Predict row and column of each patch independently.
    
            Args:
                feat_dim (int): Dimension of patch features (D)
                grid_h (int): Number of patch rows
                grid_w (int): Number of patch columns
            """
            super().__init__()
            self.grid_h = grid_h
            self.grid_w = grid_w
    
            self.row_mlp = nn.Sequential(
                nn.Linear(feat_dim, 256),
                nn.ReLU(),
                nn.Linear(256, grid_h)
            )
    
            self.col_mlp = nn.Sequential(
                nn.Linear(feat_dim, 256),
                nn.ReLU(),
                nn.Linear(256, grid_w)
            )
    
            self.ce = nn.CrossEntropyLoss()
    
            rows = torch.arange(grid_h).unsqueeze(1).repeat(1, grid_w).flatten()
            cols = torch.arange(grid_w).repeat(grid_h)
            self.register_buffer("row_labels", rows)
            self.register_buffer("col_labels", cols)
    
        def forward(self, feats):
            """
            Args:
                feats: (B, N, D) patch features, N = grid_h * grid_w
            Returns:
                avg_loss: scalar, sum of row and column classification losses
            """
            B, N, D = feats.shape
            assert N == self.grid_h * self.grid_w, f"Expected {self.grid_h*self.grid_w} patches, got {N}"
    
            x = feats.reshape(-1, D)
    
            row_labels = self.row_labels.repeat(B)
            col_labels = self.col_labels.repeat(B)
    
            row_logits = self.row_mlp(x)
            col_logits = self.col_mlp(x)
    
            loss_row = self.ce(row_logits, row_labels)
            loss_col = self.ce(col_logits, col_labels)
    
            return (loss_row + loss_col) / 2
    
    
    
    class PatchPositionCriterion(nn.Module):
        def __init__(self, feat_dim, hidden_dim=256, num_classes=None):
            """
            Args:
                feat_dim (int): Feature dimension of each patch (D)
                hidden_dim (int): Hidden layer size for MLP
                num_classes (int): Number of patches (grid_h * grid_w)
            """
            super().__init__()
            self.num_classes = num_classes
            self.mlp = nn.Sequential(
                nn.Linear(feat_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes)
            )
            self.ce = nn.CrossEntropyLoss()
    
            self.register_buffer("patch_positions", torch.arange(num_classes), persistent=False)
    
        def forward(self, feats):
            """
            Args:
                feats: (B, N, D) patch features
            Returns:
                avg_loss: scalar, mean cross-entropy over all patches
            """
            B, N, D = feats.shape
            assert N == self.num_classes, f"Expected {self.num_classes} patches, got {N}"
    
            x = feats.reshape(-1, D)
            labels = self.patch_positions.repeat(B)
            logits = self.mlp(x)
            loss = self.ce(logits, labels)
            return loss
    
    class PatchPositionRegressionCriterion(nn.Module):
        def __init__(self, feat_dim, num_classes, normalize=True):
            """
            Predict patch position index via regression (single resolution).
    
            Args:
                feat_dim (int): Feature dimension of each patch (D)
                num_classes (int): Number of patches (grid_h * grid_w)
                normalize (bool): If True, normalize position targets to [0, 1]
            """
            super().__init__()
            self.num_classes = num_classes
            self.normalize = normalize
    
            self.mlp = nn.Sequential(
                nn.Linear(feat_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )
    
            self.loss_fn = nn.SmoothL1Loss()
    
            position_targets = torch.arange(num_classes, dtype=torch.float32)
            if normalize:
                position_targets = position_targets / max(num_classes - 1, 1)
            self.register_buffer("position_targets", position_targets, persistent=False)
    
        def forward(self, feats):
            """
            Args:
                feats: (B, N, D) patch features
            Returns:
                loss: scalar, SmoothL1 loss over all patches
            """
            B, N, D = feats.shape
            assert N == self.num_classes, f"Expected {self.num_classes} patches, got {N}"
    
            x = feats.reshape(-1, D)
            targets = self.position_targets.repeat(B)
            pred = self.mlp(x).squeeze(-1)
            loss = self.loss_fn(pred, targets)
            return loss
    
    class PatchPositionRegressionCriterionDynamic(nn.Module):
        def __init__(self, feat_dim, max_patch_count, normalize=True):
            """
            Predict patch position index via regression, supporting dynamic resolutions.
    
            Args:
                feat_dim (int): Feature dimension of each patch (D)
                max_patch_count (int): Max number of patches (upper bound)
                normalize (bool): If True, normalize position targets to [0, 1]
                                  based on the *current* patch count for each batch.
            """
            super().__init__()
            self.max_patch_count = max_patch_count
            self.normalize = normalize
    
            self.mlp = nn.Sequential(
                nn.Linear(feat_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )
    
            self.loss_fn = nn.SmoothL1Loss()
    
            positions = torch.arange(max_patch_count, dtype=torch.float32)
            self.register_buffer("position_index_full", positions, persistent=False) 
    
        def forward(self, feats):
            """
            Args:
                feats: (B, N, D) patch features.
            Returns:
                loss: scalar, SmoothL1 loss over all patches.
            """
            B, N, D = feats.shape
            if N > self.max_patch_count:
                raise ValueError(f"Expected N <= max_patch_count={self.max_patch_count}, got N={N}")
    
            x = feats.reshape(-1, D)
    
            pos_idx = self.position_index_full[:N]
    
            if self.normalize:
                pos_idx = pos_idx / max(N - 1, 1)
    
            targets = pos_idx.repeat(B)
    
            pred = self.mlp(x).squeeze(-1)
    
            loss = self.loss_fn(pred, targets)
            return loss
    
    
    
    
    class MultiScaleImageDataset(Dataset):
        def __init__(self, samples, size_to_transform):
            """
            samples: list of (path, target)
            size_to_transform: dict[int, T.Compose]
            """
            self.samples = samples
            self.size_to_transform = size_to_transform
    
        def __len__(self):
            return len(self.samples)
    
        def __getitem__(self, key):
            idx, size = key
            path, target = self.samples[idx]
    
            with open(path, "rb") as f:
                img = Image.open(f).convert("RGB")
    
            transform = self.size_to_transform[size]
            img = transform(img)
    
            return img, target
    
    class CustomImageDataset(Dataset):
        def __init__(self, samples, transform=None):
            self.samples = samples
            self.transform = transform
    
        def set_transform(self, transform):
            self.transform = transform
    
        def __len__(self):
            return len(self.samples)
    
        def __getitem__(self, idx):
            path, target = self.samples[idx]
            with open(path, 'rb') as f:
                sample = Image.open(f).convert('RGB')
            if self.transform:
                sample = self.transform(sample)
            return sample, target
    
    class DynamicResolutionBatchSampler:
        """
        Yields batches of (idx, size) with dynamic batch size so that
        batch_size * size^2 approx base_batch_size * base_img_size^2.
        """
    
        def __init__(
            self,
            dataset,
            image_sizes,
            base_batch_size,
            base_img_size,
            shuffle: bool = True,
            drop_last: bool = True,
            size_schedule: str = "batch",
            hold_batches: int = 0,
            seed: int = 0,
        ):
            self.dataset_len = len(dataset)
            self.image_sizes = list(image_sizes)
            self.base_batch_size = base_batch_size
            self.base_img_size = base_img_size
            self.shuffle = shuffle
            self.drop_last = drop_last
            self.size_schedule = size_schedule
            self.hold_batches = hold_batches
            self.seed = seed
            self.epoch = 0
    
            self.pixel_budget = base_batch_size * (base_img_size ** 2)
    
            avg_size_sq = sum(s * s for s in self.image_sizes) / len(self.image_sizes)
            self.avg_batch_size = self.pixel_budget / avg_size_sq
    
        def __len__(self):
            return math.ceil(self.dataset_len / self.avg_batch_size)
    
        def set_epoch(self, epoch: int):
            self.epoch = epoch
    
        def __iter__(self):
            rng = random.Random(self.seed + self.epoch)
    
            indices = list(range(self.dataset_len))
            if self.shuffle:
                rng.shuffle(indices)
    
            ptr = 0
            n = len(indices)
            size = None
            batches_since_change = 0
            if self.size_schedule == "epoch":
                size = rng.choice(self.image_sizes)
    
            while ptr < n:
                if self.size_schedule == "batch":
                    size = rng.choice(self.image_sizes)
                elif self.size_schedule == "batches":
                    if self.hold_batches <= 0:
                        size = rng.choice(self.image_sizes)
                    elif size is None or batches_since_change >= self.hold_batches:
                        size = rng.choice(self.image_sizes)
                        batches_since_change = 0
    
                pixels_per_sample = size * size
                if pixels_per_sample > 0:
                    batch_size = max(1, self.pixel_budget // pixels_per_sample)
                else:
                    batch_size = self.base_batch_size
    
                remaining = n - ptr
                if remaining < batch_size:
                    if self.drop_last:
                        break
                    else:
                        batch_size = remaining
    
                batch_indices = indices[ptr: ptr + batch_size]
                ptr += batch_size
    
                yield [(idx, size) for idx in batch_indices]
                batches_since_change += 1
    
    
    class DistributedDynamicResolutionBatchSampler:
        """
        TPU-friendly dynamic batch sampler that keeps batch shapes identical across ranks.
        Each rank sees a disjoint shard of the shuffled indices, while sharing the same
        size schedule (so all ranks use the same resolution per step).
        """
    
        def __init__(
            self,
            dataset,
            image_sizes,
            base_batch_size,
            base_img_size,
            num_replicas: int,
            rank: int,
            shuffle: bool = True,
            drop_last: bool = True,
            size_schedule: str = "epoch",
            hold_batches: int = 0,
            seed: int = 0,
        ):
            self.dataset_len = len(dataset)
            self.image_sizes = list(image_sizes)
            self.base_batch_size = base_batch_size
            self.base_img_size = base_img_size
            self.num_replicas = num_replicas
            self.rank = rank
            self.shuffle = shuffle
            self.drop_last = drop_last
            self.size_schedule = size_schedule
            self.hold_batches = hold_batches
            self.seed = seed
            self.epoch = 0
    
            self.pixel_budget = base_batch_size * (base_img_size ** 2)
    
        def set_epoch(self, epoch: int):
            self.epoch = epoch
    
        def _shard_indices(self, rng):
            indices = list(range(self.dataset_len))
            if self.shuffle:
                rng.shuffle(indices)
            if self.drop_last:
                total = (len(indices) // self.num_replicas) * self.num_replicas
                indices = indices[:total]
            return indices[self.rank::self.num_replicas]
    
        def _num_batches(self, rng, indices):
            ptr = 0
            n = len(indices)
            count = 0
            size = None
            batches_since_change = 0
            if self.size_schedule == "epoch":
                size = rng.choice(self.image_sizes)
            while ptr < n:
                if self.size_schedule == "batch":
                    size = rng.choice(self.image_sizes)
                elif self.size_schedule == "batches":
                    if self.hold_batches <= 0:
                        size = rng.choice(self.image_sizes)
                    elif size is None or batches_since_change >= self.hold_batches:
                        size = rng.choice(self.image_sizes)
                        batches_since_change = 0
                pixels_per_sample = size * size
                if pixels_per_sample > 0:
                    batch_size = max(1, self.pixel_budget // pixels_per_sample)
                else:
                    batch_size = self.base_batch_size
    
                remaining = n - ptr
                if remaining < batch_size:
                    if self.drop_last:
                        break
                    batch_size = remaining
                ptr += batch_size
                count += 1
                batches_since_change += 1
            return count
    
        def __len__(self):
            rng = random.Random(self.seed + self.epoch)
            indices = self._shard_indices(rng)
            return self._num_batches(rng, indices)
    
        def __iter__(self):
            rng = random.Random(self.seed + self.epoch)
            indices = self._shard_indices(rng)
            ptr = 0
            n = len(indices)
            size = None
            batches_since_change = 0
            if self.size_schedule == "epoch":
                size = rng.choice(self.image_sizes)
    
            while ptr < n:
                if self.size_schedule == "batch":
                    size = rng.choice(self.image_sizes)
                elif self.size_schedule == "batches":
                    if self.hold_batches <= 0:
                        size = rng.choice(self.image_sizes)
                    elif size is None or batches_since_change >= self.hold_batches:
                        size = rng.choice(self.image_sizes)
                        batches_since_change = 0
                pixels_per_sample = size * size
                if pixels_per_sample > 0:
                    batch_size = max(1, self.pixel_budget // pixels_per_sample)
                else:
                    batch_size = self.base_batch_size
    
                remaining = n - ptr
                if remaining < batch_size:
                    if self.drop_last:
                        break
                    batch_size = remaining
    
                batch_indices = indices[ptr: ptr + batch_size]
                ptr += batch_size
                yield [(idx, size) for idx in batch_indices]
                batches_since_change += 1
    
    
    
    
    
    all_class_dirs = [
        d
        for train_path in TRAIN_PATHS
        for d in os.listdir(train_path)
        if os.path.isdir(os.path.join(train_path, d))
    ]
    selected_class_dirs = sorted(list(set(all_class_dirs)))[offset:args.num_classes+offset]
    class_to_idx = {cls_name: i for i, cls_name in enumerate(selected_class_dirs)}
    
    logger.info(f"OK: Efficiently loading the following {len(selected_class_dirs)} classes: {selected_class_dirs}")
    args.num_classes = len(selected_class_dirs)
    train_samples = []
    for train_path_part in TRAIN_PATHS:
        for class_name in selected_class_dirs:
            class_idx = class_to_idx[class_name]
            class_dir = os.path.join(train_path_part, class_name)
            if os.path.isdir(class_dir):
                for fname in os.listdir(class_dir):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        path = os.path.join(class_dir, fname)
                        item = (path, class_idx)
                        train_samples.append(item)
    
    valid_samples = []
    for class_name in selected_class_dirs:
        class_idx = class_to_idx[class_name]
        class_dir = os.path.join(VALID_PATH, class_name)
        if os.path.isdir(class_dir):
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    path = os.path.join(class_dir, fname)
                    item = (path, class_idx)
                    valid_samples.append(item)
    _debug(f"dataset scan done train={len(train_samples)} val={len(valid_samples)}")
    
    
    img_mean = [0.485, 0.456, 0.406]
    img_std  = [0.229, 0.224, 0.225]
    
    
    def make_train_transform(size: int):
        t_list = [
            T.RandomResizedCrop(size, interpolation=InterpolationMode.BICUBIC, antialias=True),
            T.RandomHorizontalFlip(),
        ]
        t_list.extend([
            T.ToTensor(),
            T.Normalize(mean=img_mean, std=img_std),
        ])
        return T.Compose(t_list)
    
    size_to_transform = {
        s: make_train_transform(s) for s in args.img_sizes
    }
    
    def make_valid_transform(img_size):
        return transforms.Compose([
            transforms.Resize(
                size=int(img_size * 1.143),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=img_mean, std=img_std),
        ])
    
    valid_transforms = make_valid_transform(args.img_sizes[0])
    
    valid_dataset = CustomImageDataset(valid_samples, transform=valid_transforms)
    
    logger.info(f"Total validation images ({args.num_classes} classes): {len(valid_dataset)}")
    
    batch_sampler = None
    train_sampler = None
    valid_sampler = None
    prefetch_kwargs = {"prefetch_factor": 2} if args.workers > 0 else {}
    pin_memory = False
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    if len(args.img_sizes) == 1:
        train_dataset = CustomImageDataset(train_samples, transform=size_to_transform[args.img_sizes[0]])
        if WORLD_SIZE > 1:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=WORLD_SIZE,
                rank=RANK,
                shuffle=True,
                drop_last=True,
                seed=args.seed,
            )
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            generator=train_generator if train_sampler is None else None,
            num_workers=args.workers,
            pin_memory=pin_memory,
            drop_last=True,
            persistent_workers=(args.workers > 0),
            **prefetch_kwargs,
        )
    else:
        train_dataset = MultiScaleImageDataset(
            samples=train_samples,
            size_to_transform=size_to_transform
        )
        if WORLD_SIZE > 1:
            batch_sampler = DistributedDynamicResolutionBatchSampler(
                dataset=train_dataset,
                image_sizes=args.img_sizes,
                base_batch_size=args.batch_size,
                base_img_size=224,
                num_replicas=WORLD_SIZE,
                rank=RANK,
                shuffle=True,
                drop_last=True,
                size_schedule=args.tpu_size_schedule,
                hold_batches=args.tpu_size_hold_batches,
                seed=args.seed,
            )
        else:
            batch_sampler = DynamicResolutionBatchSampler(
                dataset=train_dataset,
                image_sizes=args.img_sizes,
                base_batch_size=args.batch_size,
                base_img_size=224,
                shuffle=True,
                drop_last=True,
                size_schedule="batch",
                hold_batches=0,
                seed=42,
            )
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_sampler=batch_sampler,
            num_workers=args.workers,
            pin_memory=pin_memory,
            persistent_workers=(args.workers > 0),
            **prefetch_kwargs,
        )
    logger.info(f"Total training images ({args.num_classes} classes): {len(train_dataset)}")
    if WORLD_SIZE > 1:
        valid_sampler = DistributedSampler(
            valid_dataset,
            num_replicas=WORLD_SIZE,
            rank=RANK,
            shuffle=False,
            drop_last=False,
        )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=valid_sampler,
        num_workers=args.workers,
        pin_memory=pin_memory,
        persistent_workers=(args.workers > 0),
        **prefetch_kwargs,
    )
    train_loader = pl.MpDeviceLoader(train_loader, DEVICE)
    valid_loader = pl.MpDeviceLoader(valid_loader, DEVICE)
    steps_per_epoch = len(train_loader)
    optimizer_steps_per_epoch = steps_per_epoch
    logger.info(f"OK: DataLoaders for {args.num_classes} classes created successfully.")
    logger.info(f"{steps_per_epoch=}, val_steps: {len(valid_loader)}")
    logger.info(f"Effective batch size: {args.batch_size * WORLD_SIZE}")
    _debug(f"dataloaders ready steps_per_epoch={steps_per_epoch}")
    
    
    
    
    logger.info(f"Initializing model: {MODEL_NAME} for {args.num_classes} classes...")
    _debug("creating model")
    model = timm.create_model(
        MODEL_NAME,
        pretrained=False,
        use_abs_pos_emb=args.use_abs_pos_emb,
        use_rot_pos_emb=args.use_rot_pos_emb,
        num_classes=args.num_classes,
        dynamic_img_size=args.dynamic_img_size,
        img_size=args.img_sizes[0],
    ).to(DEVICE)
    _debug("model created and moved to device")
    
    
    
    logger.info(f'model.patch_embed.proj{model.patch_embed.proj}')
    _debug("building optimizer and scheduler")
    
    
    
    
    
    
    dynamic = True
    training_parameters = list(model.parameters()) 
    param_groups = []
    lr_aux = getattr(args, "lr_aux", args.lr)
    if args.use_rc_loss:
        if len(args.img_sizes)==1:
            grid_h, grid_w = model.patch_embed.grid_size
            dynamic = False
            rowcol_loss = PatchRowColRegressionCriterion(
                feat_dim=model.embed_dim,
                grid_h=grid_h,
                grid_w=grid_w,
            ).to(DEVICE)
        else:
            grid_h = grid_w = max(args.img_sizes)//args.patch_size
            rowcol_loss = PatchRowColRegressionCriterionDynamic(
                feat_dim=model.embed_dim,
                grid_h=grid_h,
                grid_w=grid_w,
            ).to(DEVICE)
        training_parameters += list(rowcol_loss.parameters())
        param_groups.append({"params": rowcol_loss.parameters(), "weight_decay": 0.0, "lr": lr_aux})
    if args.use_patch_position_loss:
        if len(args.img_sizes)==1:
            position_loss = PatchPositionRegressionCriterion(
                feat_dim=model.embed_dim,
                num_classes=model.patch_embed.num_patches
            ).to(DEVICE)
        else:
            max_grid = max(args.img_sizes)//args.patch_size
            max_patch_count = max_grid * max_grid
            position_loss = PatchPositionRegressionCriterionDynamic(
                feat_dim=model.embed_dim,
                max_patch_count=max_patch_count
            ).to(DEVICE)
        training_parameters += list(position_loss.parameters())
        param_groups.append({"params": position_loss.parameters(), "weight_decay": 0.0, "lr": lr_aux})
    
    decay_params = []
    no_decay_params = []
    
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.endswith(".bias") or ("norm" in n.lower()):
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    
    param_groups.append({
        "params": decay_params,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
    })
    param_groups.append({
        "params": no_decay_params,
        "lr": args.lr,
        "weight_decay": 0.0,
    })
    criterion = nn.CrossEntropyLoss()
    if args.composite_lr:
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    
        total = sum(p.numel() for p in model.parameters())
        opt_total = sum(p.numel() for g in optimizer.param_groups for p in g["params"])
        print("model params:", total, "optimizer params:", opt_total)
    
        seen = set()
        dups = 0
        for g in optimizer.param_groups:
            for p in g["params"]:
                pid = id(p)
                if pid in seen:
                    dups += 1
                seen.add(pid)
        print("duplicate params in groups:", dups)
    
        total_steps = args.epochs * optimizer_steps_per_epoch
    
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-7 / args.lr,
            end_factor=1.0,
            total_iters=args.warmup_steps
        )
    
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps - args.warmup_steps,
            eta_min=1e-8
        )
    
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[args.warmup_steps]
        )
    else:
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
        logger.info("OK: Model, Loss Function, and Optimizer are ready.")
    
    
        total_steps = args.epochs * optimizer_steps_per_epoch
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.eta_min)
        logger.info("OK: Step-based LR Scheduler is ready.")
    
    
    
    sys.stdout.flush()
    def get_patch_numbers(img_size, patch_size):
        """
        Calculate the number of patches in an image.
    
        Args:
            img_size (int or tuple): Size of the input image (H, W)
            patch_size (int): Size of the patch
    
        Returns:
            tuple: Number of patches in the image (H, W)
        """
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        assert 2 == len(img_size)
        hp, wp = img_size[0] // patch_size, img_size[1] // patch_size  
        return hp, wp
    
    
    ckpt_path = None
    if args.train:
        start_epoch = 0
        step = 0
        best_acc = 0.0
        if args.resume_full_ckpt and args.resume_ckpt_path:
            model.load_state_dict(resume_ckpt["model"])
            if args.resume_optimizer:
                if "optimizer" in resume_ckpt:
                    optimizer.load_state_dict(resume_ckpt["optimizer"])
            else:
                logger.info("Skipping optimizer state load (resume_optimizer=False).")
            if args.resume_scheduler:
                start_epoch = resume_ckpt.get("epoch", 0)
                step = resume_ckpt.get("step", 0)
                if resume_ckpt.get("scheduler") is not None:
                    scheduler.load_state_dict(resume_ckpt["scheduler"])
            else:
                logger.info("Skipping scheduler state load (resume_scheduler=False).")
            if args.use_rc_loss and resume_ckpt.get("rowcol_loss") is not None:
                for k in ["row_targets", "col_targets", "row_index_full", "col_index_full"]:
                    if k in resume_ckpt["rowcol_loss"]:
                        resume_ckpt["rowcol_loss"].pop(k)
                rowcol_loss.load_state_dict(resume_ckpt["rowcol_loss"])
            if args.use_patch_position_loss and resume_ckpt.get("position_loss") is not None:
                position_loss.load_state_dict(resume_ckpt["position_loss"])
            best_acc = resume_ckpt.get("best_acc", 0.0)
            logger.info(f"Resumed full checkpoint from '{args.resume_ckpt_path}' at epoch={start_epoch}, step={step}")
        logger.info(f"\nStarting training for {MODEL_NAME}...")
    
        if args.use_rc_loss or args.use_patch_position_loss:
            training_history = {
                'train_loss': [],
                'train_acc': [],
                'valid_acc': [],
                'train_time': [],
                'val_time': [],
                'epoch': [],
                'step': [],
                'base_loss': [],
                'aux_loss': [],
            }
        else:
            training_history = {
                'train_loss': [],
                'train_acc': [],
                'valid_acc': [],
                'train_time': [],
                'val_time': [],
                'epoch': [],
                'step': [],
            }
        if resume_ckpt is not None and resume_ckpt.get("training_history") is not None:
            training_history = resume_ckpt["training_history"]
        training_history.setdefault('train_time', [])
        training_history.setdefault('val_time', [])
        def _pad_history(hist, fill_value=None):
            keys = [k for k, v in hist.items() if isinstance(v, list)]
            if not keys:
                return
            max_len = max(len(hist[k]) for k in keys)
            for k in keys:
                if len(hist[k]) < max_len:
                    hist[k].extend([fill_value] * (max_len - len(hist[k])))
        if args.resume_full_ckpt:
            _pad_history(training_history)
        log_interval = getattr(args, "log_interval", 50)
        csv_interval = getattr(args, "csv_interval", 1) 
        for epoch in range(start_epoch, args.epochs):
            epoch_train_start = time.time()
            model.train()
    
            aux_loss = None
    
            running_loss_t = torch.zeros((), device=DEVICE)
            aux_loss_sum_t = torch.zeros((), device=DEVICE)
            base_loss_t = torch.zeros((), device=DEVICE)
            train_correct_t = torch.zeros((), device=DEVICE)
            train_total = 0
    
            train_total = 0
            if batch_sampler is not None:
                batch_sampler.set_epoch(epoch)
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
    
            optimizer.zero_grad(set_to_none=True)
            for step_in_epoch, (inputs, labels) in enumerate(train_loader):
                step += 1
                bs = inputs.size(0)
    
                aux_loss = None
                with _autocast():
                    feats = model.forward_features(inputs)
                    outputs = model.forward_head(feats)
                    loss = criterion(outputs, labels)
                    if args.use_rc_loss:
                        base_loss_t += loss.detach() * bs
                        if dynamic:
                            hp, wp = get_patch_numbers(inputs.shape[-2:], model.patch_embed.patch_size[0])
                            aux_loss = rowcol_loss(feats[:, model.num_prefix_tokens:, :], hp, wp)
                        else:
                            aux_loss = rowcol_loss(feats[:, model.num_prefix_tokens:, :])
    
    
                        aux_loss_sum_t += aux_loss.detach() * bs
                        loss = loss + args.rc_alpha * aux_loss
    
                    if args.use_patch_position_loss:
                        base_loss_t += loss.detach() * bs
                        aux_loss = position_loss(feats[:, model.num_prefix_tokens:, :])
                        aux_loss_sum_t += aux_loss.detach() * bs
                        loss = loss + args.rc_alpha * aux_loss
    
                loss.backward()
    
                if args.clip_value is not None:
                    torch.nn.utils.clip_grad_norm_(training_parameters, max_norm=args.clip_value)
                xm.optimizer_step(optimizer, barrier=True)
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if step % log_interval == 0:
                    _xla_sync()
    
                running_loss_t += loss.detach() * bs
                train_total += bs
    
                with torch.no_grad():
                    pred = outputs.detach().argmax(dim=1)
                    train_correct_t += (pred == labels).sum()

                if step % log_interval == 0 and IS_MASTER:
                    avg_loss = (running_loss_t / train_total).float().item()
                    avg_acc = (train_correct_t / train_total).float().item()
                    msg = f"Epoch {epoch+1}/{args.epochs} step {step}: loss={avg_loss:.4f} acc={avg_acc:.3f}"
                    if aux_loss is not None:
                        avg_aux = (aux_loss_sum_t / train_total).float().item()
                        msg += f" aux={avg_aux:.4f}"
                    if args.show_peak_gpu_mem:
                        info = _tpu_info_mem()
                        if info is not None:
                            used_mb, total_mb, peak_mb = info
                            msg += f" tpu_mem={used_mb:.0f}/{peak_mb:.0f}/{total_mb:.0f}MB"
                        elif not tpu_mem_warned:
                            tpu_mem_warned = True
                            print("TPU memory info unavailable; skipping TPU mem logging.", flush=True)
                    logger.info(msg)
    
            train_time = time.time() - epoch_train_start
            model.eval()
            val_correct_t = torch.zeros((), device=DEVICE)
            val_total_t = torch.zeros((), device=DEVICE)
            val_start = time.time()

            with torch.no_grad():
                for inputs, labels in valid_loader:
                    with _autocast():
                        outputs = model(inputs)
                    pred = outputs.argmax(dim=1)
                    val_correct_t += (pred == labels).sum()
                    val_total_t += labels.shape[0]
            val_time = time.time() - val_start
            if WORLD_SIZE > 1:
                train_total_t = torch.tensor(train_total, device=DEVICE)
                reduce_tensors = [
                    running_loss_t,
                    train_correct_t,
                    train_total_t,
                    val_correct_t,
                    val_total_t,
                    aux_loss_sum_t,
                    base_loss_t,
                ]
                xm.all_reduce(xm.REDUCE_SUM, reduce_tensors)
                train_total = int(train_total_t.item())
                val_total = int(val_total_t.item())
            else:
                val_total = int(val_total_t.item())
            if val_total == 0:
                epoch_val_acc = 0.0
            else:
                epoch_val_acc = (val_correct_t / val_total_t).item()
            is_best = False
            if best_acc < epoch_val_acc:
                best_acc = epoch_val_acc
                is_best = True
    
            epoch_train_acc  = (train_correct_t / train_total).item()
            epoch_train_loss = (running_loss_t / train_total).item()
            if IS_MASTER:
                logger.info(f"\nEpoch {epoch+1}/{args.epochs} Summary:")
                logger.info(f"\nStep {step} Summary:")
    
                if aux_loss is not None:
                    epoch_aux_loss   = (aux_loss_sum_t / train_total).item()
                    epoch_base_loss  = (base_loss_t / train_total).item()
                    training_history['aux_loss'].append(epoch_aux_loss)
                    training_history['base_loss'].append(epoch_base_loss)
                    logger.info(
                        f"  Train Loss: {epoch_train_loss:.4f} | Aux Loss: {epoch_aux_loss:.4f} | Base Loss: {epoch_base_loss:.4f} | "
                        f"Train Acc: {epoch_train_acc:.4f} | Valid Acc: {epoch_val_acc:.4f} | "
                        f"train_time: {train_time:.1f}s | val_time: {val_time:.1f}s\n"
                    )
                else:
                    logger.info(
                        f"  Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | "
                        f"Valid Acc: {epoch_val_acc:.4f} | train_time: {train_time:.1f}s | val_time: {val_time:.1f}s\n"
                    )
    
    
    
            master_history = None
            if IS_MASTER:
                training_history['train_loss'].append(epoch_train_loss)
                training_history['train_acc'].append(epoch_train_acc)
                training_history['valid_acc'].append(epoch_val_acc)  
                training_history['train_time'].append(train_time)
                training_history['val_time'].append(val_time)
                training_history['epoch'].append(epoch+1)
                training_history['step'].append(step+1)
                if (epoch + 1) % csv_interval == 0:
                    pd.DataFrame(training_history).to_csv(os.path.join(output_dir, f'{subdir_name}.csv'), index=False)
                master_history = training_history
            if args.save_full_ckpt:
                ckpt = {}
                if IS_MASTER:
                    ckpt = {
                        "epoch": epoch + 1,
                        "step": step,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict() if scheduler is not None else None,
                        "rowcol_loss": rowcol_loss.state_dict() if args.use_rc_loss else None,
                        "position_loss": position_loss.state_dict() if args.use_patch_position_loss else None,
                        "training_history": master_history,
                        "args": args,
                        "best_acc": best_acc,
                    }
                _xla_sync()
                xm.save(ckpt, last_ckpt_path, master_only=True)
                if IS_MASTER:
                    logger.info(f"Saved full checkpoint to '{last_ckpt_path}'")
    
            if args.total_run_time_hr is not None:
                elapsed = time.time() - train_start_time
                max_run_time_sec = args.total_run_time_hr * 3600
                if elapsed + (train_time + val_time) + 600 >= max_run_time_sec:
                    logger.info(
                        "Stopping training: elapsed time exceeded %.2fh.",
                        args.total_run_time_hr,
                    )
                    break
    
    
        if IS_MASTER:
            logger.info("Training complete.")
            logger.info(f"Best Accuracy: {best_acc:.4f}")
            logger.info(output_dir)
    
    
    
    
        pd.DataFrame(training_history).to_csv(os.path.join(output_dir, f'{subdir_name}.csv'), index=False)
    
    if args.val:    
        val_results = {
            'img_size': [],
            'valid_acc': []
        }
        if IS_MASTER and args.dynamic_img_size:
            logger.info("Final eval: dynamic_img_size enabled for multi-size evaluation.")

        if not args.train:
            if ckpt_path is None:
                ckpt_path = f"{args.root_dir}/{args.ckpt_path}"
            model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False))
        model.to(DEVICE)
        model.eval()
        for img_size in args.val_img_sizes:
            if hasattr(model, "set_input_size"):
                try:
                    model.set_input_size(img_size)
                except Exception:
                    pass
            elif hasattr(model, "patch_embed") and hasattr(model.patch_embed, "set_input_size"):
                try:
                    model.patch_embed.set_input_size(img_size)
                except Exception:
                    pass
            valid_dataset.set_transform(make_valid_transform(img_size))
            batch_size = max(1, int((args.batch_size * 0.8 * 224 * 224) / (img_size * img_size)))
            valid_sampler = DistributedSampler(
                valid_dataset,
                num_replicas=WORLD_SIZE,
                rank=RANK,
                shuffle=False,
                drop_last=False,
            ) if WORLD_SIZE > 1 else None
            valid_loader = DataLoader(
                dataset=valid_dataset,
                batch_size=batch_size,
                sampler=valid_sampler,
                shuffle=False if valid_sampler is None else False,
                num_workers=args.workers,
                pin_memory=pin_memory,
                persistent_workers=False,
                **prefetch_kwargs,
            )
            valid_loader = pl.MpDeviceLoader(valid_loader, DEVICE)
            val_correct_t = torch.zeros((), device=DEVICE)
            val_total_t = torch.zeros((), device=DEVICE)
            with torch.no_grad():
                for inputs, labels in valid_loader:
                    with _autocast():
                        outputs = model(inputs)
                    predicted = outputs.argmax(dim=1)
                    val_total_t += labels.shape[0]
                    val_correct_t += (predicted == labels).sum()

            if WORLD_SIZE > 1:
                xm.all_reduce(xm.REDUCE_SUM, [val_correct_t, val_total_t])
            val_total = int(val_total_t.item())
            epoch_val_acc = 0.0 if val_total == 0 else (val_correct_t / val_total_t).item()
            if IS_MASTER:
                val_results['img_size'].append(img_size)
                val_results['valid_acc'].append(epoch_val_acc)
                val_df = pd.DataFrame(val_results)
                val_df.to_csv(os.path.join(output_dir, f'{subdir_name}_eval.csv'), index=False)
                logger.info(f"{img_size=}: {epoch_val_acc=}")
    

if __name__ == "__main__":
    _spawn_tpu(main)
