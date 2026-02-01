# %%
# =================================================================================
# Step 1: Install and Import Necessary Libraries
# =================================================================================
import glob
import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset,TensorDataset, DataLoader
# from data.MultiScaleImageDataset import MultiScaleImageDataset, CustomImageDataset
# from data.DynamicResolutionBatchSampler import DynamicResolutionBatchSampler

import matplotlib.pyplot as plt
import pandas as pd
import csv
import pickle
import numpy as np
import random
from PIL import Image
from torch.nn import functional as F
import torchvision.transforms.functional as TF
import sys
import subprocess
import importlib
from types import SimpleNamespace
import gc
import time
import argparse
import logging
import shutil
import urllib.request
import zipfile
train_start_time = time.time()
# try:
#     from filelock import FileLock
# except ImportError:
#     FileLock = None

# from core.utils import log_grads
# Enable faster matmul/conv kernels on Ampere+ without extra memory cost
# if torch.cuda.is_available():
#     torch.backends.cuda.matmul.allow_tf32 = True
#     torch.backends.cudnn.allow_tf32 = True
#     # Prefer faster matmul kernels when available (Torch 2.0+)
#     if hasattr(torch, "set_float32_matmul_precision"):
#         torch.set_float32_matmul_precision("high")
# %%
# Ensure timm provides the requested model; update if missing.
# def _timm_has_model(model_name: str) -> bool:
#     try:
#         if hasattr(timm, "list_models"):
#             return model_name in timm.list_models()
#         if hasattr(timm, "models") and hasattr(timm.models, "list_models"):
#             return model_name in timm.models.list_models()
#     except Exception:
#         return False
#     return False

# from importlib.metadata import version, PackageNotFoundError
# ver = version("timm").split('.')[-1]
# print(ver)
# if int(ver) < 20:
    # !pip uninstall -y timm
subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "timm"])
LOCAL_TIMM = "/kaggle/input/timm-repos/pytorch-image-models"
sys.path.insert(0, LOCAL_TIMM)

import timm
print("timm:", timm.__version__, flush=True)
print("torch:", torch.__version__, flush=True)
# print([m for m in timm.list_models() if "dinov" in m], flush=True)

# _timm_model_name = "vit_small_patch16_dinov3"
# if not _timm_has_model(_timm_model_name):
#     print(f"timm missing {_timm_model_name} ...", flush=True)
    # sys.exit(0)
#     subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "timm"])
        
#     LOCAL_TIMM = "/kaggle/input/timm-repos/pytorch-image-models"
#     if os.path.isdir(LOCAL_TIMM):
#         sys.path.insert(0, LOCAL_TIMM)
#     import timm
# print("timm:", timm.__version__, flush=True)
# print("torch:", torch.__version__, flush=True)
# print([m for m in timm.list_models() if "dinov" in m], flush=True)

_is_kaggle_env = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.path.exists("/kaggle/working"))
# =================================================================================
# Step 2: Configuration
# =================================================================================

# --- Dynamically set root directory ---
is_kaggle = _is_kaggle_env
if _is_kaggle_env:
    is_kaggle = True
    root_dir = "/kaggle/working"
    BASE_PATH = "/kaggle/input/imagenet100/"
    print("kaggle", flush=True)
    print(os.listdir("/kaggle/input"), flush=True)
  

else:
    print("not kaggle", flush=True)
# elif os.path.exists('/home/sshuser'):
#     root_dir = '/home/sshuser'
#     BASE_PATH = f'{root_dir}/Data/imagenet100/'
# elif os.path.exists('/lc'):
#     root_dir = '/lc/logs'
#     BASE_PATH = f'/lc/data/imagenet100/'
# else:
#     root_dir = '/linux'
#     BASE_PATH = f'{root_dir}/Data/imagenet100/'
# --- Configuration via SimpleNamespace for easy interactive use ---
args = SimpleNamespace(
    # --- Model & Training Settings ---
    pos_type = None, #"alibi", # 'sin', 'alibi', 'relpos', None #,  'rpe', 'rope', 
    dynamic_img_size=False,
    model_type= "dinov3",
    use_abs_pos_emb=False,
    use_rot_pos_emb=False,
    model_size='base',
    num_classes=100,
    patch_size = 16,
    grad_accum_steps=2,
    # Adjust based on your GPU memory. BATCH_SIZE = 120, 128, 136, 392, 768, etc.
    batch_size=64, #rpe
    # batch_size=256, #rope
    # batch_size=392, 
    # batch_size=512, 
    # ViT models have a fixed input size
    # img_sizes=[224, 192, 288],
    img_sizes=[224, 192, 288],
    val_img_sizes=[160, 176, 192, 208,224, 256, 272, 288, 320, 336, 352, 368, 384, 400, 416],
    # val_img_sizes=[224],
    # lr=1e-3, #small
    lr=5e-05, #base
    lr_aux=1e-5,
    eta_min=0.0,
    weight_decay=0.01,
    epochs=130,
    # has_pos=True, # Set to True or False directly
    overlap=0,
    pretrained=None,
    seed=16,
    use_patch_position_loss=False,
    use_rc_loss=True,
    # loss_type="smooth_l1", # "mse", "smooth_l1"
    # huber_beta=None,
    # rc_alpha=300.0,
    rc_alpha=600, # base
    warmup_steps_for_aux=600,
    alpha_min=10,
    workers=5,
    randaugment=False,
    randaugment_n=2,
    randaugment_m=3,
    random_erasing=False,
    re_prob=0.0,
    train=True,
    val=False,
    ckpt_path=None,
    lock=True,
    save_full_ckpt=True,
    resume_full_ckpt=True,
    resume_ckpt_path='/kaggle/input/cls-base-colrow-ra200-wsfa600-16/ckpt/last.pth',
    resume_scheduler=True,
    resume_optimizer=True,
    resume_bs=True,
    composite_lr=True,
    warmup_steps=3000,
    clip_value=1.0,
    log_interval=500,
    csv_interval=1,
    show_peak_gpu_mem=True,
    # save_ckpt=False,
    compile_model=False,
    total_run_time_hr=12.0,
    # --- Dataset Paths ---
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
    ]
    if not args.resume_scheduler:
        skip_keys.extend([
            "epochs",
            "warmup_steps",
            "eta_min",
            "composite_lr",
        ])
    if not args.resume_bs:
        skip_keys.extend(["batch_size", "grad_accum_steps"])
    resume_ckpt = torch.load(args.resume_ckpt_path, map_location="cpu", weights_only=False)
    print(f"Resumed args from '{args.resume_ckpt_path}'")
    ckpt_args = resume_ckpt.get("args", None)
    if ckpt_args is not None:
        for k, v in vars(ckpt_args).items():
            if k not in skip_keys:
                setattr(args, k, v)

def _download_with_retries(url, dst, retries=3, timeout=30):
    tmp_path = dst + ".part"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/zip",
    }

    def _cleanup_tmp():
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

    def _finalize():
        if not zipfile.is_zipfile(tmp_path):
            raise RuntimeError("Downloaded file is not a zip.")
        os.replace(tmp_path, dst)

    def _try_curl():
        curl = shutil.which("curl")
        if not curl:
            print("download: curl not found", flush=True)
            return False
        print("download: trying curl", flush=True)
        cmd = [
            curl,
            "-L",
            "--retry",
            "3",
            "--retry-all-errors",
            "--max-redirs",
            "20",
            "--connect-timeout",
            str(timeout),
            "-H",
            f"User-Agent: {headers['User-Agent']}",
            "-H",
            f"Accept: {headers['Accept']}",
            "-o",
            tmp_path,
            url,
        ]
        subprocess.check_call(cmd)
        _finalize()
        print("download: curl ok", flush=True)
        return True

    def _try_requests():
        try:
            import requests  # type: ignore
        except Exception:
            print("download: requests not available", flush=True)
            return False
        print("download: trying requests", flush=True)
        resp = requests.get(url, stream=True, timeout=timeout, headers=headers, allow_redirects=True)
        resp.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        _finalize()
        print("download: requests ok", flush=True)
        return True

    def _try_urllib():
        print("download: trying urllib", flush=True)
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            with open(tmp_path, mode="wb") as f:
                shutil.copyfileobj(resp, f)
        _finalize()
        print("download: urllib ok", flush=True)
        return True

    methods = [_try_curl, _try_requests, _try_urllib]
    last_exc = None
    for attempt in range(1, retries + 1):
        for method in methods:
            try:
                ok = method()
                if ok:
                    return
            except Exception as exc:
                last_exc = exc
                print(f"download: method failed ({exc})", flush=True)
                _cleanup_tmp()
        if attempt < retries:
            time.sleep(2 * attempt)
    raise RuntimeError(f"Downloaded file is not a zip. Last error: {last_exc}")


def _ensure_pos_repo():
    if not _is_kaggle_env:
        return

    def _find_repo_root():
        if os.path.isdir("/kaggle/working/pos"):
            return "/kaggle/working/pos"
        for name in os.listdir("/kaggle/working"):
            cand = os.path.join("/kaggle/working", name)
            if os.path.isdir(os.path.join(cand, "core")) and os.path.isdir(os.path.join(cand, "data")):
                return cand
        return None

    url = "https://github.com/lc126eml/pos/archive/refs/heads/master.zip"
    zip_path = "/kaggle/working/pos.zip"
    if os.path.exists(zip_path) and not zipfile.is_zipfile(zip_path):
        os.remove(zip_path)
    if not os.path.exists(zip_path):
        _download_with_retries(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall("/kaggle/working")
    if os.path.exists(zip_path):
        os.remove(zip_path)

    repo_root = _find_repo_root()
    if not repo_root:
        raise RuntimeError("POS repo not found after unzip; expected /kaggle/working/pos or a repo with core/ and data/.")
    os.environ["POS_REPO_ROOT"] = repo_root
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    print(f"POS repo ready: {repo_root}", flush=True)


_ensure_pos_repo()


def _ensure_timm_pe():
    if args.pos_type is None:
        return
    repo_root = os.environ.get("POS_REPO_ROOT")
    if repo_root and os.path.isdir(os.path.join(repo_root, "timm_pe")):
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
    else:
        repo_root = None
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if os.path.isdir(os.path.join(script_dir, "timm_pe")):
            repo_root = script_dir
        elif _is_kaggle_env and os.path.isdir("/kaggle/working"):
            for name in os.listdir("/kaggle/working"):
                cand = os.path.join("/kaggle/working", name)
                if os.path.isdir(os.path.join(cand, "timm_pe")):
                    repo_root = cand
                    break
        if repo_root:
            sys.path.insert(0, repo_root)
    if repo_root is None:
        raise RuntimeError(
            f"pos_type={args.pos_type} requires timm_pe modules. "
            "Failed to locate timm_pe directory."
        )
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
            f"Failed to import timm_pe. ({exc})"
        ) from exc


_ensure_timm_pe()

if args.pos_type is not None:
    args.has_pos = True
    args.overlap = 0
    args.use_rc_loss=False
    args.use_patch_position_loss=False
    args.dynamic_img_size=False
    args.val=False
if args.use_abs_pos_emb or args.use_rot_pos_emb:
    args.overlap = 0
    args.use_patch_position_loss=False
    args.use_rc_loss = False
offset = 0
# args.batch_size = 64
# args.grad_accum_steps=2
# print(args)
MODEL_NAME = f"vit_{f'{args.pos_type}_' if args.pos_type is not None else ""}{args.model_size}_patch16_{args.model_type}"
if is_kaggle:
    output_dir = args.root_dir
    ckpt_output_dir = os.path.join(output_dir, "ckpt")
else:
    print("not kaggle")
    sys.exit(0)
last_ckpt_path = os.path.join(ckpt_output_dir, f'last.pth')

# %%
# --- Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

use_amp = torch.cuda.is_available()
use_bf16 = use_amp and torch.cuda.is_bf16_supported(including_emulation=False)
autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
use_amp = use_bf16
print(f"Using device: {DEVICE}", use_bf16, autocast_dtype)
# Speed tweaks (P100-friendly)
if torch.cuda.is_available() and use_bf16:
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if len(args.img_sizes) == 1:
        torch.backends.cudnn.benchmark = True
# sys.exit(0)
# torch.backends.cudnn.deterministic=True
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
if torch.cuda.is_available() and len(args.img_sizes) == 1:
    torch.backends.cudnn.benchmark = True
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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

logger.info(f"Using device: {DEVICE}")
logger.info(f"Using mixed precision: {'bfloat16' if use_bf16 else 'float16'}")
logger.info(args)
logger.info(output_dir)
logger.info(subdir_name)

# --- Acquire a file lock to ensure exclusive GPU usage ---
# gpu_lock = None
# if args.lock:
#     if FileLock:
#         lock_path = "/tmp/gpu.lock"
#         gpu_lock = FileLock(lock_path)
#         logger.info(f"Attempting to acquire lock on '{lock_path}'...")
#         gpu_lock.acquire()
#         logger.info("Lock acquired. It is safe to proceed.")
#         # The lock will be automatically released when the script exits.
#     else:
#         logger.warning("`filelock` library not found, skipping lock. Run `pip install filelock`.")

logger.info("Cleaning up memory...")
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
logger.info("Memory cleanup complete.")

# logger.info(args)
#%%
# List of all the partial training directories
TRAIN_PATHS = [
    os.path.join(BASE_PATH, 'train.X1'),
    os.path.join(BASE_PATH, 'train.X2'),
    os.path.join(BASE_PATH, 'train.X3'),
    os.path.join(BASE_PATH, 'train.X4'),
]

VALID_PATH = os.path.join(BASE_PATH, 'val.X')
LABEL_PATH = os.path.join(BASE_PATH, 'Labels.json')


#%%

# if args.pos_type is not None:
#     sys.path.append(r".")
#     from timm_pe.eva_relpos import *
#     from timm_pe.eva_alibi import *
#     from timm_pe.eva_sin import *
    # from vision_transformer_rope import *
    # from vision_transformer_rope2d import *
    # from vision_transformer_rpe import *
    # from vision_transformer_relpos import *
    # from vision_transformer_alibi import *
    # from vision_transformer_sin import *

# %%
# timm.list_models("vit_*_dinov2")
import torch
import torch.nn as nn
import torch.nn.functional as F
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

        # Regression heads: scalar row / scalar col
        self.row_mlp = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)   # scalar row index
        )

        self.col_mlp = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)   # scalar col index
        )

        if huber_beta is None:
            self.loss_fn = nn.SmoothL1Loss()
        else:
            self.loss_fn = nn.SmoothL1Loss(beta=0.5/self.grid_h)

        # Precompute row/col targets once (N = grid_h * grid_w)
        rows_2d = torch.arange(grid_h, dtype=torch.float32).unsqueeze(1).repeat(1, grid_w)
        cols_2d = torch.arange(grid_w, dtype=torch.float32).unsqueeze(0).repeat(grid_h, 1)

        if normalize:
            rows_2d = rows_2d / (grid_h - 1)
            cols_2d = cols_2d / (grid_w - 1)

        # Flatten to 1D (N,)
        row_targets = rows_2d.flatten()
        col_targets = cols_2d.flatten()

        # Register as buffers so they move with .to(device)
        self.register_buffer("row_targets", row_targets, persistent=False)  # (N,)
        self.register_buffer("col_targets", col_targets, persistent=False)  # (N,)

    def forward(self, feats):
        """
        Args:
            feats: (B, N, D) patch features, N = grid_h * grid_w

        Returns:
            avg_loss: scalar, average of row and column regression losses
        """
        B, N, D = feats.shape
        assert N == self.grid_h * self.grid_w, f"Expected N = grid_h * grid_w = {self.grid_h * self.grid_w}, got N = {N}"

        # (B*N, D)
        x = feats.reshape(-1, D)

        # Repeat targets for batch: (N,) -> (B*N,)
        row_targets = self.row_targets.repeat(B)
        col_targets = self.col_targets.repeat(B)

        # Predict rows and columns: (B*N, 1) -> (B*N,)
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

        # MLP for row regression (scalar output)
        self.row_mlp = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)   # scalar row index
        )

        # MLP for column regression (scalar output)
        self.col_mlp = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)   # scalar col index
        )

        self.loss_fn = nn.SmoothL1Loss()

        # Precompute integer row/col indices (max grid) as floats
        rows = torch.arange(grid_h, dtype=torch.float32).unsqueeze(1).repeat(1, grid_w)  # (grid_h, grid_w)
        cols = torch.arange(grid_w, dtype=torch.float32).unsqueeze(0).repeat(grid_h, 1)  # (grid_h, grid_w)

        self.register_buffer("row_index_full", rows, persistent=False)  # (grid_h, grid_w)
        self.register_buffer("col_index_full", cols, persistent=False)  # (grid_h, grid_w)

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

        # Fallback to maximum grid if hp/wp not given
        if hp is None:
            hp = self.grid_h
        if wp is None:
            wp = self.grid_w

        assert N == hp * wp, f"Expected N = hp * wp = {hp * wp}, got N = {N}"

        # Flatten features: (B*N, D)
        x = feats.reshape(-1, D)

        # Slice the index grids to current resolution: (hp, wp)
        row_idx_2d = self.row_index_full[:hp, :wp]  # [0..hp-1]
        col_idx_2d = self.col_index_full[:hp, :wp]  # [0..wp-1]

        if self.normalize:
            # Normalize to [0, 1] based on current hp/wp
            row_idx_2d = row_idx_2d / max(hp - 1, 1)
            col_idx_2d = col_idx_2d / max(wp - 1, 1)

        # Flatten to 1D and repeat for batch: (hp*wp,) -> (B*hp*wp,)
        row_targets = row_idx_2d.flatten().repeat(B)
        col_targets = col_idx_2d.flatten().repeat(B)

        # Predict rows and columns: (B*N, 1) -> (B*N,)
        row_pred = self.row_mlp(x).squeeze(-1)
        col_pred = self.col_mlp(x).squeeze(-1)

        # Regression losses
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

        # MLP for row prediction
        self.row_mlp = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, grid_h)
        )

        # MLP for column prediction
        self.col_mlp = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, grid_w)
        )

        self.ce = nn.CrossEntropyLoss()

        # Precompute row/col labels
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
        # assert N == self.grid_h * self.grid_w, f"Expected {self.grid_h*self.grid_w} patches, got {N}"

        x = feats.reshape(-1, D)  # (B*N, D)

        if hp is None or wp is None:
            hp = self.grid_h
            wp = self.grid_w
        # Repeat labels for batch
        row_labels = self.row_labels[:hp, :wp].flatten().repeat(B)
        col_labels = self.col_labels[:hp, :wp].flatten().repeat(B)

        # Predict rows and columns
        row_logits = self.row_mlp(x)
        col_logits = self.col_mlp(x)

        # Compute cross-entropy loss for rows and columns
        loss_row = self.ce(row_logits, row_labels)
        loss_col = self.ce(col_logits, col_labels)

        return (loss_row + loss_col) / 2  # average

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

        # MLP for row prediction
        self.row_mlp = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, grid_h)
        )

        # MLP for column prediction
        self.col_mlp = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, grid_w)
        )

        self.ce = nn.CrossEntropyLoss()

        # Precompute row/col labels
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

        x = feats.reshape(-1, D)  # (B*N, D)

        # Repeat labels for batch
        row_labels = self.row_labels.repeat(B)
        col_labels = self.col_labels.repeat(B)

        # Predict rows and columns
        row_logits = self.row_mlp(x)
        col_logits = self.col_mlp(x)

        # Compute cross-entropy loss for rows and columns
        loss_row = self.ce(row_logits, row_labels)
        loss_col = self.ce(col_logits, col_labels)

        return (loss_row + loss_col) / 2  # average


# if Use_Row_Col_Loss:
#     grid_h, grid_w = model.patch_embed.grid_size
#     rowcol_loss = PatchRowColCriterion(
#         feat_dim=model.embed_dim,
#         grid_h=grid_h,
#         grid_w=grid_w
#     ).to(DEVICE)
#     print("OK: Row-Column loss initialized.")

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

        # Precompute patch position labels once
        self.register_buffer("patch_positions", torch.arange(num_classes), persistent=False)  # shape (num_patches,)
        
    def forward(self, feats):
        """
        Args:
            feats: (B, N, D) patch features
        Returns:
            avg_loss: scalar, mean cross-entropy over all patches
        """
        B, N, D = feats.shape
        assert N == self.num_classes, f"Expected {self.num_classes} patches, got {N}"

        # Flatten batch and patches: (B*N, D)
        x = feats.reshape(-1, D)
        # Repeat labels for all images in batch: (B*N,)
        labels = self.patch_positions.repeat(B)
        # Predict positions
        logits = self.mlp(x)
        # Compute CE loss
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
            nn.Linear(256, 1)   # scalar position index
        )

        self.loss_fn = nn.SmoothL1Loss()

        # Precompute patch position targets once
        position_targets = torch.arange(num_classes, dtype=torch.float32)
        if normalize:
            position_targets = position_targets / max(num_classes - 1, 1)
        self.register_buffer("position_targets", position_targets, persistent=False)  # (N,)

    def forward(self, feats):
        """
        Args:
            feats: (B, N, D) patch features
        Returns:
            loss: scalar, SmoothL1 loss over all patches
        """
        B, N, D = feats.shape
        assert N == self.num_classes, f"Expected {self.num_classes} patches, got {N}"

        # Flatten batch and patches: (B*N, D)
        x = feats.reshape(-1, D)
        # Repeat targets for batch: (N,) -> (B*N,)
        targets = self.position_targets.repeat(B)
        # Predict positions: (B*N, 1) -> (B*N,)
        pred = self.mlp(x).squeeze(-1)
        # Compute regression loss
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
            nn.Linear(256, 1)   # scalar position index
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

        # Flatten features: (B*N, D)
        x = feats.reshape(-1, D)

        # Slice position indices to current patch count: (N,)
        pos_idx = self.position_index_full[:N]

        if self.normalize:
            pos_idx = pos_idx / max(N - 1, 1)

        # Repeat for batch: (N,) -> (B*N,)
        targets = pos_idx.repeat(B)

        # Predict positions: (B*N, 1) -> (B*N,)
        pred = self.mlp(x).squeeze(-1)

        loss = self.loss_fn(pred, targets)
        return loss
        
# if Use_Patch_Position_Loss:
#     position_loss = PatchPositionCriterion(
#         feat_dim=model.embed_dim,
#         num_classes=model.patch_embed.num_patches
#     ).to(DEVICE)


from torch.utils.data import Dataset
from PIL import Image

class MultiScaleImageDataset(Dataset):
    def __init__(self, samples, size_to_transform):
        """
        samples: list of (path, target)
        size_to_transform: dict[int, torchvision.transforms.Compose]
        """
        self.samples = samples
        self.size_to_transform = size_to_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, key):
        # key comes from the batch sampler: (idx, size)
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
    
import math
import random

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
        seed: int = 0,
    ):
        self.dataset_len = len(dataset)
        self.image_sizes = list(image_sizes)
        self.base_batch_size = base_batch_size
        self.base_img_size = base_img_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        # pixel budget based on reference configuration
        self.pixel_budget = base_batch_size * (base_img_size ** 2)

        # for __len__ (approximate)
        avg_size_sq = sum(s * s for s in self.image_sizes) / len(self.image_sizes)
        self.avg_batch_size = self.pixel_budget / avg_size_sq

    def __len__(self):
        # approximate number of batches per epoch
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

        while ptr < n:
            # 1) choose resolution for this batch
            size = rng.choice(self.image_sizes)

            # 2) compute batch size from pixel budget
            pixels_per_sample = size * size
            if pixels_per_sample > 0:
                batch_size = max(1, self.pixel_budget // pixels_per_sample)
            else:
                batch_size = self.base_batch_size

            # 3) adjust for remaining samples
            remaining = n - ptr
            if remaining < batch_size:
                if self.drop_last:
                    break
                else:
                    batch_size = remaining

            batch_indices = indices[ptr: ptr + batch_size]
            ptr += batch_size

            # 4) yield (idx, size) pairs
            yield [(idx, size) for idx in batch_indices]

# %%
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import collections

# =================================================================================
# Step 1: Configuration
# =================================================================================

# =================================================================================
# Step 2: Custom Dataset Class
# =================================================================================
# This simple Dataset class will load images from a pre-made list of file paths.

# =================================================================================
# Step 3: Efficiently Find and Load Data for Only 10 Classes
# =================================================================================

# --- Discover and select the first 10 class folders ---
# This is a fast filesystem operation. We only scan one directory to get the names.
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
# --- Manually build the list of training samples (images, labels) ---
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

# --- Manually build the list of validation samples ---
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

# =================================================================================
# Step 4: Create Datasets and DataLoaders
# =================================================================================
#%%
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

img_mean = [0.485, 0.456, 0.406]
img_std  = [0.229, 0.224, 0.225]


def make_train_transform(size: int):
    t_list = [
        T.RandomResizedCrop(size, interpolation=InterpolationMode.BICUBIC, antialias=True),
        T.RandomHorizontalFlip(),
    ]
    # if args.randaugment:
    #     t_list.append(
    #         T.RandAugment(
    #             num_ops=args.randaugment_n,
    #             magnitude=args.randaugment_m,
    #             interpolation=InterpolationMode.BICUBIC,
    #             fill=(128, 128, 128),
    #         )
    #     )
    t_list.extend([
        T.ToTensor(),
        T.Normalize(mean=img_mean, std=img_std),
    ])
    # if args.random_erasing:
    #     t_list.append(T.RandomErasing(p=args.re_prob))
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
# --- Create the final datasets from the filtered samples ---

valid_dataset = CustomImageDataset(valid_samples, transform=valid_transforms)

logger.info(f"Total validation images ({args.num_classes} classes): {len(valid_dataset)}")

# --- Create DataLoaders ---
batch_sampler = None
prefetch_kwargs = {"prefetch_factor": 2} if args.workers > 0 else {}
train_generator = torch.Generator()
train_generator.manual_seed(args.seed)
if args.resume_full_ckpt and args.resume_ckpt_path and resume_ckpt is not None:
    rng_state = resume_ckpt.get("rng_state", None)
    if isinstance(rng_state, dict):
        try:
            if "python" in rng_state:
                random.setstate(rng_state["python"])
            if "numpy" in rng_state:
                np.random.set_state(rng_state["numpy"])
            if "torch" in rng_state:
                torch.set_rng_state(rng_state["torch"])
            if torch.cuda.is_available() and rng_state.get("cuda") is not None:
                torch.cuda.set_rng_state_all(rng_state["cuda"])
            if rng_state.get("data_rng") is not None:
                train_generator.set_state(rng_state["data_rng"])
            elif rng_state.get("train_generator") is not None:
                train_generator.set_state(rng_state["train_generator"])
            logger.info("Restored RNG states from checkpoint.")
        except Exception as exc:
            logger.warning("Failed to restore RNG states from checkpoint: %s", exc)
if len(args.img_sizes) == 1:
    train_dataset = CustomImageDataset(train_samples, transform=size_to_transform[args.img_sizes[0]])
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0),
        **prefetch_kwargs,
    )
else:
    train_dataset = MultiScaleImageDataset(
        samples=train_samples,              # list of (path, target)
        size_to_transform=size_to_transform
    )
    batch_sampler = DynamicResolutionBatchSampler(
        dataset=train_dataset,
        image_sizes=args.img_sizes,
        base_batch_size=args.batch_size,    # your "reference" batch size
        base_img_size=224, #args.img_sizes[0],       # your "reference" resolution (e.g. 224)
        shuffle=True,
        drop_last=True,
        seed=42,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_sampler=batch_sampler,
        num_workers=args.workers,           # now workers do the transforms
        pin_memory=True,
        persistent_workers=(args.workers > 0),
        **prefetch_kwargs,
    )
logger.info(f"Total training images ({args.num_classes} classes): {len(train_dataset)}")
valid_loader = DataLoader(
    dataset=valid_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.workers,
    pin_memory=True,
    persistent_workers=(args.workers > 0),
    **prefetch_kwargs,
)
steps_per_epoch = len(train_loader)
accum_steps = max(1, int(getattr(args, "grad_accum_steps", 1)))
optimizer_steps_per_epoch = math.ceil(steps_per_epoch / accum_steps)
logger.info(f"OK: DataLoaders for {args.num_classes} classes created successfully.")
logger.info(f"{steps_per_epoch=}, val_steps: {len(valid_loader)}")
logger.info(f"Effective batch size: {args.batch_size * accum_steps}")

# %%
# %% [code]
# =================================================================================
# Step 3.5: Visualize a Batch of Training Data
# =================================================================================
import matplotlib.pyplot as plt
import numpy as np
import torchvision

def imshow(inp, title=None):
    """A helper function to denormalize and display an image tensor."""
    # Define the same mean and std used for normalization
    mean = np.array(img_mean)
    std = np.array(img_std)
    
    # Transpose from (C, H, W) to (H, W, C)
    inp = inp.numpy().transpose((1, 2, 0))
    # Denormalize
    inp = std * inp + mean
    # Clip values to be between 0 and 1
    inp = np.clip(inp, 0, 1)
    
    plt.imshow(inp)
    if title is not None:
        plt.title(title, fontsize=10)
    plt.axis('off')

# Get one batch of training images
# try:
#     inputs, classes = next(iter(train_loader))
    
#     # Get the class names from the dataset object
#     # class_names = meta_dict['fine_label_names']

#     # Create a grid of images
#     fig = plt.figure(figsize=(16, 8))
#     plt.suptitle("Sample Images from CIFAR-100 Dataset", fontsize=16)
    
#     # Display the first 16 images from the batch
#     for i in range(16):
#         ax = plt.subplot(4, 8, i + 1)
#         class_name = classes[i]
#         imshow(inputs[i], title=class_name)
        
#     plt.tight_layout(rect=[0, 0, 1, 0.96])
#     plt.show()

# except NameError:
#     logger.info("Could not display images. Please ensure the previous cells have been run to create 'train_loader'.")



# %%
# =================================================================================
# Step 4: Initialize the Model, Loss Function, and Optimizer
# =================================================================================
# --- Model ---
logger.info(f"Initializing model: {MODEL_NAME} for {args.num_classes} classes...")
model = timm.create_model(
    MODEL_NAME,
    pretrained=False, # As requested: trains the model from scratch
    use_abs_pos_emb=args.use_abs_pos_emb,
    use_rot_pos_emb=args.use_rot_pos_emb,
    num_classes=args.num_classes, # Set the classifier head to 100 classes
    dynamic_img_size=args.dynamic_img_size,
    img_size=args.img_sizes[0],
).to(DEVICE)
# feature_layers = [2, 5, 8, 11]
# dummy_input = torch.randn(2, 3, args.img_size, args.img_size).to(DEVICE)
# with torch.no_grad():
#     feats = model.forward_features(dummy_input)
#     multi_feats = model.forward_intermediates(dummy_input, indices=feature_layers, intermediates_only=True)


# logger.info(f"Model created successfully!")
# logger.info(f"Input shape: {dummy_input.shape}")
# logger.info(f"Output shape: {feats.shape}") 
# logger.info(f"multi_feats shape: {multi_feats[-1].shape} X {len(multi_feats)}")
# del feats, multi_feats, dummy_input
# gc.collect()

# %%
logger.info(f'model.patch_embed.proj{model.patch_embed.proj}')
# if args.overlap > 0:
#     # Customize patch embedding for overlap (e.g., patch_size=15, stride=14)
#     original_patch_size = model.patch_embed.proj.kernel_size[0]
#     new_patch_size = original_patch_size + args.overlap  # Or 15, 16, 17, etc., as desired
#     stride = original_patch_size
#     original_grid_size = args.img_sizes[0] // stride  # 16 for 224//14
#     padding = ((original_grid_size - 1) * stride + new_patch_size - args.img_sizes[0] + 1) // 2  # +1 for ceiling effect; yields 1 for patch_size=15
    
#     # Override the PatchEmbed projection (Conv2d layer)
#     in_chans = model.patch_embed.proj.in_channels  # Typically 3 for RGB
#     embed_dim = model.patch_embed.proj.out_channels  # e.g., 768 for base
#     model.patch_embed.proj = nn.Conv2d(
#         in_chans, embed_dim,
#         kernel_size=(new_patch_size, new_patch_size),
#         stride=(stride, stride),
#         padding=padding  # Updated to ensure full coverage and original grid size
#     ).to(DEVICE)
    
    # Recompute grid size and num_patches
    # grid_size_h = ((args.img_size + 2 * padding - new_patch_size) // stride) + 1
    # grid_size_w = grid_size_h  # Assuming square input
    # logger.info(new_patch_size, padding, grid_size_h, model.patch_embed.grid_size)
    # model.patch_embed.grid_size = (grid_size_h, grid_size_w)
    # model.patch_embed.num_patches = grid_size_h * grid_size_w
    # logger.info(f"Updated to patch_size={new_patch_size}, stride={stride}, padding={padding}, num_patches={model.patch_embed.num_patches}")

# if not args.has_pos and hasattr(model, 'pos_embed') and model.pos_embed is not None:
#     model.pos_embed.data.zero_()
#     model.pos_embed.requires_grad = False
#     logger.info("OK: Positional embedding has been disabled.")

# if not args.has_pos or args.pos_type is not None:
#     if hasattr(model, 'pos_embed') and model.pos_embed is not None:
#         model.pos_embed.data.zero_()
#         model.pos_embed.requires_grad = False
#         logger.info("OK: Positional embedding has been disabled.")
#     if hasattr(model, 'rope'):
#         model.rope = None

# if args.pretrained is not None:
#     state_dicts = torch.load(args.pretrained, map_location=DEVICE)
#     IncompatibleKeys = model.load_state_dict(state_dicts)
#     logger.info(IncompatibleKeys)
# %%
if args.compile_model and len(args.img_sizes)==1:
    if hasattr(torch, "compile"):
        logger.info("Compiling model with torch.compile (mode='reduce-overhead').")
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
    else:
        logger.warning("torch.compile not available; skipping compilation.")

dynamic = True
training_parameters = list(model.parameters()) 
param_groups = []
lr_aux = getattr(args, "lr_aux", args.lr)
if args.use_rc_loss:
    if len(args.img_sizes)==1:
        grid_h, grid_w = model.patch_embed.grid_size
        dynamic = False
        # from core.patch_pos import PatchRowColRegressionCriterion
        rowcol_loss = PatchRowColRegressionCriterion(
            feat_dim=model.embed_dim,
            grid_h=grid_h,
            grid_w=grid_w,
            # loss_type=args.loss_type,
        ).to(DEVICE)
    else:
        grid_h = grid_w = max(args.img_sizes)//args.patch_size
        # from core.patch_pos import PatchRowColRegressionCriterionDynamic
        rowcol_loss = PatchRowColRegressionCriterionDynamic(
            feat_dim=model.embed_dim,
            grid_h=grid_h,
            grid_w=grid_w,
            # loss_type=args.loss_type,
        ).to(DEVICE)
    training_parameters += list(rowcol_loss.parameters())
    param_groups.append({"params": rowcol_loss.parameters(), "weight_decay": 0.0, "lr": lr_aux})
if args.use_patch_position_loss:
    if len(args.img_sizes)==1:
        # from core.patch_pos import PatchPositionRegressionCriterion
        position_loss = PatchPositionRegressionCriterion(
            feat_dim=model.embed_dim,
            num_classes=model.patch_embed.num_patches
        ).to(DEVICE)
    else:
        max_grid = max(args.img_sizes)//args.patch_size
        max_patch_count = max_grid * max_grid
        # from core.patch_pos import PatchPositionRegressionCriterionDynamic
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
# --- Loss Function & Optimizer ---
criterion = nn.CrossEntropyLoss()
if args.composite_lr:
    # optimizer = torch.optim.AdamW(training_parameters, lr=args.lr, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)

    total = sum(p.numel() for p in model.parameters())
    opt_total = sum(p.numel() for g in optimizer.param_groups for p in g["params"])
    print("model params:", total, "optimizer params:", opt_total)

    # Ensure no parameter appears in multiple groups
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
    # warmup_steps = 100 #int(0.01 * total_steps)

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-7 / args.lr,   # warmup start lr = 1e-7, weight_decay=0.05
        end_factor=1.0,                # warmup end lr = base_lr
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
    # optimizer = optim.AdamW(training_parameters, lr=args.lr, weight_decay=args.weight_decay)
    logger.info("OK: Model, Loss Function, and Optimizer are ready.")

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    # logger.info("OK: Model, Loss, Optimizer, and LR Scheduler are ready.")

    total_steps = args.epochs * optimizer_steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.eta_min)
    logger.info("OK: Step-based LR Scheduler is ready.")


# %%
# dummy_input = torch.randn(2, 3, args.img_sizes[0], args.img_sizes[0]).to(DEVICE)
# with torch.no_grad():
#     feats = model.forward_features(dummy_input)
# logger.info(f"Model created successfully!")
# logger.info(f"Input shape: {dummy_input.shape}")
# logger.info(f"Output shape: {feats.shape}") 
    
sys.stdout.flush()
# %%
#%%
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


# %%
import csv

ckpt_path = None
if args.train:
    # FP16: Initialize the Gradient Scaler
    use_scaler = use_amp and (autocast_dtype == torch.float16)
    scaler = torch.amp.GradScaler(enabled=use_scaler)
    start_epoch = 0
    step = 0
    best_acc = 0.0
    if args.resume_full_ckpt and args.resume_ckpt_path:
        # resume_ckpt = ckpt
        # torch.load(args.resume_ckpt_path, map_location="cpu", weights_only=False)
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
        if resume_ckpt.get("scaler") is not None:
            scaler.load_state_dict(resume_ckpt["scaler"])
        if args.use_rc_loss and resume_ckpt.get("rowcol_loss") is not None:
            for k in ["row_targets", "col_targets", "row_index_full", "col_index_full"]:
                if k in resume_ckpt["rowcol_loss"]:
                    resume_ckpt["rowcol_loss"].pop(k)
            rowcol_loss.load_state_dict(resume_ckpt["rowcol_loss"])
        if args.use_patch_position_loss and resume_ckpt.get("position_loss") is not None:
            position_loss.load_state_dict(resume_ckpt["position_loss"])
        best_acc = resume_ckpt.get("best_acc", 0.0)
        logger.info(f"Resumed full checkpoint from '{args.resume_ckpt_path}' at epoch={start_epoch}, step={step}")
    # =================================================================================
    # Step 5: Training and Validation Loop
    # =================================================================================
    logger.info(f"\nStarting training for {MODEL_NAME}...")

    # OK: Initialize training_history as a dictionary of lists
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
    # train_epoch_times = []
    for epoch in range(start_epoch, args.epochs):
        epoch_train_start = time.time()
        # --- Training Phase ---
        model.train()
        # epoch_train_start = time.perf_counter()

        aux_loss = None

        running_loss_t = torch.zeros((), device=DEVICE)   # scalar tensor
        aux_loss_sum_t = torch.zeros((), device=DEVICE)
        base_loss_t = torch.zeros((), device=DEVICE)
        train_correct_t = torch.zeros((), device=DEVICE)
        train_total = 0

        # running_loss = 0.0
        # train_correct = 0
        train_total = 0
        # aux_loss_sum = 0.0
        # train_pbar = train_loader
        if batch_sampler is not None:
            batch_sampler.set_epoch(epoch)
        
        total_batches = len(train_loader)
        # FP16: Use autocast for the forward pass
        optimizer.zero_grad(set_to_none=True)
        for step_in_epoch, (inputs, labels) in enumerate(train_loader):
            if (step_in_epoch % accum_steps) == 0 and (total_batches - step_in_epoch) < accum_steps:
                break
            inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            bs = inputs.size(0)
            if args.show_peak_gpu_mem and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            aux_loss = None
            with torch.amp.autocast(device_type=DEVICE.type, dtype=autocast_dtype, enabled=use_amp):
                feats = model.forward_features(inputs)
                outputs = model.forward_head(feats)
                # outputs = model(inputs)
                loss = criterion(outputs, labels)
                if args.use_rc_loss:
                    base_loss_t += loss.detach() * bs
                    if dynamic:
                        hp, wp = get_patch_numbers(inputs.shape[-2:], model.patch_embed.patch_size[0])
                        aux_loss = rowcol_loss(feats[:, model.num_prefix_tokens:, :], hp, wp)
                    else:
                        aux_loss = rowcol_loss(feats[:, model.num_prefix_tokens:, :])
                    
                    # logger.info(f"grid={model.patch_embed.grid_size}, {dynamic=} num_prefix={model.num_prefix_tokens}")
                    # # once after a forward:
                    # logger.info(f"feats={feats.shape}, patch_tokens={feats[:, model.num_prefix_tokens:, :].shape[1]}")

                    aux_loss_sum_t += aux_loss.detach() * bs
                    t = min(1.0, (step + 1) / args.warmup_steps_for_aux)
                    alpha_t = args.alpha_min + (args.rc_alpha - args.alpha_min) * t
                    loss = loss + alpha_t * aux_loss
                
                elif args.use_patch_position_loss:
                    base_loss_t += loss.detach() * bs
                    aux_loss = position_loss(feats[:, model.num_prefix_tokens:, :])
                    aux_loss_sum_t += aux_loss.detach() * bs
                    t = min(1.0, (step + 1) / args.warmup_steps_for_aux)
                    alpha_t = args.alpha_min + (args.rc_alpha - args.alpha_min) * t
                    loss = loss + alpha_t * aux_loss
            
            # FP16: Scale, backward, and step (with grad accumulation)
            loss_scaled = loss / accum_steps
            scaler.scale(loss_scaled).backward()

            do_step = ((step_in_epoch + 1) % accum_steps == 0) or (step_in_epoch + 1 == len(train_loader))
            if do_step:
                if args.clip_value is not None:
                    scaler.unscale_(optimizer)
                    # log_grads(logger, model, rowcol_loss=rowcol_loss if args.use_rc_loss else None,
            #   every=331, step=step)
                    torch.nn.utils.clip_grad_norm_(training_parameters, max_norm=args.clip_value)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss_t += loss.detach() * bs
            train_total += bs

            with torch.no_grad():
                pred = outputs.detach().argmax(dim=1)
                train_correct_t += (pred == labels).sum()

            # only log every N steps (minimize sync + formatting)
            if (step + 1) % log_interval == 0:
                # now pay the sync cost, but only occasionally
                avg_loss = (running_loss_t / train_total).float().item()
                avg_acc = (train_correct_t / train_total).float().item()
                peak_mb = None
                if args.show_peak_gpu_mem and torch.cuda.is_available():
                    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                msg = f"Epoch {epoch+1}/{args.epochs} step {step+1}: loss={avg_loss:.4f} acc={avg_acc:.3f}"
                if aux_loss is not None:
                    avg_aux = (aux_loss_sum_t / train_total).float().item()
                    msg += f" aux={avg_aux:.4f}"
                if peak_mb is not None:
                    msg += f" peak_mem={peak_mb:.0f}MB"
                logger.info(msg)

            step += 1

        train_time = time.time() - epoch_train_start
        # if (step + 1) % VAL_STEPS == 0:
        # epoch_train_end = time.perf_counter()
        # train_epoch_times.append(epoch_train_end - epoch_train_start)
        # --- Validation Phase ---
        model.eval()
        val_correct_t = torch.zeros((), device=DEVICE)
        val_total = 0
        # val_pbar = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Validation]")
        val_start = time.time()
        
        with torch.inference_mode():
            for inputs, labels in valid_loader:
                inputs = inputs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                with torch.amp.autocast(device_type=DEVICE.type, dtype=autocast_dtype, enabled=use_amp):
                    outputs = model(inputs)
                pred = outputs.argmax(dim=1)
                val_correct_t += (pred == labels).sum()
                val_total += labels.size(0)

        val_time = time.time() - val_start
        epoch_val_acc = (val_correct_t / val_total).item()
        is_best = False
        if best_acc < epoch_val_acc:
            best_acc = epoch_val_acc
            is_best = True

        epoch_train_acc  = (train_correct_t / train_total).item()
        epoch_train_loss = (running_loss_t / train_total).item()
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
        

        # OK: Append the results to the correct lists within the dictionary
        
        training_history['train_loss'].append(epoch_train_loss)
        training_history['train_acc'].append(epoch_train_acc)
        training_history['valid_acc'].append(epoch_val_acc)  
        training_history['train_time'].append(train_time)
        training_history['val_time'].append(val_time)
        training_history['epoch'].append(epoch+1)
        training_history['step'].append(step+1)
        if (epoch + 1) % csv_interval == 0:
            pd.DataFrame(training_history).to_csv(os.path.join(output_dir, f'{subdir_name}.csv'), index=False)
        if args.save_full_ckpt:
            ckpt = {
                "epoch": epoch + 1,
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "scaler": scaler.state_dict() if scaler is not None else None,
                "rowcol_loss": rowcol_loss.state_dict() if args.use_rc_loss else None,
                "position_loss": position_loss.state_dict() if args.use_patch_position_loss else None,
                "training_history": training_history,
                "rng_state": {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    "data_rng": train_generator.get_state(),
                },
                "args": args,
                "best_acc": best_acc,
            }
            torch.save(ckpt, last_ckpt_path)
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
        # gc.collect()
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()

        # Update the learning rate scheduler
        # if 'scheduler' in locals():
        #     scheduler.step()

    else:
        if args.pos_type is None:
            args.val = True

    logger.info("Training complete.")
    logger.info(f"Best Accuracy: {best_acc:.4f}")
    logger.info(output_dir)

    # =================================================================================
    # Step 6: Save the Results and Model
    # =================================================================================

    # OK: Step 1: Convert the dictionary directly into a pandas DataFrame

    # OK: Step 2: Add the 'epoch' column at the beginning
    # Create the list of epochs where validation was actually performed
    # epochs_validated = range(5, EPOCHS + 1, 5) 
    # history_df.insert(0, 'epoch', epochs_validated)

    # OK: Step 3: Save the DataFrame to a CSV file
    pd.DataFrame(training_history).to_csv(os.path.join(output_dir, f'{subdir_name}.csv'), index=False)
    # times_csv_path = os.path.join(output_dir, f'{subdir_name}_train_epoch_times.csv')
    # logger.info(f"{train_epoch_times=}")
    # with open(times_csv_path, "w", newline="") as csv_file:
    #     writer = csv.writer(csv_file)
    #     for epoch_time in train_epoch_times:
    #         writer.writerow([epoch_time])
    # if args.save_ckpt:
    #     # Save the model's state dictionary
    #     ckpt_path = os.path.join(ckpt_output_dir,  f'{subdir_name}{MODEL_NAME}_final.pth')
    #     torch.save(model.state_dict(), ckpt_path)
    #     logger.info(f"OK: Model saved to '{ckpt_path}'")

if args.val:    
    val_results = {
        'img_size': [],
        'valid_acc': []
    }

    if not args.train:
        if ckpt_path is None:
            ckpt_path = f"{args.root_dir}/{args.ckpt_path}"
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False))
    model.to(DEVICE)
    model.eval()
    for img_size in args.val_img_sizes:
        valid_dataset.set_transform(make_valid_transform(img_size))
        batch_size = max(1, int((args.batch_size * 0.8 * 224 * 224) / (img_size * img_size)))
        valid_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=False,
            **prefetch_kwargs,
        )
        val_correct = 0
        val_total = 0
        with torch.inference_mode():
            for inputs, labels in valid_loader:
                inputs = inputs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                with torch.amp.autocast(device_type=DEVICE.type, dtype=autocast_dtype, enabled=use_amp):
                    outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        epoch_val_acc = val_correct / val_total
        val_results['img_size'].append(img_size)
        val_results['valid_acc'].append(epoch_val_acc)
        val_df = pd.DataFrame(val_results)
        val_df.to_csv(os.path.join(output_dir, f'{subdir_name}_eval.csv'), index=False)
        logger.info(f"{img_size=}: {epoch_val_acc=}")

# del model
# gc.collect()
# if torch.cuda.is_available():
#     torch.cuda.empty_cache()

# if gpu_lock and gpu_lock.is_locked:
#     logger.info("Manually releasing lock.")
#     gpu_lock.release()
# %%
# import matplotlib.pyplot as plt
# import pandas as pd

# if history_df is None:
#     logger.info("Training history is empty. Please run the training loop first.")
# else:
#     # --- Create a single figure and axis for the plot ---
#     fig, ax = plt.subplots(figsize=(12, 7))
#     plt.title('Training and Validation Accuracy Over Epochs', fontsize=16)
    
#     # --- Plot Training & Validation Accuracy ---
#     ax.plot(history_df['epoch'], history_df['train_acc'], 's--', color='tab:green', label='Training Accuracy')
#     ax.plot(history_df['epoch'], history_df['valid_acc'], '^-', color='tab:blue', label='Validation Accuracy')
    
#     # --- Set labels and legend ---
#     ax.set_xlabel('Epochs')
#     ax.set_ylabel('Accuracy')
#     ax.legend()
#     ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
#     # Set the y-axis to be formatted as percentages
#     ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
#     ax.set_ylim(0, 1) # Set y-axis limits from 0 to 1 for accuracy

#     # Set the x-axis to show integer epoch numbers
#     ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

#     plt.tight_layout()
#     plt.show()
# %%
# import matplotlib.pyplot as plt
# import pandas as pd

# if history_df is None:
#     logger.info("Training history is empty. Please run the training loop first.")
# else:
#     # --- Create a single figure and axis for the plot ---
#     fig, ax = plt.subplots(figsize=(12, 7))
#     plt.title('Training and Validation Accuracy Over Epochs', fontsize=16)
    
#     # --- Plot Training & Validation Accuracy ---
#     ax.plot(history_df['step'], history_df['train_acc'], 's--', color='tab:green', label='Training Accuracy')
#     ax.plot(history_df['step'], history_df['valid_acc'], '^-', color='tab:blue', label='Validation Accuracy')
    
#     # --- Set labels and legend ---
#     ax.set_xlabel('Steps')
#     ax.set_ylabel('Accuracy')
#     ax.legend()
#     ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
#     # Set the y-axis to be formatted as percentages
#     ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
#     ax.set_ylim(0, 1) # Set y-axis limits from 0 to 1 for accuracy

#     # Set the x-axis to show integer epoch numbers
#     ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

#     plt.tight_layout()
#     plt.show()
