# %%
# =================================================================================
# DINOv3 depth training (single-file, Kaggle-friendly)
# =================================================================================
import gc
import glob
import math
import os
import sys
import time
import logging
import random
import subprocess
import shutil
import urllib.request
import zipfile
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import cv2

import torchvision.transforms.functional as TF
from torchvision.transforms import ColorJitter, GaussianBlur

# =============================================================================
# Kaggle environment setup
# =============================================================================
_IS_KAGGLE = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.path.exists("/kaggle/working"))

CUDA_ALLOC_CONF_DEFAULT = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
if CUDA_ALLOC_CONF_DEFAULT:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = CUDA_ALLOC_CONF_DEFAULT

# ----------------------------------------------------------------------------
# timm: prefer local Kaggle repo
# ----------------------------------------------------------------------------
if _IS_KAGGLE:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "timm"])
    except Exception:
        pass
    LOCAL_TIMM = "/kaggle/input/timm-repos/pytorch-image-models"
    if os.path.isdir(LOCAL_TIMM):
        sys.path.insert(0, LOCAL_TIMM)
else:
    LOCAL_TIMM = os.environ.get("LOCAL_TIMM_DIR", "/home/liucong/codes/pos/timm/pytorch-image-models-main")
    if os.path.isdir(LOCAL_TIMM):
        sys.path.insert(0, LOCAL_TIMM)

import timm

print("timm:", timm.__version__, flush=True)
print("torch:", torch.__version__, flush=True)


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
    if not _IS_KAGGLE:
        return None

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
    return repo_root

# =============================================================================
# Configuration
# =============================================================================
if _IS_KAGGLE:
    train_roots_default = [
        "/kaggle/input/hsm-train-part01",
        "/kaggle/input/hsm-train-part02",
        "/kaggle/input/hsm-train-part03",
        "/kaggle/input/hsm-train-part04",
        "/kaggle/input/hsm-train-part05",
    ]
    eval_root_default = "/kaggle/input/hsm-test-val"
    output_root_default = "/kaggle/working"
else:
    # Fallback for local usage
    if os.path.exists("/lc"):
        output_root_default = "/lc/logs"
        eval_root_default = "/lc/data/3D"
    elif os.path.exists("/home/liucong"):
        output_root_default = "/home/liucong/codes/pos/logs"
        eval_root_default = "/home/liucong/data/3d"
    else:
        output_root_default = "/tmp"
        eval_root_default = "/tmp"
    train_roots_default = [os.path.join(eval_root_default, "hypersim_processed", "train")]

args = SimpleNamespace(
    # Data
    train_roots=train_roots_default,
    eval_root=eval_root_default,
    eval_split="test",  # "val" or "test" when eval_root has subdirs
    model_type="dinov3",
    use_abs_pos_emb=False,
    use_rot_pos_emb=False,
    model_size='base',
    train_sizes=[(224, 224)],
    eval_size=(224, 224),
    final_eval_size=[(224, 224)],
    # final_eval_size=[(224, 224), (240,240), (240, 320), (288, 384), (336, 448), (384, 512)],
    color_jitter_prob=0.5,
    scale_jitter=(1.0, 1.2),
    scale_jitter_sw=(1.0, 1.01),
    batch_size=24,
    eval_batch_size=24,
    image_list_path=None,
    grad_accum_steps=1,
    patch_size=16,
    lr=5.0e-5, #7e-5
    lr_aux=1e-5,
    eta_min=1e-7,
    epochs=130,
    break_at_epoch=None,
    has_pos=False,
    weight_decay=0.05,
    overlap=0,
    seed=18,
    val_steps=None,
    use_rc_loss=True,
    loss_type="smooth_l1",
    rc_alpha=200,
    warmup_steps_for_aux=600,
    alpha_min=10,
    workers=2 if _IS_KAGGLE else 8,
    composite_lr=True,
    warmup_steps=3000,
    warmup_ratio=None,
    clip_value=1.0,
    debug_loss_stats=False,
    debug_loss_interval=1,
    depth_decoder="dpt",  # "simple", "lite4", or "dpt"
    log_interval=500,
    show_peak_gpu_mem=True,
    depth_eval_mode="relative",  # "relative", "metric", or "scale_invariant"
    align_mode="mean_std",
    silog_w=0.0,
    depth_norm="median",
    ssim_norm_mode="per_image",
    ssim_percentiles=(5.0, 95.0),
    eval_crop_mode="crop",
    eval_dataset="hypersim",  # "hypersim" or "nyu"
    eval_depth_min=1e-3,
    eval_depth_max=None,
    eval_prescale=1.07,
    train_depth_valid_thresh=0.1,
    eval_depth_valid_thresh=0.01,
    use_sliding_window=False,
    sw_window_size=None,
    sw_overlap=0.25,
    debug_dataset=False,
    output_dir=output_root_default,
    csv_interval=5,
    prefetch_factor=2,
    compile_model=False,
    save_full_ckpt=True,
    resume_full_ckpt=True,
    resume_ckpt_path='/kaggle/input/depth-base-colrow-gpu-518/ckpt/last.pth',
    resume_args=True,
    resume_scheduler=True,
    resume_optimizer=False,
    resume_bs=True,
    resume_img_size=False,
    total_run_time_hr=12.0,
    train=True,
    val=False,
    final_use_sliding_window=False,
    final_sw_window_size=None,
    final_sw_overlap=0.25,
    cuda_alloc_conf=CUDA_ALLOC_CONF_DEFAULT,
)

ckpt = None
if args.resume_full_ckpt and args.resume_ckpt_path:
    if not os.path.exists(args.resume_ckpt_path):
        search_root = "/kaggle/input" if _IS_KAGGLE else os.path.dirname(args.resume_ckpt_path)
        candidates = sorted(glob.glob(os.path.join(search_root, "**", "last.pth"), recursive=True))
        if candidates:
            args.resume_ckpt_path = candidates[0]
    ckpt = torch.load(args.resume_ckpt_path, map_location="cpu", weights_only=False)
    if args.resume_args:
        skip_keys = [
            "resume_full_ckpt",
            "resume_ckpt_path",
            "resume_bs",
            "resume_scheduler",
            "resume_optimizer",
            "total_run_time_hr",
            "break_at_epoch",
            "loss_type",
            "rc_alpha",
            "train",
            "val",
            "final_eval_size",
            "eval_crop_mode",
            "eval_prescale",
        ]
        if not args.resume_scheduler:
            skip_keys.extend(["epochs", "warmup_steps", "warmup_ratio", "eta_min", "composite_lr"])
        if not args.resume_bs:
            skip_keys.extend(["batch_size", "grad_accum_steps", "eval_batch_size"])
        if not args.resume_img_size:
            skip_keys.extend(["train_sizes", "eval_size", "final_eval_size"])
        skip_keys.extend(["image_list_path"])
        ckpt_args = ckpt.get("args", None)
        if ckpt_args is not None:
            for k, v in vars(ckpt_args).items():
                if k not in skip_keys:
                    setattr(args, k, v)

if args.use_abs_pos_emb or args.use_rot_pos_emb:
    args.overlap = 0
    args.use_rc_loss = False
if args.eval_dataset == "nyu" and args.eval_depth_max is None:
    args.eval_depth_max = 10.0

_ensure_pos_repo()

# =============================================================================
# Depth augmentations (from depth/aug.py)
# =============================================================================
from depth.aug import TrainDepthAug, EvalDepthPreprocess, EvalDepthPreprocessNoResize

# =============================================================================
# Dataset (from depth/hypersim_simple_dataset.py)
# =============================================================================
from depth.hypersim_simple_dataset import HyperSimSimple

# =============================================================================
# Depth losses (from depth/depth_loss.py)
# =============================================================================
from depth.depth_loss import MonocularDepthHybridLoss, compute_scale_and_shift

# =============================================================================
# Depth heads (from depth/depth_head.py)
# =============================================================================
from depth.depth_head import Lite4LayerDepthHead, SimpleDepthDecoderV2, DPTHead

# =============================================================================
# Patch position losses (from core/patch_pos.py)
# =============================================================================
from core.patch_pos import PatchRowColRegressionCriterion, PatchRowColRegressionCriterionDynamic

# =============================================================================
# Model setup
# =============================================================================
MODEL_NAME = f"vit_{args.model_size}_patch16_{args.model_type}"
TRAIN_SIZE = tuple(args.train_sizes[0])
EVAL_SIZE = tuple(args.eval_size)
EPOCHS = args.epochs
SEED = args.seed

if args.final_sw_window_size is None:
    args.final_sw_window_size = EVAL_SIZE
if args.final_sw_overlap is None:
    args.final_sw_overlap = 0.25

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if len(args.train_sizes) == 1:
        torch.backends.cudnn.benchmark = True

use_amp = torch.cuda.is_available() and (not _IS_KAGGLE)
use_bf16 = use_amp and torch.cuda.is_bf16_supported(including_emulation=False)
autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
if _IS_KAGGLE:
    use_bf16 = False
    autocast_dtype = None

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def _seed_worker(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


data_rng = torch.Generator()
data_rng.manual_seed(SEED)

# =============================================================================
# Logging
# =============================================================================
subdir_name = (
    f"{args.model_size}"
    f"{'_abs_pos' if args.use_abs_pos_emb else ''}"
    f"{'_rot_pos' if args.use_rot_pos_emb else ''}"
    f"_rc_{args.use_rc_loss}_lr{int(args.lr/1e-5)}"
    f"_{args.depth_eval_mode}_{args.depth_norm}"
    f"_dec_{args.depth_decoder}"
    f"_h{TRAIN_SIZE[0]}w{TRAIN_SIZE[1]}"
    f"_s{args.seed}"
)
if args.use_rc_loss:
    subdir_name += f"_alpha_{int(args.rc_alpha)}"

run_tag = time.strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(args.output_dir)
ckpt_output_dir = os.path.join(output_dir, "ckpt")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(ckpt_output_dir, exist_ok=True)
last_ckpt_path = os.path.join(ckpt_output_dir, "last.pth")

log_file_path = os.path.join(output_dir, f"{subdir_name}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()],
)
logger = logging.getLogger()

logger.info("Arguments: %s", args)
logger.info("Using device: %s", DEVICE)
logger.info("Using mixed precision: %s", "disabled" if not use_amp else ("bfloat16" if use_bf16 else "float16"))
# logger.info("Output dir: %s", output_dir)
logger.info("Subdir: %s", subdir_name)
if args.resume_full_ckpt and args.resume_ckpt_path and ckpt is not None:
    rng_state = ckpt.get("rng_state", None)
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
                data_rng.set_state(rng_state["data_rng"])
            logger.info("Restored RNG states from checkpoint.")
        except Exception as exc:
            logger.warning("Failed to restore RNG states from checkpoint: %s", exc)

# =============================================================================
# Dataset and DataLoader
# =============================================================================
logger.info("Creating datasets...")
try:
    train_dataset = None
    train_loader = None
    if args.train:
        train_dataset = HyperSimSimple(
            roots=args.train_roots,
            split=None,
            resolution=(TRAIN_SIZE[1], TRAIN_SIZE[0]),
            pair_transform=TrainDepthAug(
                target_size=TRAIN_SIZE,
                scale_jitter=args.scale_jitter_sw if args.use_sliding_window else args.scale_jitter,
                color_jitter_prob=args.color_jitter_prob,
                normalize=True,
                depth_valid_thresh=args.train_depth_valid_thresh,
            ),
            image_list_path=args.image_list_path,
        )
    eval_root = args.eval_root
    eval_split = args.eval_split
    valid_dataset = HyperSimSimple(
        roots=[eval_root],
        split=eval_split,
        resolution=(EVAL_SIZE[1], EVAL_SIZE[0]),
        pair_transform=(
            EvalDepthPreprocessNoResize(
                ensure_multiple_of=args.patch_size,
                normalize=True,
            )
            if args.use_sliding_window
            else EvalDepthPreprocess(
                target_size=EVAL_SIZE,
                target_by="height",
                eval_crop_mode=args.eval_crop_mode,
                eval_prescale=args.eval_prescale,
                ensure_multiple_of=args.patch_size,
                normalize=True,
                depth_valid_thresh=args.eval_depth_valid_thresh,
            )
        ),
        image_list_path=args.image_list_path,
    )

    loader_kwargs = dict(
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=(args.workers > 0),
        worker_init_fn=_seed_worker,
        generator=data_rng,
    )
    if args.workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    if args.train:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    valid_kwargs = dict(
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
        persistent_workers=(args.workers > 0),
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_seed_worker,
        generator=data_rng,
    )
    if args.workers > 0:
        valid_kwargs["prefetch_factor"] = args.prefetch_factor
    def _make_valid_loader(batch_size):
        kwargs = dict(valid_kwargs)
        kwargs["batch_size"] = batch_size
        return DataLoader(valid_dataset, **kwargs)
    valid_loader = _make_valid_loader(args.eval_batch_size)

    if args.train:
        steps_per_epoch = len(train_loader)
        accum_steps = max(1, int(getattr(args, "grad_accum_steps", 1)))
        optimizer_steps_per_epoch = math.ceil(steps_per_epoch / accum_steps)
        logger.info("DataLoaders created: train=%s, val=%s", len(train_dataset), len(valid_dataset))
    else:
        logger.info("DataLoaders created: val=%s", len(valid_dataset))
except Exception as e:
    logger.error("Error creating datasets: %s", e)
    if _IS_KAGGLE and os.path.isdir("/kaggle/input"):
        logger.error("Available /kaggle/input entries: %s", os.listdir("/kaggle/input"))
    raise


# =============================================================================
# Model and optimizer
# =============================================================================
logger.info("Creating %s via timm...", MODEL_NAME)
model = timm.create_model(
    MODEL_NAME,
    pretrained=False,
    use_abs_pos_emb=args.use_abs_pos_emb,
    use_rot_pos_emb=args.use_rot_pos_emb,
    num_classes=0,
    dynamic_img_size=True,
    img_size=TRAIN_SIZE,
).to(DEVICE)

for param in model.parameters():
    param.requires_grad = True

feature_layers = [2, 5, 8, 11]

decoder_type = getattr(args, "depth_decoder", "simple")
if decoder_type == "lite4":
    decoder = Lite4LayerDepthHead(embed_dim=model.embed_dim).to(DEVICE)
elif decoder_type == "simple":
    decoder = SimpleDepthDecoderV2(embed_dim=model.embed_dim).to(DEVICE)
elif decoder_type == "dpt":
    patch_size = model.patch_embed.patch_size
    if isinstance(patch_size, tuple):
        patch_size = patch_size[0]
    decoder = DPTHead(
        in_channels=model.embed_dim,
        features=256,
        out_channels=[256, 512, 1024, 1024],
        use_bn=False,
        use_clstoken=False,
        patch_size=int(patch_size),
    ).to(DEVICE)
else:
    raise ValueError(f"Unsupported depth_decoder='{decoder_type}'. Use 'simple', 'lite4', or 'dpt'.")

if args.compile_model:
    try:
        model = torch.compile(model)
        decoder = torch.compile(decoder)
        logger.info("torch.compile enabled for model.")
    except Exception as e:
        logger.warning("torch.compile failed; continuing without it. Error: %s", e)


def _infer_grid_hw(model, inputs):
    patch_size = model.patch_embed.patch_size
    if isinstance(patch_size, tuple):
        ph, pw = patch_size
    else:
        ph = pw = patch_size
    return (inputs.shape[-2] // ph, inputs.shape[-1] // pw)

def predict_depth(model, decoder, inputs, feature_layers, grid_hw=None):
    if grid_hw is None:
        grid_hw = _infer_grid_hw(model, inputs)
    h, w = inputs.shape[-2], inputs.shape[-1]
    if args.depth_decoder == "lite4":
        features = model.forward_intermediates(
            inputs,
            indices=feature_layers,
            norm=False,
            intermediates_only=True,
            output_fmt="NLC",
        )
        pred_depths = decoder(features, grid_hw=grid_hw, out_hw=inputs.shape[-2:])
    elif args.depth_decoder == "dpt":
        patch_size = model.patch_embed.patch_size
        if isinstance(patch_size, tuple):
            patch_size = patch_size[0]
        if (h % patch_size != 0) or (w % patch_size != 0):
            raise ValueError(
                f"Input size {(h, w)} must be divisible by patch_size={patch_size} for DPT decoder."
            )
        patch_h, patch_w = h // patch_size, w // patch_size
        features = model.forward_intermediates(
            inputs,
            indices=feature_layers,
            norm=False,
            intermediates_only=True,
            output_fmt="NLC",
        )
        if use_amp:
            dpt_feats_fp32 = [f.float() for f in features]
            with torch.amp.autocast(device_type=DEVICE.type, enabled=False):
                pred_depths = decoder(dpt_feats_fp32, patch_h=patch_h, patch_w=patch_w)
        else:
            pred_depths = decoder(features, patch_h=patch_h, patch_w=patch_w)
        if pred_depths.dim() == 3:
            pred_depths = pred_depths.unsqueeze(1)
    else:
        features = model.forward_features(inputs)
        pred_depths = decoder(features, grid_hw=grid_hw, out_hw=inputs.shape[-2:])
    return pred_depths, features


def sliding_window_predict(model, decoder, inputs, feature_layers, window_size, overlap):
    if isinstance(window_size, int):
        win_h = win_w = window_size
    else:
        win_h, win_w = window_size
    stride_h = max(1, int(win_h * (1.0 - overlap)))
    stride_w = max(1, int(win_w * (1.0 - overlap)))
    b, _, h, w = inputs.shape
    out = torch.zeros((b, 1, h, w), device=inputs.device, dtype=inputs.dtype)
    weight = torch.zeros((b, 1, h, w), device=inputs.device, dtype=inputs.dtype)

    for bi in range(b):
        for top in range(0, h, stride_h):
            for left in range(0, w, stride_w):
                bottom = min(top + win_h, h)
                right = min(left + win_w, w)
                patch = inputs[bi:bi + 1, :, top:bottom, left:right]
                pad_h = win_h - (bottom - top)
                pad_w = win_w - (right - left)
                if pad_h > 0 or pad_w > 0:
                    patch = F.pad(patch, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
                pred_patch, _ = predict_depth(model, decoder, patch, feature_layers)
                pred_patch = pred_patch[..., :bottom - top, :right - left]
                out[bi:bi + 1, :, top:bottom, left:right] += pred_patch
                weight[bi:bi + 1, :, top:bottom, left:right] += 1.0

    out = out / weight.clamp_min(1e-6)
    return out


training_parameters = list(model.parameters()) + list(decoder.parameters())
param_groups = []
lr_aux = getattr(args, "lr_aux", args.lr)
if args.use_rc_loss:
    if len(args.train_sizes) == 1:
        grid_h, grid_w = model.patch_embed.grid_size
        rowcol_loss = PatchRowColRegressionCriterion(
            feat_dim=model.embed_dim,
            grid_h=grid_h,
            grid_w=grid_w,
            loss_type=args.loss_type,
        ).to(DEVICE)
    else:
        max_side = max(max(h, w) for (h, w) in args.train_sizes)
        grid_h = grid_w = max_side // args.patch_size
        rowcol_loss = PatchRowColRegressionCriterionDynamic(
            feat_dim=model.embed_dim,
            grid_h=grid_h,
            grid_w=grid_w,
            loss_type=args.loss_type,
        ).to(DEVICE)
    training_parameters += list(rowcol_loss.parameters())
    param_groups.append({"params": rowcol_loss.parameters(), "weight_decay": 0.0, "lr": lr_aux})


decay_params = []
no_decay_params = []
for n, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if n.endswith(".bias") or ("norm" in n.lower()):
        no_decay_params.append(p)
    else:
        decay_params.append(p)

for n, p in decoder.named_parameters():
    if not p.requires_grad:
        continue
    if n.endswith(".bias") or ("norm" in n.lower()):
        no_decay_params.append(p)
    else:
        decay_params.append(p)

param_groups.append({"params": decay_params, "lr": args.lr, "weight_decay": args.weight_decay})
param_groups.append({"params": no_decay_params, "lr": args.lr, "weight_decay": 0.0})

if args.depth_eval_mode not in ("relative", "metric", "scale_invariant"):
    raise ValueError(f"Unsupported depth_eval_mode='{args.depth_eval_mode}'.")

l1_w = 1.0
grad_w = 0.5
silog_w = args.silog_w
silog_on_aligned = False
criterion = MonocularDepthHybridLoss(
    l1_w=l1_w,
    grad_w=grad_w,
    silog_w=silog_w,
    silog_beta=0.15,
    scales=4,
    reduction="batch-based",
    eps=1e-8,
    silog_on_aligned=silog_on_aligned,
    align_mode=getattr(args, "align_mode", "scale_shift"),
)

optimizer = None
if args.train:
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader)
    accum_steps = max(1, int(getattr(args, "grad_accum_steps", 1)))
    optimizer_steps_per_epoch = math.ceil(steps_per_epoch / accum_steps)

    total_steps = EPOCHS * optimizer_steps_per_epoch
    if args.composite_lr:
        warmup_steps = args.warmup_steps
        if args.warmup_ratio is not None:
            warmup_steps = int(max(1, total_steps * float(args.warmup_ratio)))
        warmup_steps = min(warmup_steps, max(1, total_steps - 1))
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-7 / args.lr,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=1e-8,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=args.eta_min,
        )
else:
    scheduler = None

logger.info("Loss, Optimizer, and Scheduler are ready.")


def _compute_scale_align_pred(gt, pred, mask, mode):
    if mode == "mean":
        denom = mask.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1)
        gt_mean = (gt * mask).sum(dim=(1, 2, 3), keepdim=True) / denom
        pred_mean = (pred * mask).sum(dim=(1, 2, 3), keepdim=True) / denom
        scale = gt_mean / pred_mean.clamp_min(1e-8)
        return scale.clamp_min(1e-8)
    if mode == "median":
        out = []
        for b in range(gt.shape[0]):
            mb = mask[b, 0] > 0.5
            gt_vals = gt[b, 0][mb]
            pred_vals = pred[b, 0][mb]
            if gt_vals.numel() == 0 or pred_vals.numel() == 0:
                out.append(torch.tensor(1.0, device=gt.device, dtype=gt.dtype))
            else:
                out.append(gt_vals.median() / pred_vals.median().clamp_min(1e-8))
        return torch.stack(out, dim=0).view(gt.shape[0], 1, 1, 1).clamp_min(1e-8)
    raise ValueError(f"Unsupported depth_norm='{mode}'.")


def compute_depth_metrics(pred, target, mask=None, *, return_count: bool = False, mode: str | None = None):
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    if target.dim() == 3:
        target = target.unsqueeze(1)
    if pred.dim() != 4 or target.dim() != 4:
        raise ValueError(f"Expected (B,1,H,W) or (B,H,W); got pred={pred.shape}, target={target.shape}")

    dmin = args.eval_depth_min if args.eval_depth_min is not None else 0.0
    dmax = args.eval_depth_max if args.eval_depth_max is not None else float("inf")
    eps = 1e-8
    thresh = max(dmin, eps)
    valid_mask = torch.isfinite(target) & torch.isfinite(pred)
    valid_mask = valid_mask & (target > thresh) & (target <= dmax)
    if mask is not None:
        valid_mask = valid_mask & mask.bool()

    valid_mask_f = valid_mask.float()
    denom = valid_mask_f.sum(dim=(1, 2, 3))
    valid_img = denom > 0
    if not valid_img.any():
        return ({}, 0) if return_count else {}
    denom = denom.clamp_min(1)

    eval_mode = mode if mode is not None else args.depth_eval_mode
    if eval_mode in ("relative", "scale_invariant"):
        scale, shift = compute_scale_and_shift(pred[:, 0], target[:, 0], valid_mask_f[:, 0])
        pred_cmp = scale.view(-1, 1, 1, 1) * pred + shift.view(-1, 1, 1, 1)
        target_cmp = target
    else:
        pred_cmp = pred
        target_cmp = target

    pred_cmp = pred_cmp.clamp(min=thresh, max=dmax)
    target_cmp = target_cmp.clamp(min=thresh, max=dmax)

    diff = pred_cmp - target_cmp
    pred_c = pred_cmp
    target_c = target_cmp
    ratio = torch.maximum(pred_c / target_c, target_c / pred_c)

    def masked_mean_per_image(x):
        return (x * valid_mask_f).sum(dim=(1, 2, 3)) / denom

    abs_rel = masked_mean_per_image(torch.abs(diff) / target_c)
    l1 = masked_mean_per_image(torch.abs(diff))
    rmse = torch.sqrt(masked_mean_per_image(diff ** 2))
    a1 = masked_mean_per_image((ratio < 1.25).float())
    a2 = masked_mean_per_image((ratio < 1.25 ** 2).float())
    a3 = masked_mean_per_image((ratio < 1.25 ** 3).float())

    metrics = {
        "abs_rel": abs_rel[valid_img].mean(),
        "l1": l1[valid_img].mean(),
        "rmse": rmse[valid_img].mean(),
        "a1": a1[valid_img].mean(),
        "a2": a2[valid_img].mean(),
        "a3": a3[valid_img].mean(),
    }

    out = {k: v.item() for k, v in metrics.items()}
    return (out, int(valid_img.sum().item())) if return_count else out


def _extract_meta(metas, idx):
    if metas is None:
        return None
    if isinstance(metas, dict):
        out = {}
        for k, v in metas.items():
            if torch.is_tensor(v):
                out[k] = float(v[idx].item())
            else:
                out[k] = float(v[idx])
        return out
    return None


def _crop_to_valid_region(pred, target, meta):
    if meta is None:
        mask = torch.ones_like(target, dtype=torch.bool)
        return pred, target, mask
    rh = int(round(meta.get("resized_h", target.shape[-2])))
    rw = int(round(meta.get("resized_w", target.shape[-1])))
    rh = max(1, min(rh, target.shape[-2]))
    rw = max(1, min(rw, target.shape[-1]))
    pred = pred[..., :rh, :rw]
    target = target[..., :rh, :rw]
    mask = torch.zeros_like(target, dtype=torch.bool)
    mask[..., :rh, :rw] = True
    if args.eval_crop_mode == "nyu":
        top, bottom, left, right = 45, 471, 41, 601
        scale_h = float(meta.get("scale_h", 1.0))
        scale_w = float(meta.get("scale_w", 1.0))
        t = int(round(top * scale_h))
        b = int(round(bottom * scale_h))
        l = int(round(left * scale_w))
        r = int(round(right * scale_w))
        t = max(0, min(t, target.shape[-2] - 1))
        b = max(t + 1, min(b, target.shape[-2]))
        l = max(0, min(l, target.shape[-1] - 1))
        r = max(l + 1, min(r, target.shape[-1]))
        pred = pred[..., t:b, l:r]
        target = target[..., t:b, l:r]
        mask = mask[..., t:b, l:r]
    return pred, target, mask


# =============================================================================
# Training / validation
# =============================================================================
use_scaler = use_amp and (autocast_dtype == torch.float16)
scaler = torch.amp.GradScaler(DEVICE.type, enabled=use_scaler)
logger.info("Starting training for %s", MODEL_NAME)
train_start_time = time.time()
start_epoch = 0
global_step = 0

if args.resume_full_ckpt and args.resume_ckpt_path and ckpt is not None:
    model.load_state_dict(ckpt.get("model", {}), strict=False)
    decoder.load_state_dict(ckpt.get("decoder", {}), strict=False)
    if args.resume_optimizer:
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
    else:
        logger.info("Skipping optimizer state load (resume_optimizer=False).")
    if args.resume_scheduler:
        start_epoch = int(ckpt.get("epoch", 0))
        if scheduler is not None and "scheduler" in ckpt and ckpt["scheduler"] is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
    else:
        logger.info("Skipping scheduler state load (resume_scheduler=False).")
    global_step = int(ckpt.get("step", 0))
    if "scaler" in ckpt and ckpt["scaler"] is not None:
        scaler.load_state_dict(ckpt["scaler"])
    if args.use_rc_loss and "rowcol_loss" in ckpt and ckpt["rowcol_loss"] is not None:
        for k in ["row_targets", "col_targets", "row_index_full", "col_index_full"]:
            if k in ckpt["rowcol_loss"]:
                ckpt["rowcol_loss"].pop(k)
        rowcol_loss.load_state_dict(ckpt["rowcol_loss"])
    logger.info(
        "Resumed full checkpoint from '%s' at epoch %s, step %s",
        args.resume_ckpt_path,
        start_epoch,
        global_step,
    )
    training_history = ckpt.get("training_history", None)

if not isinstance(locals().get("training_history", None), dict):
    training_history = {
        "train_loss": [],
        "valid_abs_rel": [],
        "valid_l1": [],
        "valid_rmse": [],
        "valid_a1": [],
        "train_time": [],
        "val_time": [],
        "epoch": [],
    }
if args.use_rc_loss:
    training_history.setdefault("base_loss", [])
    training_history.setdefault("aux_loss", [])

training_history.setdefault("train_time", [])
training_history.setdefault("val_time", [])
training_history.setdefault("valid_l1", [])
training_history.setdefault("valid_abs_rel", [])
training_history.setdefault("valid_rmse", [])
training_history.setdefault("valid_a1", [])
training_history.setdefault("train_loss", [])
training_history.setdefault("epoch", [])


def _pad_history(hist, fill_value=None):
    keys = [k for k, v in hist.items() if isinstance(v, list)]
    if not keys:
        return
    max_len = max(len(hist[k]) for k in keys)
    for k in keys:
        if len(hist[k]) < max_len:
            hist[k].extend([fill_value] * (max_len - len(hist[k])))


def _history_to_frame(hist):
    list_keys = [k for k, v in hist.items() if isinstance(v, list)]
    if not list_keys:
        scalar_data = {k: v for k, v in hist.items() if not isinstance(v, list)}
        return pd.DataFrame([scalar_data]) if scalar_data else pd.DataFrame()
    max_len = max(len(hist[k]) for k in list_keys)
    data = {}
    for k, v in hist.items():
        if isinstance(v, list):
            if len(v) < max_len:
                data[k] = v + [None] * (max_len - len(v))
            else:
                data[k] = v
        else:
            data[k] = [v] * max_len
    return pd.DataFrame(data)


if args.resume_full_ckpt:
    _pad_history(training_history)


def _append_eval_log(log_path, row):
    if not row:
        return
    df = pd.DataFrame([row])
    header = not os.path.exists(log_path)
    df.to_csv(log_path, mode="a", header=header, index=False)


def _scaled_eval_batch_size(size_hw, base_size_hw, base_batch_size):
    if not size_hw or not base_size_hw:
        return max(1, int(base_batch_size))
    h, w = int(size_hw[0]), int(size_hw[1])
    bh, bw = int(base_size_hw[0]), int(base_size_hw[1])
    area = max(1, h * w)
    base_area = max(1, bh * bw)
    scale = base_area / area
    return max(1, int(round(base_batch_size * scale)))


def train_one_epoch(
    model,
    decoder,
    loader,
    criterion,
    optimizer,
    scheduler,
    scaler,
    feature_layers,
    epoch,
    total_epochs,
    global_step,
):
    model.train()
    decoder.train()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    running_loss_t = torch.zeros((), device=DEVICE)
    base_loss_t = torch.zeros((), device=DEVICE)
    aux_loss_sum_t = torch.zeros((), device=DEVICE)
    total_samples = 0
    log_interval = getattr(args, "log_interval", 50)
    accum_steps = max(1, int(getattr(args, "grad_accum_steps", 1)))
    total_batches = len(loader)
    optimizer.zero_grad(set_to_none=True)
    for i, (inputs, gt_depths, metas) in enumerate(loader):
        if (i % accum_steps) == 0 and (total_batches - i) < accum_steps:
            break
        inputs = inputs.to(DEVICE, non_blocking=True)
        gt_depths = gt_depths.to(DEVICE, non_blocking=True)
        bs = inputs.size(0)
        aux_loss = None
        do_step = ((i + 1) % accum_steps == 0) or (i + 1 == len(loader))
        opt_step = (i // accum_steps) + 1
        debug_this_step = args.debug_loss_stats and do_step and (opt_step % args.debug_loss_interval == 0)
        with torch.amp.autocast(device_type=DEVICE.type, dtype=autocast_dtype, enabled=use_amp):
            pred_depths, features = predict_depth(model, decoder, inputs, feature_layers)
            raw_pred_depths = pred_depths
            pred_depths = torch.nan_to_num(pred_depths, nan=0.0, posinf=0.0, neginf=0.0)
            valid = (gt_depths > 0)
            # if (valid.sum() == 0) or (pred_depths.sum() < 1e-8):
            #     logger.warning("Skipping batch: no valid depth pixels after sanitization.")
            #     logger.warning("pred nan/inf: nan=%s +inf=%s -inf=%s",
            #                    torch.isnan(raw_pred_depths).sum().item(),
            #                    torch.isposinf(raw_pred_depths).sum().item(),
            #                    torch.isneginf(raw_pred_depths).sum().item())
            #     sys.exit(0)
            base_loss = criterion(pred_depths, gt_depths, mask=valid.float())
            loss = base_loss

        if args.use_rc_loss:
            last_feat = features[-1] if isinstance(features, (list, tuple)) else features
            if args.depth_decoder in ("lite4", "dpt"):
                aux_loss = rowcol_loss(last_feat)
            else:
                aux_loss = rowcol_loss(last_feat[:, model.num_prefix_tokens:, :])
            t = min(1.0, (global_step + 1) / args.warmup_steps_for_aux)
            alpha_t = args.alpha_min + (args.rc_alpha - args.alpha_min) * t
            loss = base_loss + alpha_t * aux_loss
            aux_loss_sum_t += aux_loss.detach() * bs
        base_loss_t += base_loss.detach() * bs

        loss_scaled = loss / accum_steps
        # if debug_this_step:
        #     loss_val = loss.detach().float().item()
        #     if not math.isfinite(loss_val):
        #         logger.warning("[debug] loss_nonfinite=%s", loss_val)
        scaler.scale(loss_scaled).backward()
        if do_step:
            if args.clip_value is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(training_parameters, max_norm=args.clip_value)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        running_loss_t += loss.detach() * bs
        total_samples += bs

        if (i + 1) % log_interval == 0:
            avg_loss = (running_loss_t / max(total_samples, 1)).float().item()
            # mem_str = ""
            # if torch.cuda.is_available():
            #     mem_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
            #     mem_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
            #     mem_str = f" mem={mem_alloc:.0f}/{mem_reserved:.0f}MB"
            peak_str = ""
            if args.show_peak_gpu_mem and torch.cuda.is_available():
                peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
                peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
                peak_str = f" peak={peak_alloc:.0f}/{peak_reserved:.0f}MB"
            if aux_loss is not None:
                avg_aux = (aux_loss_sum_t / max(total_samples, 1)).float().item()
                logger.info(
                    "Epoch %s/%s step %s: loss=%.4f aux=%.4f%s",
                    epoch + 1,
                    total_epochs,
                    i + 1,
                    avg_loss,
                    avg_aux,
                    peak_str,
                )
            else:
                logger.info(
                    "Epoch %s/%s step %s: loss=%.4f%s",
                    epoch + 1,
                    total_epochs,
                    i + 1,
                    avg_loss,
                    peak_str,
                )
        global_step += 1

    denom = max(total_samples, 1)
    avg_loss = (running_loss_t / denom).float().item()
    avg_aux = (aux_loss_sum_t / denom).float().item()
    avg_base = (base_loss_t / denom).float().item()
    return avg_loss, avg_aux, avg_base, global_step


def validate(model, decoder, loader, criterion, feature_layers, max_steps=None, *, use_sliding_window=None, sw_window_size=None, sw_overlap=None):
    model.eval()
    decoder.eval()
    use_sw = args.use_sliding_window if use_sliding_window is None else bool(use_sliding_window)
    window_size = args.sw_window_size if sw_window_size is None else sw_window_size
    overlap = args.sw_overlap if sw_overlap is None else sw_overlap
    val_metrics = {"abs_rel": 0, "l1": 0, "rmse": 0, "a1": 0, "a2": 0, "a3": 0}
    steps = 0
    batch_count = 0

    with torch.inference_mode():
        for val_inputs, gt_depths, metas in loader:
            val_inputs = val_inputs.to(DEVICE, non_blocking=True)
            gt_depths = gt_depths.to(DEVICE, non_blocking=True)
            with torch.amp.autocast(device_type=DEVICE.type, dtype=autocast_dtype, enabled=use_amp):
                if use_sw:
                    window_size = window_size or EVAL_SIZE
                    val_pred_depths = sliding_window_predict(
                        model,
                        decoder,
                        val_inputs,
                        feature_layers,
                        window_size=window_size,
                        overlap=overlap,
                    )
                else:
                    val_pred_depths, _ = predict_depth(model, decoder, val_inputs, feature_layers)

            can_batch = (
                (not use_sw)
                and (args.eval_crop_mode is None)
                and isinstance(metas, dict)
                and ("pad_h" in metas) and ("pad_w" in metas)
            )
            if can_batch:
                pad_h = metas["pad_h"]
                pad_w = metas["pad_w"]
                if torch.is_tensor(pad_h):
                    pad_h_ok = bool((pad_h == 0).all())
                    pad_w_ok = bool((pad_w == 0).all())
                else:
                    pad_h_ok = all(v == 0 for v in pad_h)
                    pad_w_ok = all(v == 0 for v in pad_w)
                if pad_h_ok and pad_w_ok:
                    batch_metrics, count = compute_depth_metrics(
                        val_pred_depths, gt_depths, return_count=True, mode=args.depth_eval_mode
                    )
                    if batch_metrics:
                        for k in val_metrics:
                            val_metrics[k] += batch_metrics.get(k, 0) * count
                        steps += count
                else:
                    can_batch = False
            if not can_batch:
                for b in range(val_inputs.size(0)):
                    meta_b = _extract_meta(metas, b)
                    pred_b, gt_b, mask_b = _crop_to_valid_region(
                        val_pred_depths[b:b + 1], gt_depths[b:b + 1], meta_b
                    )
                    batch_metrics = compute_depth_metrics(pred_b, gt_b, mask=mask_b, mode=args.depth_eval_mode)
                    if not batch_metrics:
                        continue
                    for k in val_metrics:
                        val_metrics[k] += batch_metrics.get(k, 0)
                    steps += 1
            batch_count += 1
            if max_steps and batch_count >= max_steps:
                break

    denom = max(steps, 1)
    return 0.0, {k: v / denom for k, v in val_metrics.items()}


# =============================================================================
# Main train loop
# =============================================================================
if args.break_at_epoch is not None and start_epoch >= args.break_at_epoch:
    args.train = False
if args.train:
    logger.info("Starting training...")
    for epoch in range(start_epoch, EPOCHS):
        train_start = time.time()
        avg_train_loss, avg_aux_loss, base_loss, global_step = train_one_epoch(
            model,
            decoder,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            scaler,
            feature_layers,
            epoch,
            EPOCHS,
            global_step,
        )
        train_time = time.time() - train_start
        val_start = time.time()
        _, avg_val_metrics = validate(
            model, decoder, valid_loader, criterion, feature_layers, max_steps=args.val_steps
        )
        val_time = time.time() - val_start

        logger.info("\n--- Epoch %s Validation Summary ---", epoch + 1)
        if args.use_rc_loss:
            logger.info(
                "  Train Loss: %.4f | aux_loss: %.4f | base_loss: %.4f | train_time: %.1fs | val_time: %.1fs",
                avg_train_loss, avg_aux_loss, base_loss, train_time, val_time,
            )
        else:
            logger.info(
                "  Train Loss: %.4f | train_time: %.1fs | val_time: %.1fs",
                avg_train_loss, train_time, val_time,
            )
        logger.info(
            " Valid AbsRel: %.4f | Valid L1: %.4f | Valid RMSE: %.4f | Valid a1: %.4f\n",
            avg_val_metrics["abs_rel"], avg_val_metrics["l1"], avg_val_metrics["rmse"], avg_val_metrics["a1"],
        )

        training_history["train_loss"].append(avg_train_loss)
        if args.use_rc_loss:
            training_history["base_loss"].append(base_loss)
            training_history["aux_loss"].append(avg_aux_loss)
        training_history["valid_abs_rel"].append(avg_val_metrics["abs_rel"])
        training_history["valid_l1"].append(avg_val_metrics["l1"])
        training_history["valid_rmse"].append(avg_val_metrics["rmse"])
        training_history["valid_a1"].append(avg_val_metrics["a1"])
        training_history["train_time"].append(train_time)
        training_history["val_time"].append(val_time)
        training_history["epoch"].append(epoch + 1)

        if args.csv_interval and (epoch + 1) % args.csv_interval == 0:
            history_df = _history_to_frame(training_history)
            history_df.to_csv(os.path.join(output_dir, f"{subdir_name}.csv"), index=False)

        if args.save_full_ckpt:
            ckpt = {
                "epoch": epoch + 1,
                "step": int(global_step),
                "model": model.state_dict(),
                "decoder": decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "scaler": scaler.state_dict() if scaler is not None else None,
                "rowcol_loss": rowcol_loss.state_dict() if args.use_rc_loss else None,
                "training_history": training_history,
                "rng_state": {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    "data_rng": data_rng.get_state(),
                },
                "args": args,
            }
            torch.save(ckpt, last_ckpt_path)
            logger.info("Saved full checkpoint to '%s'", last_ckpt_path)

        if args.total_run_time_hr is not None:
            elapsed = time.time() - train_start_time
            max_run_time_sec = args.total_run_time_hr * 3600
            if elapsed  + (train_time + val_time) + 900>= max_run_time_sec:
                logger.info("Stopping training: elapsed %.0fs reached limit %.2fh.", elapsed, args.total_run_time_hr)
                break
        if args.break_at_epoch is not None and (epoch + 1) >= args.break_at_epoch:
            logger.info("Stopping training: reached break_at_epoch=%s.", args.break_at_epoch)
            break

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("Training complete.")
else:
    logger.info("Skipping training (args.train=False).")
    if not (args.resume_full_ckpt and args.resume_ckpt_path):
        logger.warning("No checkpoint specified; evaluation will use randomly initialized weights.")

if args.val:
    if not args.train and args.resume_full_ckpt and args.resume_ckpt_path and ckpt is not None:
        model.load_state_dict(ckpt.get("model", {}), strict=False)
        decoder.load_state_dict(ckpt.get("decoder", {}), strict=False)
        logger.info("Loaded checkpoint for evaluation from %s.", args.resume_ckpt_path)
    elif not args.train and args.resume_full_ckpt:
        logger.warning("Evaluation requested but no checkpoint loaded; results may be random.")

    base_size = tuple(args.eval_size)
    final_sizes = args.final_eval_size
    if not isinstance(final_sizes, (list, tuple)):
        final_sizes = [final_sizes]
    final_eval_log = os.path.join(output_dir, f"{subdir_name}_final_eval.csv")
    for size in final_sizes:
        size_hw = tuple(size)
        valid_dataset.resolution = (size_hw[1], size_hw[0])
        valid_dataset._setup_resolution()
        if args.use_sliding_window:
            valid_dataset.pair_transform = EvalDepthPreprocessNoResize(
                ensure_multiple_of=args.patch_size,
                normalize=True,
            )
        else:
            valid_dataset.pair_transform = EvalDepthPreprocess(
                target_size=size_hw,
                target_by="height",
                eval_crop_mode=args.eval_crop_mode,
                eval_prescale=args.eval_prescale,
                ensure_multiple_of=args.patch_size,
                normalize=True,
                depth_valid_thresh=args.eval_depth_valid_thresh,
            )
        eval_bs = _scaled_eval_batch_size(size_hw, base_size, args.eval_batch_size)
        eval_loader = _make_valid_loader(eval_bs)
        sw_window_size = args.final_sw_window_size or size_hw
        val_start = time.time()
        _, avg_val_metrics = validate(
            model,
            decoder,
            eval_loader,
            criterion,
            feature_layers,
            max_steps=args.val_steps,
        )
        val_time = time.time() - val_start
        # logger.info(
        #     "Final Eval @%s: AbsRel %.4f | L1 %.4f | RMSE %.4f | a1 %.4f | time %.1fs",
        #     size_hw,
        #     avg_val_metrics["abs_rel"],
        #     avg_val_metrics["l1"],
        #     avg_val_metrics["rmse"],
        #     avg_val_metrics["a1"],
        #     val_time,
        # )
        print(
            size_hw,
            avg_val_metrics["abs_rel"],
            avg_val_metrics["a1"]
        )
        _append_eval_log(
            final_eval_log,
            {
                "run_tag": run_tag,
                "subdir_name": subdir_name,
                "output_dir": output_dir,
                "eval_size": str(size_hw),
                "eval_batch_size": int(eval_bs),
                "valid_abs_rel": float(avg_val_metrics["abs_rel"]),
                "valid_l1": float(avg_val_metrics["l1"]),
                "valid_rmse": float(avg_val_metrics["rmse"]),
                "valid_a1": float(avg_val_metrics["a1"]),
                "val_time": float(val_time),
            },
        )

history_df = _history_to_frame(training_history)
if args.train:
    history_df.to_csv(os.path.join(output_dir, f"{subdir_name}.csv"), index=False)

if (not history_df.empty) and history_df["valid_a1"].notna().any():
    best_a1 = history_df["valid_a1"].max()
    best_epoch = history_df.loc[history_df["valid_a1"].idxmax(), "epoch"]
    logger.info("Best a1: %.4f at epoch %s", best_a1, best_epoch)

if (not history_df.empty) and history_df["valid_abs_rel"].notna().any():
    best_a1_row = history_df.loc[history_df["valid_a1"].idxmax()]
    best_a1_epoch = int(best_a1_row["epoch"])
    best_a1_val = best_a1_row["valid_a1"]

    best_abs_rel_row = history_df.loc[history_df["valid_abs_rel"].idxmin()]
    best_abs_rel_epoch = int(best_abs_rel_row["epoch"])
    best_abs_rel_val = best_abs_rel_row["valid_abs_rel"]

    best_rmse_row = history_df.loc[history_df["valid_rmse"].idxmin()]
    best_rmse_epoch = int(best_rmse_row["epoch"])
    best_rmse_val = best_rmse_row["valid_rmse"]

    logger.info("\n--- Best Validation Metrics from History ---")
    logger.info("  Best a1:      %.4f (Epoch %s)", best_a1_val, best_a1_epoch)
    logger.info("  Best AbsRel:  %.4f (Epoch %s)", best_abs_rel_val, best_abs_rel_epoch)
    logger.info("  Best RMSE:    %.4f (Epoch %s)", best_rmse_val, best_rmse_epoch)
    logger.info("------------------------------------------")

logger.info("Output dir: %s", output_dir)
logger.info("Subdir: %s", subdir_name)

del model, decoder
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
