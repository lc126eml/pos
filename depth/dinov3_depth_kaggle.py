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
    eval_split="val",  # "val" or "test" when eval_root has subdirs
    model_type="dinov3",
    use_abs_pos_emb=False,
    use_rot_pos_emb=True,
    model_size='base',
    train_sizes=[(224, 224)],
    eval_size=(224, 224),
    final_eval_size=(224, 224),
    color_jitter_prob=0.5,
    scale_jitter=(1.0, 1.2),
    scale_jitter_sw=(1.0, 1.01),
    batch_size=24,
    grad_accum_steps=1,
    patch_size=16,
    lr=7e-5, #7e-5
    lr_aux=1e-5,
    eta_min=1e-7,
    epochs=120,
    break_at_epoch=None,
    has_pos=False,
    weight_decay=0.05,
    overlap=0,
    seed=53,
    val_steps=None,
    use_rc_loss=False,
    loss_type="smooth_l1",
    rc_alpha=200.0,
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
    resume_ckpt_path='/kaggle/input/depth-base-rope-d-653/ckpt/last.pth',
    resume_args=True,
    resume_scheduler=True,
    resume_optimizer=True,
    resume_bs=True,
    resume_img_size=False,
    total_run_time_hr=12.0,
    train=True,
    val=False,
    final_use_sliding_window=True,
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
            "eval_crop_mode",
            "eval_prescale",
        ]
        if not args.resume_scheduler:
            skip_keys.extend(["epochs", "warmup_steps", "warmup_ratio", "eta_min", "composite_lr"])
        if not args.resume_bs:
            skip_keys.extend(["batch_size", "grad_accum_steps"])
        if not args.resume_img_size:
            skip_keys.extend(["train_sizes", "eval_size", "final_eval_size"])
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

# =============================================================================
# Depth augmentations (inline from depth/aug.py)
# =============================================================================
ImageLike = Union[Image.Image, np.ndarray, torch.Tensor]
DepthLike = Union[np.ndarray, torch.Tensor]


def _to_pil_rgb(image: ImageLike) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, torch.Tensor):
        x = image.detach().cpu()
        if x.ndim == 3 and x.shape[0] in (1, 3):
            x = x.permute(1, 2, 0)
        image = x.numpy()
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0.0, 1.0) if arr.max() <= 1.5 else np.clip(arr / 255.0, 0.0, 1.0)
            arr = (arr * 255.0).round().astype(np.uint8)
        return Image.fromarray(arr)
    raise TypeError(f"Unsupported image type: {type(image)}")


def _to_depth_1chw(depth: DepthLike) -> torch.Tensor:
    if isinstance(depth, torch.Tensor):
        d = depth.detach().float().cpu()
        if d.ndim == 2:
            d = d.unsqueeze(0)
        elif d.ndim == 3 and d.shape[-1] == 1:
            d = d.permute(2, 0, 1)
        if d.ndim != 3 or d.shape[0] != 1:
            raise ValueError(f"Depth must be [H,W] or [H,W,1] or [1,H,W]; got {tuple(d.shape)}")
        return d
    if isinstance(depth, np.ndarray):
        d = torch.from_numpy(depth).float()
        if d.ndim == 2:
            d = d.unsqueeze(0)
        elif d.ndim == 3 and d.shape[-1] == 1:
            d = d.permute(2, 0, 1)
        if d.ndim != 3 or d.shape[0] != 1:
            raise ValueError(f"Depth must be [H,W] or [H,W,1]; got {depth.shape}")
        return d
    raise TypeError(f"Unsupported depth type: {type(depth)}")


def _pil_to_tensor01(pil_img: Image.Image) -> torch.Tensor:
    x = torch.from_numpy(np.array(pil_img)).float() / 255.0
    return x.permute(2, 0, 1).contiguous()


def _normalize_img(img_t: torch.Tensor, mean, std) -> torch.Tensor:
    mean_v = mean if isinstance(mean, (list, tuple)) else [float(mean)]
    std_v = std if isinstance(std, (list, tuple)) else [float(std)]
    if len(mean_v) == 1:
        mean_v = mean_v * 3
    if len(std_v) == 1:
        std_v = std_v * 3
    mean_t = torch.tensor(mean_v, dtype=img_t.dtype, device=img_t.device).view(3, 1, 1)
    std_t = torch.tensor(std_v, dtype=img_t.dtype, device=img_t.device).view(3, 1, 1)
    return (img_t - mean_t) / std_t


def _resize_depth_with_mask(
    depth_1chw: torch.Tensor,
    size_hw: Tuple[int, int],
    *,
    valid_thresh: float = 0.1,
    eps: float = 1e-6,
) -> torch.Tensor:
    d = depth_1chw.unsqueeze(0).float()
    valid = torch.isfinite(d) & (d > 0)
    m = valid.float()
    dm = d * m

    H, W = d.shape[-2:]
    Ht, Wt = size_hw
    is_down = (Ht <= H) and (Wt <= W)

    if is_down:
        dm_rs = F.interpolate(dm, size=size_hw, mode="area")
        m_rs = F.interpolate(m, size=size_hw, mode="area")
    else:
        dm_rs = F.interpolate(dm, size=size_hw, mode="bilinear", align_corners=False)
        m_rs = F.interpolate(m, size=size_hw, mode="bilinear", align_corners=False)

    d_rs = dm_rs / (m_rs + eps)
    valid_rs = m_rs > valid_thresh
    d_rs = torch.where(valid_rs, d_rs, torch.zeros_like(d_rs))

    return d_rs.squeeze(0)


def _round_to_multiple(x: int, m: int) -> int:
    return int(round(x / m) * m)


def train_aug_depth_ar_resize_random_crop(
    image: ImageLike,
    depth: DepthLike,
    target_size: Tuple[int, int],
    *,
    hflip_prob: float = 0.5,
    scale_jitter: Optional[Tuple[float, Optional[float]]] = (1.0, None),
    color_jitter: Optional[Dict[str, float]] = None,
    color_jitter_prob: float = 0.8,
    gamma_jitter: Optional[Tuple[float, float]] = (0.9, 1.1),
    gamma_jitter_prob: float = 0.0,
    grayscale_prob: float = 0.05,
    blur_prob: float = 0.0,
    blur_kernel: Tuple[int, int] = (5, 5),
    blur_sigma: Tuple[float, float] = (0.1, 1.0),
    noise_std: Optional[Tuple[float, float]] = (0.0, 0.0),
    normalize: bool = True,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ensure_multiple_of: Optional[int] = None,
    depth_valid_thresh: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    Ht, Wt = target_size
    pil = _to_pil_rgb(image)
    d = _to_depth_1chw(depth)

    W0, H0 = pil.size

    base = max(Ht / H0, Wt / W0)
    if scale_jitter is None:
        jitter = 1.0
    else:
        j0, j1 = scale_jitter
        j0 = max(1.0, float(j0))
        if j1 is None:
            j1 = (1.0 / base) if base < 1.0 else j0
        j1 = max(j0, float(j1))
        jitter = random.uniform(j0, j1)
    scale = base * jitter

    newH = int(round(H0 * scale))
    newW = int(round(W0 * scale))

    if newH < Ht or newW < Wt:
        scale = base
        newH = int(round(H0 * scale))
        newW = int(round(W0 * scale))

    if ensure_multiple_of is not None and ensure_multiple_of > 1:
        newH = max(Ht, _round_to_multiple(newH, ensure_multiple_of))
        newW = max(Wt, _round_to_multiple(newW, ensure_multiple_of))

    resample = Image.BOX if scale < 1.0 else Image.BICUBIC
    pil = pil.resize((newW, newH), resample=resample)
    d = _resize_depth_with_mask(d, (newH, newW), valid_thresh=depth_valid_thresh)

    top = 0 if newH == Ht else random.randint(0, newH - Ht)
    left = 0 if newW == Wt else random.randint(0, newW - Wt)

    pil = TF.crop(pil, top=top, left=left, height=Ht, width=Wt)
    d = d[:, top:top + Ht, left:left + Wt]

    if random.random() < hflip_prob:
        pil = TF.hflip(pil)
        d = torch.flip(d, dims=[2])

    if (
        color_jitter is not None
        and any(v > 0 for v in color_jitter.values())
        and (color_jitter_prob > 0)
        and (random.random() < color_jitter_prob)
    ):
        pil = ColorJitter(**color_jitter)(pil)

    if gamma_jitter is not None and gamma_jitter_prob > 0 and random.random() < gamma_jitter_prob:
        g0, g1 = gamma_jitter
        if g0 != 1.0 or g1 != 1.0:
            gamma = random.uniform(g0, g1)
            pil = TF.adjust_gamma(pil, gamma=gamma)

    if grayscale_prob > 0 and random.random() < grayscale_prob:
        pil = TF.to_grayscale(pil, num_output_channels=3)

    if blur_prob > 0 and random.random() < blur_prob:
        pil = GaussianBlur(kernel_size=blur_kernel, sigma=blur_sigma)(pil)

    img_t = _pil_to_tensor01(pil)

    if noise_std is not None:
        n0, n1 = noise_std
        if n1 > 0:
            std = random.uniform(n0, n1)
            if std > 0:
                img_t = (img_t + torch.randn_like(img_t) * std).clamp(0.0, 1.0)

    if normalize:
        img_t = _normalize_img(img_t, mean, std)

    return img_t, d.contiguous().float()


def eval_preprocess_depth_keep_ar(
    image: ImageLike,
    depth: DepthLike,
    target_size: Tuple[int, int],
    *,
    target_by: str = "height",
    eval_crop_mode: str = "pad",
    eval_prescale: float = 1.0,
    ensure_multiple_of: Optional[int] = 32,
    normalize: bool = True,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    depth_valid_thresh: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    pil = _to_pil_rgb(image)
    d = _to_depth_1chw(depth)

    W0, H0 = pil.size
    Ht, Wt = target_size

    prescale = max(1e-6, float(eval_prescale))
    if target_by == "height":
        newH = int(round(Ht * prescale))
        scale = newH / H0
        newW = int(round(W0 * scale))
    elif target_by == "long_side":
        long_target = int(round(max(Ht, Wt) * prescale))
        long0 = max(H0, W0)
        scale = long_target / long0
        newH = int(round(H0 * scale))
        newW = int(round(W0 * scale))
    else:
        raise ValueError(f"target_by must be 'height' or 'long_side', got {target_by}")

    resample = Image.BOX if scale < 1.0 else Image.BICUBIC
    pil_rs = pil.resize((newW, newH), resample=resample)
    d_rs = _resize_depth_with_mask(d, (newH, newW), valid_thresh=depth_valid_thresh)
    resize_h = newH
    resize_w = newW

    crop_top = crop_left = 0
    pad_left = pad_top = pad_right = pad_bottom = 0
    if eval_crop_mode not in ("pad", "crop", "crop_or_pad"):
        raise ValueError(f"eval_crop_mode must be 'pad', 'crop', or 'crop_or_pad', got {eval_crop_mode}")
    if eval_crop_mode == "crop" or (eval_crop_mode == "crop_or_pad" and newH >= Ht and newW >= Wt):
        top = max(0, (newH - Ht) // 2)
        left = max(0, (newW - Wt) // 2)
        pil_rs = TF.crop(pil_rs, top=top, left=left, height=Ht, width=Wt)
        d_rs = d_rs[:, top:top + Ht, left:left + Wt]
        crop_top = top
        crop_left = left
        newH, newW = Ht, Wt
    elif eval_crop_mode in ("pad", "crop_or_pad"):
        if newH < Ht or newW < Wt:
            pad_h = max(0, Ht - newH)
            pad_w = max(0, Wt - newW)
            pad_top = pad_h // 2
            pad_left = pad_w // 2
            pad_bottom = pad_h - pad_top
            pad_right = pad_w - pad_left
            mean_v = mean if isinstance(mean, (list, tuple)) else [float(mean)]
            if len(mean_v) == 1:
                mean_v = mean_v * 3
            pad_fill = tuple(int(round(m * 255.0)) for m in mean_v[:3])
            pil_rs = TF.pad(pil_rs, padding=(pad_left, pad_top, pad_right, pad_bottom), fill=pad_fill)
            d_rs = F.pad(d_rs, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0)
            newH = newH + pad_top + pad_bottom
            newW = newW + pad_left + pad_right

    pad_h = 0
    pad_w = 0
    if ensure_multiple_of is not None and ensure_multiple_of > 1:
        pad_h = (ensure_multiple_of - (newH % ensure_multiple_of)) % ensure_multiple_of
        pad_w = (ensure_multiple_of - (newW % ensure_multiple_of)) % ensure_multiple_of
        if pad_h or pad_w:
            mean_v = mean if isinstance(mean, (list, tuple)) else [float(mean)]
            if len(mean_v) == 1:
                mean_v = mean_v * 3
            pad_fill = tuple(int(round(m * 255.0)) for m in mean_v[:3])
            pil_rs = TF.pad(pil_rs, padding=(0, 0, pad_w, pad_h), fill=pad_fill)
            d_rs = F.pad(d_rs, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

    meta = {
        "orig_h": float(H0), "orig_w": float(W0),
        "resized_h": float(newH), "resized_w": float(newW),
        "resize_h": float(resize_h), "resize_w": float(resize_w),
        "scale_h": float(resize_h) / float(H0),
        "scale_w": float(resize_w) / float(W0),
        "pad_h": float(pad_h), "pad_w": float(pad_w),
        "padded_h": float(newH + pad_h), "padded_w": float(newW + pad_w),
        "crop_top": float(crop_top), "crop_left": float(crop_left),
        "pad_left": float(pad_left), "pad_top": float(pad_top),
        "pad_right": float(pad_right), "pad_bottom": float(pad_bottom),
    }
    img_t = _pil_to_tensor01(pil_rs)
    if normalize:
        img_t = _normalize_img(img_t, mean, std)
    return img_t, d_rs.contiguous().float(), meta


def eval_preprocess_depth_no_resize(
    image: ImageLike,
    depth: DepthLike,
    *,
    ensure_multiple_of: Optional[int] = None,
    normalize: bool = True,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    pil = _to_pil_rgb(image)
    d = _to_depth_1chw(depth)

    W0, H0 = pil.size
    img_t = _pil_to_tensor01(pil)

    pad_h = 0
    pad_w = 0
    if ensure_multiple_of is not None and ensure_multiple_of > 1:
        pad_h = (ensure_multiple_of - (H0 % ensure_multiple_of)) % ensure_multiple_of
        pad_w = (ensure_multiple_of - (W0 % ensure_multiple_of)) % ensure_multiple_of
        if pad_h or pad_w:
            img_t = TF.pad(img_t, padding=(0, 0, pad_w, pad_h), fill=0)
            d = F.pad(d, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

    if normalize:
        img_t = _normalize_img(img_t, mean, std)

    meta = {
        "orig_h": float(H0), "orig_w": float(W0),
        "resized_h": float(H0), "resized_w": float(W0),
        "scale_h": 1.0, "scale_w": 1.0,
        "pad_h": float(pad_h), "pad_w": float(pad_w),
        "padded_h": float(H0 + pad_h), "padded_w": float(W0 + pad_w),
    }
    return img_t, d.contiguous().float(), meta


class TrainDepthAug:
    def __init__(
        self,
        target_size: Tuple[int, int],
        *,
        hflip_prob: float = 0.5,
        scale_jitter: Optional[Tuple[float, Optional[float]]] = (1.0, None),
        color_jitter: Optional[Dict[str, float]] = None,
        color_jitter_prob: float = 0.5,
        gamma_jitter: Optional[Tuple[float, float]] = (0.9, 1.1),
        gamma_jitter_prob: float = 0.0,
        grayscale_prob: float = 0.05,
        blur_prob: float = 0.0,
        blur_kernel: Tuple[int, int] = (5, 5),
        blur_sigma: Tuple[float, float] = (0.1, 1.0),
        noise_std: Optional[Tuple[float, float]] = (0.0, 0.0),
        normalize: bool = True,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        ensure_multiple_of: Optional[int] = None,
        depth_valid_thresh: float = 0.1,
    ) -> None:
        self.target_size = target_size
        self.hflip_prob = hflip_prob
        self.scale_jitter = scale_jitter
        if color_jitter is None:
            color_jitter = dict(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)
        self.color_jitter = dict(color_jitter)
        self.color_jitter_prob = color_jitter_prob
        self.gamma_jitter = gamma_jitter
        self.gamma_jitter_prob = gamma_jitter_prob
        self.grayscale_prob = grayscale_prob
        self.blur_prob = blur_prob
        self.blur_kernel = blur_kernel
        self.blur_sigma = blur_sigma
        self.noise_std = noise_std
        self.normalize = normalize
        mean_v = mean if isinstance(mean, (list, tuple)) else [float(mean)]
        std_v = std if isinstance(std, (list, tuple)) else [float(std)]
        if len(mean_v) == 1:
            mean_v = mean_v * 3
        if len(std_v) == 1:
            std_v = std_v * 3
        self._mean_t = torch.tensor(mean_v).view(3, 1, 1)
        self._std_t = torch.tensor(std_v).view(3, 1, 1)
        self.ensure_multiple_of = ensure_multiple_of
        self.depth_valid_thresh = depth_valid_thresh

    def __call__(self, image: ImageLike, depth: DepthLike) -> Tuple[torch.Tensor, torch.Tensor]:
        img_t, depth_t = train_aug_depth_ar_resize_random_crop(
            image,
            depth,
            self.target_size,
            hflip_prob=self.hflip_prob,
            scale_jitter=self.scale_jitter,
            color_jitter=self.color_jitter,
            color_jitter_prob=self.color_jitter_prob,
            gamma_jitter=self.gamma_jitter,
            gamma_jitter_prob=self.gamma_jitter_prob,
            grayscale_prob=self.grayscale_prob,
            blur_prob=self.blur_prob,
            blur_kernel=self.blur_kernel,
            blur_sigma=self.blur_sigma,
            noise_std=self.noise_std,
            normalize=False,
            ensure_multiple_of=self.ensure_multiple_of,
            depth_valid_thresh=self.depth_valid_thresh,
        )
        if self.normalize:
            mean_t = self._mean_t.to(dtype=img_t.dtype)
            std_t = self._std_t.to(dtype=img_t.dtype)
            img_t = (img_t - mean_t) / std_t
        return img_t, depth_t


class EvalDepthPreprocess:
    def __init__(
        self,
        target_size: Tuple[int, int],
        *,
        target_by: str = "height",
        eval_crop_mode: str = "pad",
        eval_prescale: float = 1.0,
        ensure_multiple_of: Optional[int] = 32,
        normalize: bool = True,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        depth_valid_thresh: float = 0.0,
    ) -> None:
        self.target_size = target_size
        self.target_by = target_by
        self.eval_crop_mode = eval_crop_mode
        self.eval_prescale = eval_prescale
        self.ensure_multiple_of = ensure_multiple_of
        self.normalize = normalize
        mean_v = mean if isinstance(mean, (list, tuple)) else [float(mean)]
        std_v = std if isinstance(std, (list, tuple)) else [float(std)]
        if len(mean_v) == 1:
            mean_v = mean_v * 3
        if len(std_v) == 1:
            std_v = std_v * 3
        self._mean_t = torch.tensor(mean_v).view(3, 1, 1)
        self._std_t = torch.tensor(std_v).view(3, 1, 1)
        self.depth_valid_thresh = depth_valid_thresh

    def __call__(self, image: ImageLike, depth: DepthLike) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        img_t, depth_t, meta = eval_preprocess_depth_keep_ar(
            image,
            depth,
            self.target_size,
            target_by=self.target_by,
            eval_crop_mode=self.eval_crop_mode,
            eval_prescale=self.eval_prescale,
            ensure_multiple_of=self.ensure_multiple_of,
            normalize=False,
            depth_valid_thresh=self.depth_valid_thresh,
        )
        if self.normalize:
            mean_t = self._mean_t.to(dtype=img_t.dtype)
            std_t = self._std_t.to(dtype=img_t.dtype)
            img_t = (img_t - mean_t) / std_t
        return img_t, depth_t, meta


class EvalDepthPreprocessNoResize:
    def __init__(
        self,
        *,
        ensure_multiple_of: Optional[int] = None,
        normalize: bool = True,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.ensure_multiple_of = ensure_multiple_of
        self.normalize = normalize
        mean_v = mean if isinstance(mean, (list, tuple)) else [float(mean)]
        std_v = std if isinstance(std, (list, tuple)) else [float(std)]
        if len(mean_v) == 1:
            mean_v = mean_v * 3
        if len(std_v) == 1:
            std_v = std_v * 3
        self._mean_t = torch.tensor(mean_v).view(3, 1, 1)
        self._std_t = torch.tensor(std_v).view(3, 1, 1)

    def __call__(self, image: ImageLike, depth: DepthLike) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        img_t, depth_t, meta = eval_preprocess_depth_no_resize(
            image,
            depth,
            ensure_multiple_of=self.ensure_multiple_of,
            normalize=False,
        )
        if self.normalize:
            mean_t = self._mean_t.to(dtype=img_t.dtype)
            std_t = self._std_t.to(dtype=img_t.dtype)
            img_t = (img_t - mean_t) / std_t
        return img_t, depth_t, meta


# =============================================================================
# Dataset (multi-root HyperSim)
# =============================================================================
class HyperSimSimple(Dataset):
    def __init__(
        self,
        roots: Union[str, Sequence[str]],
        resolution: Tuple[int, int],
        split: Optional[str] = None,
        sample_rate: float = 1.0,
        pair_transform=None,
        **kwargs,
    ):
        super().__init__()
        if isinstance(roots, (list, tuple)):
            root_list = list(roots)
        else:
            root_list = [roots]

        if split is not None:
            root_list = [os.path.join(r, split) if os.path.isdir(os.path.join(r, split)) else r for r in root_list]

        self.roots = root_list
        self.resolution = resolution
        self._setup_resolution()
        self.dataset_label = "HyperSimSimple"
        self.is_train = (split == "train")

        if pair_transform is None:
            target_size = (self.resolution[1], self.resolution[0]) if isinstance(self.resolution, (list, tuple)) else (self.resolution, self.resolution)
            if self.is_train:
                self.pair_transform = TrainDepthAug(
                    target_size=target_size,
                    normalize=True,
                    depth_valid_thresh=0.1,
                )
            else:
                self.pair_transform = EvalDepthPreprocess(
                    target_size=target_size,
                    target_by="height",
                    ensure_multiple_of=32,
                    normalize=True,
                    depth_valid_thresh=0.0,
                )
        else:
            self.pair_transform = pair_transform

        self.image_paths: List[str] = []
        for root in self.roots:
            if not os.path.isdir(root):
                continue
            self.image_paths.extend(glob.glob(os.path.join(root, "**", "*_rgb.png"), recursive=True))

        if not self.image_paths:
            raise FileNotFoundError(f"No '*_rgb.png' files found in roots: {self.roots}")

        if sample_rate < 1.0:
            num_samples = int(len(self.image_paths) * sample_rate)
            self.image_paths = random.sample(self.image_paths, num_samples)

        self.image_paths.sort()

    def _setup_resolution(self):
        if isinstance(self.resolution, int):
            self.resolution = (self.resolution, self.resolution)
        elif isinstance(self.resolution, (list, tuple)):
            assert len(self.resolution) == 2, "Resolution must be an int or a (width, height) tuple."

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if idx >= len(self.image_paths):
            raise IndexError("Index out of range")

        impath = self.image_paths[idx]
        depthpath = impath.replace("_rgb.png", "_depth.npy")

        rgb_bgr = cv2.imread(impath, cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            raise IOError(f"Could not load image={impath} with cv2")
        rgb_image = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        depthmap = np.load(depthpath)
        depthmap[~np.isfinite(depthmap)] = 0.0
        depthmap = depthmap.astype(np.float32)

        out = self.pair_transform(rgb_image, depthmap)
        if isinstance(out, tuple) and len(out) == 3:
            img_t, depth_t, meta = out
        else:
            img_t, depth_t = out
            h, w = depth_t.shape[-2], depth_t.shape[-1]
            meta = {
                "orig_h": float(h),
                "orig_w": float(w),
                "resized_h": float(h),
                "resized_w": float(w),
                "scale_h": 1.0,
                "scale_w": 1.0,
                "pad_h": 0.0,
                "pad_w": 0.0,
                "padded_h": float(h),
                "padded_w": float(w),
            }
        return img_t, depth_t, meta


# =============================================================================
# Depth losses (inline from depth/depth_loss.py)
# =============================================================================

def _ensure_4d(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        return x.unsqueeze(1)
    if x.dim() == 4:
        return x
    raise ValueError(f"Expected (B,H,W) or (B,1,H,W); got {tuple(x.shape)}")


def _default_mask(gt: torch.Tensor, eps: float) -> torch.Tensor:
    return (torch.isfinite(gt) & (gt > eps)).float()


def compute_scale_and_shift(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det > 0

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    x_0[~valid] = 1.0
    x_1[~valid] = 0.0

    return x_0, x_1


def _reduction_batch(image_loss: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    denom = torch.sum(M)
    if denom == 0:
        return image_loss.sum() * 0.0
    return torch.sum(image_loss) / denom


def _gradient_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                   reduction=_reduction_batch) -> torch.Tensor:
    M = torch.sum(mask, (1, 2))
    diff = (prediction - target) * mask

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = mask[:, :, 1:] * mask[:, :, :-1]
    grad_x = grad_x * mask_x

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = mask[:, 1:, :] * mask[:, :-1, :]
    grad_y = grad_y * mask_y

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))
    return reduction(image_loss, M)


def _l1_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
             reduction=_reduction_batch) -> torch.Tensor:
    M = torch.sum(mask, (1, 2))
    res = torch.abs(prediction - target)
    image_loss = torch.sum(mask * res, (1, 2))
    return reduction(image_loss, M)


class SILogLoss(nn.Module):
    def __init__(self, beta: float = 0.15, correction: int = 1, per_image: bool = True):
        super().__init__()
        self.beta = float(beta)
        self.correction = int(correction)
        self.per_image = bool(per_image)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        pred = _ensure_4d(pred).float()
        target = _ensure_4d(target).float()
        mask = _ensure_4d(mask).float()

        with torch.amp.autocast(device_type=pred.device.type, enabled=False):
            valid = (mask > 0.5) & torch.isfinite(pred) & torch.isfinite(target)
            mask = valid.float()
            pred = torch.where(valid, pred, torch.ones_like(pred))
            target = torch.where(valid, target, torch.ones_like(target))
            alpha = 1e-7
            pred = torch.clamp(pred, min=alpha)
            target = torch.clamp(target, min=alpha)

            g = torch.log(pred) - torch.log(target)
            if self.per_image:
                B = g.shape[0]
                g_flat = g.reshape(B, -1)
                m_flat = mask.reshape(B, -1)
                mask_sum = m_flat.sum(dim=1)
                valid_img = mask_sum > 0

                mask_sum_safe = mask_sum.clamp_min(1.0)
                mean = (g_flat * m_flat).sum(dim=1) / mask_sum_safe
                denom_var = (mask_sum_safe - float(self.correction)).clamp_min(1.0)
                var = ((g_flat - mean[:, None]) ** 2 * m_flat).sum(dim=1) / denom_var
                Dg = var + self.beta * mean.pow(2)
                loss_per_img = 10.0 * torch.sqrt(Dg)
                if valid_img.any():
                    return loss_per_img[valid_img].mean()
                return pred.sum() * 0.0

            denom = mask.sum().clamp_min(1.0)
            mean = (g * mask).sum() / denom
            denom_var = (denom - float(self.correction)).clamp_min(1.0)
            var = ((g - mean) ** 2 * mask).sum() / denom_var
            Dg = var + self.beta * mean.pow(2)
            return 10.0 * torch.sqrt(Dg)


class MonocularDepthHybridLoss(nn.Module):
    def __init__(
        self,
        l1_w: float = 1.0,
        grad_w: float = 0.5,
        silog_w: float = 0.0,
        silog_beta: float = 0.15,
        scales: int = 4,
        reduction: str = "batch-based",
        eps: float = 1e-8,
        silog_on_aligned: bool = False,
    ):
        super().__init__()
        self.l1_w = float(l1_w)
        self.grad_w = float(grad_w)
        self.silog_w = float(silog_w)
        self.scales = int(scales)
        self.eps = float(eps)
        self.silog_on_aligned = bool(silog_on_aligned)

        if reduction == "batch-based":
            self._reduction = _reduction_batch
        else:
            raise ValueError("Only 'batch-based' reduction is supported.")

        self._silog = SILogLoss(beta=silog_beta)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None,
                interpolate: bool = True):
        prediction = _ensure_4d(prediction)
        target = _ensure_4d(target)

        if interpolate and prediction.shape[-2:] != target.shape[-2:]:
            prediction = F.interpolate(prediction, target.shape[-2:], mode="bilinear", align_corners=True)

        if mask is None:
            mask = _default_mask(target, self.eps)
        else:
            mask = _ensure_4d(mask).float()

        pred_hw = prediction[:, 0]
        tgt_hw = target[:, 0]
        m_hw = mask[:, 0]

        scale, shift = compute_scale_and_shift(pred_hw, tgt_hw, m_hw)
        pred_aligned = scale.view(-1, 1, 1, 1) * prediction + shift.view(-1, 1, 1, 1)

        total = pred_aligned.new_zeros(())
        if self.l1_w > 0:
            total = total + self.l1_w * _l1_loss(pred_aligned[:, 0], target[:, 0], m_hw, self._reduction)

        if self.grad_w > 0:
            grad_total = pred_aligned.new_zeros(())
            for scale_i in range(self.scales):
                step = 2 ** scale_i
                grad_total = grad_total + _gradient_loss(
                    pred_aligned[:, 0, ::step, ::step],
                    target[:, 0, ::step, ::step],
                    m_hw[:, ::step, ::step],
                    reduction=self._reduction,
                )
            total = total + self.grad_w * grad_total

        if self.silog_w > 0:
            silog_pred = pred_aligned if self.silog_on_aligned else prediction
            total = total + self.silog_w * self._silog(silog_pred, target, mask)

        return total


# =============================================================================
# Depth heads (inline from depth/depth_head.py and depth_anything/dpt.py)
# =============================================================================
class DWConvBlock(nn.Module):
    def __init__(self, c, gn_groups=16):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False)
        self.pw = nn.Conv2d(c, c, 1, bias=False)
        self.gn = nn.GroupNorm(min(gn_groups, c), c)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.gn(self.pw(self.dw(x))))


class Lite4LayerDepthHead(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        fuse_ch: int = 128,
        dec_ch: int = 128,
        use_softplus: bool = True,
    ):
        super().__init__()
        self.use_softplus = use_softplus
        self.ln = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(4)])
        self.proj = nn.ModuleList([nn.Linear(embed_dim, fuse_ch) for _ in range(4)])
        self.layer_mix = nn.ModuleList([DWConvBlock(fuse_ch) for _ in range(4)])
        self.fuse = nn.Sequential(
            nn.Conv2d(4 * fuse_ch, dec_ch, kernel_size=1, bias=False),
            nn.GroupNorm(min(16, dec_ch), dec_ch),
            nn.GELU(),
        )
        self.refine1 = DWConvBlock(dec_ch)
        self.refine2 = DWConvBlock(dec_ch)
        self.head = nn.Conv2d(dec_ch, 1, kernel_size=3, padding=1)
        self.softplus = nn.Softplus(beta=1.0, threshold=20.0)

    def _tokens_to_map(self, t, gh, gw, ln, proj):
        t = ln(t)
        t = proj(t)
        t = t.permute(0, 2, 1).contiguous().view(t.size(0), -1, gh, gw)
        return t

    def forward(self, feats4, grid_hw=None, out_hw=None):
        assert len(feats4) == 4
        N = feats4[0].shape[1]
        for t in feats4[1:]:
            assert t.shape[1] == N

        if grid_hw is None:
            gh = int(math.sqrt(N))
            gw = N // gh
            assert gh * gw == N
        else:
            gh, gw = grid_hw
            assert gh * gw == N

        maps = []
        for i in range(4):
            m = self._tokens_to_map(feats4[i], gh, gw, self.ln[i], self.proj[i])
            m = self.layer_mix[i](m)
            maps.append(m)

        x = torch.cat(maps, dim=1)
        x = self.fuse(x)

        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = x + self.refine1(x)

        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = x + self.refine2(x)

        if out_hw is not None:
            x = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)

        logits = self.head(x)
        if self.use_softplus:
            depth = self.softplus(logits) + 1e-6
        else:
            depth = torch.exp(torch.clamp(logits, min=-10, max=10))

        return depth


class SimpleDepthDecoderV2(nn.Module):
    def __init__(self, embed_dim=768, mid_ch=256, out_range=None):
        super().__init__()
        self.out_range = out_range
        self.in_proj = nn.Conv2d(embed_dim, mid_ch, kernel_size=1)

        def block(ch):
            return nn.Sequential(
                nn.GroupNorm(32, ch),
                nn.GELU(),
                nn.Conv2d(ch, ch, kernel_size=3, padding=1),
                nn.GroupNorm(32, ch),
                nn.GELU(),
                nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            )

        self.refine1 = block(mid_ch)
        self.refine2 = block(mid_ch // 2)
        self.refine3 = block(mid_ch // 4)

        self.reduce2 = nn.Conv2d(mid_ch, mid_ch // 2, kernel_size=1)
        self.reduce3 = nn.Conv2d(mid_ch // 2, mid_ch // 4, kernel_size=1)

        self.head = nn.Conv2d(mid_ch // 4, 1, kernel_size=3, padding=1)
        self.softplus = nn.Softplus(beta=1.0, threshold=20.0)

    def forward(self, features, grid_hw=None, out_hw=None):
        B, Np1, D = features.shape
        x = features[:, 1:, :]
        N = x.shape[1]

        if grid_hw is None:
            gh = int(math.sqrt(N))
            gw = N // gh
            assert gh * gw == N
        else:
            gh, gw = grid_hw
            assert gh * gw == N

        x = x.permute(0, 2, 1).reshape(B, D, gh, gw)
        x = self.in_proj(x)

        x = x + self.refine1(x)
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.reduce2(x)
        x = x + self.refine2(x)

        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.reduce3(x)
        x = x + self.refine3(x)

        if out_hw is not None:
            x = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)

        logits = self.head(x)
        depth = self.softplus(logits) + 1e-6

        if self.out_range is not None:
            dmin, dmax = self.out_range
            depth = dmin + (dmax - dmin) * torch.sigmoid(logits)

        return depth


class ResidualConvUnit(nn.Module):
    def __init__(self, features, activation, bn):
        super().__init__()
        self.bn = bn
        self.groups = 1
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        if self.bn:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)
        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn:
            out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn:
            out = self.bn2(out)
        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True, size=None):
        super().__init__()
        self.deconv = deconv
        self.align_corners = align_corners
        self.groups = 1
        self.expand = expand
        out_features = features
        if self.expand:
            out_features = features // 2
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)
        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)
        self.skip_add = nn.quantized.FloatFunctional()
        self.size = size

    def forward(self, *xs, size=None):
        output = xs[0]
        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)
        output = self.resConfUnit2(output)
        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}
        output = F.interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        output = self.out_conv(output)
        return output


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()
    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape
    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8
    scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    return scratch


class DPTHead(nn.Module):
    def __init__(
        self,
        in_channels,
        features=256,
        use_bn=False,
        out_channels=[256, 512, 1024, 1024],
        use_clstoken=False,
        patch_size: int = 14,
    ):
        super().__init__()
        self.use_clstoken = use_clstoken
        self.patch_size = int(patch_size)

        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])

        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])

        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()
                    )
                )

        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        head_features_1 = features
        head_features_2 = 32

        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.Softplus(beta=1.0, threshold=20.0)
        )

    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (int(patch_h * self.patch_size), int(patch_w * self.patch_size)), mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out)
        return out


# =============================================================================
# Patch position losses (inline from core/patch_pos.py)
# =============================================================================
class PatchRowColRegressionCriterion(nn.Module):
    def __init__(
        self,
        feat_dim,
        grid_h,
        grid_w,
        normalize=True,
        huber_beta=None,
        loss_type: str = "smooth_l1",
    ):
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

        if loss_type == "l1":
            self.loss_fn = nn.L1Loss()
        elif loss_type == "smooth_l1":
            if huber_beta is None:
                self.loss_fn = nn.SmoothL1Loss()
            else:
                self.loss_fn = nn.SmoothL1Loss(beta=huber_beta)
        elif loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

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
        B, N, D = feats.shape
        assert N == self.grid_h * self.grid_w

        x = feats.reshape(-1, D)
        row_targets = self.row_targets.repeat(B)
        col_targets = self.col_targets.repeat(B)

        row_pred = self.row_mlp(x).squeeze(-1)
        col_pred = self.col_mlp(x).squeeze(-1)

        loss_row = self.loss_fn(row_pred, row_targets)
        loss_col = self.loss_fn(col_pred, col_targets)
        return (loss_row + loss_col) / 2.0


class PatchRowColRegressionCriterionDynamic(nn.Module):
    def __init__(
        self,
        feat_dim,
        grid_h,
        grid_w,
        normalize=True,
        loss_type: str = "smooth_l1",
    ):
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

        if loss_type == "l1":
            self.loss_fn = nn.L1Loss()
        elif loss_type == "smooth_l1":
            self.loss_fn = nn.SmoothL1Loss()
        elif loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

        rows = torch.arange(grid_h, dtype=torch.float32).unsqueeze(1).repeat(1, grid_w)
        cols = torch.arange(grid_w, dtype=torch.float32).unsqueeze(0).repeat(grid_h, 1)
        self.register_buffer("row_index_full", rows, persistent=False)
        self.register_buffer("col_index_full", cols, persistent=False)

    def forward(self, feats, hp=None, wp=None):
        B, N, D = feats.shape
        if hp is None:
            hp = self.grid_h
        if wp is None:
            wp = self.grid_w
        assert N == hp * wp

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

# =============================================================================
# Dataset and DataLoader
# =============================================================================
logger.info("Creating datasets...")
try:
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

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    valid_kwargs = dict(
        batch_size=args.batch_size,
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
    valid_loader = DataLoader(valid_dataset, **valid_kwargs)

    steps_per_epoch = len(train_loader)
    accum_steps = max(1, int(getattr(args, "grad_accum_steps", 1)))
    optimizer_steps_per_epoch = math.ceil(steps_per_epoch / accum_steps)

    logger.info("DataLoaders created: train=%s, val=%s", len(train_dataset), len(valid_dataset))
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


def _prep_dpt_features(features, grid_hw):
    gh, gw = grid_hw
    tokens_needed = gh * gw
    prepped = []
    for f in features:
        if f.shape[1] == tokens_needed + 1:
            f = f[:, 1:, :]
        prepped.append((f, None))
    return prepped


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
        dpt_feats = _prep_dpt_features(features, (patch_h, patch_w))
        if use_amp:
            dpt_feats_fp32 = [(f.float(), aux) for (f, aux) in dpt_feats]
            with torch.amp.autocast(device_type=DEVICE.type, enabled=False):
                pred_depths = decoder(dpt_feats_fp32, patch_h=patch_h, patch_w=patch_w)
        else:
            pred_depths = decoder(dpt_feats, patch_h=patch_h, patch_w=patch_w)
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
)

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
        if "scheduler" in ckpt and ckpt["scheduler"] is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
    else:
        logger.info("Skipping scheduler state load (resume_scheduler=False).")
    if "scaler" in ckpt and ckpt["scaler"] is not None:
        scaler.load_state_dict(ckpt["scaler"])
    if args.use_rc_loss and "rowcol_loss" in ckpt and ckpt["rowcol_loss"] is not None:
        for k in ["row_targets", "col_targets", "row_index_full", "col_index_full"]:
            if k in ckpt["rowcol_loss"]:
                ckpt["rowcol_loss"].pop(k)
        rowcol_loss.load_state_dict(ckpt["rowcol_loss"])
    logger.info("Resumed full checkpoint from '%s' at epoch %s", args.resume_ckpt_path, start_epoch)
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


def train_one_epoch(model, decoder, loader, criterion, optimizer, scheduler, scaler, feature_layers, epoch, total_epochs):
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
    optimizer.zero_grad(set_to_none=True)
    for i, (inputs, gt_depths, metas) in enumerate(loader):
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
            loss = base_loss + args.rc_alpha * aux_loss
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

    denom = max(total_samples, 1)
    avg_loss = (running_loss_t / denom).float().item()
    avg_aux = (aux_loss_sum_t / denom).float().item()
    avg_base = (base_loss_t / denom).float().item()
    return avg_loss, avg_aux, avg_base


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
        avg_train_loss, avg_aux_loss, base_loss = train_one_epoch(
            model, decoder, train_loader, criterion, optimizer, scheduler, scaler, feature_layers, epoch, EPOCHS
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
                "model": model.state_dict(),
                "decoder": decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "scaler": scaler.state_dict() if scaler is not None else None,
                "rowcol_loss": rowcol_loss.state_dict() if args.use_rc_loss else None,
                "training_history": training_history,
                "args": args,
            }
            torch.save(ckpt, last_ckpt_path)
            logger.info("Saved full checkpoint to '%s'", last_ckpt_path)

        if args.total_run_time_hr is not None:
            elapsed = time.time() - train_start_time
            max_run_time_sec = args.total_run_time_hr * 3600
            if elapsed  + (train_time + val_time) + 1200>= max_run_time_sec:
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
