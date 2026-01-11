# %%
# =================================================================================
# DINOv3 segmentation training (single-file, Kaggle-friendly)
# =================================================================================
import math
import os
import sys
import time
import logging
import random
import gc
import subprocess
from types import SimpleNamespace
from typing import Tuple, Optional, Dict, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import csv
from PIL import Image
from tqdm import tqdm

import torchvision.transforms.functional as TF
from torchvision.transforms import ColorJitter
from importlib.metadata import version, PackageNotFoundError
ver = version("timm").split('.')[-1]
print(ver)
if int(ver) < 20:
    # !pip uninstall -y timm
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "timm"])
    LOCAL_TIMM = "/kaggle/input/timm-repos/pytorch-image-models"
    sys.path.insert(0, LOCAL_TIMM)

import timm
print("timm:", timm.__version__, flush=True)
print("torch:", torch.__version__, flush=True)
# =============================================================================
# Kaggle environment setup
# =============================================================================
_IS_KAGGLE = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.path.exists("/kaggle/working"))



# =============================================================================
# Segmentation augmentations
# =============================================================================
ImageLike = Union[Image.Image, np.ndarray, torch.Tensor]
MaskLike = Union[Image.Image, np.ndarray, torch.Tensor]

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
        return Image.fromarray(arr, mode="RGB")
    raise TypeError(f"Unsupported image type: {type(image)}")

def _to_pil_mask(mask: MaskLike) -> Image.Image:
    if isinstance(mask, Image.Image):
        return mask.convert("I")
    if isinstance(mask, torch.Tensor):
        m = mask.detach().cpu()
        if m.ndim == 3 and m.shape[0] == 1:
            m = m.squeeze(0)
        if m.ndim == 3 and m.shape[-1] == 1:
            m = m.squeeze(-1)
        mask = m.numpy()
    if isinstance(mask, np.ndarray):
        arr = mask
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr.squeeze(-1)
        if arr.ndim != 2:
            raise ValueError(f"Mask must be [H,W]; got {arr.shape}")
        if arr.dtype != np.int32:
            arr = arr.astype(np.int32, copy=False)
        return Image.fromarray(arr, mode="I")
    raise TypeError(f"Unsupported mask type: {type(mask)}")

def _pil_to_tensor01(pil_img: Image.Image) -> torch.Tensor:
    x = torch.from_numpy(np.array(pil_img)).float() / 255.0
    return x.permute(2, 0, 1).contiguous()

def _mask_to_tensor(mask_pil: Image.Image) -> torch.Tensor:
    arr = np.array(mask_pil, dtype=np.int64)
    return torch.from_numpy(arr)

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

def train_aug_seg_resize_random_crop(
    image: ImageLike,
    mask: MaskLike,
    target_size: Tuple[int, int],
    *,
    hflip_prob: float = 0.5,
    scale_jitter: Optional[Tuple[float, Optional[float]]] = (1.0, None),
    cat_max_ratio: Optional[float] = None,
    cat_max_ratio_tries: int = 10,
    ignore_index: Optional[int] = None,
    color_jitter: Optional[Dict[str, float]] = None,
    color_jitter_prob: float = 0.0,
    normalize: bool = True,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> Tuple[torch.Tensor, torch.Tensor]:
    Ht, Wt = target_size
    pil = _to_pil_rgb(image)
    m = _to_pil_mask(mask)

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
        newH = int(round(H0 * base))
        newW = int(round(W0 * base))

    resample = Image.BOX if scale < 1.0 else Image.BICUBIC
    pil = pil.resize((newW, newH), resample=resample)
    m = m.resize((newW, newH), resample=Image.NEAREST)

    def _is_crop_ok(mask_crop: Image.Image) -> bool:
        if cat_max_ratio is None or cat_max_ratio >= 1.0:
            return True
        mask_np = np.array(mask_crop, dtype=np.int64)
        if ignore_index is not None:
            mask_np = mask_np[mask_np != ignore_index]
        if mask_np.size == 0:
            return True
        counts = np.bincount(mask_np.reshape(-1))
        max_ratio = counts.max() / counts.sum()
        return max_ratio < cat_max_ratio

    top = 0
    left = 0
    m_crop = None
    tries = max(1, int(cat_max_ratio_tries))
    for _ in range(tries):
        top = 0 if newH == Ht else random.randint(0, newH - Ht)
        left = 0 if newW == Wt else random.randint(0, newW - Wt)
        m_crop = TF.crop(m, top=top, left=left, height=Ht, width=Wt)
        if _is_crop_ok(m_crop):
            break
    pil = TF.crop(pil, top=top, left=left, height=Ht, width=Wt)
    if m_crop is None:
        m_crop = TF.crop(m, top=top, left=left, height=Ht, width=Wt)
    m = m_crop

    if random.random() < hflip_prob:
        pil = TF.hflip(pil)
        m = TF.hflip(m)

    if (
        color_jitter is not None
        and any(v > 0 for v in color_jitter.values())
        and (color_jitter_prob > 0)
        and (random.random() < color_jitter_prob)
    ):
        pil = ColorJitter(**color_jitter)(pil)

    img_t = _pil_to_tensor01(pil)
    if normalize:
        img_t = _normalize_img(img_t, mean, std)
    mask_t = _mask_to_tensor(m)
    return img_t, mask_t

def eval_preprocess_seg_keep_ar(
    image: ImageLike,
    mask: MaskLike,
    target_size: Tuple[int, int],
    *,
    target_by: str = "shorter",
    eval_crop_mode: str = "pad",
    normalize: bool = True,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    pil = _to_pil_rgb(image)
    m = _to_pil_mask(mask)
    W0, H0 = pil.size
    Ht, Wt = target_size

    if target_by == "shorter":
        scale = min(Ht, Wt) / min(H0, W0)
    elif target_by == "longer":
        scale = max(Ht, Wt) / max(H0, W0)
    else:
        raise ValueError(f"target_by must be 'shorter' or 'longer', got {target_by}")

    newH = int(round(H0 * scale))
    newW = int(round(W0 * scale))
    resample = Image.BOX if scale < 1.0 else Image.BICUBIC
    pil = pil.resize((newW, newH), resample=resample)
    m = m.resize((newW, newH), resample=Image.NEAREST)

    top = max(0, (newH - Ht) // 2)
    left = max(0, (newW - Wt) // 2)
    pad = (0, 0, 0, 0)
    mean_v = mean if isinstance(mean, (list, tuple)) else [float(mean)]
    if len(mean_v) == 1:
        mean_v = mean_v * 3
    pad_fill = tuple(int(round(v * 255.0)) for v in mean_v[:3])

    if eval_crop_mode == "crop":
        pil = TF.crop(pil, top=top, left=left, height=Ht, width=Wt)
        m = TF.crop(m, top=top, left=left, height=Ht, width=Wt)
    elif eval_crop_mode == "pad":
        if newH < Ht or newW < Wt:
            pad_h = max(0, Ht - newH)
            pad_w = max(0, Wt - newW)
            pad_top = pad_h // 2
            pad_left = pad_w // 2
            pad_bottom = pad_h - pad_top
            pad_right = pad_w - pad_left
            pil = TF.pad(pil, padding=(pad_left, pad_top, pad_right, pad_bottom), fill=pad_fill)
            m = TF.pad(m, padding=(pad_left, pad_top, pad_right, pad_bottom), fill=0)
            pad = (pad_left, pad_top, pad_right, pad_bottom)
    elif eval_crop_mode == "crop_or_pad":
        if newH >= Ht and newW >= Wt:
            pil = TF.crop(pil, top=top, left=left, height=Ht, width=Wt)
            m = TF.crop(m, top=top, left=left, height=Ht, width=Wt)
        else:
            pad_h = max(0, Ht - newH)
            pad_w = max(0, Wt - newW)
            pad_top = pad_h // 2
            pad_left = pad_w // 2
            pad_bottom = pad_h - pad_top
            pad_right = pad_w - pad_left
            pil = TF.pad(pil, padding=(pad_left, pad_top, pad_right, pad_bottom), fill=pad_fill)
            m = TF.pad(m, padding=(pad_left, pad_top, pad_right, pad_bottom), fill=0)
            pad = (pad_left, pad_top, pad_right, pad_bottom)
    else:
        raise ValueError(f"eval_crop_mode must be 'pad', 'crop', or 'crop_or_pad', got {eval_crop_mode}")

    img_t = _pil_to_tensor01(pil)
    if normalize:
        img_t = _normalize_img(img_t, mean, std)
    mask_t = _mask_to_tensor(m)

    meta = {
        "orig_h": float(H0), "orig_w": float(W0),
        "resized_h": float(newH), "resized_w": float(newW),
        "scale_h": float(newH) / float(H0),
        "scale_w": float(newW) / float(W0),
        "crop_top": float(top), "crop_left": float(left),
        "pad_left": float(pad[0]), "pad_top": float(pad[1]),
        "pad_right": float(pad[2]), "pad_bottom": float(pad[3]),
        "out_h": float(pil.size[1]), "out_w": float(pil.size[0]),
    }
    return img_t, mask_t, meta

class TrainSegAug:
    def __init__(
        self,
        target_size: Tuple[int, int],
        *,
        hflip_prob: float = 0.5,
        scale_jitter: Optional[Tuple[float, Optional[float]]] = (1.0, None),
        cat_max_ratio: Optional[float] = None,
        cat_max_ratio_tries: int = 10,
        ignore_index: Optional[int] = None,
        color_jitter: Optional[Dict[str, float]] = None,
        color_jitter_prob: float = 0.0,
        normalize: bool = True,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.target_size = target_size
        self.hflip_prob = hflip_prob
        self.scale_jitter = scale_jitter
        self.cat_max_ratio = cat_max_ratio
        self.cat_max_ratio_tries = cat_max_ratio_tries
        self.ignore_index = ignore_index
        self.color_jitter = None if color_jitter is None else dict(color_jitter)
        self.color_jitter_prob = color_jitter_prob
        self.normalize = normalize
        mean_v = mean if isinstance(mean, (list, tuple)) else [float(mean)]
        std_v = std if isinstance(std, (list, tuple)) else [float(std)]
        if len(mean_v) == 1:
            mean_v = mean_v * 3
        if len(std_v) == 1:
            std_v = std_v * 3
        self._mean_t = torch.tensor(mean_v).view(3, 1, 1)
        self._std_t = torch.tensor(std_v).view(3, 1, 1)

    def __call__(self, image: ImageLike, mask: MaskLike) -> Tuple[torch.Tensor, torch.Tensor]:
        img_t, mask_t = train_aug_seg_resize_random_crop(
            image,
            mask,
            self.target_size,
            hflip_prob=self.hflip_prob,
            scale_jitter=self.scale_jitter,
            cat_max_ratio=self.cat_max_ratio,
            cat_max_ratio_tries=self.cat_max_ratio_tries,
            ignore_index=self.ignore_index,
            color_jitter=self.color_jitter,
            color_jitter_prob=self.color_jitter_prob,
            normalize=False,
        )
        if self.normalize:
            mean_t = self._mean_t.to(dtype=img_t.dtype)
            std_t = self._std_t.to(dtype=img_t.dtype)
            img_t = (img_t - mean_t) / std_t
        return img_t, mask_t

class EvalSegPreprocess:
    def __init__(
        self,
        target_size: Tuple[int, int],
        *,
        target_by: str = "shorter",
        eval_crop_mode: str = "pad",
        normalize: bool = True,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.target_size = target_size
        self.target_by = target_by
        self.eval_crop_mode = eval_crop_mode
        self.normalize = normalize
        mean_v = mean if isinstance(mean, (list, tuple)) else [float(mean)]
        std_v = std if isinstance(std, (list, tuple)) else [float(std)]
        if len(mean_v) == 1:
            mean_v = mean_v * 3
        if len(std_v) == 1:
            std_v = std_v * 3
        self._mean_t = torch.tensor(mean_v).view(3, 1, 1)
        self._std_t = torch.tensor(std_v).view(3, 1, 1)

    def __call__(self, image: ImageLike, mask: MaskLike):
        img_t, mask_t, meta = eval_preprocess_seg_keep_ar(
            image,
            mask,
            self.target_size,
            target_by=self.target_by,
            eval_crop_mode=self.eval_crop_mode,
            normalize=False,
        )
        if self.normalize:
            mean_t = self._mean_t.to(dtype=img_t.dtype)
            std_t = self._std_t.to(dtype=img_t.dtype)
            img_t = (img_t - mean_t) / std_t
        return img_t, mask_t, meta


def eval_preprocess_seg_multiscale(
    image: ImageLike,
    mask: MaskLike,
    base_target_size: Tuple[int, int],
    *,
    scales: Tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5),
    flip: bool = True,
    target_by: str = "shorter",
    eval_crop_mode: str = "pad",
    normalize: bool = True,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> list[Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]]:
    out_list = []
    for s in scales:
        ts = (int(round(base_target_size[0] * s)), int(round(base_target_size[1] * s)))
        img_t, mask_t, meta = eval_preprocess_seg_keep_ar(
            image,
            mask,
            ts,
            target_by=target_by,
            eval_crop_mode=eval_crop_mode,
            normalize=normalize,
            mean=mean,
            std=std,
        )
        meta["scale"] = float(s)
        meta["flip"] = False
        out_list.append((img_t, mask_t, meta))
        if flip:
            img_f = torch.flip(img_t, dims=[2])
            mask_f = torch.flip(mask_t, dims=[1])
            meta_f = dict(meta)
            meta_f["flip"] = True
            out_list.append((img_f, mask_f, meta_f))
    return out_list


class EvalSegPreprocessMSFlip:
    def __init__(
        self,
        base_target_size: Tuple[int, int],
        *,
        scales: Tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5),
        flip: bool = True,
        target_by: str = "shorter",
        eval_crop_mode: str = "pad",
        normalize: bool = True,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.base_target_size = base_target_size
        self.scales = scales
        self.flip = flip
        self.target_by = target_by
        self.eval_crop_mode = eval_crop_mode
        self.normalize = normalize
        self.mean = mean
        self.std = std

    def __call__(
        self, image: ImageLike, mask: MaskLike
    ) -> list[Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]]:
        return eval_preprocess_seg_multiscale(
            image,
            mask,
            self.base_target_size,
            scales=self.scales,
            flip=self.flip,
            target_by=self.target_by,
            eval_crop_mode=self.eval_crop_mode,
            normalize=self.normalize,
            mean=self.mean,
            std=self.std,
        )

# =============================================================================
# Segmentation heads and losses
# =============================================================================
class PPMliteFCNHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        grid_size: tuple,
        out_size: tuple,
        mid_channels: int = 256,
        ppm_bins=(1, 2, 3),
        ppm_channels: int = 64,
        dropout: float = 0.1,
        norm: str = "gn",
    ):
        super().__init__()
        self.grid_size = grid_size
        self.out_size = out_size
        self.ppm_bins = tuple(ppm_bins)

        def norm2d(c: int):
            if norm == "bn":
                return nn.BatchNorm2d(c)
            if norm == "gn":
                g = 32 if c >= 32 else max(1, c // 4)
                return nn.GroupNorm(g, c)
            raise ValueError(f"Unknown norm='{norm}', use 'bn' or 'gn'.")

        self.in_proj = nn.Sequential(
            nn.Conv2d(embed_dim, mid_channels, kernel_size=1, bias=False),
            norm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.ppm = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(mid_channels, ppm_channels, kernel_size=1, bias=False),
                norm2d(ppm_channels),
                nn.ReLU(inplace=True),
            )
            for _ in self.ppm_bins
        ])

        in_fuse = mid_channels + len(self.ppm_bins) * ppm_channels
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_fuse, mid_channels, kernel_size=3, padding=1, bias=False),
            norm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        self.classifier = nn.Conv2d(mid_channels, num_classes, kernel_size=1)

    def forward(self, x_tokens: torch.Tensor, *, grid_size=None, out_size=None) -> torch.Tensor:
        B, N, C = x_tokens.shape
        Hp, Wp = grid_size if grid_size is not None else self.grid_size
        assert N == Hp * Wp, f"Expected N={Hp*Wp} tokens, got N={N}"

        x = x_tokens.transpose(1, 2).contiguous().view(B, C, Hp, Wp)
        x = self.in_proj(x)

        ppm_outs = [x]
        for bin_sz, branch in zip(self.ppm_bins, self.ppm):
            pooled = F.adaptive_avg_pool2d(x, output_size=(bin_sz, bin_sz))
            pooled = branch(pooled)
            up = F.interpolate(pooled, size=(Hp, Wp), mode="bilinear", align_corners=False)
            ppm_outs.append(up)

        x = torch.cat(ppm_outs, dim=1)
        x = self.bottleneck(x)
        logits = self.classifier(x)

        out_size = out_size if out_size is not None else self.out_size
        logits = F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)
        return logits

class FCNSegHead(nn.Module):
    def __init__(self, embed_dim, num_classes, grid_size, out_size, mid_channels=256, dropout=0.1, norm="gn"):
        super().__init__()
        self.grid_size = grid_size
        self.out_size = out_size

        if norm == "bn":
            norm2d = nn.BatchNorm2d
        elif norm == "gn":
            norm2d = lambda c: nn.GroupNorm(32, c)
        else:
            raise ValueError(f"Unknown norm: {norm}")

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, mid_channels, kernel_size=3, padding=1, bias=False),
            norm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(mid_channels, num_classes, kernel_size=1),
        )

    def forward(self, x_tokens, *, grid_size=None, out_size=None):
        B, N, C = x_tokens.shape
        H, W = grid_size if grid_size is not None else self.grid_size
        assert N == H * W, f"Token count N={N} != H*W={H*W}"

        x = x_tokens.transpose(1, 2).contiguous().view(B, C, H, W)
        logits = self.proj(x)
        out_size = out_size if out_size is not None else self.out_size
        logits = F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)
        return logits

class MMSegCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=-1, loss_weight=1.0, avg_non_ignore=True):
        super().__init__()
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight
        self.avg_non_ignore = avg_non_ignore

    def forward(self, logits, target):
        if not self.avg_non_ignore:
            return self.loss_weight * F.cross_entropy(
                logits, target, ignore_index=self.ignore_index, reduction="mean"
            )
        loss = F.cross_entropy(logits, target, ignore_index=self.ignore_index, reduction="none")
        valid = (target != self.ignore_index)
        denom = valid.sum().clamp_min(1).to(loss.dtype)
        return self.loss_weight * (loss[valid].sum() / denom)

class PatchRowColRegressionCriterion(nn.Module):
    def __init__(self, feat_dim, grid_h, grid_w, normalize=True, huber_beta=None):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.normalize = normalize

        self.row_mlp = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.col_mlp = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        if huber_beta is None:
            self.loss_fn = nn.SmoothL1Loss()
        else:
            self.loss_fn = nn.SmoothL1Loss(beta=0.5 / self.grid_h)

        rows_2d = torch.arange(grid_h, dtype=torch.float32).unsqueeze(1).repeat(1, grid_w)
        cols_2d = torch.arange(grid_w, dtype=torch.float32).unsqueeze(0).repeat(grid_h, 1)
        if normalize:
            rows_2d = rows_2d / (grid_h - 1)
            cols_2d = cols_2d / (grid_w - 1)
        self.register_buffer("row_targets", rows_2d.flatten())
        self.register_buffer("col_targets", cols_2d.flatten())

    def forward(self, feats):
        B, N, D = feats.shape
        assert N == self.grid_h * self.grid_w, f"Expected N={self.grid_h*self.grid_w}, got N={N}"

        x = feats.reshape(-1, D)
        row_targets = self.row_targets.repeat(B)
        col_targets = self.col_targets.repeat(B)
        row_pred = self.row_mlp(x).squeeze(-1)
        col_pred = self.col_mlp(x).squeeze(-1)
        loss_row = self.loss_fn(row_pred, row_targets)
        loss_col = self.loss_fn(col_pred, col_targets)
        return (loss_row + loss_col) / 2.0

# =============================================================================
# Configuration
# =============================================================================
if _IS_KAGGLE:
    root_dir = "/kaggle/working"
    base_path_default =  "/kaggle/input/ade20k/ADEChallengeData2016"
args = SimpleNamespace(
    model_type="dinov3",
    use_abs_pos_emb=False,
    use_rot_pos_emb=False,
    model_size="base",
    num_classes=150,
    batch_size=16,
    grad_accum_steps=4,
    train_img_size=512,
    eval_img_size=512,
    use_ms_flip_eval=False,
    scale_jitter=(1.0, None),
    use_cat_max_ratio=False,
    cat_max_ratio=0.75,
    cat_max_ratio_tries=10,
    ms_scales=(0.75, 1.0, 1.25, 1.5),
    eval_crop_mode="crop_or_pad",
    final_ms_flip_eval=True,
    lr=7e-4,
    lr_aux=1e-5,
    eta_min=1e-8,
    composite_lr=True,
    warmup_steps=3000,
    weight_decay=0.01,
    epochs=130,
    overlap=0,
    start_epoch=0,
    seed=55,
    use_rc_loss=True,
    huber_beta=0.1,
    rc_alpha=30.0,
    workers=2 if _IS_KAGGLE else 5,
    color_jitter={"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.05},
    color_jitter_prob=0.1,
    train=True,
    val=False,
    ckpt_path=None,
    lock=False if _IS_KAGGLE else True,
    clip_value=1.0,
    output_dir=os.path.join(root_dir, "seg"),
    log_interval=50,
    csv_interval=3,
    compile_model=False,
    save_full_ckpt=True,
    resume_full_ckpt=False,
    resume_ckpt_path=None,
    resume_bs=False,
    total_run_time_sec=None,
    base_path=base_path_default,
)

ckpt = None
if args.resume_full_ckpt and args.resume_ckpt_path:
    resume_full_ckpt = args.resume_full_ckpt
    resume_ckpt_path = args.resume_ckpt_path
    batch_size = args.batch_size
    grad_accum_steps = args.grad_accum_steps
    ckpt = torch.load(resume_ckpt_path, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args", None)
    if ckpt_args is not None:
        for k, v in vars(ckpt_args).items():
            setattr(args, k, v)
    args.resume_full_ckpt = resume_full_ckpt
    args.resume_ckpt_path = resume_ckpt_path
    if not args.resume_bs:
        args.batch_size = batch_size
        args.grad_accum_steps = grad_accum_steps

if args.use_abs_pos_emb or args.use_rot_pos_emb:
    args.overlap = 0
    args.use_rc_loss = False

if args.eval_img_size != args.train_img_size:
    print("Best practice is to keep eval_img_size == train_img_size; overriding.", flush=True)
    args.eval_img_size = args.train_img_size

MODEL_NAME = f"vit_{args.model_size}_patch16_{args.model_type}"
TRAIN_IMAGE_PATH = os.path.join(args.base_path, "images", "training")
TRAIN_ANNOTATION_PATH = os.path.join(args.base_path, "annotations", "training")
VALID_IMAGE_PATH = os.path.join(args.base_path, "images", "validation")
VALID_ANNOTATION_PATH = os.path.join(args.base_path, "annotations", "validation")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

use_amp = False
use_bf16 = False
autocast_dtype = None

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def _seed_worker(worker_id):
    worker_seed = args.seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

data_rng = torch.Generator()
data_rng.manual_seed(args.seed)

subdir_name = (
    f"{args.model_size}"
    f"{'_abs_pos' if args.use_abs_pos_emb else ''}"
    f"{'_rot_pos' if args.use_rot_pos_emb else ''}_rc_{args.use_rc_loss}_lr{int(args.lr/1e-5)}"
)
if args.use_rc_loss:
    subdir_name += f"_overlap_{args.overlap}_alpha_{int(args.rc_alpha)}"

output_dir = os.path.join(args.output_dir, subdir_name)
ckpt_output_dir = os.path.join(output_dir, "ckpt")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(ckpt_output_dir, exist_ok=True)
last_ckpt_path = os.path.join(ckpt_output_dir, "last.pth")
if args.resume_full_ckpt and args.resume_ckpt_path is None:
    args.resume_ckpt_path = last_ckpt_path

log_file_path = os.path.join(output_dir, f"{subdir_name}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()],
)
logger = logging.getLogger()

logger.info("Using device: %s", DEVICE)
logger.info("Using mixed precision: disabled (fp32)")
logger.info("Arguments: %s", args)
logger.info("Output dir: %s", output_dir)
logger.info("Subdir name: %s", subdir_name)

if not os.path.isdir(TRAIN_IMAGE_PATH):
    logger.error("Missing training images at %s", TRAIN_IMAGE_PATH)
    if _IS_KAGGLE and os.path.isdir("/kaggle/input"):
        logger.error("Available /kaggle/input entries: %s", os.listdir("/kaggle/input"))
    raise FileNotFoundError(f"Training images not found: {TRAIN_IMAGE_PATH}")

# =============================================================================
# Dataset and dataloaders
# =============================================================================
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, pair_transform):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])
        self.pair_transform = pair_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        ann_path = os.path.join(self.annotation_dir, img_name.replace(".jpg", ".png"))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(ann_path).convert("L")

        out = self.pair_transform(image, mask)
        if isinstance(out, tuple) and len(out) == 3:
            image_t, mask_t, _ = out
        else:
            image_t, mask_t = out
        mask_t = mask_t.long() - 1
        return image_t, mask_t

train_dataset = SegmentationDataset(
    TRAIN_IMAGE_PATH,
    TRAIN_ANNOTATION_PATH,
    pair_transform=TrainSegAug(
        target_size=(args.train_img_size, args.train_img_size),
        scale_jitter=args.scale_jitter,
        cat_max_ratio=(args.cat_max_ratio if args.use_cat_max_ratio else None),
        cat_max_ratio_tries=args.cat_max_ratio_tries,
        ignore_index=0 if args.use_cat_max_ratio else None,
        color_jitter=args.color_jitter,
        color_jitter_prob=args.color_jitter_prob,
        normalize=True,
    ),
)
valid_dataset = SegmentationDataset(
    VALID_IMAGE_PATH,
    VALID_ANNOTATION_PATH,
    pair_transform=EvalSegPreprocess(
        target_size=(args.eval_img_size, args.eval_img_size),
        target_by="shorter",
        eval_crop_mode=args.eval_crop_mode,
        normalize=True,
    ),
)

loader_kwargs = dict(
    num_workers=args.workers,
    pin_memory=True,
    worker_init_fn=_seed_worker,
    generator=data_rng,
    persistent_workers=(args.workers > 0),
)
if args.workers > 0:
    loader_kwargs["prefetch_factor"] = 2

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
    **loader_kwargs,
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    drop_last=False,
    **loader_kwargs,
)

steps_per_epoch = len(train_loader)
accum_steps = max(1, int(getattr(args, "grad_accum_steps", 1)))
optimizer_steps_per_epoch = math.ceil(steps_per_epoch / accum_steps)
logger.info("DataLoaders created: train=%s, val=%s", len(train_dataset), len(valid_dataset))

# =============================================================================
# Model, head, optimizer
# =============================================================================
logger.info("Initializing model: %s for %s classes", MODEL_NAME, args.num_classes)
model = timm.create_model(
    MODEL_NAME,
    pretrained=False,
    use_abs_pos_emb=args.use_abs_pos_emb,
    use_rot_pos_emb=args.use_rot_pos_emb,
    num_classes=0,
    dynamic_img_size=True,
    img_size=args.train_img_size,
).to(DEVICE)

grid_h, grid_w = model.patch_embed.grid_size
decoder = PPMliteFCNHead(
    embed_dim=model.embed_dim,
    num_classes=args.num_classes,
    grid_size=(grid_h, grid_w),
    out_size=(args.train_img_size, args.train_img_size),
    mid_channels=256,
    ppm_bins=(1, 2, 3),
    ppm_channels=64,
    dropout=0.1,
    norm="gn",
).to(DEVICE)

logger.info("model.patch_embed.proj %s", model.patch_embed.proj)
if args.overlap > 0:
    original_patch_size = model.patch_embed.proj.kernel_size[0]
    new_patch_size = original_patch_size + args.overlap
    stride = original_patch_size
    original_grid_size = args.train_img_size // stride
    padding = ((original_grid_size - 1) * stride + new_patch_size - args.train_img_size + 1) // 2
    in_chans = model.patch_embed.proj.in_channels
    embed_dim = model.patch_embed.proj.out_channels
    model.patch_embed.proj = nn.Conv2d(
        in_chans, embed_dim,
        kernel_size=(new_patch_size, new_patch_size),
        stride=(stride, stride),
        padding=padding,
    ).to(DEVICE)

if args.compile_model:
    if hasattr(torch, "compile"):
        logger.info("Compiling model with torch.compile (mode='reduce-overhead').")
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
        decoder = torch.compile(decoder)
    else:
        logger.warning("torch.compile not available; skipping compilation.")

dynamic = True
training_parameters = list(model.parameters()) + list(decoder.parameters())
param_groups = []
lr_aux = getattr(args, "lr_aux", args.lr)
if args.use_rc_loss:
    grid_h, grid_w = model.patch_embed.grid_size
    dynamic = False
    rowcol_loss = PatchRowColRegressionCriterion(
        feat_dim=model.embed_dim,
        grid_h=grid_h,
        grid_w=grid_w,
        huber_beta=args.huber_beta,
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

ce_criterion = MMSegCrossEntropyLoss(ignore_index=-1, avg_non_ignore=True)
optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
total_steps = args.epochs * optimizer_steps_per_epoch
if args.composite_lr:
    warmup_steps = min(args.warmup_steps, max(1, total_steps - 1))
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
        optimizer, T_max=total_steps, eta_min=args.eta_min
    )
logger.info("Initialized loss, optimizer, and scheduler.")

# =============================================================================
# Helpers
# =============================================================================
def _infer_grid_hw(model, inputs):
    patch_size = model.patch_embed.patch_size
    if isinstance(patch_size, tuple):
        ph, pw = patch_size
    else:
        ph = pw = patch_size
    return (inputs.shape[-2] // ph, inputs.shape[-1] // pw)

def _round_to_multiple(x: int, m: int) -> int:
    return max(m, int(round(x / m) * m))

def _ms_flip_predict(model, decoder, inputs, num_classes, scales, flip, patch_size):
    if isinstance(patch_size, tuple):
        ph, pw = patch_size
    else:
        ph = pw = patch_size
    b, _, h0, w0 = inputs.shape
    logits_sum = torch.zeros((b, num_classes, h0, w0), device=inputs.device, dtype=inputs.dtype)
    count = 0
    for s in scales:
        hs = _round_to_multiple(int(round(h0 * s)), ph)
        ws = _round_to_multiple(int(round(w0 * s)), pw)
        x_s = F.interpolate(inputs, size=(hs, ws), mode="bilinear", align_corners=False)
        grid_hw = _infer_grid_hw(model, x_s)
        feats = model.forward_features(x_s)
        logits = decoder(feats[:, model.num_prefix_tokens:, :], grid_size=grid_hw, out_size=(hs, ws))
        logits = F.interpolate(logits, size=(h0, w0), mode="bilinear", align_corners=False)
        logits_sum += logits
        count += 1
        if flip:
            x_f = torch.flip(x_s, dims=[3])
            feats_f = model.forward_features(x_f)
            logits_f = decoder(feats_f[:, model.num_prefix_tokens:, :], grid_size=grid_hw, out_size=(hs, ws))
            logits_f = torch.flip(logits_f, dims=[3])
            logits_f = F.interpolate(logits_f, size=(h0, w0), mode="bilinear", align_corners=False)
            logits_sum += logits_f
            count += 1
    return logits_sum / max(count, 1)

@torch.no_grad()
def fast_confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int = -1):
    pred = pred.view(-1).to(torch.int64)
    target = target.view(-1).to(torch.int64)
    valid = target != ignore_index
    pred = pred[valid]
    target = target[valid]
    idx = target * num_classes + pred
    conf = torch.bincount(idx, minlength=num_classes * num_classes)
    return conf.view(num_classes, num_classes)

# =============================================================================
# Train / validation
# =============================================================================
ckpt_path = None
if args.train:
    logger.info("Starting training for %s", MODEL_NAME)
    train_start_time = time.time()
    start_epoch = 0
    training_history = None
    if args.resume_full_ckpt and args.resume_ckpt_path and ckpt is not None:
        model.load_state_dict(ckpt.get("model", {}), strict=False)
        decoder.load_state_dict(ckpt.get("decoder", {}), strict=False)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt and ckpt["scheduler"] is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        if args.use_rc_loss and "rowcol_loss" in ckpt and ckpt["rowcol_loss"] is not None:
            rowcol_loss.load_state_dict(ckpt["rowcol_loss"])
        start_epoch = int(ckpt.get("epoch", 0))
        logger.info("Resumed full checkpoint from %s at epoch %s", args.resume_ckpt_path, start_epoch)
        training_history = ckpt.get("training_history", None)

    if not isinstance(training_history, dict):
        if args.use_rc_loss:
            training_history = {
                "train_loss": [],
                "train_acc": [],
                "valid_acc": [],
                "valid_miou": [],
                "train_time": [],
                "val_time": [],
                "epoch": [],
                "step": [],
                "base_loss": [],
                "aux_loss": [],
            }
        else:
            training_history = {
                "train_loss": [],
                "train_acc": [],
                "valid_acc": [],
                "valid_miou": [],
                "train_time": [],
                "val_time": [],
                "epoch": [],
                "step": [],
            }
    training_history.setdefault("train_time", [])
    training_history.setdefault("val_time", [])

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

    step = int(training_history.get("step", [0])[-1]) if training_history.get("step") else 0
    best_acc = 0.0
    log_interval = getattr(args, "log_interval", 50)
    csv_interval = getattr(args, "csv_interval", 1)

    for epoch in range(start_epoch, args.epochs):
        epoch_train_start = time.time()
        model.train()
        decoder.train()
        running_loss_t = torch.zeros((), device=DEVICE)
        base_loss_t = torch.zeros((), device=DEVICE)
        aux_loss_sum_t = torch.zeros((), device=DEVICE)
        train_correct_t = torch.zeros((), device=DEVICE)
        train_total_t = torch.zeros((), device=DEVICE)
        train_samples_t = torch.zeros((), device=DEVICE)
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Training]", mininterval=0.5)

        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (inputs, labels) in enumerate(train_pbar):
            inputs = inputs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            bs = inputs.size(0)
            aux_loss = None

            feats = model.forward_features(inputs)
            grid_hw = _infer_grid_hw(model, inputs)
            outputs = decoder(
                feats[:, model.num_prefix_tokens:, :],
                grid_size=grid_hw,
                out_size=inputs.shape[-2:],
            )
            loss = ce_criterion(outputs, labels)
            base_loss = loss

            if args.use_rc_loss:
                aux_loss = rowcol_loss(feats[:, model.num_prefix_tokens:, :])
                aux_loss_sum_t += aux_loss.detach() * bs
                loss = base_loss + args.rc_alpha * aux_loss

            loss_scaled = loss / accum_steps
            loss_scaled.backward()

            do_step = ((batch_idx + 1) % accum_steps == 0) or (batch_idx + 1 == len(train_loader))
            if do_step:
                if args.clip_value is not None:
                    torch.nn.utils.clip_grad_norm_(training_parameters, max_norm=args.clip_value)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                pred = outputs.detach().argmax(dim=1)
                mask = (labels >= 0)
                valid_pixels = mask.sum()
                train_correct_t += ((pred == labels) & mask).sum()
                train_total_t += valid_pixels
                train_samples_t += bs

            running_loss_t += loss.detach() * valid_pixels
            if args.use_rc_loss:
                base_loss_t += base_loss.detach() * valid_pixels

            if (step + 1) % log_interval == 0:
                avg_loss = (running_loss_t / train_total_t.clamp_min(1)).float().item()
                avg_acc = (train_correct_t / train_total_t.clamp_min(1)).float().item()
                if args.use_rc_loss:
                    avg_aux = (aux_loss_sum_t / train_samples_t.clamp_min(1)).float().item()
                    train_pbar.set_postfix_str(f"loss={avg_loss:.4f} acc={avg_acc:.3f} aux={avg_aux:.4f}")
                else:
                    train_pbar.set_postfix_str(f"loss={avg_loss:.4f} acc={avg_acc:.3f}")

            step += 1

        train_time = time.time() - epoch_train_start
        model.eval()
        decoder.eval()
        val_correct_t = torch.zeros((), device=DEVICE)
        val_total_t = torch.zeros((), device=DEVICE)
        confmat = torch.zeros((args.num_classes, args.num_classes), device=DEVICE, dtype=torch.int64)

        val_pbar = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Validation]", mininterval=0.5)
        val_start = time.time()
        with torch.inference_mode():
            for inputs, labels in val_pbar:
                inputs = inputs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                if args.use_ms_flip_eval:
                    outputs = _ms_flip_predict(
                        model,
                        decoder,
                        inputs,
                        args.num_classes,
                        args.ms_scales,
                        True,
                        model.patch_embed.patch_size,
                    )
                else:
                    feats = model.forward_features(inputs)
                    grid_hw = _infer_grid_hw(model, inputs)
                    outputs = decoder(
                        feats[:, model.num_prefix_tokens:, :],
                        grid_size=grid_hw,
                        out_size=inputs.shape[-2:],
                    )
                pred = outputs.argmax(dim=1)
                mask = (labels >= 0)
                val_correct_t += ((pred == labels) & mask).sum()
                val_total_t += mask.sum()
                confmat += fast_confusion_matrix(pred, labels, args.num_classes, ignore_index=-1)

        val_time = time.time() - val_start
        confmat_f = confmat.to(torch.float32)
        intersection = torch.diag(confmat_f)
        union = confmat_f.sum(dim=1) + confmat_f.sum(dim=0) - intersection
        valid = union > 0
        epoch_val_miou = (intersection[valid] / union[valid]).mean().item() if valid.any() else 0.0

        epoch_val_acc = (val_correct_t / val_total_t.clamp_min(1)).float().item()
        epoch_train_acc = (train_correct_t / train_total_t.clamp_min(1)).float().item()
        denom_pixels = train_total_t.clamp_min(1).float()
        denom_samples = train_samples_t.clamp_min(1).float()
        epoch_train_loss = (running_loss_t / denom_pixels).float().item()
        if best_acc < epoch_val_acc:
            best_acc = epoch_val_acc

        logger.info("Epoch %s/%s Summary:", epoch + 1 + args.start_epoch, args.epochs)
        logger.info("Step %s Summary:", step)

        if args.use_rc_loss:
            epoch_aux_loss = (aux_loss_sum_t / denom_samples).float().item()
            epoch_base_loss = (base_loss_t / denom_pixels).float().item()
            logger.info(
                "  Train Loss: %.4f | Aux Loss: %.4f | Base Loss: %.4f | Train Acc: %.4f | "
                "Valid Acc: %.4f | Valid mIoU: %.4f | train_time: %.1fs | val_time: %.1fs",
                epoch_train_loss, epoch_aux_loss, epoch_base_loss, epoch_train_acc, epoch_val_acc,
                epoch_val_miou, train_time, val_time,
            )
            training_history["aux_loss"].append(epoch_aux_loss)
            training_history["base_loss"].append(epoch_base_loss)
        else:
            logger.info(
                "  Train Loss: %.4f | Train Acc: %.4f | Valid Acc: %.4f | Valid mIoU: %.4f | "
                "train_time: %.1fs | val_time: %.1fs",
                epoch_train_loss, epoch_train_acc, epoch_val_acc, epoch_val_miou, train_time, val_time,
            )

        training_history["train_loss"].append(epoch_train_loss)
        training_history["train_acc"].append(epoch_train_acc)
        training_history["valid_acc"].append(epoch_val_acc)
        training_history["valid_miou"].append(epoch_val_miou)
        training_history["train_time"].append(train_time)
        training_history["val_time"].append(val_time)
        training_history["epoch"].append(epoch + 1)
        training_history["step"].append(step)

        if (epoch + 1) % csv_interval == 0:
            pd.DataFrame(training_history).to_csv(os.path.join(output_dir, f"{subdir_name}.csv"), index=False)
        if args.save_full_ckpt:
            ckpt = {
                "epoch": epoch + 1,
                "step": step,
                "model": model.state_dict(),
                "decoder": decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "scaler": None,
                "rowcol_loss": rowcol_loss.state_dict() if args.use_rc_loss else None,
                "training_history": training_history,
                "args": args,
            }
            torch.save(ckpt, last_ckpt_path)
            logger.info("Saved full checkpoint to %s", last_ckpt_path)

        if args.total_run_time_sec is not None:
            elapsed = time.time() - train_start_time
            if elapsed >= args.total_run_time_sec:
                logger.info("Stopping training: elapsed %.0fs reached limit %.0fs.", elapsed, args.total_run_time_sec)
                break

    logger.info("Training complete.")
    logger.info("Best Accuracy: %.4f", best_acc)
    logger.info(output_dir)

    if args.final_ms_flip_eval and not args.use_ms_flip_eval:
        logger.info("Running final multi-scale + flip evaluation...")
        model.eval()
        decoder.eval()
        val_correct_t = torch.zeros((), device=DEVICE)
        val_total_t = torch.zeros((), device=DEVICE)
        confmat = torch.zeros((args.num_classes, args.num_classes), device=DEVICE, dtype=torch.int64)
        val_pbar = tqdm(valid_loader, desc="Final MS+Flip [Validation]", mininterval=0.5)
        with torch.inference_mode():
            for inputs, labels in val_pbar:
                inputs = inputs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                outputs = _ms_flip_predict(
                    model,
                    decoder,
                    inputs,
                    args.num_classes,
                    args.ms_scales,
                    True,
                    model.patch_embed.patch_size,
                )
                pred = outputs.argmax(dim=1)
                mask = (labels >= 0)
                val_correct_t += ((pred == labels) & mask).sum()
                val_total_t += mask.sum()
                confmat += fast_confusion_matrix(pred, labels, args.num_classes, ignore_index=-1)

        confmat_f = confmat.to(torch.float32)
        intersection = torch.diag(confmat_f)
        union = confmat_f.sum(dim=1) + confmat_f.sum(dim=0) - intersection
        valid = union > 0
        final_ms_miou = (intersection[valid] / union[valid]).mean().item() if valid.any() else 0.0
        final_ms_acc = (val_correct_t / val_total_t.clamp_min(1)).float().item()
        logger.info("Final MS+Flip Acc: %.4f | Final MS+Flip mIoU: %.4f", final_ms_acc, final_ms_miou)
        training_history["final_ms_flip_acc"] = final_ms_acc
        training_history["final_ms_flip_miou"] = final_ms_miou
        pd.DataFrame(training_history).to_csv(os.path.join(output_dir, f"{subdir_name}.csv"), index=False)
    history_df = pd.DataFrame(training_history)
    history_df.to_csv(os.path.join(output_dir, f'{subdir_name}.csv'), index=False)
    # save_checkpoint(model, decoder, ckpt_output_dir, "final")

    best_miou = history_df['valid_miou'].max()
    best_epoch = history_df.loc[history_df['valid_miou'].idxmax(), 'epoch']
    logger.info(f"Best miou: {best_miou:.4f} at epoch {best_epoch}")

    # Find the epoch with the best validation a1 score
    best_miou_row = history_df.loc[history_df['valid_miou'].idxmax()]
    best_miou_epoch = int(best_miou_row['epoch'])
    best_miou_val = best_miou_row['valid_miou']

    # Find the epoch with the best validation abs_rel
    best_acc_row = history_df.loc[history_df['valid_acc'].idxmax()]
    best_acc_epoch = int(best_acc_row['epoch'])
    best_acc_val = best_acc_row['valid_acc']

    logger.info("\n--- Best Validation Metrics from History ---")
    logger.info(f"  Best miou:      {best_miou_val:.4f} (Epoch {best_miou_epoch})")
    logger.info(f"  Best acc:  {best_acc_val:.4f} (Epoch {best_acc_epoch})")
    logger.info("------------------------------------------")
