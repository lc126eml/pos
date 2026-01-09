import random
from typing import Tuple, Optional, Dict, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from torchvision.transforms import ColorJitter, GaussianBlur
import torchvision.transforms.functional as TF


ImageLike = Union[Image.Image, np.ndarray, torch.Tensor]
DepthLike = Union[np.ndarray, torch.Tensor]


def _to_pil_rgb(image: ImageLike) -> Image.Image:
    """Convert input to PIL RGB."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, torch.Tensor):
        x = image.detach().cpu()
        if x.ndim == 3 and x.shape[0] in (1, 3):  # CHW -> HWC
            x = x.permute(1, 2, 0)
        x = x.numpy()
        image = x
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        if arr.dtype != np.uint8:
            # assume float in [0,1] or [0,255]
            arr = np.clip(arr, 0.0, 1.0) if arr.max() <= 1.5 else np.clip(arr / 255.0, 0.0, 1.0)
            arr = (arr * 255.0).round().astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")
    raise TypeError(f"Unsupported image type: {type(image)}")


def _to_depth_1chw(depth: DepthLike) -> torch.Tensor:
    """
    Convert depth to torch float32 [1,H,W]. Invalid depth may be <=0 or non-finite.
    """
    if isinstance(depth, torch.Tensor):
        d = depth.detach().float().cpu()
        if d.ndim == 2:
            d = d.unsqueeze(0)
        elif d.ndim == 3 and d.shape[-1] == 1:  # HWC1 -> 1HW
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
    """PIL RGB -> torch float32 [3,H,W] in [0,1]."""
    x = torch.from_numpy(np.array(pil_img)).float() / 255.0  # [H,W,3]
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


def _resize_depth_with_mask(depth_1chw: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
    """
    Resize depth with bilinear interpolation while preserving invalid regions.
    depth_1chw: [1,H,W] -> returns [1,Ht,Wt] with invalid set to 0.
    """
    d = depth_1chw.unsqueeze(0)  # [1,1,H,W]
    valid = torch.isfinite(d) & (d > 0)
    d0 = torch.where(valid, d, torch.zeros_like(d))

    d_rs = F.interpolate(d0, size=size_hw, mode="bilinear", align_corners=False)
    m_rs = F.interpolate(valid.float(), size=size_hw, mode="nearest") > 0.5
    d_rs = torch.where(m_rs, d_rs, torch.zeros_like(d_rs))

    return d_rs.squeeze(0)  # [1,Ht,Wt]


def _round_to_multiple(x: int, m: int) -> int:
    return int(round(x / m) * m)


def train_aug_depth_ar_resize_random_crop(
    image: ImageLike,
    depth: DepthLike,
    target_size: Tuple[int, int],
    *,
    hflip_prob: float = 0.5,
    scale_jitter: Optional[Tuple[float, float]] = (0.85, 1.15),
    color_jitter: Optional[Dict[str, float]] = None,
    color_jitter_prob: float = 0.8,
    gamma_jitter: Optional[Tuple[float, float]] = (0.9, 1.1),
    grayscale_prob: float = 0.05,
    blur_prob: float = 0.05,
    blur_kernel: Tuple[int, int] = (5, 5),
    blur_sigma: Tuple[float, float] = (0.1, 1.0),
    noise_std: Optional[Tuple[float, float]] = (0.0, 0.01),
    normalize: bool = True,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ensure_multiple_of: Optional[int] = None,  # e.g., 32 (applies to the *resized pre-crop* size)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Training augmentation (no padding):
      1) Resize with aspect ratio (optionally mild scale jitter), ensuring resized dims >= target
      2) Random crop to target_size
      3) Random horizontal flip
      4) Color + gamma jitter, optional grayscale, blur, and noise (RGB only)

    Returns:
      img_t:   [3,Ht,Wt] float32 in [0,1]
      depth_t: [1,Ht,Wt] float32 (invalid -> 0)
    """
    Ht, Wt = target_size
    pil = _to_pil_rgb(image)
    d = _to_depth_1chw(depth)

    W0, H0 = pil.size

    # Base scale so that both dimensions can contain the target crop.
    base = max(Ht / H0, Wt / W0)
    jitter = random.uniform(*scale_jitter) if scale_jitter is not None else 1.0
    scale = base * jitter

    newH = int(round(H0 * scale))
    newW = int(round(W0 * scale))

    # Ensure we can crop (guard against jitter < 1.0)
    if newH < Ht or newW < Wt:
        scale = base
        newH = int(round(H0 * scale))
        newW = int(round(W0 * scale))

    if ensure_multiple_of is not None and ensure_multiple_of > 1:
        newH = max(Ht, _round_to_multiple(newH, ensure_multiple_of))
        newW = max(Wt, _round_to_multiple(newW, ensure_multiple_of))

    # Resize (synced)
    pil = pil.resize((newW, newH), resample=Image.BILINEAR)
    d = _resize_depth_with_mask(d, (newH, newW))

    # Random crop (synced)
    top = 0 if newH == Ht else random.randint(0, newH - Ht)
    left = 0 if newW == Wt else random.randint(0, newW - Wt)

    pil = TF.crop(pil, top=top, left=left, height=Ht, width=Wt)
    d = d[:, top:top + Ht, left:left + Wt]

    # Horizontal flip (synced)
    if random.random() < hflip_prob:
        pil = TF.hflip(pil)
        d = torch.flip(d, dims=[2])  # W dimension

    # Color jitter (RGB only)
    if (
        color_jitter is not None
        and any(v > 0 for v in color_jitter.values())
        and (color_jitter_prob > 0)
        and (random.random() < color_jitter_prob)
    ):
        pil = ColorJitter(**color_jitter)(pil)

    # Gamma jitter (RGB only)
    if gamma_jitter is not None:
        g0, g1 = gamma_jitter
        if g0 != 1.0 or g1 != 1.0:
            gamma = random.uniform(g0, g1)
            pil = TF.adjust_gamma(pil, gamma=gamma)

    # Random grayscale (RGB only)
    if grayscale_prob > 0 and random.random() < grayscale_prob:
        pil = TF.to_grayscale(pil, num_output_channels=3)

    # Random blur (RGB only)
    if blur_prob > 0 and random.random() < blur_prob:
        pil = GaussianBlur(kernel_size=blur_kernel, sigma=blur_sigma)(pil)

    img_t = _pil_to_tensor01(pil)

    # Additive noise in [0,1] (RGB only)
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
    target_by: str = "height",          # "height" (most common) or "long_side"
    ensure_multiple_of: Optional[int] = 32,
    normalize: bool = True,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    Validation/Test preprocessing (no padding):
      - Deterministic resize keeping aspect ratio
      - Optional padding to ensure multiple-of for network stride
      - Single forward pass on the resized tensor
      - (Typically) upsample prediction back to original (H,W) externally for metrics

    Interpretation of target_size:
      - If target_by="height": uses target_size[0] as the inference height (recommended default).
      - If target_by="long_side": uses max(target_size) as the inference long side.

    Returns:
      img_t:   [3,H',W'] float32 in [0,1]
      depth_t: [1,H',W'] float32 resized with mask (invalid -> 0)
      meta: dict with orig/resized sizes and scale (useful for resizing predictions back):
            {
              "orig_h","orig_w","resized_h","resized_w",
              "scale_h","scale_w"
            }
    """
    pil = _to_pil_rgb(image)
    d = _to_depth_1chw(depth)

    W0, H0 = pil.size
    Ht, Wt = target_size

    if target_by == "height":
        newH = int(Ht)
        scale = newH / H0
        newW = int(round(W0 * scale))
    elif target_by == "long_side":
        long_target = int(max(Ht, Wt))
        long0 = max(H0, W0)
        scale = long_target / long0
        newH = int(round(H0 * scale))
        newW = int(round(W0 * scale))
    else:
        raise ValueError(f"target_by must be 'height' or 'long_side', got {target_by}")

    # Deterministic resize (synced)
    pil_rs = pil.resize((newW, newH), resample=Image.BILINEAR)
    d_rs = _resize_depth_with_mask(d, (newH, newW))

    pad_h = 0
    pad_w = 0
    if ensure_multiple_of is not None and ensure_multiple_of > 1:
        pad_h = (ensure_multiple_of - (newH % ensure_multiple_of)) % ensure_multiple_of
        pad_w = (ensure_multiple_of - (newW % ensure_multiple_of)) % ensure_multiple_of
        if pad_h or pad_w:
            pil_rs = TF.pad(pil_rs, padding=(0, 0, pad_w, pad_h), fill=0)
            d_rs = F.pad(d_rs, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

    meta = {
        "orig_h": float(H0), "orig_w": float(W0),
        "resized_h": float(newH), "resized_w": float(newW),
        "scale_h": float(newH) / float(H0),
        "scale_w": float(newW) / float(W0),
        "pad_h": float(pad_h), "pad_w": float(pad_w),
        "padded_h": float(newH + pad_h), "padded_w": float(newW + pad_w),
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
    """
    Validation/Test preprocessing without resizing:
      - Convert to tensor, keep original resolution
      - Optional padding to ensure multiple-of for network stride
    """
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
        scale_jitter: Optional[Tuple[float, float]] = (0.85, 1.15),
        color_jitter: Optional[Dict[str, float]] = None,
        color_jitter_prob: float = 0.8,
        gamma_jitter: Optional[Tuple[float, float]] = (0.9, 1.1),
        grayscale_prob: float = 0.05,
        blur_prob: float = 0.05,
        blur_kernel: Tuple[int, int] = (5, 5),
        blur_sigma: Tuple[float, float] = (0.1, 1.0),
        noise_std: Optional[Tuple[float, float]] = (0.0, 0.01),
        normalize: bool = True,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        ensure_multiple_of: Optional[int] = None,
    ) -> None:
        self.target_size = target_size
        self.hflip_prob = hflip_prob
        self.scale_jitter = scale_jitter
        if color_jitter is None:
            color_jitter = dict(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)
        self.color_jitter = dict(color_jitter)
        self.color_jitter_prob = color_jitter_prob
        self.gamma_jitter = gamma_jitter
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
            grayscale_prob=self.grayscale_prob,
            blur_prob=self.blur_prob,
            blur_kernel=self.blur_kernel,
            blur_sigma=self.blur_sigma,
            noise_std=self.noise_std,
            normalize=False,
            ensure_multiple_of=self.ensure_multiple_of,
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
        ensure_multiple_of: Optional[int] = 32,
        normalize: bool = True,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.target_size = target_size
        self.target_by = target_by
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

    def __call__(
        self, image: ImageLike, depth: DepthLike
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        img_t, depth_t, meta = eval_preprocess_depth_keep_ar(
            image,
            depth,
            self.target_size,
            target_by=self.target_by,
            ensure_multiple_of=self.ensure_multiple_of,
            normalize=False,
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

    def __call__(
        self, image: ImageLike, depth: DepthLike
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
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
