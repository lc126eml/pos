import torch
import torch.nn as nn
import torch.nn.functional as F


def _compute_scale_only(
    pred_hw: torch.Tensor,
    tgt_hw: torch.Tensor,
    mask_hw: torch.Tensor,
    eps: float = 1e-8,
    clamp_min: float = 1e-8,
) -> torch.Tensor:
    num = torch.sum(mask_hw * pred_hw * tgt_hw, dim=(1, 2))
    den = torch.sum(mask_hw * pred_hw * pred_hw, dim=(1, 2)).clamp_min(eps)
    s = num / den
    return s.clamp_min(clamp_min)

def _compute_scale_and_shift(
    pred_hw: torch.Tensor,
    tgt_hw: torch.Tensor,
    mask_hw: torch.Tensor,
    eps: float = 1e-8,
):
    # least-squares fit: tgt ~= a * pred + b
    m = mask_hw
    p = pred_hw
    t = tgt_hw
    m_sum = m.sum(dim=(1, 2)).clamp_min(eps)
    p_sum = (m * p).sum(dim=(1, 2))
    t_sum = (m * t).sum(dim=(1, 2))
    p2_sum = (m * p * p).sum(dim=(1, 2))
    pt_sum = (m * p * t).sum(dim=(1, 2))

    denom = (p2_sum * m_sum - p_sum * p_sum).clamp_min(eps)
    a = (pt_sum * m_sum - p_sum * t_sum) / denom
    b = (t_sum - a * p_sum) / m_sum
    return a, b

def compute_depth_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    return_count: bool = False,
    mode: str = "relative",          # "relative" (scale-only) or "metric"
    align_mode: str = "scale",       # "scale" or "scale_shift"
    depth_min: float = 0.0,
    depth_max: float = float("inf"),
    eps: float = 1e-8,
):
    """
    Evaluation for linear depth focusing on AbsRel + a1 (and optional others).
    Setting: scale-ambiguous monocular depth -> use scale-only alignment (s * pred).

    Differences vs your code:
      - Uses SCALE-ONLY alignment for linear depth when mode == "relative"
      - Keeps masking/reduction per-image, then averages across valid images
      - Avoids shift (non-physical for linear depth)

    Args:
      pred, target: (B,H,W) or (B,1,H,W)
      mask: optional validity mask (same shape as target or broadcastable), treated as boolean
      mode: "relative" (scale-only alignment) or "metric" (no alignment)
      depth_min/depth_max: evaluation range
      eps: numerical stability
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    if target.dim() == 3:
        target = target.unsqueeze(1)
    if pred.dim() != 4 or target.dim() != 4:
        raise ValueError(f"Expected (B,1,H,W) or (B,H,W); got pred={pred.shape}, target={target.shape}")

    # Ensure finite predictions; target validity handled by mask below
    pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)

    # Build validity mask from target + range, optionally AND with user mask
    thresh = max(float(depth_min), float(eps))
    valid_mask = torch.isfinite(target) & (target > thresh) & (target <= float(depth_max))
    if mask is not None:
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        valid_mask = valid_mask & mask.bool()

    valid_mask_f = valid_mask.float()  # (B,1,H,W)
    denom = valid_mask_f.sum(dim=(1, 2, 3))  # (B,)
    valid_img = denom > 0
    if not valid_img.any():
        return ({}, 0) if return_count else {}
    denom_safe = denom.clamp_min(1.0)

    # -------------------------
    # Scale-only alignment (linear depth) for "relative" mode
    # -------------------------
    if mode in ("relative", "scale_invariant"):
        p = pred[:, 0]
        t = target[:, 0]
        m = valid_mask_f[:, 0]

        if align_mode == "scale_shift":
            a, b = _compute_scale_and_shift(p, t, m, eps=eps)
            pred_cmp = a.view(-1, 1, 1, 1) * pred + b.view(-1, 1, 1, 1)
        elif align_mode == "scale":
            s = _compute_scale_only(p, t, m, eps=eps, clamp_min=0.0)
            pred_cmp = s.view(-1, 1, 1, 1) * pred
        else:
            raise ValueError(f"Unsupported align_mode='{align_mode}'. Use 'scale' or 'scale_shift'.")
        target_cmp = target
    else:
        pred_cmp = pred
        target_cmp = target

    # Clamp to evaluation range (avoid divide-by-zero in ratios)
    pred_cmp = pred_cmp.clamp(min=thresh, max=depth_max)
    target_cmp = target_cmp.clamp(min=thresh, max=depth_max)

    diff = pred_cmp - target_cmp
    ratio = torch.maximum(pred_cmp / target_cmp, target_cmp / pred_cmp)

    def masked_mean_per_image(x: torch.Tensor) -> torch.Tensor:
        # x: (B,1,H,W)
        return (x * valid_mask_f).sum(dim=(1, 2, 3)) / denom_safe

    # Metrics (per-image then mean over valid images)
    abs_rel = masked_mean_per_image(torch.abs(diff) / target_cmp)
    l1 = masked_mean_per_image(torch.abs(diff))
    rmse = torch.sqrt(masked_mean_per_image(diff ** 2))
    a1 = masked_mean_per_image((ratio < 1.25).float())
    a2 = masked_mean_per_image((ratio < 1.25 ** 2).float())
    a3 = masked_mean_per_image((ratio < 1.25 ** 3).float())

    metrics = {
        "abs_rel": abs_rel[valid_img].mean(),
        "mae": l1[valid_img].mean(),
        "rmse": rmse[valid_img].mean(),
        "a1": a1[valid_img].mean(),
        "a2": a2[valid_img].mean(),
        "a3": a3[valid_img].mean(),
    }

    out = {k: v.item() for k, v in metrics.items()}
    return (out, int(valid_img.sum().item())) if return_count else out

class MonocularDepthLoss(nn.Module):
    """
    Standard-component loss for linear depth under scale ambiguity:
      - scale-only alignment: p' = s * p
      - SiLog on aligned depth
      - optional L1 on aligned depth
      - optional multi-scale gradient loss on aligned error

    Reduction behavior is configurable:
      - reduction="per_image": compute per-image mean over valid pixels, then average over valid images
      - reduction="batch": compute mean over all valid pixels in the batch (global)

    Notes:
      - This does NOT invent new losses; it only controls reduction.
      - Masking is safe (finite + positive + optional depth range)
      - AMP-safe: SiLog statistics computed with autocast disabled
    """
    def __init__(
        self,
        silog_w: float = 1.0,
        grad_w: float = 0.1,
        l1_w: float = 0.0,
        silog_beta: float = 0.0,
        scales: int = 4,
        eps: float = 1e-7,
        min_valid_pixels: int = 100,
        min_depth: float | None = None,
        max_depth: float | None = None,
        clamp_scale_min: float = 0.0,
        clamp_scale_max: float | None = None,
        reduction: str = "per_image",   # "per_image" or "batch"
        align_mode: str = "scale",      # "scale" or "scale_shift"
    ):
        super().__init__()
        self.silog_w = float(silog_w)
        self.grad_w = float(grad_w)
        self.l1_w = float(l1_w)

        self.beta = float(silog_beta)
        self.scales = int(scales)
        self.eps = float(eps)
        self.min_valid_pixels = int(min_valid_pixels)

        self.min_depth = min_depth
        self.max_depth = max_depth

        self.clamp_scale_min = float(clamp_scale_min)
        self.clamp_scale_max = clamp_scale_max
        self.align_mode = align_mode

        reduction = str(reduction).lower()
        if reduction not in ("per_image", "batch"):
            raise ValueError("reduction must be 'per_image' or 'batch'")
        self.reduction = reduction

        if self.silog_w <= 0 and self.grad_w <= 0 and self.l1_w <= 0:
            raise ValueError("At least one of silog_w, grad_w, l1_w must be > 0.")

    @staticmethod
    def _ensure_4d(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            return x.unsqueeze(1)
        if x.dim() == 4:
            return x
        raise ValueError(f"Expected (B,H,W) or (B,1,H,W); got {tuple(x.shape)}")

    def _default_mask(self, target: torch.Tensor) -> torch.Tensor:
        m = torch.isfinite(target) & (target > self.eps)
        if self.min_depth is not None:
            m = m & (target > float(self.min_depth))
        if self.max_depth is not None:
            m = m & (target < float(self.max_depth))
        return m.float()

    @torch.no_grad()
    def _compute_scale_only(self, pred_hw: torch.Tensor, tgt_hw: torch.Tensor, mask_hw: torch.Tensor) -> torch.Tensor:
        num = torch.sum(mask_hw * pred_hw * tgt_hw, dim=(1, 2))
        den = torch.sum(mask_hw * pred_hw * pred_hw, dim=(1, 2)).clamp_min(self.eps)
        s = num / den

        if self.clamp_scale_max is None:
            s = torch.clamp(s, min=self.clamp_scale_min)
        else:
            s = torch.clamp(s, min=self.clamp_scale_min, max=float(self.clamp_scale_max))
        return s

    @torch.no_grad()
    def _compute_scale_and_shift(self, pred_hw: torch.Tensor, tgt_hw: torch.Tensor, mask_hw: torch.Tensor):
        return _compute_scale_and_shift(pred_hw, tgt_hw, mask_hw, eps=self.eps)

    def _reduce(self, sum_per_img: torch.Tensor, count_per_img: torch.Tensor) -> torch.Tensor:
        """
        Flexible reduction for masked losses.
        Inputs:
          sum_per_img:   (B,) sum over pixels for each image (masked)
          count_per_img: (B,) count of valid pixels for each image
        """
        valid = count_per_img > 0
        if not valid.any():
            return sum_per_img.sum() * 0.0

        if self.reduction == "per_image":
            return (sum_per_img[valid] / count_per_img[valid]).mean()

        # "batch": mean over all valid pixels in batch
        total_sum = sum_per_img[valid].sum()
        total_cnt = count_per_img[valid].sum().clamp_min(1.0)
        return total_sum / total_cnt

    def _silog(self, pred_hw: torch.Tensor, tgt_hw: torch.Tensor, mask_hw: torch.Tensor) -> torch.Tensor:
        # per-image log-statistics (standard SiLog), then reduced according to `reduction`
        with torch.amp.autocast(device_type=pred_hw.device.type, enabled=False):
            p = pred_hw.float().clamp_min(self.eps)
            t = tgt_hw.float().clamp_min(self.eps)
            g = torch.log(p) - torch.log(t)  # (B,H,W)

            B = g.shape[0]
            g_flat = g.reshape(B, -1)
            m_flat = mask_hw.float().reshape(B, -1)

            n = m_flat.sum(dim=1)  # (B,)
            valid = n > 0
            if not valid.any():
                return g.sum() * 0.0

            n_safe = n.clamp_min(1.0)
            mean_g = (g_flat * m_flat).sum(dim=1) / n_safe
            mean_g2 = ((g_flat * g_flat) * m_flat).sum(dim=1) / n_safe
            var_g = (mean_g2 - mean_g * mean_g).clamp_min(0.0)

            dg = var_g + self.beta * mean_g.pow(2)
            loss_per_img = 10.0 * torch.sqrt(dg.clamp_min(self.eps))  # (B,)

        # Reduce across images: either per-image average (default) or batch-style over valid images.
        # For SiLog, "batch" doesn't naturally mean pixel-weighted (it's already per-image),
        # so we interpret "batch" as mean over valid images as well.
        # If you want pixel-weighted SiLog, you'd need a different definition (not recommended).
        return loss_per_img[valid].mean()

    def _l1(self, pred_hw: torch.Tensor, tgt_hw: torch.Tensor, mask_hw: torch.Tensor) -> torch.Tensor:
        diff = (pred_hw - tgt_hw).abs() * mask_hw
        sum_per_img = diff.sum(dim=(1, 2))
        cnt_per_img = mask_hw.sum(dim=(1, 2))
        return self._reduce(sum_per_img, cnt_per_img)

    def _grad_error(self, pred_hw: torch.Tensor, tgt_hw: torch.Tensor, mask_hw: torch.Tensor) -> torch.Tensor:
        diff = (pred_hw - tgt_hw) * mask_hw

        gx = (diff[:, :, 1:] - diff[:, :, :-1]).abs()
        mx = mask_hw[:, :, 1:] * mask_hw[:, :, :-1]
        gx = gx * mx

        gy = (diff[:, 1:, :] - diff[:, :-1, :]).abs()
        my = mask_hw[:, 1:, :] * mask_hw[:, :-1, :]
        gy = gy * my

        sum_per_img = gx.sum(dim=(1, 2)) + gy.sum(dim=(1, 2))
        cnt_per_img = mask_hw.sum(dim=(1, 2))
        return self._reduce(sum_per_img, cnt_per_img)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
        interpolate: bool = True,
        return_scale: bool = False,
    ):
        pred4 = self._ensure_4d(prediction).float()
        tgt4 = self._ensure_4d(target).float()

        if interpolate and pred4.shape[-2:] != tgt4.shape[-2:]:
            pred4 = F.interpolate(pred4, tgt4.shape[-2:], mode="bilinear", align_corners=True)

        if mask is None:
            mask4 = self._default_mask(tgt4)
        else:
            mask4 = self._ensure_4d(mask).float()
            mask4 = mask4 * self._default_mask(tgt4)

        # enforce min_valid_pixels by zeroing masks for invalid images
        m_hw = mask4[:, 0]
        if self.min_valid_pixels > 0:
            valid_count = m_hw.sum(dim=(1, 2))
            keep = valid_count >= float(self.min_valid_pixels)
            if not torch.all(keep):
                keep_f = keep.float()
                mask4 = mask4 * keep_f[:, None, None, None]
                m_hw = mask4[:, 0]

        p_hw = pred4[:, 0]
        t_hw = tgt4[:, 0]

        with torch.no_grad():
            if self.align_mode == "scale_shift":
                a, b = self._compute_scale_and_shift(p_hw, t_hw, m_hw)
                p_aligned = a[:, None, None] * p_hw + b[:, None, None]
            elif self.align_mode == "scale":
                s = self._compute_scale_only(p_hw, t_hw, m_hw)
                p_aligned = s[:, None, None] * p_hw
            else:
                raise ValueError(f"Unsupported align_mode='{self.align_mode}'. Use 'scale' or 'scale_shift'.")

        total = p_aligned.new_zeros(())

        if self.silog_w > 0:
            total = total + self.silog_w * self._silog(p_aligned, t_hw, m_hw)

        if self.l1_w > 0:
            total = total + self.l1_w * self._l1(p_aligned, t_hw, m_hw)

        if self.grad_w > 0:
            g_total = p_aligned.new_zeros(())
            for i in range(self.scales):
                step = 2 ** i
                g_total = g_total + self._grad_error(
                    p_aligned[:, ::step, ::step],
                    t_hw[:, ::step, ::step],
                    m_hw[:, ::step, ::step],
                )
            total = total + self.grad_w * g_total

        if return_scale:
            if self.align_mode == "scale_shift":
                return total, (a, b)
            return total, s
        return total
