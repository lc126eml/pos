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


def compute_depth_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    return_count: bool = False,
    mode: str = "relative",          # "relative" (scale-only) or "metric"
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

        # s = sum(m*p*t) / sum(m*p^2)
        num = torch.sum(m * p * t, dim=(1, 2))
        den = torch.sum(m * p * p, dim=(1, 2)).clamp_min(eps)
        s = num / den
        # linear depth: disallow negative scale
        s = torch.clamp(s, min=0.0)

        pred_cmp = s.view(-1, 1, 1, 1) * pred
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
    Loss for linear depth when evaluation focuses on AbsRel and δ<1.25 (a1) under scale ambiguity.

    Standard components only:
      - scale-only per-image alignment: p' = s * p
      - SILog (beta default 0.0) on aligned depth
      - optional small multi-scale gradient loss on aligned error
      - optional small L1 on aligned depth

    Notes:
      - scale-only is the physically meaningful invariance for linear depth
      - all terms are computed in the same aligned space to avoid gradient conflicts
      - safe masking (finite + positive + optional depth range)
      - AMP-safe: SILog stats computed with autocast disabled
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
        # s = sum(m*p*t) / sum(m*p^2)
        num = torch.sum(mask_hw * pred_hw * tgt_hw, dim=(1, 2))
        den = torch.sum(mask_hw * pred_hw * pred_hw, dim=(1, 2)).clamp_min(self.eps)
        s = num / den

        if self.clamp_scale_max is None:
            s = torch.clamp(s, min=self.clamp_scale_min)
        else:
            s = torch.clamp(s, min=self.clamp_scale_min, max=float(self.clamp_scale_max))
        return s

    @staticmethod
    def _reduce_per_image(sum_per_img: torch.Tensor, valid_count: torch.Tensor) -> torch.Tensor:
        valid = valid_count > 0
        if not valid.any():
            return sum_per_img.sum() * 0.0
        return (sum_per_img[valid] / valid_count[valid]).mean()

    def _silog(self, pred_hw: torch.Tensor, tgt_hw: torch.Tensor, mask_hw: torch.Tensor) -> torch.Tensor:
        # AMP-safe log-statistics
        with torch.amp.autocast(device_type=pred_hw.device.type, enabled=False):
            p = pred_hw.float().clamp_min(self.eps)
            t = tgt_hw.float().clamp_min(self.eps)
            g = torch.log(p) - torch.log(t)

            B = g.shape[0]
            g_flat = g.reshape(B, -1)
            m_flat = mask_hw.float().reshape(B, -1)

            n = m_flat.sum(dim=1)
            valid = n > 0
            if not valid.any():
                return g.sum() * 0.0

            n_safe = n.clamp_min(1.0)
            mean_g = (g_flat * m_flat).sum(dim=1) / n_safe
            mean_g2 = ((g_flat * g_flat) * m_flat).sum(dim=1) / n_safe
            var_g = (mean_g2 - mean_g * mean_g).clamp_min(0.0)

            dg = var_g + self.beta * mean_g.pow(2)
            loss = 10.0 * torch.sqrt(dg.clamp_min(self.eps))

            return loss[valid].mean()

    def _l1(self, pred_hw: torch.Tensor, tgt_hw: torch.Tensor, mask_hw: torch.Tensor) -> torch.Tensor:
        diff = (pred_hw - tgt_hw).abs() * mask_hw
        sum_per_img = diff.sum(dim=(1, 2))
        n = mask_hw.sum(dim=(1, 2))
        return self._reduce_per_image(sum_per_img, n)

    def _grad_error(self, pred_hw: torch.Tensor, tgt_hw: torch.Tensor, mask_hw: torch.Tensor) -> torch.Tensor:
        diff = (pred_hw - tgt_hw) * mask_hw

        gx = (diff[:, :, 1:] - diff[:, :, :-1]).abs()
        mx = mask_hw[:, :, 1:] * mask_hw[:, :, :-1]
        gx = gx * mx

        gy = (diff[:, 1:, :] - diff[:, :-1, :]).abs()
        my = mask_hw[:, 1:, :] * mask_hw[:, :-1, :]
        gy = gy * my

        sum_per_img = gx.sum(dim=(1, 2)) + gy.sum(dim=(1, 2))
        n = mask_hw.sum(dim=(1, 2))
        return self._reduce_per_image(sum_per_img, n)

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

        # filter images with too few valid pixels (stability)
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
            s = self._compute_scale_only(p_hw, t_hw, m_hw)

        p_aligned = s[:, None, None] * p_hw

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
            return total, s
        return total
