import torch
import torch.nn as nn
import torch.nn.functional as F


def _ensure_4d(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        return x.unsqueeze(1)
    if x.dim() == 4:
        return x
    raise ValueError(f"Expected (B,H,W) or (B,1,H,W); got {tuple(x.shape)}")


def _default_mask(gt: torch.Tensor, eps: float) -> torch.Tensor:
    return (torch.isfinite(gt) & (gt > eps)).float()


def compute_scale_and_shift(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    # Solves least-squares for s,t in s*pred + t against target
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

    # Identity transform for invalid cases to avoid collapsing predictions to zero.
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
    """SILog loss (pixel-wise)."""
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
    """
    Hybrid monocular depth loss:
      - Scale-and-shift invariant L1 on aligned prediction (MiDaS-style)
      - Multi-scale gradient loss on aligned prediction (MiDaS-style)
      - Optional SILog on aligned prediction (ZoeDepth/AdaBins-style)
    """
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

        # squeeze channel for scale/shift (B,H,W)
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
