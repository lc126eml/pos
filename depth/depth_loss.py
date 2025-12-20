import torch
import torch.nn as nn
import torch.nn.functional as F


def _ensure_4d(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        return x.unsqueeze(1)
    if x.dim() == 4:
        return x
    raise ValueError(f"Expected (B,H,W) or (B,1,H,W); got {tuple(x.shape)}")


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # x, mask: (B,1,H,W)
    denom = mask.sum(dim=(2, 3), keepdim=True).clamp_min(eps)
    return (x * mask).sum(dim=(2, 3), keepdim=True) / denom


def _masked_median_per_image(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Robust but slower. Returns (B,1,1,1).
    Implemented with a per-sample loop; typically acceptable for depth batches.
    """
    B = x.shape[0]
    out = []
    for b in range(B):
        xb = x[b, 0]  # (H,W)
        mb = mask[b, 0] > 0.5
        vals = xb[mb]
        if vals.numel() == 0:
            out.append(torch.tensor(1.0, device=x.device, dtype=x.dtype))
        else:
            out.append(vals.median())
    out = torch.stack(out, dim=0).view(B, 1, 1, 1).clamp_min(eps)
    return out


def _erode_mask(mask: torch.Tensor, k: int = 3) -> torch.Tensor:
    if k <= 1:
        return mask
    inv = 1.0 - mask
    inv_dil = F.max_pool2d(inv, kernel_size=k, stride=1, padding=k // 2)
    eroded = 1.0 - inv_dil
    return (eroded > 0.5).float()


def ssim_distance_map_unit01(pred01: torch.Tensor, tgt01: torch.Tensor, window: int = 3) -> torch.Tensor:
    """
    SSIM distance map in [0,1], assumes inputs are in [0,1].
    Reflection padding reduces border artifacts.
    """
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2

    pad = window // 2
    pred = F.pad(pred01, (pad, pad, pad, pad), mode="reflect")
    tgt  = F.pad(tgt01,  (pad, pad, pad, pad), mode="reflect")

    mu_p = F.avg_pool2d(pred, window, stride=1, padding=0)
    mu_t = F.avg_pool2d(tgt,  window, stride=1, padding=0)

    sigma_p  = F.avg_pool2d(pred * pred, window, stride=1, padding=0) - mu_p * mu_p
    sigma_t  = F.avg_pool2d(tgt  * tgt,  window, stride=1, padding=0) - mu_t * mu_t
    sigma_pt = F.avg_pool2d(pred * tgt,  window, stride=1, padding=0) - mu_p * mu_t

    ssim = ((2 * mu_p * mu_t + C1) * (2 * sigma_pt + C2)) / (
        (mu_p * mu_p + mu_t * mu_t + C1) * (sigma_p + sigma_t + C2)
    )
    return torch.clamp((1.0 - ssim) * 0.5, 0.0, 1.0)


class MonocularDepthLoss(nn.Module):
    """
    Composite depth loss with INTERNAL normalization suitable for unnormalized metric depth.

    Key idea:
      - Normalize both pred and GT by a per-image GT-derived scale so typical magnitude ~ 1.
      - Use that normalized depth for L1 / grad / SSIM for stable optimization.
      - SILog can be used on normalized depths as well (often redundant but fine).

    If you want strict metric scale supervision, use scale_mode='dataset' and provide dataset_scale.
    """
    def __init__(
        self,
        silog_w: float = 0.0,
        l1_w: float = 1.0,
        grad_w: float = 0.5,
        ssim_w: float = 0.2,
        l_inf_w: float = 0.0,
        lambda_var: float = 0.0,   # metric-friendly: avoid pushing scale-invariance
        valid_mask: bool = True,
        eps: float = 1e-8,
        # normalization
        scale_mode: str = "gt_mean",   # 'gt_mean' | 'gt_median' | 'dataset' | 'none'
        dataset_scale: float | None = None,
        scale_detach: bool = True,     # stop-grad through scale to avoid cheating
        # gradient loss domain
        grad_use_log: bool = False,
        # SSIM settings
        ssim_norm_mode: str = "per_image",  # "per_image"
        ssim_min: float | None = None,
        ssim_max: float | None = None,
        ssim_percentiles: tuple[float, float] = (5.0, 95.0),
        ssim_window: int = 3,
        ssim_erode_mask: bool = True,
        ssim_log_range: float = 4.0,   # map log-depth in [1/r, r] to [0,1]; default r=4
    ):
        super().__init__()
        self.silog_w = float(silog_w)
        self.l1_w = float(l1_w)
        self.grad_w = float(grad_w)
        self.ssim_w = float(ssim_w)
        self.l_inf_w = float(l_inf_w)
        self.lambda_var = float(lambda_var)
        self.use_default_valid_mask = bool(valid_mask)
        self.eps = float(eps)

        self.scale_mode = str(scale_mode)
        self.dataset_scale = dataset_scale
        self.scale_detach = bool(scale_detach)
        self.grad_use_log = bool(grad_use_log)
        self.ssim_norm_mode = str(ssim_norm_mode)
        self.ssim_min = ssim_min
        self.ssim_max = ssim_max
        self.ssim_percentiles = ssim_percentiles

        self.ssim_window = int(ssim_window)
        self.ssim_erode_mask = bool(ssim_erode_mask)
        self.ssim_log_range = float(ssim_log_range)

        if self.scale_mode == "dataset" and (self.dataset_scale is None or self.dataset_scale <= 0):
            raise ValueError("scale_mode='dataset' requires dataset_scale > 0")

    def forward(self, pred_depth, gt_depth, valid_mask=None):
        pred_depth = _ensure_4d(pred_depth)
        gt_depth = _ensure_4d(gt_depth)

        if valid_mask is None and self.use_default_valid_mask:
            valid_mask = (gt_depth > self.eps).float()
        elif valid_mask is not None:
            valid_mask = _ensure_4d(valid_mask).float()
        else:
            valid_mask = torch.ones_like(gt_depth, dtype=gt_depth.dtype, device=gt_depth.device)

        # fp32 for AMP/bf16 stability
        pred = pred_depth.float()
        gt = gt_depth.float()
        mask = valid_mask.float()

        # --- normalization (core) ---
        scale = self._compute_scale(gt, mask)  # (B,1,1,1)
        if self.scale_detach:
            scale = scale.detach()

        pred_n = pred / scale
        gt_n   = gt   / scale

        loss_dict = {}
        total = pred_n.new_zeros(())

        if self.silog_w > 0:
            silog = self._silog_loss(pred_n, gt_n, mask)
            total = total + self.silog_w * silog
            loss_dict["silog"] = float(silog.detach().item())

        if self.l1_w > 0:
            l1 = self._l1_loss(pred_n, gt_n, mask)
            total = total + self.l1_w * l1
            loss_dict["l1"] = float(l1.detach().item())

        if self.grad_w > 0:
            if self.grad_use_log:
                pred_g = torch.log(torch.clamp(pred_n, min=self.eps))
                gt_g = torch.log(torch.clamp(gt_n, min=self.eps))
            else:
                pred_g = pred_n
                gt_g = gt_n
            grad = self._gradient_loss(pred_g, gt_g, mask)
            total = total + self.grad_w * grad
            loss_dict["grad"] = float(grad.detach().item())

        if self.ssim_w > 0:
            ssim_l = self._ssim_loss(pred_n, gt_n, mask)
            total = total + self.ssim_w * ssim_l
            loss_dict["ssim"] = float(ssim_l.detach().item())

        if self.l_inf_w > 0:
            linf = self._l_inf_loss(pred_n, gt_n, mask)
            total = total + self.l_inf_w * linf
            loss_dict["l_inf"] = float(linf.detach().item())

        # helpful for debugging
        loss_dict["scale_mean"] = float(scale.mean().detach().item())

        return total, loss_dict

    def _compute_scale(self, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.scale_mode == "none":
            return torch.ones((gt.shape[0], 1, 1, 1), device=gt.device, dtype=gt.dtype)

        if self.scale_mode == "dataset":
            return torch.full((gt.shape[0], 1, 1, 1), float(self.dataset_scale),
                              device=gt.device, dtype=gt.dtype).clamp_min(self.eps)

        if self.scale_mode == "gt_mean":
            s = _masked_mean(gt, mask, eps=self.eps)
            return s.clamp_min(self.eps)

        if self.scale_mode == "gt_median":
            s = _masked_median_per_image(gt, mask, eps=self.eps)
            return s.clamp_min(self.eps)

        raise ValueError(f"Unknown scale_mode: {self.scale_mode}")

    def _silog_loss(self, pred, gt, mask):
        pred_c = torch.clamp(pred, min=self.eps)
        gt_c   = torch.clamp(gt,   min=self.eps)

        log_diff = (torch.log(pred_c) - torch.log(gt_c)) * mask
        valid = mask.sum(dim=(1, 2, 3), keepdim=True).clamp_min(self.eps)

        mean_sq = (log_diff * log_diff).sum(dim=(1, 2, 3), keepdim=True) / valid
        mean    = log_diff.sum(dim=(1, 2, 3), keepdim=True) / valid

        var = mean_sq - self.lambda_var * (mean * mean)
        silog = torch.sqrt(torch.clamp(var, min=self.eps))
        return silog.mean()

    def _l1_loss(self, pred, gt, mask):
        valid = mask.sum(dim=(1, 2, 3), keepdim=True).clamp_min(self.eps)
        l1 = (torch.abs(pred - gt) * mask).sum(dim=(1, 2, 3), keepdim=True) / valid
        return l1.mean()

    def _gradient_loss(self, pred, gt, mask):
        pred_gx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_gy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        gt_gx   = gt[:,   :, :, 1:] - gt[:,   :, :, :-1]
        gt_gy   = gt[:,   :, 1:, :] - gt[:,   :, :-1, :]

        mx = mask[:, :, :, 1:] * mask[:, :, :, :-1]
        my = mask[:, :, 1:, :] * mask[:, :, :-1, :]

        vx = mx.sum(dim=(1, 2, 3), keepdim=True).clamp_min(self.eps)
        vy = my.sum(dim=(1, 2, 3), keepdim=True).clamp_min(self.eps)

        lx = (torch.abs(pred_gx - gt_gx) * mx).sum(dim=(1, 2, 3), keepdim=True) / vx
        ly = (torch.abs(pred_gy - gt_gy) * my).sum(dim=(1, 2, 3), keepdim=True) / vy
        return (lx + ly).mean()

    def _ssim_loss(self, pred_n, gt_n, mask):
        """
        SSIM on normalized depths, with log mapping around 1:
          - pred_n/gt_n are already scaled so typical magnitude ~ 1
          - apply log, then map a fixed multiplicative window [1/r, r] to [0,1]
        """
        ssim_mask = _erode_mask(mask, k=self.ssim_window) if self.ssim_erode_mask else mask

        pred_c = torch.clamp(pred_n, min=self.eps)
        gt_c   = torch.clamp(gt_n,   min=self.eps)

        pred_l = torch.log(pred_c)
        gt_l   = torch.log(gt_c)

        if self.ssim_norm_mode == "per_image":
            p_lo, p_hi = self.ssim_percentiles
            pred01_list = []
            gt01_list = []
            for b in range(pred_l.shape[0]):
                mb = ssim_mask[b, 0] > 0.5
                vals = gt_l[b, 0][mb]
                if vals.numel() == 0:
                    min_l = torch.tensor(0.0, device=pred_l.device, dtype=pred_l.dtype)
                    max_l = torch.tensor(1.0, device=pred_l.device, dtype=pred_l.dtype)
                else:
                    min_l = torch.quantile(vals, p_lo / 100.0)
                    max_l = torch.quantile(vals, p_hi / 100.0)
                denom = (max_l - min_l).clamp_min(self.eps)
                pred01_list.append(torch.clamp((pred_l[b:b+1] - min_l) / denom, 0.0, 1.0))
                gt01_list.append(torch.clamp((gt_l[b:b+1] - min_l) / denom, 0.0, 1.0))
            pred01 = torch.cat(pred01_list, dim=0)
            gt01 = torch.cat(gt01_list, dim=0)
        else:
            raise ValueError(f"Unsupported ssim_norm_mode='{self.ssim_norm_mode}'.")

        dist = ssim_distance_map_unit01(pred01, gt01, window=self.ssim_window)

        valid = ssim_mask.sum(dim=(1, 2, 3), keepdim=True).clamp_min(self.eps)
        loss = (dist * ssim_mask).sum(dim=(1, 2, 3), keepdim=True) / valid
        return loss.mean()

    def _l_inf_loss(self, pred, gt, mask):
        diff = torch.abs(pred - gt) * mask
        max_per = diff.flatten(1).amax(dim=1)
        return max_per.mean()
