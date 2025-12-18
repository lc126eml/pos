import torch
import torch.nn as nn
import torch.nn.functional as F

class PPMliteFCNHead(nn.Module):
    """
    Lightweight FCN + PPM head for ViT token features.

    Input:
      x_tokens: (B, N, C) where N = H*W (patch grid)
    Output:
      logits: (B, num_classes, out_h, out_w)
    """
    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        grid_size: tuple,          # (H_patches, W_patches)
        out_size: tuple,           # (H_img, W_img)
        mid_channels: int = 256,
        ppm_bins=(1, 2, 3),        # "lite": fewer bins than (1,2,3,6)
        ppm_channels: int = 64,    # channels per PPM branch
        dropout: float = 0.1,
        norm: str = "gn",          # "gn" recommended unless per-GPU batch is large
    ):
        super().__init__()
        self.grid_size = grid_size
        self.out_size = out_size
        self.ppm_bins = tuple(ppm_bins)

        def norm2d(c: int):
            if norm == "bn":
                return nn.BatchNorm2d(c)
            elif norm == "gn":
                # 32 groups is a common default; clamp to valid range
                g = 32 if c >= 32 else max(1, c // 4)
                return nn.GroupNorm(g, c)
            else:
                raise ValueError(f"Unknown norm='{norm}', use 'bn' or 'gn'.")

        # Project ViT embed_dim -> mid_channels (FCN-style)
        self.in_proj = nn.Sequential(
            nn.Conv2d(embed_dim, mid_channels, kernel_size=1, bias=False),
            norm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        # PPM branches (adaptive pooling -> 1x1 conv -> upsample back)
        self.ppm = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(mid_channels, ppm_channels, kernel_size=1, bias=False),
                norm2d(ppm_channels),
                nn.ReLU(inplace=True),
            )
            for _ in self.ppm_bins
        ])

        # Fuse (mid + sum(branches)) via a bottleneck conv
        in_fuse = mid_channels + len(self.ppm_bins) * ppm_channels
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_fuse, mid_channels, kernel_size=3, padding=1, bias=False),
            norm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

        self.classifier = nn.Conv2d(mid_channels, num_classes, kernel_size=1)

    def forward(self, x_tokens: torch.Tensor) -> torch.Tensor:
        B, N, C = x_tokens.shape
        Hp, Wp = self.grid_size
        assert N == Hp * Wp, f"Expected N={Hp*Wp} tokens, got N={N}"

        # (B, N, C) -> (B, C, Hp, Wp)
        x = x_tokens.transpose(1, 2).contiguous().view(B, C, Hp, Wp)

        x = self.in_proj(x)  # (B, mid, Hp, Wp)

        ppm_outs = [x]
        for bin_sz, branch in zip(self.ppm_bins, self.ppm):
            pooled = F.adaptive_avg_pool2d(x, output_size=(bin_sz, bin_sz))
            pooled = branch(pooled)
            up = F.interpolate(pooled, size=(Hp, Wp), mode="bilinear", align_corners=False)
            ppm_outs.append(up)

        x = torch.cat(ppm_outs, dim=1)
        x = self.bottleneck(x)
        logits = self.classifier(x)

        # Upsample to image resolution
        logits = F.interpolate(logits, size=self.out_size, mode="bilinear", align_corners=False)
        return logits


class MMSegCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=-1, loss_weight=1.0, avg_non_ignore=True):
        super().__init__()
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight
        self.avg_non_ignore = avg_non_ignore

    def forward(self, logits, target):
        # logits: (N, C, H, W), target: (N, H, W)
        if not self.avg_non_ignore:
            return self.loss_weight * F.cross_entropy(
                logits, target, ignore_index=self.ignore_index, reduction='mean'
            )

        # reduction='none' then average only over valid pixels
        loss = F.cross_entropy(
            logits, target, ignore_index=self.ignore_index, reduction='none'
        )  # (N, H, W)
        valid = (target != self.ignore_index)
        denom = valid.sum().clamp_min(1).to(loss.dtype)
        return self.loss_weight * (loss[valid].sum() / denom)
        
class MMSegDiceLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=-1, smooth=1.0, loss_weight=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.loss_weight = loss_weight

    def forward(self, logits, target):
        # logits: (N, C, H, W), target: (N, H, W)
        N, C, H, W = logits.shape
        prob = F.softmax(logits, dim=1)

        valid = (target != self.ignore_index)
        target_clamped = target.clamp_min(0)  # safe for ignore pixels

        # Flatten
        prob = prob.view(N, C, -1)                 # (N, C, HW)
        target_flat = target_clamped.view(N, -1)   # (N, HW)
        valid_flat = valid.view(N, -1).float()     # (N, HW)

        total = 0.0
        count = 0

        for c in range(C):
            p = prob[:, c, :] * valid_flat
            t = (target_flat == c).float() * valid_flat

            inter = (p * t).sum(dim=1)
            union = p.sum(dim=1) + t.sum(dim=1)

            dice = (2.0 * inter + self.smooth) / (union + self.smooth)
            total += (1.0 - dice).mean()
            count += 1

        return self.loss_weight * (total / max(count, 1))
