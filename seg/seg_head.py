from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

def _make_group_norm(num_channels: int, max_groups: int = 32) -> nn.GroupNorm:
    groups = min(max_groups, num_channels)
    while groups > 1 and (num_channels % groups) != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)

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
        align_corners: bool = False,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.out_size = out_size
        self.ppm_bins = tuple(ppm_bins)
        self.align_corners = align_corners

        def norm2d(c: int):
            if norm == "bn":
                return nn.BatchNorm2d(c)
            elif norm == "gn":
                return _make_group_norm(c)
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

    def forward(self, x_tokens: torch.Tensor, *, grid_size=None, out_size=None) -> torch.Tensor:
        B, N, C = x_tokens.shape
        Hp, Wp = grid_size if grid_size is not None else self.grid_size
        assert N == Hp * Wp, f"Expected N={Hp*Wp} tokens, got N={N}"

        # (B, N, C) -> (B, C, Hp, Wp)
        x = x_tokens.transpose(1, 2).contiguous().view(B, C, Hp, Wp)

        x = self.in_proj(x)  # (B, mid, Hp, Wp)

        ppm_outs = [x]
        for bin_sz, branch in zip(self.ppm_bins, self.ppm):
            pooled = F.adaptive_avg_pool2d(x, output_size=(bin_sz, bin_sz))
            pooled = branch(pooled)
            up = F.interpolate(pooled, size=(Hp, Wp), mode="bilinear", align_corners=self.align_corners)
            ppm_outs.append(up)

        x = torch.cat(ppm_outs, dim=1)
        x = self.bottleneck(x)
        logits = self.classifier(x)

        # Upsample to image resolution
        out_size = out_size if out_size is not None else self.out_size
        logits = F.interpolate(logits, size=out_size, mode="bilinear", align_corners=self.align_corners)
        return logits

class UPerNetTokenHead(nn.Module):
    """
    UPerNet-style head for ViT token features (multi-level).

    Expected features order: low-level (highest resolution) -> high-level.
    Each entry can be tokens (B, N, C) or feature maps (B, C, H, W).
    """
    def __init__(
        self,
        embed_dims: Sequence[int],
        num_classes: int,
        *,
        grid_size: Optional[tuple] = None,
        grid_sizes: Optional[Sequence[tuple]] = None,
        out_size: Optional[tuple] = None,
        fpn_channels: int = 256,
        ppm_bins=(1, 2, 3, 6),
        dropout: float = 0.1,
        norm: str = "gn",
        align_corners: bool = False,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.grid_sizes = grid_sizes
        self.out_size = out_size
        self.align_corners = align_corners
        self.ppm_bins = tuple(ppm_bins)

        def norm2d(c: int):
            if norm == "bn":
                return nn.BatchNorm2d(c)
            if norm == "gn":
                return _make_group_norm(c)
            raise ValueError(f"Unknown norm='{norm}', use 'bn' or 'gn'.")

        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_c, fpn_channels, kernel_size=1, bias=False),
                norm2d(fpn_channels),
                nn.ReLU(inplace=True),
            )
            for in_c in embed_dims
        ])
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1, bias=False),
                norm2d(fpn_channels),
                nn.ReLU(inplace=True),
            )
            for _ in embed_dims
        ])

        self.ppm = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_channels, fpn_channels, kernel_size=1, bias=False),
                norm2d(fpn_channels),
                nn.ReLU(inplace=True),
            )
            for _ in self.ppm_bins
        ])
        ppm_in = fpn_channels * (1 + len(self.ppm_bins))
        self.ppm_bottleneck = nn.Sequential(
            nn.Conv2d(ppm_in, fpn_channels, kernel_size=3, padding=1, bias=False),
            norm2d(fpn_channels),
            nn.ReLU(inplace=True),
        )

        fuse_in = fpn_channels * len(embed_dims)
        self.fuse = nn.Sequential(
            nn.Conv2d(fuse_in, fpn_channels, kernel_size=3, padding=1, bias=False),
            norm2d(fpn_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        self.classifier = nn.Conv2d(fpn_channels, num_classes, kernel_size=1)

    def _tokens_to_map(self, x: torch.Tensor, grid_hw: tuple) -> torch.Tensor:
        if x.dim() == 4:
            return x
        B, N, C = x.shape
        Hp, Wp = grid_hw
        assert N == Hp * Wp, f"Expected N={Hp*Wp} tokens, got N={N}"
        return x.transpose(1, 2).contiguous().view(B, C, Hp, Wp)

    def forward(self, features, *, grid_sizes=None, out_size=None) -> torch.Tensor:
        if grid_sizes is None:
            grid_sizes = self.grid_sizes
        maps = []
        for i, feat in enumerate(features):
            if feat.dim() == 4:
                maps.append(feat)
                continue
            if grid_sizes is not None:
                grid_hw = grid_sizes[i]
            elif self.grid_size is not None:
                grid_hw = self.grid_size
            else:
                raise ValueError("grid_size(s) required for token inputs.")
            maps.append(self._tokens_to_map(feat, grid_hw))

        laterals = [conv(m) for conv, m in zip(self.lateral_convs, maps)]

        top = laterals[-1]
        ppm_outs = [top]
        for bin_sz, ppm_conv in zip(self.ppm_bins, self.ppm):
            pooled = F.adaptive_avg_pool2d(top, output_size=(bin_sz, bin_sz))
            pooled = ppm_conv(pooled)
            up = F.interpolate(pooled, size=top.shape[2:], mode="bilinear", align_corners=self.align_corners)
            ppm_outs.append(up)
        laterals[-1] = self.ppm_bottleneck(torch.cat(ppm_outs, dim=1))

        for i in range(len(laterals) - 1, 0, -1):
            up = F.interpolate(laterals[i], size=laterals[i - 1].shape[2:], mode="bilinear",
                               align_corners=self.align_corners)
            laterals[i - 1] = laterals[i - 1] + up

        fpn_outs = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]
        base_size = fpn_outs[0].shape[2:]
        fused = [fpn_outs[0]]
        for feat in fpn_outs[1:]:
            fused.append(F.interpolate(feat, size=base_size, mode="bilinear", align_corners=self.align_corners))
        x = torch.cat(fused, dim=1)
        x = self.fuse(x)
        logits = self.classifier(x)

        out_size = out_size if out_size is not None else self.out_size
        if out_size is not None:
            logits = F.interpolate(logits, size=out_size, mode="bilinear", align_corners=self.align_corners)
        return logits

class FCNSegHead(nn.Module):
    def __init__(self, embed_dim, num_classes, grid_size, out_size,
                 mid_channels=256, dropout=0.1, norm='gn'):
        super().__init__()
        self.grid_size = grid_size
        self.out_size = out_size

        if norm == 'bn':
            Norm2d = nn.BatchNorm2d
        elif norm == 'gn':
            # GN can be more stable for small batch
            Norm2d = _make_group_norm
        else:
            raise ValueError(f"Unknown norm: {norm}")

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, mid_channels, kernel_size=3, padding=1, bias=False),
            Norm2d(mid_channels),
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
        logits = F.interpolate(logits, size=out_size, mode='bilinear', align_corners=False)
        return logits
class LinearSegHead(nn.Module):
    def __init__(self, embed_dim, num_classes, grid_size, out_size, dropout=0.1):
        super().__init__()
        self.grid_size = grid_size  # (H, W) in patches
        self.out_size = out_size    # (img_size, img_size)
        self.dropout = nn.Dropout2d(dropout)
        self.cls = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, x_tokens, *, grid_size=None, out_size=None):
        # x_tokens: (B, N, C), N = H*W
        B, N, C = x_tokens.shape
        H, W = grid_size if grid_size is not None else self.grid_size
        assert N == H * W, f"Token count N={N} != H*W={H*W}"

        x = x_tokens.transpose(1, 2).contiguous().view(B, C, H, W)  # (B, C, H, W)
        x = self.dropout(x)
        logits = self.cls(x)  # (B, num_classes, H, W)
        out_size = out_size if out_size is not None else self.out_size
        logits = F.interpolate(logits, size=out_size, mode='bilinear', align_corners=False)
        return logits

# =================================================================================
# An Improved, Progressive Decoder
# =================================================================================
class ProgressiveSegDecoder(nn.Module):
    """
    A more robust decoder that progressively upsamples features.
    This is a common pattern inspired by architectures like U-Net and FPN.
    """
    def __init__(self, in_channels, num_classes, grid_size):
        super().__init__()
        self.grid_size = grid_size
        
        # The embedding dimension from the ViT
        embed_dim = in_channels
        
        # A series of upsampling blocks
        # Each block consists of Upsample -> Conv -> BatchNorm -> ReLU
        # This allows the model to learn to refine features at increasing resolutions.
        self.decoder = nn.Sequential(
            # First, project the flattened patches into a channel-rich 2D grid
            nn.Conv2d(embed_dim, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # Upsample x2 (e.g., 16x16 -> 28x28)
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Upsample x2 (e.g., 28x28 -> 56x56)
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Upsample x4 to get to a higher resolution (e.g., 56x56 -> 224x224)
            # Another option is to continue with x2 upsampling for more refinement
            nn.Upsample(size=(args.img_size, args.img_size), mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Final 1x1 convolution to map to the number of classes
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # x has shape (B, N, C) where N = (grid_size*grid_size)
        B, N, C = x.shape
        
        # Reshape to a 2D grid: (B, C, H, W)
        x = x.permute(0, 2, 1).view(B, C, self.grid_size, self.grid_size)
        
        # Pass through the progressive decoder
        return self.decoder(x)
