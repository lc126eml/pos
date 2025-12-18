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
            Norm2d = lambda c: nn.GroupNorm(32, c)
        else:
            raise ValueError(f"Unknown norm: {norm}")

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, mid_channels, kernel_size=3, padding=1, bias=False),
            Norm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(mid_channels, num_classes, kernel_size=1),
        )

    def forward(self, x_tokens):
        B, N, C = x_tokens.shape
        H, W = self.grid_size
        assert N == H * W, f"Token count N={N} != H*W={H*W}"

        x = x_tokens.transpose(1, 2).contiguous().view(B, C, H, W)
        logits = self.proj(x)
        logits = F.interpolate(logits, size=self.out_size, mode='bilinear', align_corners=False)
        return logits
class LinearSegHead(nn.Module):
    def __init__(self, embed_dim, num_classes, grid_size, out_size, dropout=0.1):
        super().__init__()
        self.grid_size = grid_size  # (H, W) in patches
        self.out_size = out_size    # (img_size, img_size)
        self.dropout = nn.Dropout2d(dropout)
        self.cls = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, x_tokens):
        # x_tokens: (B, N, C), N = H*W
        B, N, C = x_tokens.shape
        H, W = self.grid_size
        assert N == H * W, f"Token count N={N} != H*W={H*W}"

        x = x_tokens.transpose(1, 2).contiguous().view(B, C, H, W)  # (B, C, H, W)
        x = self.dropout(x)
        logits = self.cls(x)  # (B, num_classes, H, W)
        logits = F.interpolate(logits, size=self.out_size, mode='bilinear', align_corners=False)
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