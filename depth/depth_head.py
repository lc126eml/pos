import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union

class DWConvBlock(nn.Module):
    """Depthwise-separable conv block: light but effective."""
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
        fuse_ch: int = 128,     # per-layer projected channels
        dec_ch: int = 128,      # decoder channels
        use_softplus: bool = True,
    ):
        super().__init__()
        self.use_softplus = use_softplus

        # Token-space normalization + projection for each of 4 layers
        self.ln = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(4)])
        self.proj = nn.ModuleList([nn.Linear(embed_dim, fuse_ch) for _ in range(4)])

        # Optional light per-layer spatial mixing after reshape
        self.layer_mix = nn.ModuleList([DWConvBlock(fuse_ch) for _ in range(4)])

        # Fuse 4 maps -> dec_ch
        self.fuse = nn.Sequential(
            nn.Conv2d(4 * fuse_ch, dec_ch, kernel_size=1, bias=False),
            nn.GroupNorm(min(16, dec_ch), dec_ch),
            nn.GELU(),
        )

        # Progressive refinement (two stages is usually enough)
        self.refine1 = DWConvBlock(dec_ch)
        self.refine2 = DWConvBlock(dec_ch)

        self.head = nn.Conv2d(dec_ch, 1, kernel_size=3, padding=1)
        self.softplus = nn.Softplus(beta=1.0, threshold=20.0)

    def _tokens_to_map(self, t, gh, gw, ln, proj):
        # t: (B, N, D) -> (B, C, gh, gw)
        t = ln(t)
        t = proj(t)                      # (B, N, C)
        t = t.permute(0, 2, 1).contiguous().view(t.size(0), -1, gh, gw)
        return t

    def forward(self, feats4, grid_hw=None, out_hw=None):
        """
        feats4: list/tuple of 4 tensors, each (B, N, D) patch tokens only
                e.g. [feat_l3, feat_l6, feat_l9, feat_l12]
        grid_hw: (gh, gw) patch grid. Recommended.
        out_hw:  (H, W) output depth size. Recommended.

        Returns: depth (B, 1, H, W) if out_hw provided, else upsampled by fixed factors.
        """
        assert len(feats4) == 4
        B, Nt, D = feats4[0].shape

        toks = feats4
        N = toks[0].shape[1]
        for t in toks[1:]:
            assert t.shape[1] == N, "All 4 layers must have same number of patch tokens."

        if grid_hw is None:
            gh = int(math.sqrt(N))
            gw = N // gh
            assert gh * gw == N, "Cannot infer grid; please pass grid_hw."
        else:
            gh, gw = grid_hw
            assert gh * gw == N

        maps = []
        for i in range(4):
            m = self._tokens_to_map(toks[i], gh, gw, self.ln[i], self.proj[i])
            m = self.layer_mix[i](m)
            maps.append(m)

        x = torch.cat(maps, dim=1)   # (B, 4*fuse_ch, gh, gw)
        x = self.fuse(x)             # (B, dec_ch, gh, gw)

        # Progressive upsample + refine (lightweight)
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
            # alternative: log-depth output (return logits, apply exp in loss/inference)
            depth = torch.exp(torch.clamp(logits, min=-10, max=10))

        return depth

class SimpleDepthDecoderV2(nn.Module):
    def __init__(self, embed_dim=768, mid_ch=256, out_range=None):
        """
        out_range: optional (min_depth, max_depth) for bounded prediction.
        """
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

        # stable positivity
        self.softplus = nn.Softplus(beta=1.0, threshold=20.0)

    def forward(self, features, grid_hw=None, out_hw=None):
        """
        features: (B, 1+N, D) with CLS at index 0
        grid_hw: (gh, gw) patch grid size (recommended)
        out_hw: (H, W) desired output resolution; if None, will upsample by ~patch grid scale heuristically.
        """
        B, Np1, D = features.shape
        x = features[:, 1:, :]  # (B, N, D)
        N = x.shape[1]

        if grid_hw is None:
            gh = int(math.sqrt(N))
            gw = N // gh
            assert gh * gw == N, f"Cannot infer grid from N={N}; please pass grid_hw."
        else:
            gh, gw = grid_hw
            assert gh * gw == N, f"grid_hw {grid_hw} mismatches N={N}"

        x = x.permute(0, 2, 1).reshape(B, D, gh, gw)
        x = self.in_proj(x)

        # Progressive upsample + refinement (3 stages)
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

        # Positive depth
        depth = self.softplus(logits) + 1e-6

        # Optional bounding (useful if your dataset has a known valid range)
        if self.out_range is not None:
            dmin, dmax = self.out_range
            # map to [dmin, dmax] smoothly
            depth = dmin + (dmax - dmin) * torch.sigmoid(logits)

        return depth

class SimpleDepthDecoderBak(nn.Module):
    def __init__(self, embed_dim=768, patch_size=14, img_size=224):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.grid_h = self.grid_w = img_size // patch_size
        
        self.proj = nn.Conv2d(embed_dim, 64, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(size=(img_size, img_size), mode='bilinear', align_corners=False)
        self.depth_head = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, features):
        B, N_plus_1, D = features.shape
        N = self.grid_h * self.grid_w
        assert N == N_plus_1 - 1, f"Expected {N} patches + CLS, got {N_plus_1}"
        features = features[:, 1:, :]  # Skip CLS token: (B, N, D)
        
        features = features.permute(0, 2, 1).reshape(B, D, self.grid_h, self.grid_w)
        x = self.act(self.proj(features))
        x = self.upsample(x)
        depth = torch.exp(self.depth_head(x))
        # depth = torch.clamp(, min=1e-6, max=100.0)  # Ensure positive, bounded
        return depth
# %%
# =================================================================================
# Step 1: DPT Head Implementation (from reference file)
# =================================================================================

def activate_head(out, activation="inv_log", conf_activation="expp1"):
    """
A compatible activation head function for DPTHead.
    Since output_dim=1, it primarily ensures the depth prediction is positive.
    """
    if out.shape[1] > 1:
        preds = out[:, 0:1, :, :]
        conf = out[:, 1:2, :, :]
    else:
        preds = out
        conf = torch.ones_like(preds)

    # Ensure depth predictions are positive, as loss function uses log
    preds = F.relu(preds) + 1e-6 
    
    return preds, conf


def _make_scratch(in_shape: List[int], out_shape: int, groups: int = 1, expand: bool = False) -> nn.Module:
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

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(
            in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
        )
    return scratch


def _make_fusion_block(features: int, size: int = None, has_residual: bool = True, groups: int = 1) -> nn.Module:
    """A helper function to create a FeatureFusionBlock."""
    return FeatureFusionBlock(
        features,
        nn.ReLU(inplace=True),
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=size,
        has_residual=has_residual,
        groups=groups,
    )


class ResidualConvUnit(nn.Module):
    """Residual convolution module."""
    def __init__(self, features, activation, bn, groups=1):
        super().__init__()
        self.bn = bn
        self.groups = groups
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        self.norm1 = None
        self.norm2 = None
        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        if self.norm1 is not None:
            out = self.norm1(out)
        out = self.activation(out)
        out = self.conv2(out)
        if self.norm2 is not None:
            out = self.norm2(out)
        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""
    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True, size=None, has_residual=True, groups=1):
        super(FeatureFusionBlock, self).__init__()
        self.deconv = deconv
        self.align_corners = align_corners
        self.groups = groups
        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=self.groups)
        if has_residual:
            self.resConfUnit1 = ResidualConvUnit(features, activation, bn, groups=self.groups)
        self.has_residual = has_residual
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn, groups=self.groups)
        self.skip_add = nn.quantized.FloatFunctional()
        self.size = size

    def forward(self, *xs, size=None):
        output = xs[0]
        if self.has_residual:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)
        output = self.resConfUnit2(output)
        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}
        output = custom_interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        output = self.out_conv(output)
        return output


def custom_interpolate(x: torch.Tensor, size: Tuple[int, int] = None, scale_factor: float = None, mode: str = "bilinear", align_corners: bool = True) -> torch.Tensor:
    if size is None:
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))
    INT_MAX = 1610612736
    input_elements = size[0] * size[1] * x.shape[0] * x.shape[1]
    if input_elements > INT_MAX:
        chunks = torch.chunk(x, chunks=(input_elements // INT_MAX) + 1, dim=0)
        interpolated_chunks = [nn.functional.interpolate(chunk, size=size, mode=mode, align_corners=align_corners) for chunk in chunks]
        x = torch.cat(interpolated_chunks, dim=0)
        return x.contiguous()
    else:
        return nn.functional.interpolate(x, size=size, mode=mode, align_corners=align_corners)


class DPTHead(nn.Module):
    def __init__(
        self,
        dim_in: int,
        patch_size: int = 14,
        output_dim: int = 1,
        activation: str = "inv_log",
        conf_activation: str = "expp1",
        features: int = 256,
        out_channels: List[int] = [256, 512, 1024, 1024],
        intermediate_layer_idx: List[int] = [0, 1, 2, 3], # Use indices for the feature list
        pos_embed: bool = False, # Disabled to avoid dependency issues
        feature_only: bool = False,
        down_ratio: int = 1,
    ) -> None:
        super(DPTHead, self).__init__()
        self.patch_size = patch_size
        self.activation = activation
        self.conf_activation = conf_activation
        self.pos_embed = pos_embed
        self.feature_only = feature_only
        self.down_ratio = down_ratio
        self.intermediate_layer_idx = intermediate_layer_idx
        
        self.norm = nn.LayerNorm(dim_in)
        self.projects = nn.ModuleList([nn.Conv2d(in_channels=dim_in, out_channels=oc, kernel_size=1, stride=1, padding=0) for oc in out_channels])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(in_channels=out_channels[0], out_channels=out_channels[0], kernel_size=4, stride=4, padding=0),
            nn.ConvTranspose2d(in_channels=out_channels[1], out_channels=out_channels[1], kernel_size=2, stride=2, padding=0),
            nn.Identity(),
            nn.Conv2d(in_channels=out_channels[3], out_channels=out_channels[3], kernel_size=3, stride=2, padding=1),
        ])
        
        self.scratch = _make_scratch(out_channels, features, expand=False)
        self.scratch.stem_transpose = None
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)
        head_features_1 = features
        head_features_2 = 32
        
        if feature_only:
            self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1, kernel_size=3, stride=1, padding=1)
        else:
            self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
            conv2_in_channels = head_features_1 // 2
            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(conv2_in_channels, head_features_2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_features_2, output_dim, kernel_size=1, stride=1, padding=0),
            )

    def forward(self, features: List[torch.Tensor], images: torch.Tensor, patch_start_idx: int, frames_chunk_size: int = 8) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, S, _, H, W = images.shape
        
        if frames_chunk_size is None or frames_chunk_size >= S:
            return self._forward_impl(features, images, patch_start_idx)
            
        assert frames_chunk_size > 0
        all_preds, all_conf = [], []
        for frames_start_idx in range(0, S, frames_chunk_size):
            frames_end_idx = min(frames_start_idx + frames_chunk_size, S)
            if self.feature_only:
                chunk_output = self._forward_impl(features, images, patch_start_idx, frames_start_idx, frames_end_idx)
                all_preds.append(chunk_output)
            else:
                chunk_preds, chunk_conf = self._forward_impl(features, images, patch_start_idx, frames_start_idx, frames_end_idx)
                all_preds.append(chunk_preds)
                all_conf.append(chunk_conf)
        if self.feature_only:
            return torch.cat(all_preds, dim=1)
        else:
            return torch.cat(all_preds, dim=1), torch.cat(all_conf, dim=1)

    def _forward_impl(self, features: List[torch.Tensor], images: torch.Tensor, patch_start_idx: int, frames_start_idx: int = None, frames_end_idx: int = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if frames_start_idx is not None and frames_end_idx is not None:
            images = images[:, frames_start_idx:frames_end_idx].contiguous()
            
        B, S, _, H, W = images.shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size
        out = []
        dpt_idx = 0
        
        # The `features` is the list of features from the backbone's intermediate layers
        # We iterate through the indices [0, 1, 2, 3] to get the 4 feature maps
        for layer_idx in self.intermediate_layer_idx:
            x = features[layer_idx][:, patch_start_idx:] # Use features directly, remove CLS/pose tokens
            x = x.unsqueeze(1) # Add sequence dimension for compatibility
            if frames_start_idx is not None and frames_end_idx is not None:
                x = x[:, frames_start_idx:frames_end_idx]
                
            x = x.view(B * S, -1, x.shape[-1])
            x = self.norm(x)
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            x = self.projects[dpt_idx](x)
            if self.pos_embed:
                x = self._apply_pos_embed(x, W, H)
            x = self.resize_layers[dpt_idx](x)
            out.append(x)
            dpt_idx += 1
            
        out = self.scratch_forward(out)
        out = custom_interpolate(out, (int(patch_h * self.patch_size / self.down_ratio), int(patch_w * self.patch_size / self.down_ratio)), mode="bilinear", align_corners=True)
        if self.pos_embed:
            out = self._apply_pos_embed(out, W, H)
        if self.feature_only:
            return out.view(B, S, *out.shape[1:])
            
        out = self.scratch.output_conv2(out)
        preds, conf = activate_head(out, activation=self.activation, conf_activation=self.conf_activation)
        preds = preds.view(B, S, *preds.shape[1:])
        conf = conf.view(B, S, *conf.shape[1:])
        return preds, conf

    def _apply_pos_embed(self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
        # This method is not used if pos_embed is False
        pass

    def scratch_forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        layer_1, layer_2, layer_3, layer_4 = features
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        out = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        del layer_4_rn, layer_4
        out = self.scratch.refinenet3(out, layer_3_rn, size=layer_2_rn.shape[2:])
        del layer_3_rn, layer_3
        out = self.scratch.refinenet2(out, layer_2_rn, size=layer_1_rn.shape[2:])
        del layer_2_rn, layer_2
        out = self.scratch.refinenet1(out, layer_1_rn)
        del layer_1_rn, layer_1
        out = self.scratch.output_conv1(out)
        return out
