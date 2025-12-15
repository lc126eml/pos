""" Relative Position Vision Transformer (ViT) in PyTorch

NOTE: these models are experimental / WIP, expect changes

Hacked together by / Copyright 2022, Ross Wightman
"""
import logging
import math
from functools import partial
from typing import List, Optional, Tuple, Type, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch
import torch.nn as nn
from torch.jit import Final

from timm.data import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.layers import (
    PatchEmbed,
    Mlp,
    DropPath,
    RelPosMlp,
    RelPosBias,
    use_fused_attn,
    LayerType,
    AttentionRope,
    create_rope_embed,
)
from timm.models._builder import build_model_with_cfg
from timm.models._features import feature_take_indices
from timm.models._manipulate import named_apply, checkpoint
from timm.models._registry import generate_default_cfgs, register_model
from timm.models.vision_transformer import get_init_weights_vit

__all__ = ['VisionTransformerRope']  # model_registry will add each entrypoint fn to this

_logger = logging.getLogger(__name__)




class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class RopeBlock(nn.Module):
    """Transformer block using AttentionRope from timm with RoPE embeddings."""

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_norm=False,
            num_prefix_tokens=0,
            init_values=None,
            proj_drop=0.,
            attn_drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            rotate_half: bool = False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        # Use AttentionRope from timm.layers (same implementation as eva.py)
        self.attn = AttentionRope(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qkv_fused=True,
            num_prefix_tokens=num_prefix_tokens,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            attn_head_dim=None,
            norm_layer=norm_layer,
            qk_norm=qk_norm,
            scale_norm=False,
            proj_bias=True,
            rotate_half=rotate_half,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(
            self,
            x,
            freqs_cis: Optional[torch.Tensor] = None,
            rope: Optional[torch.Tensor] = None,
    ):
        # Keep freqs_cis arg for backward-compatibility; treat it as rope if rope is not given
        rope = rope if rope is not None else freqs_cis
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), rope=rope)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class ResPostRopeBlock(nn.Module):
    """Post-norm variant using AttentionRope."""

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_norm=False,
            num_prefix_tokens=0,
            init_values=None,
            proj_drop=0.,
            attn_drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            rotate_half: bool = False,
    ):
        super().__init__()
        self.init_values = init_values

        self.attn = AttentionRope(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qkv_fused=True,
            num_prefix_tokens=num_prefix_tokens,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            attn_head_dim=None,
            norm_layer=norm_layer,
            qk_norm=qk_norm,
            scale_norm=False,
            proj_bias=True,
            rotate_half=rotate_half,
        )
        self.norm1 = norm_layer(dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.norm2 = norm_layer(dim)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.init_weights()

    def init_weights(self):
        # NOTE this init overrides that base model init with specific changes for the block type
        if self.init_values is not None:
            nn.init.constant_(self.norm1.weight, self.init_values)
            nn.init.constant_(self.norm2.weight, self.init_values)

    def forward(
            self,
            x,
            freqs_cis: Optional[torch.Tensor] = None,
            rope: Optional[torch.Tensor] = None,
    ):
        rope = rope if rope is not None else freqs_cis
        x = x + self.drop_path1(self.norm1(self.attn(x, rope=rope)))
        x = x + self.drop_path2(self.norm2(self.mlp(x)))
        return x


class VisionTransformerRope(nn.Module):
    """ Vision Transformer w/ Rotary Position Embedding

    Differing from classic vit, this impl
      * uses rotary position embedding
      * defaults to no class token (can be enabled)
      * defaults to global avg pool for head (can be changed)
      * layer-scale (residual branch gain) enabled

    RoPE implementation is now aligned with eva.py, using create_rope_embed + AttentionRope.
    """

    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: Literal['', 'avg', 'token', 'map'] = 'avg',
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            init_values: Optional[float] = 1.0e-5,
            class_token: bool = False,
            fc_norm: bool = False,
            rope_theta: float = 100.0,
            # rope_temperature: float = 100.,
            rope_type: Optional[str] = 'dinov3',
            rope_rotate_half: bool = True,
            drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            weight_init: Literal['skip', 'jax', 'moco', ''] = 'skip',
            fix_init: bool = False,
            embed_layer: Type[nn.Module] = PatchEmbed,
            norm_layer: Optional[LayerType] = None,
            act_layer: Optional[LayerType] = None,
            block_fn: Type[nn.Module] = RopeBlock,
    ):
        """
        Args:
            img_size: input image size
            patch_size: patch size
            in_chans: number of input channels
            num_classes: number of classes for classification head
            global_pool: type of global pooling for final sequence (default: 'avg')
            embed_dim: embedding dimension
            depth: depth of transformer
            num_heads: number of attention heads
            mlp_ratio: ratio of mlp hidden dim to embedding dim
            qkv_bias: enable bias for qkv if True
            qk_norm: Enable normalization of query and key in attention
            init_values: layer-scale init values
            class_token: use class token (default: False)
            fc_norm: use pre classifier norm instead of pre-pool
            rope_theta: theta / temperature for rotary position embedding
            drop_rate: dropout rate
            proj_drop_rate: projection dropout rate
            attn_drop_rate: attention dropout rate
            drop_path_rate: stochastic depth rate
            weight_init: weight init scheme
            fix_init: apply weight initialization fix (scaling w/ layer index)
            embed_layer: patch embedding layer
            norm_layer: normalization layer
            act_layer: MLP activation layer
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'token')
        assert class_token or global_pool != 'token'
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.head_hidden_size = self.embed_dim = embed_dim  # for consistency with other models
        self.num_heads = num_heads
        self.num_prefix_tokens = 1 if class_token else 0
        self.grad_checkpointing = False

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        feat_size = self.patch_embed.grid_size
        r = self.patch_embed.feat_ratio() if hasattr(self.patch_embed, 'feat_ratio') else patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, self.num_prefix_tokens, embed_dim)) if class_token else None

        # New RoPE setup (aligned with eva.py)
        self.rope_theta = rope_theta
        self.use_rot_pos_emb = True
        if self.use_rot_pos_emb:
            rope_kwargs = dict(
                dim=embed_dim,
                num_heads=num_heads,
                feat_shape=self.patch_embed.grid_size,   # (H_patches, W_patches)
                temperature=self.rope_theta,
                grid_indexing='ij',
            )
            # Use 'cat' RoPE type (as in EVA for 2D ViT / DINO-style models)
            self.rope = create_rope_embed(rope_type=rope_type, **rope_kwargs)
        else:
            self.rope = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                num_prefix_tokens=self.num_prefix_tokens,
                init_values=init_values,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                rotate_half=rope_rotate_half,
            )
            for i in range(depth)])
        self.feature_info = [
            dict(module=f'blocks.{i}', num_chs=embed_dim, reduction=r) for i in range(depth)]
        self.norm = norm_layer(embed_dim) if not fc_norm else nn.Identity()

        # Classifier Head
        self.fc_norm = norm_layer(embed_dim) if fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if weight_init != 'skip':
            self.init_weights(weight_init)
        if fix_init:
            self.fix_init_weight()

    def init_weights(self, mode=''):
        assert mode in ('jax', 'moco', '')
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(get_init_weights_vit(mode), self)

    def fix_init_weight(self):
        def rescale(param, _layer_id):
            param.div_(math.sqrt(2.0 * _layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    @torch.jit.ignore
    def no_weight_decay(self):
        nwd = {'cls_token'}
        # Optionally mark RoPE parameters as no-weight-decay, similar to eva.py
        if getattr(self, "rope", None) is not None and hasattr(self.rope, "no_weight_decay"):
            nwd |= {f"rope.{p}" for p in self.rope.no_weight_decay()}
        return nwd

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^cls_token|patch_embed',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'token')
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _get_rope_embed(self, grid_size: Tuple[int, int], device: torch.device):
        """Helper to obtain RoPE tensor for a given feature grid_size."""
        if self.rope is None or not self.use_rot_pos_emb:
            return None
        # For variable resolutions, update RoPE feature shape if supported
        if hasattr(self.rope, "update_feat_shape"):
            self.rope.update_feat_shape(grid_size)
        rope = self.rope.get_embed()
        return rope.to(device=device)

    def forward_intermediates(
            self,
            x: torch.Tensor,
            indices: Optional[Union[int, List[int]]] = None,
            return_prefix_tokens: bool = False,
            norm: bool = False,
            stop_early: bool = False,
            output_fmt: str = 'NCHW',
            intermediates_only: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        """ Forward features that returns intermediates.

        Args:
            x: Input image tensor
            indices: Take last n blocks if int, all if None, select matching indices if sequence
            return_prefix_tokens: Return both prefix and spatial intermediate tokens
            norm: Apply norm layer to all intermediates
            stop_early: Stop iterating over blocks when last desired intermediate hit
            output_fmt: Shape of intermediate feature outputs
            intermediates_only: Only return intermediate features
        Returns:

        """
        assert output_fmt in ('NCHW', 'NLC'), 'Output format must be one of NCHW or NLC.'
        reshape = output_fmt == 'NCHW'
        intermediates = []
        take_indices, max_index = feature_take_indices(len(self.blocks), indices)

        # forward pass
        B, C, height, width = x.shape
        x = self.patch_embed(x)
        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        current_grid_size = (height // self.patch_embed.patch_size[0], width // self.patch_embed.patch_size[1])
        rope = self._get_rope_embed(current_grid_size, x.device)

        if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
            blocks = self.blocks
        else:
            blocks = self.blocks[:max_index + 1]
        for i, blk in enumerate(blocks):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x, rope=rope)
            else:
                x = blk(x, rope=rope)
            if i in take_indices:
                intermediates.append(self.norm(x) if norm else x)

        # process intermediates
        if self.num_prefix_tokens:
            prefix_tokens = [y[:, 0:self.num_prefix_tokens] for y in intermediates]
            intermediates = [y[:, self.num_prefix_tokens:] for y in intermediates]
        if reshape:
            H, W = (height // self.patch_embed.patch_size[0], width // self.patch_embed.patch_size[1])
            intermediates = [y.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for y in intermediates]
        if not torch.jit.is_scripting() and return_prefix_tokens:
            intermediates = list(zip(intermediates, prefix_tokens))

        if intermediates_only:
            return intermediates

        x = self.norm(x)

        return x, intermediates

    def prune_intermediate_layers(
            self,
            indices: Union[int, List[int]] = 1,
            prune_norm: bool = False,
            prune_head: bool = True,
    ):
        """ Prune layers not required for specified intermediates.
        """
        take_indices, max_index = feature_take_indices(len(self.blocks), indices)
        self.blocks = self.blocks[:max_index + 1]  # truncate blocks
        if prune_norm:
            self.norm = nn.Identity()
        if prune_head:
            self.fc_norm = nn.Identity()
            self.reset_classifier(0, '')
        return take_indices

    def forward_features(self, x):
        B, C, height, width = x.shape
        x = self.patch_embed(x)
        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        current_grid_size = (height // self.patch_embed.patch_size[0], width // self.patch_embed.patch_size[1])
        rope = self._get_rope_embed(current_grid_size, x.device)

        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x, rope=rope)
            else:
                x = blk(x, rope=rope)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _create_vision_transformer_rope(variant, pretrained=False, **kwargs):
    out_indices = kwargs.pop('out_indices', 3)
    model = build_model_with_cfg(
        VisionTransformerRope, variant, pretrained,
        feature_cfg=dict(out_indices=out_indices, feature_cls='getter'),
        **kwargs,
    )
    return model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    'vit_rope_base_patch32_plus_rpn_256.sw_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/vit_replos_base_patch32_plus_rpn_256-sw-dd486f51.pth',
        hf_hub_id='timm/',
        input_size=(3, 256, 256)),
    'vit_rope_base_patch16_plus_240.untrained': _cfg(url='', input_size=(3, 240, 240)),

    'vit_rope_small_patch16_224.sw_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/vit_relpos_small_patch16_224-sw-ec2778b4.pth',
        hf_hub_id='timm/'),
    'vit_rope_small_patch14_224.untrained': _cfg(),
    'vit_rope_small_patch14_dinov2.untrained': _cfg(input_size=(3, 518, 518), crop_pct=1.0),
    'vit_rope_base_patch14_dinov2.untrained': _cfg(input_size=(3, 518, 518), crop_pct=1.0),
    'vit_rope_large_patch14_dinov2.untrained': _cfg(input_size=(3, 518, 518), crop_pct=1.0),
    'vit_rope_medium_patch16_224.sw_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/vit_relpos_medium_patch16_224-sw-11c174af.pth',
        hf_hub_id='timm/'),
    'vit_rope_base_patch16_224.sw_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/vit_relpos_base_patch16_224-sw-49049aed.pth',
        hf_hub_id='timm/'),

    'vit_srope_small_patch16_224.sw_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/vit_srelpos_small_patch16_224-sw-6cdb8849.pth',
        hf_hub_id='timm/'),
    'vit_srope_medium_patch16_224.sw_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/vit_srelpos_medium_patch16_224-sw-ad702b8c.pth',
        hf_hub_id='timm/'),

    'vit_rope_medium_patch16_cls_224.sw_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/vit_relpos_medium_patch16_cls_224-sw-cfe8e259.pth',
        hf_hub_id='timm/'),
    'vit_rope_base_patch16_cls_224.untrained': _cfg(),
    'vit_rope_base_patch16_clsgap_224.sw_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/vit_relpos_base_patch16_gapcls_224-sw-1a341d6c.pth',
        hf_hub_id='timm/'),

    'vit_rope_small_patch16_rpn_224.untrained': _cfg(),
    'vit_rope_medium_patch16_rpn_224.sw_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/vit_relpos_medium_patch16_rpn_224-sw-5d2befd8.pth',
        hf_hub_id='timm/'),
    'vit_rope_base_patch16_rpn_224.untrained': _cfg(),
})


@register_model
def vit_rope_base_patch32_plus_rpn_256(pretrained=False, **kwargs) -> VisionTransformerRope:
    """ ViT-Base (ViT-B/32+) w/ rotary position embedding and residual post-norm, no class token
    """
    model_args = dict(patch_size=32, embed_dim=896, depth=12, num_heads=14, block_fn=ResPostRopeBlock)
    model = _create_vision_transformer_rope(
        'vit_rope_base_patch32_plus_rpn_256', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_rope_base_patch16_plus_240(pretrained=False, **kwargs) -> VisionTransformerRope:
    """ ViT-Base (ViT-B/16+) w/ rotary position embedding, no class token
    """
    model_args = dict(patch_size=16, embed_dim=896, depth=12, num_heads=14)
    model = _create_vision_transformer_rope(
        'vit_rope_base_patch16_plus_240', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_rope_small_patch16_224(pretrained=False, **kwargs) -> VisionTransformerRope:
    """ ViT-Base (ViT-B/16) w/ rotary position embedding, no class token
    """
    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, qkv_bias=False, fc_norm=True)
    model = _create_vision_transformer_rope(
        'vit_rope_small_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_rope_small_patch14_dinov2(pretrained=False, patch_size=14, **kwargs) -> VisionTransformerRope:
    """ ViT-S/14 DINOv2 counterpart with RoPE pos embed
    """
    model_args = dict(patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, init_values=1e-5, class_token=True, **kwargs)
    model = _create_vision_transformer_rope(
        'vit_rope_small_patch14_dinov2', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_rope_base_patch14_dinov2(pretrained=False, patch_size=14, **kwargs) -> VisionTransformerRope:
    """ ViT-B/14 DINOv2 counterpart with RoPE pos embed
    """
    model_args = dict(patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, init_values=1e-5, class_token=True, **kwargs)
    model = _create_vision_transformer_rope(
        'vit_rope_base_patch14_dinov2', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_rope_large_patch14_dinov2(pretrained=False, patch_size=14, **kwargs) -> VisionTransformerRope:
    """ ViT-L/14 DINOv2 counterpart with RoPE pos embed
    """
    model_args = dict(patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, init_values=1e-5, class_token=True, **kwargs)
    model = _create_vision_transformer_rope(
        'vit_rope_large_patch14_dinov2', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_rope_medium_patch16_224(pretrained=False, **kwargs) -> VisionTransformerRope:
    """ ViT-Base (ViT-B/16) w/ rotary position embedding, no class token
    """
    model_args = dict(
        patch_size=16, embed_dim=512, depth=12, num_heads=8, qkv_bias=False, fc_norm=True)
    model = _create_vision_transformer_rope(
        'vit_rope_medium_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_rope_base_patch16_224(pretrained=False, **kwargs) -> VisionTransformerRope:
    """ ViT-Base (ViT-B/16) w/ rotary position embedding, no class token
    """
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=False, fc_norm=True)
    model = _create_vision_transformer_rope(
        'vit_rope_base_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_srope_small_patch16_224(pretrained=False, **kwargs) -> VisionTransformerRope:
    """ ViT-Base (ViT-B/16) w/ shared rotary position embedding, no class token
    """
    model_args = dict(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, qkv_bias=False, fc_norm=False)
    model = _create_vision_transformer_rope(
        'vit_srope_small_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_srope_medium_patch16_224(pretrained=False, **kwargs) -> VisionTransformerRope:
    """ ViT-Base (ViT-B/16) w/ shared rotary position embedding, no class token
    """
    model_args = dict(
        patch_size=16, embed_dim=512, depth=12, num_heads=8, qkv_bias=False, fc_norm=False)
    model = _create_vision_transformer_rope(
        'vit_srope_medium_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_rope_medium_patch16_cls_224(pretrained=False, **kwargs) -> VisionTransformerRope:
    """ ViT-Base (ViT-M/16) w/ rotary position embedding, class token present
    """
    model_args = dict(
        patch_size=16, embed_dim=512, depth=12, num_heads=8, qkv_bias=False, fc_norm=False,
        class_token=True, global_pool='token')
    model = _create_vision_transformer_rope(
        'vit_rope_medium_patch16_cls_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_rope_base_patch16_cls_224(pretrained=False, **kwargs) -> VisionTransformerRope:
    """ ViT-Base (ViT-B/16) w/ rotary position embedding, class token present
    """
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=False, class_token=True, global_pool='token')
    model = _create_vision_transformer_rope(
        'vit_rope_base_patch16_cls_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_rope_base_patch16_clsgap_224(pretrained=False, **kwargs) -> VisionTransformerRope:
    """ ViT-Base (ViT-B/16) w/ rotary position embedding, class token present
    NOTE this config is a bit of a mistake, class token was enabled but global avg-pool w/ fc-norm was not disabled
    Leaving here for comparisons w/ a future re-train as it performs quite well.
    """
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=False, fc_norm=True, class_token=True)
    model = _create_vision_transformer_rope(
        'vit_rope_base_patch16_clsgap_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_rope_small_patch16_rpn_224(pretrained=False, **kwargs) -> VisionTransformerRope:
    """ ViT-Base (ViT-B/16) w/ rotary position embedding and residual post-norm, no class token
    """
    model_args = dict(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, qkv_bias=False, block_fn=ResPostRopeBlock)
    model = _create_vision_transformer_rope(
        'vit_rope_small_patch16_rpn_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_rope_medium_patch16_rpn_224(pretrained=False, **kwargs) -> VisionTransformerRope:
    """ ViT-Base (ViT-B/16) w/ rotary position embedding and residual post-norm, no class token
    """
    model_args = dict(
        patch_size=16, embed_dim=512, depth=12, num_heads=8, qkv_bias=False, block_fn=ResPostRopeBlock)
    model = _create_vision_transformer_rope(
        'vit_rope_medium_patch16_rpn_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model
