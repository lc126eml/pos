""" Vision Transformer (ViT) with 2D Sinusoidal Position Embedding in PyTorch

Based on timm code by Ross Wightman
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
import torch.nn.functional as F
import numpy as np

from timm.data import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.layers import PatchEmbed, Mlp, DropPath, LayerType
from timm.models._builder import build_model_with_cfg
from timm.models._features import feature_take_indices
from timm.models._manipulate import named_apply, checkpoint
from timm.models._registry import register_model
from timm.models.vision_transformer import get_init_weights_vit, Attention, Block

__all__ = ['VisionTransformerSin']

_logger = logging.getLogger(__name__)


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    Create 2D sin/cos positional embeddings.
    Args:
        embed_dim: embedding dimension
        grid_size: int of the grid height and width
        cls_token: bool whether to add a position for the class token
    Returns:
        pos_embed: (grid_size*grid_size, embed_dim) or (1+grid_size*grid_size, embed_dim) (w/ cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    return np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    return np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)


class VisionTransformerSin(nn.Module):
    """ Vision Transformer w/ 2D Sinusoidal Position Embedding
    """

    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: Literal['', 'avg', 'token'] = 'token',
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            init_values: Optional[float] = None,
            class_token: bool = True,
            fc_norm: bool = False,
            drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            weight_init: Literal['skip', 'jax', 'moco', ''] = 'skip',
            fix_init: bool = False,
            embed_layer: Type[nn.Module] = PatchEmbed,
            norm_layer: Optional[LayerType] = None,
            act_layer: Optional[LayerType] = None,
            block_fn: Type[nn.Module] = Block
    ):
        super().__init__()
        assert global_pool in ('', 'avg', 'token')
        assert class_token or global_pool != 'token'
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim
        self.num_prefix_tokens = 1 if class_token else 0
        self.grad_checkpointing = False

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        grid_size = self.patch_embed.grid_size
        r = self.patch_embed.feat_ratio() if hasattr(self.patch_embed, 'feat_ratio') else patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, self.num_prefix_tokens, embed_dim)) if class_token else None

        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + self.num_prefix_tokens, embed_dim), requires_grad=False)
        pos_embed_data = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], grid_size[0], cls_token=class_token)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed_data).float().unsqueeze(0))

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                init_values=init_values,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
            )
            for i in range(depth)])
        self.feature_info = [
            dict(module=f'blocks.{i}', num_chs=embed_dim, reduction=r) for i in range(depth)]
        self.norm = norm_layer(embed_dim) if not fc_norm else nn.Identity()

        self.fc_norm = norm_layer(embed_dim) if fc_norm else nn.Identity()
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if weight_init != 'skip':
            self.init_weights(weight_init)
        if fix_init:
            self.fix_init_weight()

    def init_weights(self, mode=''):
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(get_init_weights_vit(mode), self)

    def fix_init_weight(self):
        def rescale(param, _layer_id):
            param.div_(math.sqrt(2.0 * _layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _pos_embed(self, x):
        pos_embed = self.pos_embed

        if x.shape[1] != pos_embed.shape[1]:
            # Positional embedding interpolation for different image sizes
            num_patches = x.shape[1] - self.num_prefix_tokens
            if self.num_prefix_tokens:
                cls_pos_embed = pos_embed[:, :self.num_prefix_tokens]
                patch_pos_embed = pos_embed[:, self.num_prefix_tokens:]
            else:
                cls_pos_embed = None
                patch_pos_embed = pos_embed

            gs_old = self.patch_embed.grid_size
            gs_new = int(math.sqrt(num_patches))
            _logger.info(f'Resizing position embedding from {gs_old} to {gs_new}')
            patch_pos_embed = patch_pos_embed.reshape(1, gs_old[0], gs_old[1], -1).permute(0, 3, 1, 2)
            patch_pos_embed = F.interpolate(patch_pos_embed, size=(gs_new, gs_new), mode='bicubic', align_corners=False)
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, self.embed_dim)
            
            if cls_pos_embed is not None:
                pos_embed = torch.cat((cls_pos_embed, patch_pos_embed), dim=1)
            else:
                pos_embed = patch_pos_embed

        return x + pos_embed.to(x.device)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^cls_token|patch_embed',
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
        assert output_fmt in ('NCHW', 'NLC'), 'Output format must be one of NCHW or NLC.'
        reshape = output_fmt == 'NCHW'
        intermediates = []
        take_indices, max_index = feature_take_indices(len(self.blocks), indices)

        x = self.patch_embed(x)
        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self._pos_embed(x)
        x = self.pos_drop(x)

        if torch.jit.is_scripting() or not stop_early:
            blocks = self.blocks
        else:
            blocks = self.blocks[:max_index + 1]
        for i, blk in enumerate(blocks):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x)
            else:
                x = blk(x)
            if i in take_indices:
                intermediates.append(self.norm(x) if norm else x)

        if self.num_prefix_tokens:
            prefix_tokens = [y[:, 0:self.num_prefix_tokens] for y in intermediates]
            intermediates = [y[:, self.num_prefix_tokens:] for y in intermediates]
        if reshape:
            B = x.shape[0]
            H, W = self.patch_embed.grid_size
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
        take_indices, max_index = feature_take_indices(len(self.blocks), indices)
        self.blocks = self.blocks[:max_index + 1]
        if prune_norm:
            self.norm = nn.Identity()
        if prune_head:
            self.fc_norm = nn.Identity()
            self.reset_classifier(0, '')
        return take_indices

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self._pos_embed(x)
        x = self.pos_drop(x)
        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x)
            else:
                x = blk(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool == 'avg':
            x = x[:, self.num_prefix_tokens:].mean(dim=1)
        elif self.global_pool == 'token':
            x = x[:, 0]
        
        x = self.fc_norm(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

def _create_vision_transformer_sin(variant, pretrained=False, **kwargs):
    model = build_model_with_cfg(
        VisionTransformerSin, variant, pretrained,
        feature_cfg=dict(out_indices=(3, 5, 7, 11), feature_cls='getter'),
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

default_cfgs = {
    'vit_sin_small_patch16_224': _cfg(),
    'vit_sin_base_patch16_224': _cfg(),
    'vit_sin_small_patch14_dinov2': _cfg(input_size=(3, 518, 518), crop_pct=1.0),
    'vit_sin_base_patch14_dinov2': _cfg(input_size=(3, 518, 518), crop_pct=1.0),
    'vit_sin_large_patch14_dinov2': _cfg(input_size=(3, 518, 518), crop_pct=1.0),
}

@register_model
def vit_sin_small_patch14_dinov2(pretrained=False, patch_size=14, **kwargs) -> VisionTransformerSin:
    """ ViT-S/14 DINOv2 counterpart with sinusoidal pos embed
    """
    model_args = dict(patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, init_values=1e-5, **kwargs)
    model = _create_vision_transformer_sin('vit_sin_small_patch14_dinov2', pretrained=pretrained, **model_args)
    return model

@register_model
def vit_sin_base_patch14_dinov2(pretrained=False, patch_size=14, **kwargs) -> VisionTransformerSin:
    """ ViT-B/14 DINOv2 counterpart with sinusoidal pos embed
    """
    model_args = dict(patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, init_values=1e-5, **kwargs)
    model = _create_vision_transformer_sin('vit_sin_base_patch14_dinov2', pretrained=pretrained, **model_args)
    return model

@register_model
def vit_sin_large_patch14_dinov2(pretrained=False, patch_size=14, **kwargs) -> VisionTransformerSin:
    """ ViT-L/14 DINOv2 counterpart with sinusoidal pos embed
    """
    model_args = dict(patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, init_values=1e-5, **kwargs)
    model = _create_vision_transformer_sin('vit_sin_large_patch14_dinov2', pretrained=pretrained, **model_args)
    return model

@register_model
def vit_sin_small_patch16_224(pretrained=False, **kwargs) -> VisionTransformerSin:
    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer_sin('vit_sin_small_patch16_224', pretrained=pretrained, **model_args)
    return model

@register_model
def vit_sin_base_patch16_224(pretrained=False, **kwargs) -> VisionTransformerSin:
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer_sin('vit_sin_base_patch16_224', pretrained=pretrained, **model_args)
    return model