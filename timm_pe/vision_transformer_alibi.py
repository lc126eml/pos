""" Vision Transformer (ViT) w/ ALiBi Postional Encoding in PyTorch

NOTE: these models are experimental / WIP, expect changes

Hacked together by / Copyright 2024, Ross Wightman
"""
import logging
import math
import itertools
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
from timm.layers import PatchEmbed, Mlp, DropPath, use_fused_attn, LayerType
from timm.models._builder import build_model_with_cfg
from timm.models._features import feature_take_indices
from timm.models._manipulate import named_apply, checkpoint
from timm.models._registry import generate_default_cfgs, register_model
from timm.models.vision_transformer import get_init_weights_vit

__all__ = ['VisionTransformerAlibi']  # model_registry will add each entrypoint fn to this

_logger = logging.getLogger(__name__)

class AlibiAttention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, alibi: Optional[torch.Tensor] = None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.fused_attn:
            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=alibi,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if alibi is not None:
                attn = attn + alibi
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class AlibiBlock(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_norm=False,
            init_values=None,
            proj_drop=0.,
            attn_drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            **kwargs,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = AlibiAttention(
            dim,
            num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
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

    def forward(self, x, alibi: Optional[torch.Tensor] = None):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), alibi=alibi)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class AlibiAttentionWithClsToken(AlibiAttention):
    def forward(self, x, alibi: Optional[torch.Tensor] = None):
        # In this version, the alibi bias is only applied to patch tokens
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        x = super().forward(x, alibi=alibi)
        return x


class VisionTransformerAlibi(nn.Module):
    """ Vision Transformer w/ ALiBi Position Embedding
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
            init_values: Optional[float] = 1e-6,
            class_token: bool = False,
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
            block_fn: Type[nn.Module] = AlibiBlock
    ):
        super().__init__()
        assert global_pool in ('', 'avg', 'token')
        assert class_token or global_pool != 'token'
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.head_hidden_size = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_prefix_tokens = 1 if class_token else 0
        self.grad_checkpointing = False

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        r = self.patch_embed.feat_ratio() if hasattr(self.patch_embed, 'feat_ratio') else patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, self.num_prefix_tokens, embed_dim)) if class_token else None

        self.alibi = get_2dalibi(self.num_heads, num_patches)

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
        return {'cls_token'}

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

    def forward_features(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        alibi_bias = self.alibi.to(x.device)

        # The alibi bias is applied only to the attention between patch tokens
        # The class token does not get any bias.
        if self.cls_token is not None:
            alibi_bias = torch.nn.functional.pad(alibi_bias, (self.num_prefix_tokens, 0, self.num_prefix_tokens, 0))

        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x, alibi=alibi_bias)
            else:
                x = blk(x, alibi=alibi_bias)
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

def get_2dalibi(num_heads, num_patches):
    # inspired by: https://github.com/ofirpress/attention_with_linear_biases
    points = list(itertools.product(range(int(math.sqrt(num_patches))), range(int(math.sqrt(num_patches)))))

    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2 ** (-2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2 * closest_power_of_2)[0::2][
                                                               :n - closest_power_of_2]

    slopes = torch.Tensor(get_slopes(num_heads)).unsqueeze(1)
    idxs = []
    for p1 in points:
        for p2 in points:
            dist = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
            idxs.append(dist * slopes * -1)
    all_bias = torch.cat(idxs, dim=1)
    return all_bias.view(1, num_heads, num_patches, num_patches)

def _create_vision_transformer_alibi(variant, pretrained=False, **kwargs):
    out_indices = kwargs.pop('out_indices', 3)
    model = build_model_with_cfg(
        VisionTransformerAlibi, variant, pretrained,
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
    'vit_alibi_small_patch16_224.untrained': _cfg(),
    'vit_alibi_base_patch16_224.untrained': _cfg(),
    'vit_alibi_small_patch14_dinov2.untrained': _cfg(input_size=(3, 518, 518), crop_pct=1.0),
    'vit_alibi_base_patch14_dinov2.untrained': _cfg(input_size=(3, 518, 518), crop_pct=1.0),
    'vit_alibi_large_patch14_dinov2.untrained': _cfg(input_size=(3, 518, 518), crop_pct=1.0),
})


@register_model
def vit_alibi_small_patch16_224(pretrained=False, **kwargs) -> VisionTransformerAlibi:
    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, qkv_bias=False, fc_norm=True)
    model = _create_vision_transformer_alibi(
        'vit_alibi_small_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_alibi_base_patch16_224(pretrained=False, **kwargs) -> VisionTransformerAlibi:
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=False, fc_norm=True)
    model = _create_vision_transformer_alibi(
        'vit_alibi_base_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model
@register_model
def vit_alibi_small_patch14_dinov2(pretrained=False, patch_size=14, **kwargs) -> VisionTransformerAlibi:
    """ ViT-S/14 DINOv2 counterpart with ALiBi pos embed
    """
    model_args = dict(patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, init_values=1e-5, class_token=True, **kwargs)
    model = _create_vision_transformer_alibi(
        'vit_alibi_small_patch14_dinov2', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_alibi_base_patch14_dinov2(pretrained=False, patch_size=14, **kwargs) -> VisionTransformerAlibi:
    """ ViT-B/14 DINOv2 counterpart with ALiBi pos embed
    """
    model_args = dict(patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, init_values=1e-5, class_token=True, **kwargs)
    model = _create_vision_transformer_alibi(
        'vit_alibi_base_patch14_dinov2', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_alibi_large_patch14_dinov2(pretrained=False, patch_size=14, **kwargs) -> VisionTransformerAlibi:
    """ ViT-L/14 DINOv2 counterpart with ALiBi pos embed
    """
    model_args = dict(patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, init_values=1e-5, class_token=True, **kwargs)
    model = _create_vision_transformer_alibi(
        'vit_alibi_large_patch14_dinov2', pretrained=pretrained, **dict(model_args, **kwargs))
    return model
