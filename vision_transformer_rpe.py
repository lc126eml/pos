""" Vision Transformer (ViT) w/ Image Relative Position Encoding (iRPE) in PyTorch

A PyTorch implement of Vision Transformers as described in
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929

`timm` style implementation w/ iRPE taken from
'Rethinking and Improving Relative Position Encoding for Vision Transformer' - https://arxiv.org/abs/2107.14222

Hacked together by / Copyright 2020 Ross Wightman
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
import numpy as np

from timm.data import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.layers import PatchEmbed, Mlp, DropPath, LayerType, to_2tuple, trunc_normal_
from timm.models._builder import build_model_with_cfg
from timm.models._features import feature_take_indices
from timm.models._manipulate import named_apply, checkpoint
from timm.models._registry import register_model
from timm.models.vision_transformer import get_init_weights_vit

__all__ = ['VisionTransformerRPE']

_logger = logging.getLogger(__name__)


# region iRPE code from Cream-main/iRPE/DeiT-with-iRPE/irpe.py

class METHOD:
    EUCLIDEAN = 0
    QUANT = 1
    PRODUCT = 3
    CROSS = 4
    CROSS_ROWS = 41
    CROSS_COLS = 42


@torch.no_grad()
def piecewise_index(relative_position, alpha, beta, gamma, dtype):
    rp_abs = relative_position.abs()
    mask = rp_abs <= alpha
    not_mask = ~mask
    rp_out = relative_position[not_mask]
    rp_abs_out = rp_abs[not_mask]
    y_out = (torch.sign(rp_out) * (alpha +
                                   torch.log(rp_abs_out / alpha) /
                                   math.log(gamma / alpha) *
                                   (beta - alpha)).round().clip(max=beta)).to(dtype)
    idx = relative_position.clone()
    if idx.dtype in [torch.float32, torch.float64]:
        idx = idx.round().to(dtype)
    idx[not_mask] = y_out
    return idx


def get_absolute_positions(height, width, dtype, device):
    rows = torch.arange(height, dtype=dtype, device=device).view(height, 1).repeat(1, width)
    cols = torch.arange(width, dtype=dtype, device=device).view(1, width).repeat(height, 1)
    return torch.stack([rows, cols], 2)


@torch.no_grad()
def _rp_2d_euclidean(diff, **kwargs):
    dis = diff.square().sum(2).float().sqrt().round()
    return piecewise_index(dis, **kwargs)


@torch.no_grad()
def _rp_2d_quant(diff, **kwargs):
    dis = diff.square().sum(2)
    return piecewise_index(dis, **kwargs)


@torch.no_grad()
def _rp_2d_product(diff, **kwargs):
    beta_int = int(kwargs['beta'])
    S = 2 * beta_int + 1
    r = piecewise_index(diff[:, :, 0], **kwargs) + beta_int
    c = piecewise_index(diff[:, :, 1], **kwargs) + beta_int
    pid = r * S + c
    return pid


@torch.no_grad()
def _rp_2d_cross_rows(diff, **kwargs):
    dis = diff[:, :, 0]
    return piecewise_index(dis, **kwargs)


@torch.no_grad()
def _rp_2d_cross_cols(diff, **kwargs):
    dis = diff[:, :, 1]
    return piecewise_index(dis, **kwargs)


_METHOD_FUNC = {
    METHOD.EUCLIDEAN: _rp_2d_euclidean,
    METHOD.QUANT: _rp_2d_quant,
    METHOD.PRODUCT: _rp_2d_product,
    METHOD.CROSS_ROWS: _rp_2d_cross_rows,
    METHOD.CROSS_COLS: _rp_2d_cross_cols,
}


def get_num_buckets(method, alpha, beta, gamma):
    beta_int = int(beta)
    if method == METHOD.PRODUCT:
        num_buckets = (2 * beta_int + 1) ** 2
    else:
        num_buckets = 2 * beta_int + 1
    return num_buckets


BUCKET_IDS_BUF = dict()


@torch.no_grad()
def get_bucket_ids_2d(method, height, width, skip, alpha, beta, gamma, dtype=torch.long, device=torch.device('cpu')):
    bucket_ids, num_buckets, L = get_bucket_ids_2d_without_skip(
        method, height, width, alpha, beta, gamma, dtype, device)

    if skip > 0:
        new_bids = bucket_ids.new_empty(size=(skip + L, skip + L))
        extra_bucket_id = num_buckets
        num_buckets += 1
        new_bids[:skip] = extra_bucket_id
        new_bids[:, :skip] = extra_bucket_id
        new_bids[skip:, skip:] = bucket_ids
        bucket_ids = new_bids
    bucket_ids = bucket_ids.contiguous()
    return bucket_ids, num_buckets


@torch.no_grad()
def get_bucket_ids_2d_without_skip(method, height, width, alpha, beta, gamma, dtype=torch.long, device=torch.device('cpu')):
    key = (method, alpha, beta, gamma, dtype, device)
    value = BUCKET_IDS_BUF.get(key, None)
    if value is None or value[-2] < height or value[-1] < width:
        if value is None:
            max_height, max_width = height, width
        else:
            max_height = max(value[-2], height)
            max_width = max(value[-1], width)
        func = _METHOD_FUNC.get(method, None)
        pos = get_absolute_positions(max_height, max_width, dtype, device)
        max_L = max_height * max_width
        pos1 = pos.view((max_L, 1, 2))
        pos2 = pos.view((1, max_L, 2))
        diff = pos1 - pos2
        bucket_ids = func(diff, alpha=alpha, beta=beta, gamma=gamma, dtype=dtype)
        beta_int = int(beta)
        if method != METHOD.PRODUCT:
            bucket_ids += beta_int
        bucket_ids = bucket_ids.view(max_height, max_width, max_height, max_width)
        num_buckets = get_num_buckets(method, alpha, beta, gamma)
        value = (bucket_ids, num_buckets, height, width)
        BUCKET_IDS_BUF[key] = value
    L = height * width
    bucket_ids = value[0][:height, :width, :height, :width].reshape(L, L)
    num_buckets = value[1]
    return bucket_ids, num_buckets, L


class iRPE(nn.Module):
    _rp_bucket_buf = (None, None, None)

    def __init__(self, head_dim, num_heads=8, mode=None, method=None, transposed=True, num_buckets=None, initializer=None, rpe_config=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.mode = mode
        self.method = method
        self.transposed = transposed
        self.num_buckets = num_buckets
        self.initializer = initializer if initializer is not None else lambda x: None
        self.rpe_config = rpe_config
        self._ctx_rp_bucket_flatten = None  # Good practice to initialize
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        if self.transposed:
            if self.mode == 'bias':
                self.lookup_table_bias = nn.Parameter(torch.zeros(self.num_heads, self.num_buckets))
                self.initializer(self.lookup_table_bias)
            elif self.mode == 'contextual':
                self.lookup_table_weight = nn.Parameter(torch.zeros(self.num_heads, self.head_dim, self.num_buckets))
                self.initializer(self.lookup_table_weight)
        else:
            if self.mode == 'contextual':
                self.lookup_table_weight = nn.Parameter(torch.zeros(self.num_heads, self.num_buckets, self.head_dim))
                self.initializer(self.lookup_table_weight)

    def forward(self, x, height=None, width=None):
        rp_bucket, self._ctx_rp_bucket_flatten = self._get_rp_bucket(x, height=height, width=width)
        if self.transposed:
            return self.forward_rpe_transpose(x, rp_bucket)
        return self.forward_rpe_no_transpose(x, rp_bucket)

    def _get_rp_bucket(self, x, height=None, width=None):
        B, H, L, D = x.shape
        device = x.device
        if height is None:
            E = int(math.sqrt(L - self.rpe_config.skip))
            height = width = E
        key = (height, width, device)
        if self._rp_bucket_buf[0] == key:
            return self._rp_bucket_buf[1:3]
        
        config = self.rpe_config
        rp_bucket, num_buckets = get_bucket_ids_2d(
            method=self.method, height=height, width=width, skip=config.skip,
            alpha=config.alpha, beta=config.beta, gamma=config.gamma, dtype=torch.long, device=device)
        assert num_buckets == self.num_buckets
        
        _ctx_rp_bucket_flatten = None
        if self.mode == 'contextual' and self.transposed:
            offset = torch.arange(0, L * self.num_buckets, self.num_buckets, dtype=rp_bucket.dtype, device=rp_bucket.device).view(-1, 1)
            _ctx_rp_bucket_flatten = (rp_bucket + offset).flatten()
        self._rp_bucket_buf = (key, rp_bucket, _ctx_rp_bucket_flatten)
        return rp_bucket, _ctx_rp_bucket_flatten

    def forward_rpe_transpose(self, x, rp_bucket):
        B = len(x)
        L_query, L_key = rp_bucket.shape
        if self.mode == 'bias':
            return self.lookup_table_bias[:, rp_bucket.flatten()].view(1, self.num_heads, L_query, L_key)
        elif self.mode == 'contextual':
            lookup_table = torch.matmul(
                x.transpose(0, 1).reshape(-1, B * L_query, self.head_dim), self.lookup_table_weight
            ).view(-1, B, L_query, self.num_buckets).transpose(0, 1)
            return lookup_table.flatten(2)[:, :, self._ctx_rp_bucket_flatten].view(B, -1, L_query, L_key)

    def forward_rpe_no_transpose(self, x, rp_bucket):
        B = len(x)
        L_query, L_key = rp_bucket.shape
        weight = self.lookup_table_weight[:, rp_bucket.flatten()].view(self.num_heads, L_query, L_key, self.head_dim)
        return torch.matmul(x.permute(1, 2, 0, 3), weight).permute(2, 0, 1, 3)


class iRPE_Cross(nn.Module):
    def __init__(self, method, **kwargs):
        super().__init__()
        assert method == METHOD.CROSS
        self.rp_rows = iRPE(**kwargs, method=METHOD.CROSS_ROWS)
        self.rp_cols = iRPE(**kwargs, method=METHOD.CROSS_COLS)

    def forward(self, x, height=None, width=None):
        rows = self.rp_rows(x, height=height, width=width)
        cols = self.rp_cols(x, height=height, width=width)
        return rows + cols


class _Config:
    pass


def get_single_rpe_config(ratio=1.9, method=METHOD.PRODUCT, mode='contextual', shared_head=True, skip=0):
    config = _Config()
    config.shared_head = shared_head
    config.mode = mode
    config.method = method
    config.alpha = 1 * ratio
    config.beta = 2 * ratio
    config.gamma = 8 * ratio
    config.num_buckets = get_num_buckets(method, config.alpha, config.beta, config.gamma)
    if skip > 0:
        config.num_buckets += 1
    config.skip = skip
    return config


def get_rpe_config(ratio=1.9, method=METHOD.PRODUCT, mode='contextual', shared_head=True, skip=0, rpe_on='k'):
    if isinstance(method, str):
        method_mapping = dict(euc=METHOD.EUCLIDEAN, quant=METHOD.QUANT, cross=METHOD.CROSS, product=METHOD.PRODUCT)
        method = method_mapping[method.lower()]
    if mode == 'ctx':
        mode = 'contextual'
    config = _Config()
    kwargs = dict(ratio=ratio, method=method, mode=mode, shared_head=shared_head, skip=skip)
    config.rpe_q = get_single_rpe_config(**kwargs) if 'q' in rpe_on else None
    config.rpe_k = get_single_rpe_config(**kwargs) if 'k' in rpe_on else None
    config.rpe_v = get_single_rpe_config(**kwargs) if 'v' in rpe_on else None
    return config


def build_rpe(config, head_dim, num_heads):
    if config is None:
        return None, None, None
    rpes = [config.rpe_q, config.rpe_k, config.rpe_v]
    transposeds = [True, True, False]

    def _build_single_rpe(rpe, transposed):
        if rpe is None:
            return None
        rpe_cls = iRPE if rpe.method != METHOD.CROSS else iRPE_Cross
        return rpe_cls(
            head_dim=head_dim, num_heads=1 if rpe.shared_head else num_heads,
            mode=rpe.mode, method=rpe.method, transposed=transposed,
            num_buckets=rpe.num_buckets, rpe_config=rpe)
    return [_build_single_rpe(rpe, transposed) for rpe, transposed in zip(rpes, transposeds)]

# endregion


class RPEAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_norm=False, attn_drop=0., proj_drop=0., rpe_config=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rpe_q, self.rpe_k, self.rpe_v = build_rpe(rpe_config, head_dim=head_dim, num_heads=num_heads)

    def forward(self, x, height=None, width=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.rpe_k is not None:
            attn += self.rpe_k(q, height=height, width=width)
        if self.rpe_q is not None:
            attn += self.rpe_q(k, height=height, width=width).transpose(2, 3)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v

        if self.rpe_v is not None:
            out += self.rpe_v(attn, height=height, width=width)

        x = out.transpose(1, 2).reshape(B, N, C)
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

class RPEBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_norm=False, init_values=None,
                 proj_drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, rpe_config=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = RPEAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm,
            attn_drop=attn_drop, proj_drop=proj_drop, rpe_config=rpe_config)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=proj_drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, height=None, width=None):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), height=height, width=width)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class VisionTransformerRPE(nn.Module):
    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, qk_norm=False,
            init_values=None, class_token=True, fc_norm=False, drop_rate=0., proj_drop_rate=0.,
            attn_drop_rate=0., drop_path_rate=0., weight_init='', fix_init=False,
            embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=RPEBlock, rpe_config=None):
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
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        r = self.patch_embed.feat_ratio() if hasattr(self.patch_embed, 'feat_ratio') else patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, self.num_prefix_tokens, embed_dim)) if class_token else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_prefix_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            block_fn(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_norm=qk_norm,
                init_values=init_values, proj_drop=proj_drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, rpe_config=rpe_config)
            for i in range(depth)])
        self.feature_info = [dict(module=f'blocks.{i}', num_chs=embed_dim, reduction=r) for i in range(depth)]
        self.norm = norm_layer(embed_dim) if not fc_norm else nn.Identity()

        self.fc_norm = norm_layer(embed_dim) if fc_norm else nn.Identity()
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=.02)
        if weight_init != 'skip':
            self.apply(self._init_weights)
        if fix_init:
            self.fix_init_weight()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def fix_init_weight(self):
        def rescale(param, _layer_id):
            param.div_(math.sqrt(2.0 * _layer_id))
        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(stem=r'^cls_token|patch_embed|pos_embed', blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))])

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_intermediates(self, x, indices=None, return_prefix_tokens=False, norm=False, stop_early=False, output_fmt='NCHW', intermediates_only=False):
        assert output_fmt in ('NCHW', 'NLC'), 'Output format must be one of NCHW or NLC.'
        reshape = output_fmt == 'NCHW'
        intermediates = []
        take_indices, max_index = feature_take_indices(len(self.blocks), indices)

        B, C, H, W = x.shape
        x = self.patch_embed(x)
        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        if torch.jit.is_scripting() or not stop_early:
            blocks = self.blocks
        else:
            blocks = self.blocks[:max_index + 1]
        for i, blk in enumerate(blocks):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x, H // self.patch_embed.patch_size[0], W // self.patch_embed.patch_size[1])
            else:
                x = blk(x, H // self.patch_embed.patch_size[0], W // self.patch_embed.patch_size[1])
            if i in take_indices:
                intermediates.append(self.norm(x) if norm else x)

        if self.num_prefix_tokens:
            prefix_tokens = [y[:, 0:self.num_prefix_tokens] for y in intermediates]
            intermediates = [y[:, self.num_prefix_tokens:] for y in intermediates]
        if reshape:
            H_feat, W_feat = self.patch_embed.dynamic_feat_size((H, W))
            intermediates = [y.reshape(B, H_feat, W_feat, -1).permute(0, 3, 1, 2).contiguous() for y in intermediates]
        if not torch.jit.is_scripting() and return_prefix_tokens:
            intermediates = list(zip(intermediates, prefix_tokens))

        if intermediates_only:
            return intermediates

        x = self.norm(x)
        return x, intermediates

    def forward_features(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x, H // self.patch_embed.patch_size[0], W // self.patch_embed.patch_size[1])
            else:
                x = blk(x, H // self.patch_embed.patch_size[0], W // self.patch_embed.patch_size[1])
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


def _create_vision_transformer_rpe(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', False):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')
    model = build_model_with_cfg(VisionTransformerRPE, variant, pretrained, **kwargs)
    return model


def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head', **kwargs
    }


@register_model
def vit_rpe_small_patch14_dinov2(pretrained=False, patch_size=14, **kwargs):
    """ ViT-S/14 DINOv2 counterpart with iRPE pos embed
    """
    rpe_config = get_rpe_config(ratio=1.9, method=METHOD.PRODUCT, mode='contextual', shared_head=True, skip=1)
    model_kwargs = dict(patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, init_values=1e-5, rpe_config=rpe_config, class_token=True, **kwargs)
    model = _create_vision_transformer_rpe('vit_rpe_small_patch14_dinov2', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_rpe_base_patch14_dinov2(pretrained=False, patch_size=14, **kwargs):
    """ ViT-B/14 DINOv2 counterpart with iRPE pos embed
    """
    rpe_config = get_rpe_config(ratio=1.9, method=METHOD.PRODUCT, mode='contextual', shared_head=True, skip=1)
    model_kwargs = dict(patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, init_values=1e-5, rpe_config=rpe_config, class_token=True, **kwargs)
    model = _create_vision_transformer_rpe('vit_rpe_base_patch14_dinov2', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_rpe_large_patch14_dinov2(pretrained=False, patch_size=14, **kwargs):
    """ ViT-L/14 DINOv2 counterpart with iRPE pos embed
    """
    rpe_config = get_rpe_config(ratio=1.9, method=METHOD.PRODUCT, mode='contextual', shared_head=True, skip=1)
    model_kwargs = dict(patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, init_values=1e-5, rpe_config=rpe_config, class_token=True, **kwargs)
    model = _create_vision_transformer_rpe('vit_rpe_large_patch14_dinov2', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def vit_rpe_small_patch16_224(pretrained=False, **kwargs):
    rpe_config = get_rpe_config(ratio=1.9, method=METHOD.PRODUCT, mode='contextual', shared_head=True, skip=1)
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, rpe_config=rpe_config, **kwargs)
    model = _create_vision_transformer_rpe('vit_rpe_small_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_rpe_base_patch16_224(pretrained=False, **kwargs):
    rpe_config = get_rpe_config(ratio=1.9, method=METHOD.PRODUCT, mode='contextual', shared_head=True, skip=1)
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, rpe_config=rpe_config, **kwargs)
    model = _create_vision_transformer_rpe('vit_rpe_base_patch16_224', pretrained=pretrained, **model_kwargs)
    return model
