# %%
# =================================================================================
# Step 1: Install and Import Necessary Libraries
# =================================================================================
import glob
import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset,TensorDataset, DataLoader
# from data.MultiScaleImageDataset import MultiScaleImageDataset, CustomImageDataset
# from data.DynamicResolutionBatchSampler import DynamicResolutionBatchSampler

import matplotlib.pyplot as plt
import pandas as pd
import csv
import pickle
import numpy as np
import random
from PIL import Image
from torch.nn import functional as F
import torchvision.transforms.functional as TF
import sys
import subprocess
import importlib
from types import SimpleNamespace
import gc
import time
import argparse
import logging
train_start_time = time.time()
# try:
#     from filelock import FileLock
# except ImportError:
#     FileLock = None

# from core.utils import log_grads
# Enable faster matmul/conv kernels on Ampere+ without extra memory cost
# if torch.cuda.is_available():
#     torch.backends.cuda.matmul.allow_tf32 = True
#     torch.backends.cudnn.allow_tf32 = True
#     # Prefer faster matmul kernels when available (Torch 2.0+)
#     if hasattr(torch, "set_float32_matmul_precision"):
#         torch.set_float32_matmul_precision("high")
# %%
# Ensure timm provides the requested model; update if missing.
# def _timm_has_model(model_name: str) -> bool:
#     try:
#         if hasattr(timm, "list_models"):
#             return model_name in timm.list_models()
#         if hasattr(timm, "models") and hasattr(timm.models, "list_models"):
#             return model_name in timm.models.list_models()
#     except Exception:
#         return False
#     return False

# from importlib.metadata import version, PackageNotFoundError
# ver = version("timm").split('.')[-1]
# print(ver)
# if int(ver) < 20:
    # !pip uninstall -y timm
subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "timm"])
LOCAL_TIMM = "/kaggle/input/timm-repos/pytorch-image-models"
sys.path.insert(0, LOCAL_TIMM)

import timm
print("timm:", timm.__version__, flush=True)
print("torch:", torch.__version__, flush=True)
# print([m for m in timm.list_models() if "dinov" in m], flush=True)

# _timm_model_name = "vit_small_patch16_dinov3"
# if not _timm_has_model(_timm_model_name):
#     print(f"timm missing {_timm_model_name} ...", flush=True)
    # sys.exit(0)
#     subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "timm"])
        
#     LOCAL_TIMM = "/kaggle/input/timm-repos/pytorch-image-models"
#     if os.path.isdir(LOCAL_TIMM):
#         sys.path.insert(0, LOCAL_TIMM)
#     import timm
# print("timm:", timm.__version__, flush=True)
# print("torch:", torch.__version__, flush=True)
# print([m for m in timm.list_models() if "dinov" in m], flush=True)

_is_kaggle_env = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.path.exists("/kaggle/working"))
# =================================================================================
# Step 2: Configuration
# =================================================================================

# --- Dynamically set root directory ---
is_kaggle = _is_kaggle_env
if _is_kaggle_env:
    is_kaggle = True
    root_dir = "/kaggle/working"
    BASE_PATH = "/kaggle/input/imagenet100/"
    print("kaggle", flush=True)
    print(os.listdir("/kaggle/input"), flush=True)
  

else:
    print("not kaggle", flush=True)
# elif os.path.exists('/home/sshuser'):
#     root_dir = '/home/sshuser'
#     BASE_PATH = f'{root_dir}/Data/imagenet100/'
# elif os.path.exists('/lc'):
#     root_dir = '/lc/logs'
#     BASE_PATH = f'/lc/data/imagenet100/'
# else:
#     root_dir = '/linux'
#     BASE_PATH = f'{root_dir}/Data/imagenet100/'
# --- Configuration via SimpleNamespace for easy interactive use ---
args = SimpleNamespace(
    # --- Model & Training Settings ---
    pos_type = 'alibi', #"alibi", # 'sin', 'alibi', 'relpos', None #,  'rpe', 'rope', 
    dynamic_img_size=False,
    model_type= "dinov3",
    use_abs_pos_emb=False,
    use_rot_pos_emb=False,
    model_size='base',
    num_classes=100,
    patch_size = 16,
    grad_accum_steps=2,
    # Adjust based on your GPU memory. BATCH_SIZE = 120, 128, 136, 392, 768, etc.
    batch_size=64, #rpe
    # batch_size=256, #rope
    # batch_size=392, 
    # batch_size=512, 
    # ViT models have a fixed input size
    # img_sizes=[224, 192, 288],
    img_sizes=[224],
    val_img_sizes=[160, 176, 192, 208,224, 256, 272, 288, 320, 336, 352, 368, 384, 400, 416],
    # val_img_sizes=[224],
    # lr=1e-3, #small
    lr=3e-4, #base
    lr_aux=1e-5,
    eta_min=0.0,
    weight_decay=0.01,
    epochs=130,
    # has_pos=True, # Set to True or False directly
    overlap=0,
    pretrained=None,
    seed=26,
    use_patch_position_loss=False,
    use_rc_loss=False,
    # loss_type="smooth_l1", # "mse", "smooth_l1"
    # huber_beta=None,
    # rc_alpha=300.0,
    rc_alpha=600.0, # base
    warmup_steps_for_aux=1,
    workers=5,
    randaugment=False,
    randaugment_n=2,
    randaugment_m=3,
    random_erasing=False,
    re_prob=0.0,
    train=True,
    val=False,
    ckpt_path=None,
    lock=True,
    save_full_ckpt=True,
    resume_full_ckpt=True,
    resume_ckpt_path='/kaggle/input/cls-base-alibi-desc-326/ckpt/last.pth',
    resume_scheduler=True,
    resume_optimizer=True,
    resume_bs=True,
    composite_lr=True,
    warmup_steps=3000,
    clip_value=1.0,
    log_interval=100,
    csv_interval=1,
    show_peak_gpu_mem=True,
    # save_ckpt=False,
    compile_model=False,
    total_run_time_hr=12.0,
    # --- Dataset Paths ---
    root_dir=root_dir,
)
resume_ckpt=None
if args.resume_full_ckpt and args.resume_ckpt_path:
    if not os.path.exists(args.resume_ckpt_path):
        resume_dir = os.path.dirname(args.resume_ckpt_path)
        parts = os.path.normpath(resume_dir).split(os.sep)
        if os.path.isabs(resume_dir):
            prefix_parts = parts[1:4]
            search_root = os.path.join(os.sep, *prefix_parts)
        else:
            prefix_parts = parts[:3]
            search_root = os.path.join(*prefix_parts)
        candidates = sorted(
            glob.glob(os.path.join(search_root, "**", "last.pth"), recursive=True)
        )
        if candidates:
            args.resume_ckpt_path = candidates[0]
    skip_keys = [
        "resume_full_ckpt",
        "resume_ckpt_path",
        "resume_bs",
        "resume_scheduler",
        "resume_optimizer",
        "total_run_time_hr",
    ]
    if not args.resume_scheduler:
        skip_keys.extend([
            "epochs",
            "warmup_steps",
            "eta_min",
            "composite_lr",
        ])
    if not args.resume_bs:
        skip_keys.extend(["batch_size", "grad_accum_steps"])
    resume_ckpt = torch.load(args.resume_ckpt_path, map_location="cpu", weights_only=False)
    print(f"Resumed args from '{args.resume_ckpt_path}'")
    ckpt_args = resume_ckpt.get("args", None)
    if ckpt_args is not None:
        for k, v in vars(ckpt_args).items():
            if k not in skip_keys:
                setattr(args, k, v)
if args.pos_type is not None:
    args.has_pos = True
    args.overlap = 0
    args.use_rc_loss=False
    args.use_patch_position_loss=False
    args.dynamic_img_size=False
    args.val=False
if args.use_abs_pos_emb or args.use_rot_pos_emb:
    args.overlap = 0
    args.use_patch_position_loss=False
    args.use_rc_loss = False
if args.model_size == "base":
    args.rc_alpha = 600.0
else:
    args.rc_alpha = 300.0

offset = 0
# args.batch_size = 64
# args.grad_accum_steps=2
# print(args)
if args.pos_type is not None: 
    pos_str = f"{args.pos_type}_"
else:
    pos_str= ""
MODEL_NAME = f"vit_{pos_str}{args.model_size}_patch16_{args.model_type}"
# MODEL_NAME = f"vit_{args.model_size}_patch16_{args.model_type}"
if is_kaggle:
    output_dir = args.root_dir
    ckpt_output_dir = os.path.join(output_dir, "ckpt")
else:
    print("not kaggle")
    sys.exit(0)
last_ckpt_path = os.path.join(ckpt_output_dir, f'last.pth')

# %%
# --- Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

use_amp = torch.cuda.is_available()
use_bf16 = use_amp and torch.cuda.is_bf16_supported(including_emulation=False)
autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
use_amp = use_bf16
print(f"Using device: {DEVICE}", use_bf16, autocast_dtype)
# Speed tweaks (P100-friendly)
if torch.cuda.is_available() and use_bf16:
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if len(args.img_sizes) == 1:
        torch.backends.cudnn.benchmark = True
# sys.exit(0)
# torch.backends.cudnn.deterministic=True
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
if torch.cuda.is_available() and len(args.img_sizes) == 1:
    torch.backends.cudnn.benchmark = True
pos_prefix = ""
if args.pos_type is not None:
    pos_prefix = f"{args.pos_type}_"

abs_pos = ""
if args.use_abs_pos_emb:
    abs_pos = "_abs_pos"

rot_pos = ""
if args.use_rot_pos_emb:
    rot_pos = "_rot_pos"

patch_pos = ""
if args.use_patch_position_loss:
    patch_pos = "_patch_pos"

subdir_name = (
    f"{pos_prefix}{args.model_size}{abs_pos}{rot_pos}_overlap_{args.overlap}_"
    f"rc_{args.use_rc_loss}{patch_pos}_alpha_{int(args.rc_alpha)}lr{int(args.lr/1e-5)}_s{args.seed}"
).replace(',', '_').replace('[', '_').replace(']', '_').replace(' ', '')
if not is_kaggle:
    output_dir = os.path.join(output_dir, subdir_name)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(ckpt_output_dir, exist_ok=True)

log_file_path = os.path.join(output_dir, f'{subdir_name}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

logger.info(f"Using device: {DEVICE}")
logger.info(f"Using mixed precision: {'bfloat16' if use_bf16 else 'float16'}")
logger.info(args)
logger.info(output_dir)
logger.info(subdir_name)

# --- Acquire a file lock to ensure exclusive GPU usage ---
# gpu_lock = None
# if args.lock:
#     if FileLock:
#         lock_path = "/tmp/gpu.lock"
#         gpu_lock = FileLock(lock_path)
#         logger.info(f"Attempting to acquire lock on '{lock_path}'...")
#         gpu_lock.acquire()
#         logger.info("Lock acquired. It is safe to proceed.")
#         # The lock will be automatically released when the script exits.
#     else:
#         logger.warning("`filelock` library not found, skipping lock. Run `pip install filelock`.")

logger.info("Cleaning up memory...")
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
logger.info("Memory cleanup complete.")

# logger.info(args)
#%%
# List of all the partial training directories
TRAIN_PATHS = [
    os.path.join(BASE_PATH, 'train.X1'),
    os.path.join(BASE_PATH, 'train.X2'),
    os.path.join(BASE_PATH, 'train.X3'),
    os.path.join(BASE_PATH, 'train.X4'),
]

VALID_PATH = os.path.join(BASE_PATH, 'val.X')
LABEL_PATH = os.path.join(BASE_PATH, 'Labels.json')

if args.pos_type == 'relpos':
    import math
    import os
    from functools import partial
    from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
    from timm.layers import (
        PatchEmbed,
        Mlp,
        GluMlp,
        SwiGLU,
        LayerNorm,
        DropPath, calculate_drop_path_rates,
        PatchDropoutWithIndices,
        create_rope_embed,
        apply_rot_embed_cat,
        apply_keep_indices_nlc,
        trunc_normal_,
        resample_patch_embed,
        resample_abs_pos_embed,
        global_pool_nlc,
        to_2tuple,
        use_fused_attn,
        maybe_add_mask,
        AttentionRope,
        AttentionPoolLatent,
        RelPosMlp,
        RelPosBias,
    )
    from timm.models._builder import build_model_with_cfg
    from timm.models._features import feature_take_indices
    from timm.models._manipulate import named_apply, checkpoint
    from timm.models._registry import generate_default_cfgs, register_model


    __all__ = ['Eva']


    class EvaAttention(nn.Module):
        """ EVA Attention with ROPE, no k-bias, and fused/unfused qkv options
        """
        fused_attn: torch.jit.Final[bool]

        def __init__(
                self,
                dim: int,
                num_heads: int = 8,
                qkv_bias: bool = True,
                qkv_fused: bool = True,
                qkv_bias_separate: bool = False,
                num_prefix_tokens: int = 1,
                attn_drop: float = 0.,
                proj_drop: float = 0.,
                attn_head_dim: Optional[int] = None,
                norm_layer: Optional[Callable] = None,
                qk_norm: bool = False,
                scale_norm: bool = True,
                rotate_half: bool = False,
                rel_pos_cls: Optional[Type[nn.Module]] = None,
                device=None,
                dtype=None,
        ):
            """
            Args:
                dim: Input dimension of the token embeddings
                num_heads: Number of attention heads
                qkv_bias: Whether to add a bias term to the query, key, and value projections
                qkv_fused: Whether qkv projections are fused into one projection or separate
                qkv_bias_separate: Whether to apply bias to qkv as a separate addition or part of F.linear() call
                num_prefix_tokens: Number of reg/cls tokens at the beginning of the sequence that
                    should not have position embeddings applied
                attn_drop: Dropout rate for attention weights
                proj_drop: Dropout rate for the output projection
                attn_head_dim: Dimension of each attention head (if None, computed as dim // num_heads)
                norm_layer: Normalization layer constructor to use for QK and scale normalization
                qk_norm: Enable normalization of query (Q) and key (K) vectors with norm_layer
                scale_norm: Enable normalization (scaling) of attention output with norm_layer
                rotate_half: Use half rotation layout instead of interleaved
                rel_pos_cls: Relative position bias module constructor (RelPosMlp or RelPosBias)
            """
            dd = {'device': device, 'dtype': dtype}
            super().__init__()
            if scale_norm or qk_norm:
                assert norm_layer is not None, 'norm_layer must be provided if qk_norm or scale_norm is True'
            self.num_heads = num_heads
            head_dim = dim // num_heads
            if attn_head_dim is not None:
                head_dim = attn_head_dim
            attn_dim = head_dim * self.num_heads
            self.scale = head_dim ** -0.5
            self.head_dim = head_dim
            self.num_prefix_tokens = num_prefix_tokens
            self.fused_attn = use_fused_attn()
            self.qkv_bias_separate = qkv_bias_separate
            self.rotate_half = rotate_half
            self.rel_pos = rel_pos_cls(num_heads=num_heads, **dd) if rel_pos_cls is not None else None

            if qkv_fused:
                self.qkv = nn.Linear(dim, attn_dim * 3, bias=False, **dd)
                self.q_proj = self.k_proj = self.v_proj = None
                if qkv_bias:
                    self.q_bias = nn.Parameter(torch.zeros(attn_dim, **dd))
                    self.register_buffer('k_bias', torch.zeros(attn_dim, **dd), persistent=False)
                    self.v_bias = nn.Parameter(torch.zeros(attn_dim, **dd))
                else:
                    self.q_bias = self.k_bias = self.v_bias = None
            else:
                self.q_proj = nn.Linear(dim, attn_dim, bias=qkv_bias, **dd)
                self.k_proj = nn.Linear(dim, attn_dim, bias=False, **dd)
                self.v_proj = nn.Linear(dim, attn_dim, bias=qkv_bias, **dd)
                self.qkv = None
                self.q_bias = self.k_bias = self.v_bias = None
            self.q_norm = norm_layer(self.head_dim, **dd) if qk_norm else nn.Identity()
            self.k_norm = norm_layer(self.head_dim, **dd) if qk_norm else nn.Identity()
            self.attn_drop = nn.Dropout(attn_drop)
            self.norm = norm_layer(attn_dim, **dd) if scale_norm else nn.Identity()
            self.proj = nn.Linear(attn_dim, dim, **dd)
            self.proj_drop = nn.Dropout(proj_drop)

        def forward(
                self,
                x,
                rope: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None,
                shared_rel_pos: Optional[torch.Tensor] = None,
        ):
            """Forward pass for the attention module.

            Args:
                x: Input tensor of shape (batch_size, sequence_length, embedding_dim)
                rope: Rotary position embeddings tensor for position-aware attention
                attn_mask: Optional attention mask to apply during attention computation
                shared_rel_pos: Pre-computed relative position bias shared across blocks

            Returns:
                Tensor of shape (batch_size, sequence_length, embedding_dim)
            """
            B, N, C = x.shape

            if self.qkv is not None:
                if self.q_bias is None:
                    qkv = self.qkv(x)
                else:
                    qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
                    if self.qkv_bias_separate:
                        qkv = self.qkv(x)
                        qkv += qkv_bias
                    else:
                        qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
                qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
            else:
                q = self.q_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)  # B, num_heads, N, C
                k = self.k_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)
                v = self.v_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)

            q, k = self.q_norm(q), self.k_norm(k)

            if rope is not None:
                npt = self.num_prefix_tokens
                half = getattr(self, 'rotate_half', False)
                q = torch.cat([q[:, :, :npt, :], apply_rot_embed_cat(q[:, :, npt:, :], rope, half=half)], dim=2).type_as(v)
                k = torch.cat([k[:, :, :npt, :], apply_rot_embed_cat(k[:, :, npt:, :], rope, half=half)], dim=2).type_as(v)

            rel_pos_bias = None
            if self.rel_pos is not None:
                rel_pos_bias = self.rel_pos.get_bias()
            elif shared_rel_pos is not None:
                rel_pos_bias = shared_rel_pos

            if self.fused_attn:
                if attn_mask is not None:
                    if attn_mask.dtype == torch.bool:
                        attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype).masked_fill(attn_mask, float('-inf'))
                    elif attn_mask.dtype != q.dtype:
                        attn_mask = attn_mask.to(dtype=q.dtype)
                    rel_pos_bias = attn_mask if rel_pos_bias is None else rel_pos_bias + attn_mask
                x = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=rel_pos_bias,
                    dropout_p=self.attn_drop.p if self.training else 0.,
                )
            else:
                q = q * self.scale
                attn = (q @ k.transpose(-2, -1))
                if self.rel_pos is not None:
                    attn = self.rel_pos(attn, shared_rel_pos=shared_rel_pos)
                elif rel_pos_bias is not None:
                    attn = attn + rel_pos_bias
                attn = maybe_add_mask(attn, attn_mask)
                attn = attn.softmax(dim=-1)

                attn = self.attn_drop(attn)
                x = attn @ v

            x = x.transpose(1, 2).reshape(B, N, C)
            x = self.norm(x)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x


    class EvaBlock(nn.Module):

        def __init__(
                self,
                dim: int,
                num_heads: int,
                qkv_bias: bool = True,
                qkv_fused: bool = True,
                mlp_ratio: float = 4.,
                swiglu_mlp: bool = False,
                swiglu_align_to: int = 0,
                scale_mlp: bool = False,
                scale_attn_inner: bool = False,
                num_prefix_tokens: int = 1,
                attn_type: str = 'eva',
                rotate_half: bool = False,
                proj_drop: float = 0.,
                attn_drop: float = 0.,
                drop_path: float = 0.,
                rel_pos_cls: Optional[Type[nn.Module]] = None,
                init_values: Optional[float] = None,
                act_layer: Callable = nn.GELU,
                norm_layer: Callable = LayerNorm,
                attn_head_dim: Optional[int] = None,
                device=None,
                dtype=None,
                **kwargs,
        ):
            """ Initialize the EVA transformer block.

            Args:
            dim: Input dimension of the token embeddings
                num_heads: Number of attention heads
                qkv_bias: Whether to use bias terms in query, key, value projections
                qkv_fused: Whether to use a single projection for query, key, value
                mlp_ratio: Ratio of MLP hidden dimension to input dimension
                swiglu_mlp: Whether to use SwiGLU activation in the MLP
                scale_mlp: Whether to use normalization in the MLP
                scale_attn_inner: Whether to use normalization within the attention mechanism
                num_prefix_tokens: Number of tokens at the beginning of the sequence (class tokens, etc.)
                attn_type: Type of attention module to use ('eva' or 'rope')
                proj_drop: Dropout rate for projection layers
                attn_drop: Dropout rate for attention matrix
                drop_path: Stochastic depth rate
                init_values: Initial value for LayerScale, None = no LayerScale
                act_layer: Activation layer constructor
                norm_layer: Normalization layer constructor
                attn_head_dim: Dimension of each attention head (if None, computed as dim // num_heads)
            """
            dd = {'device': device, 'dtype': dtype}
            super().__init__()

            self.norm1 = norm_layer(dim, **dd)
            attn_cls = AttentionRope if attn_type == 'rope' else EvaAttention
            attn_kwargs = dict(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qkv_fused=qkv_fused,
                num_prefix_tokens=num_prefix_tokens,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                attn_head_dim=attn_head_dim,
                norm_layer=norm_layer,
                scale_norm=scale_attn_inner,
                rotate_half=rotate_half,
                **dd,
            )
            if attn_type != 'rope':
                attn_kwargs.update(dict(rel_pos_cls=rel_pos_cls))
            self.attn = attn_cls(**attn_kwargs)
            self.gamma_1 = nn.Parameter(init_values * torch.ones(dim, **dd)) if init_values is not None else None
            self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

            self.norm2 = norm_layer(dim, **dd)
            hidden_features = int(dim * mlp_ratio)
            if swiglu_mlp:
                if scale_mlp or swiglu_align_to:
                    # when norm in SwiGLU used or alignment enabled, an impl with separate fc for gate & x is used
                    self.mlp = SwiGLU(
                        in_features=dim,
                        hidden_features=hidden_features,
                        norm_layer=norm_layer if scale_mlp else None,
                        drop=proj_drop,
                        align_to=swiglu_align_to,
                        **dd,
                    )
                else:
                    # w/o any extra norm, an impl with packed weights is used
                    self.mlp = GluMlp(
                        in_features=dim,
                        hidden_features=hidden_features * 2,
                        norm_layer=norm_layer if scale_mlp else None,
                        act_layer=nn.SiLU,
                        gate_last=False,
                        drop=proj_drop,
                        **dd,
                    )
            else:
                self.mlp = Mlp(
                    in_features=dim,
                    hidden_features=hidden_features,
                    act_layer=act_layer,
                    norm_layer=norm_layer if scale_mlp else None,
                    drop=proj_drop,
                    **dd,
                )
            self.gamma_2 = nn.Parameter(init_values * torch.ones(dim, **dd)) if init_values is not None else None
            self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        def forward(
                self,
                x: torch.Tensor,
                rope: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None,
                shared_rel_pos: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            if self.gamma_1 is None:
                x = x + self.drop_path1(self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask, shared_rel_pos=shared_rel_pos))
                x = x + self.drop_path2(self.mlp(self.norm2(x)))
            else:
                x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask, shared_rel_pos=shared_rel_pos))
                x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
            return x


    class EvaBlockPostNorm(nn.Module):
        """ EVA block w/ post-norm and support for swiglu, MLP norm scale, ROPE. """
        def __init__(
                self,
                dim: int,
                num_heads: int,
                qkv_bias: bool = True,
                qkv_fused: bool = True,
                mlp_ratio: float = 4.,
                attn_type: str = 'eva',
                rotate_half: bool = False,
                swiglu_mlp: bool = False,
                swiglu_align_to: int = 0,
                scale_mlp: bool = False,
                scale_attn_inner: bool = False,
                num_prefix_tokens: int = 1,
                proj_drop: float = 0.,
                attn_drop: float = 0.,
                drop_path: float = 0.,
                rel_pos_cls: Optional[Type[nn.Module]] = None,
                init_values: Optional[float] = None,  # ignore for post-norm
                act_layer: Callable = nn.GELU,
                norm_layer: Callable = nn.LayerNorm,
                attn_head_dim: Optional[int] = None,
                device=None,
                dtype=None,
        ):
            """ Initialize the post-norm EVA transformer block.

            Args:
            dim: Input dimension of the token embeddings
                num_heads: Number of attention heads
                qkv_bias: Whether to use bias terms in query, key, value projections
                qkv_fused: Whether to use a single projection for query, key, value
                mlp_ratio: Ratio of MLP hidden dimension to input dimension
                swiglu_mlp: Whether to use SwiGLU activation in the MLP
                scale_mlp: Whether to use normalization in the MLP
                scale_attn_inner: Whether to use normalization within the attention mechanism
                num_prefix_tokens: Number of tokens at the beginning of the sequence (class tokens, etc.)
                attn_type: Type of attention module to use ('eva' or 'rope')
                proj_drop: Dropout rate for projection layers
                attn_drop: Dropout rate for attention matrix
                drop_path: Stochastic depth rate
                init_values: Initial value for LayerScale, None = no LayerScale (NOTE: ignored for post-norm block)
                act_layer: Activation layer constructor
                norm_layer: Normalization layer constructor
                attn_head_dim: Dimension of each attention head (if None, computed as dim // num_heads)
            """
            dd = {'device': device, 'dtype': dtype}
            super().__init__()

            attn_cls = AttentionRope if attn_type == 'rope' else EvaAttention
            attn_kwargs = dict(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qkv_fused=qkv_fused,
                num_prefix_tokens=num_prefix_tokens,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                attn_head_dim=attn_head_dim,
                norm_layer=norm_layer,
                scale_norm=scale_attn_inner,
                rotate_half=rotate_half,
                **dd,
            )
            if attn_type != 'rope':
                attn_kwargs.update(dict(rel_pos_cls=rel_pos_cls))
            self.attn = attn_cls(**attn_kwargs)
            self.norm1 = norm_layer(dim, **dd)
            self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

            hidden_features = int(dim * mlp_ratio)
            if swiglu_mlp:
                if scale_mlp:
                    # when norm in SwiGLU used, an impl with separate fc for gate & x is used
                    self.mlp = SwiGLU(
                        in_features=dim,
                        hidden_features=hidden_features,
                        norm_layer=norm_layer if scale_mlp else None,
                        drop=proj_drop,
                        align_to=swiglu_align_to,
                        **dd,
                    )
                else:
                    # w/o any extra norm, an impl with packed fc1 weights is used, matches existing GluMLP
                    self.mlp = GluMlp(
                        in_features=dim,
                        hidden_features=hidden_features * 2,
                        norm_layer=norm_layer if scale_mlp else None,
                        act_layer=nn.SiLU,
                        gate_last=False,
                        drop=proj_drop,
                        **dd,
                    )
            else:
                self.mlp = Mlp(
                    in_features=dim,
                    hidden_features=hidden_features,
                    act_layer=act_layer,
                    norm_layer=norm_layer if scale_mlp else None,
                    drop=proj_drop,
                    **dd,
                )
            self.norm2 = norm_layer(dim, **dd)
            self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        def forward(
                self,
                x: torch.Tensor,
                rope: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None,
                shared_rel_pos: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            x = x + self.drop_path1(self.norm1(self.attn(x, rope=rope, attn_mask=attn_mask, shared_rel_pos=shared_rel_pos)))
            x = x + self.drop_path2(self.norm2(self.mlp(x)))
            return x

if args.pos_type == 'relpos':
    class Eva(nn.Module):
        """ Eva Vision Transformer w/ Abs & Rotary Pos Embed

        This class implements the EVA and EVA02 models that were based on the BEiT ViT variant
        * EVA - abs pos embed, global avg pool
        * EVA02 - abs + rope pos embed, global avg pool, SwiGLU, scale Norm in MLP (ala normformer)
        """

        def __init__(
                self,
                img_size: Union[int, Tuple[int, int]] = 224,
                patch_size: Union[int, Tuple[int, int]] = 16,
                in_chans: int = 3,
                num_classes: int = 1000,
                global_pool: str = 'avg',
                embed_dim: int = 768,
                depth: int = 12,
                num_heads: int = 12,
                qkv_bias: bool = True,
                qkv_fused: bool = True,
                mlp_ratio: float = 4.,
                swiglu_mlp: bool = False,
                swiglu_align_to: int = 0,
                scale_mlp: bool = False,
                scale_attn_inner: bool = False,
                attn_type: str = 'eva',
                drop_rate: float = 0.,
                pos_drop_rate: float = 0.,
                patch_drop_rate: float = 0.,
                proj_drop_rate: float = 0.,
                attn_drop_rate: float = 0.,
                drop_path_rate: float = 0.,
                norm_layer: Callable = LayerNorm,
                init_values: Optional[float] = None,
                class_token: bool = True,
                num_reg_tokens: int = 0,
                no_embed_class: bool = False,
                use_abs_pos_emb: bool = True,
                use_rot_pos_emb: bool = False,
                use_relpos_pos_emb: bool = False,
                rel_pos_type: str = 'mlp',
                rel_pos_dim: Optional[int] = None,
                shared_rel_pos: bool = False,
                rope_type: Optional[str] = 'cat',
                rope_grid_offset: float = 0.,
                rope_grid_indexing: str = 'ij',
                rope_temperature: float = 10000.,
                rope_rotate_half: bool = False,
                use_post_norm: bool = False,
                use_pre_transformer_norm: bool = False,
                use_post_transformer_norm: Optional[bool] = None,
                use_fc_norm: Optional[bool] = None,
                attn_pool_num_heads: Optional[int] = None,
                attn_pool_mlp_ratio: Optional[float] = None,
                dynamic_img_size: bool = False,
                dynamic_img_pad: bool = False,
                ref_feat_shape: Optional[Union[Tuple[int, int], int]] = None,
                head_init_scale: float = 0.001,
                device=None,
                dtype=None,
        ):
            """Initialize the EVA Vision Transformer model.

            Args:
                img_size: Input image size (single int for square, or tuple for rectangular)
                patch_size: Patch size to divide image into tokens (single int for square, or tuple)
                in_chans: Number of input image channels
                num_classes: Number of classes (output dim) for classification head (final projection), 0 for pass-through
                global_pool: Type of global pooling for final sequence ('avg', 'token', 'map', etc.)
                embed_dim: Embedding dimension for tokens
                depth: Number of transformer blocks
                num_heads: Number of attention heads
                qkv_bias: Enable bias for query, key, value projections
                qkv_fused: Use a single projection for query, key, value
                mlp_ratio: Ratio of mlp hidden dim to embedding dim
                swiglu_mlp: Use SwiGLU activation in MLP
                scale_mlp: Apply scaling normalization in MLP (normformer style)
                scale_attn_inner: Apply scaling normalization inside attention
                attn_type: Type of attention module to use
                drop_rate: Dropout rate after final projection and pooling
                pos_drop_rate: Dropout rate for positional embeddings
                patch_drop_rate: Rate of dropping patches during training
                proj_drop_rate: Dropout rate for projections
                attn_drop_rate: Dropout rate for attention
                drop_path_rate: Stochastic depth rate
                norm_layer: Normalization layer constructor
                init_values: Initial layer-scale values
                class_token: Use class token
                num_reg_tokens: Number of additional learnable 'register' tokens to add to the sequence
                no_embed_class: Don't include position embeddings for class (or reg) tokens
                use_abs_pos_emb: Use absolute (learned) positional embeddings
                use_rot_pos_emb: Use rotary position embeddings
                use_relpos_pos_emb: Use relative position bias/encoding (Swin/BEiT style)
                rel_pos_type: Relative position encoding type ('mlp', 'mlp-swin', or 'bias')
                rel_pos_dim: Hidden dimension for relative position MLP
                shared_rel_pos: Share a single relative position module across blocks
                rope_type: Type of RoPE to use ('cat', 'mixed', 'dinov3', etc.).
                rope_grid_offset: Offset for rotary position embedding grid
                rope_grid_indexing: Indexing mode for rotary position embeddings ('ij' or 'xy')
                rope_temperature: Temperature parameter for ROPE frequency computation
                rope_rotate_half: Use half rotation layout (rotate D/2 dims), else use interleaved rotation layout
                use_post_norm: Use post-norm transformer block type
                use_pre_transformer_norm: Use normalization layer before transformer blocks
                use_post_transformer_norm: Use normalization layer after transformer blocks
                use_fc_norm: Use normalization layer after pooling, before final classifier
                attn_pool_num_heads: Number of heads in attention pooling
                attn_pool_mlp_ratio: MLP ratio in attention pooling
                dynamic_img_size: Support dynamic image sizes in forward pass
                dynamic_img_pad: Apply dynamic padding for irregular image sizes
                ref_feat_shape: Reference feature shape for rotary position embedding scale
                head_init_scale: Initialization scale for classification head weights
            """
            super().__init__()
            dd = {'device': device, 'dtype': dtype}
            assert global_pool in ('', 'avg', 'avgmax', 'max', 'token', 'map')
            self.num_classes = num_classes
            self.global_pool = global_pool
            self.num_features = self.head_hidden_size = self.embed_dim = embed_dim  # for consistency with other models
            self.num_prefix_tokens = (1 if class_token else 0) + num_reg_tokens
            self.no_embed_class = no_embed_class
            self.dynamic_img_size = dynamic_img_size
            self.grad_checkpointing = False

            # resolve norm / pool usage
            activate_pre_norm = use_pre_transformer_norm
            if use_fc_norm is not None:
                activate_fc_norm = use_fc_norm  # pass through if explicit
            else:
                activate_fc_norm = global_pool == 'avg'  # default on if avg pool used
            if use_post_transformer_norm is not None:
                activate_post_norm = use_post_transformer_norm  # pass through if explicit
            else:
                activate_post_norm = not activate_fc_norm  # default on if fc_norm isn't active

            embed_args = {}
            if dynamic_img_size:
                # flatten deferred until after pos embed
                embed_args.update(dict(strict_img_size=False, output_fmt='NHWC'))
            self.patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
                dynamic_img_pad=dynamic_img_pad,
                bias=not use_pre_transformer_norm,
                **embed_args,
                **dd,
            )
            num_patches = self.patch_embed.num_patches
            r = self.patch_embed.feat_ratio() if hasattr(self.patch_embed, 'feat_ratio') else patch_size

            self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim, **dd)) if class_token else None
            self.reg_token = nn.Parameter(torch.empty(1, num_reg_tokens, embed_dim, **dd)) if num_reg_tokens else None
            self.cls_embed = class_token and self.reg_token is None

            num_pos_tokens = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
            self.pos_embed = nn.Parameter(torch.empty(1, num_pos_tokens, embed_dim, **dd)) if use_abs_pos_emb else None
            self.pos_drop = nn.Dropout(p=pos_drop_rate)
            if patch_drop_rate > 0:
                self.patch_drop = PatchDropoutWithIndices(patch_drop_rate, num_prefix_tokens=self.num_prefix_tokens)
            else:
                self.patch_drop = None

            rel_pos_cls = None
            self.shared_rel_pos = None
            self.use_relpos_pos_emb = use_relpos_pos_emb
            if use_relpos_pos_emb:
                assert not dynamic_img_size, 'relpos currently requires dynamic_img_size=False (fixed patch grid).'
                assert patch_drop_rate == 0., 'relpos currently requires patch_drop_rate=0 to keep a dense patch grid.'
                rel_pos_args = dict(window_size=self.patch_embed.grid_size, prefix_tokens=self.num_prefix_tokens)
                if rel_pos_type.startswith('mlp'):
                    if rel_pos_dim:
                        rel_pos_args['hidden_dim'] = rel_pos_dim
                    if 'swin' in rel_pos_type:
                        rel_pos_args['mode'] = 'swin'
                    rel_pos_cls = partial(RelPosMlp, **rel_pos_args)
                else:
                    rel_pos_cls = partial(RelPosBias, **rel_pos_args)
                if shared_rel_pos:
                    self.shared_rel_pos = rel_pos_cls(num_heads=num_heads, **dd)
                    rel_pos_cls = None

            self.rope_mixed = False
            if use_rot_pos_emb:
                ref_feat_shape = to_2tuple(ref_feat_shape) if ref_feat_shape is not None else None

                # Setup RoPE kwargs
                rope_kwargs = dict(
                    dim=embed_dim,
                    num_heads=num_heads,
                    feat_shape=None if dynamic_img_size else self.patch_embed.grid_size,
                    temperature=rope_temperature,
                    grid_indexing=rope_grid_indexing,
                    **dd,
                )
                if rope_type == 'mixed':
                    rope_kwargs.update(dict(depth=depth))
                    self.rope_mixed = True
                elif rope_type == 'cat':
                    rope_kwargs.update(dict(
                        in_pixels=False,
                        grid_offset=rope_grid_offset,
                        ref_feat_shape=ref_feat_shape,
                    ))

                self.rope = create_rope_embed(rope_type=rope_type, **rope_kwargs)
            else:
                self.rope = None

            self.norm_pre = norm_layer(embed_dim, **dd) if activate_pre_norm else nn.Identity()

            dpr = calculate_drop_path_rates(drop_path_rate, depth)  # stochastic depth decay rule
            block_fn = EvaBlockPostNorm if use_post_norm else EvaBlock
            self.blocks = nn.ModuleList([
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    qkv_fused=qkv_fused,
                    mlp_ratio=mlp_ratio,
                    swiglu_mlp=swiglu_mlp,
                    swiglu_align_to=swiglu_align_to,
                    scale_mlp=scale_mlp,
                    scale_attn_inner=scale_attn_inner,
                    attn_type=attn_type,
                    rotate_half=rope_rotate_half,
                    rel_pos_cls=rel_pos_cls,
                    num_prefix_tokens=self.num_prefix_tokens,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    init_values=init_values,
                    **dd,
                )
                for i in range(depth)])
            self.feature_info = [
                dict(module=f'blocks.{i}', num_chs=embed_dim, reduction=r) for i in range(depth)]

            self.norm = norm_layer(embed_dim, **dd) if activate_post_norm else nn.Identity()

            if global_pool == 'map':
                self.attn_pool = AttentionPoolLatent(
                    self.embed_dim,
                    num_heads=attn_pool_num_heads or num_heads,
                    mlp_ratio=attn_pool_mlp_ratio or mlp_ratio,
                    norm_layer=norm_layer,
                    act_layer=nn.GELU,
                    **dd,
                )
            else:
                self.attn_pool = None
            self.fc_norm = norm_layer(embed_dim, **dd) if activate_fc_norm else nn.Identity()
            self.head_drop = nn.Dropout(drop_rate)
            self.head = nn.Linear(embed_dim, num_classes, **dd) if num_classes > 0 else nn.Identity()

            self.init_weights(head_init_scale=head_init_scale)

        def init_weights(self, head_init_scale=None):
            self.apply(self._init_weights)
            if self.pos_embed is not None:
                trunc_normal_(self.pos_embed, std=.02)
            if self.cls_token is not None:
                trunc_normal_(self.cls_token, std=.02)
            if self.reg_token is not None:
                trunc_normal_(self.reg_token, std=.02)
            self.fix_init_weight()
            if head_init_scale and isinstance(self.head, nn.Linear):
                trunc_normal_(self.head.weight, std=.02)
                self.head.weight.data.mul_(head_init_scale)
                self.head.bias.data.mul_(head_init_scale)

        def fix_init_weight(self) -> None:
            """Fix initialization weights by rescaling based on layer depth."""
            def rescale(param, layer_id):
                param.div_(math.sqrt(2.0 * layer_id))

            for layer_id, layer in enumerate(self.blocks):
                rescale(layer.attn.proj.weight.data, layer_id + 1)
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

        def _init_weights(self, m: nn.Module) -> None:
            """Initialize weights for Linear layers.

            Args:
                m: Module to initialize.
            """
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        @torch.jit.ignore
        def no_weight_decay(self) -> Set[str]:
            """Parameters to exclude from weight decay."""
            nwd = {'pos_embed', 'cls_token'}
            if (rope := getattr(self, "rope", None)) and hasattr(rope, "no_weight_decay"):
                return nwd | {f"rope.{p}" for p in rope.no_weight_decay()}
            return nwd

        @torch.jit.ignore
        def set_grad_checkpointing(self, enable: bool = True) -> None:
            """Enable or disable gradient checkpointing."""
            self.grad_checkpointing = enable

        @torch.jit.ignore
        def group_matcher(self, coarse: bool = False) -> Dict[str, Any]:
            """Create layer groupings for optimization."""
            matcher = dict(
                stem=r'^cls_token|pos_embed|patch_embed',  # stem and embed
                blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))],
            )
            return matcher

        @torch.jit.ignore
        def get_classifier(self) -> nn.Module:
            return self.head

        def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None) -> None:
            """Reset the classifier head.

            Args:
                num_classes: Number of output classes.
                global_pool: Global pooling type.
            """
            self.num_classes = num_classes
            if global_pool is not None:
                self.global_pool = global_pool
            self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        def set_input_size(
                self,
                img_size: Optional[Tuple[int, int]] = None,
                patch_size: Optional[Tuple[int, int]] = None,
        ) -> None:
            """Update the input image resolution and patch size.

            Args:
                img_size: New input resolution, if None current resolution is used.
                patch_size: New patch size, if None existing patch size is used.
            """
            prev_grid_size = self.patch_embed.grid_size
            self.patch_embed.set_input_size(img_size=img_size, patch_size=patch_size)

            if self.pos_embed is not None:
                num_prefix_tokens = 0 if self.no_embed_class else self.num_prefix_tokens
                num_new_tokens = self.patch_embed.num_patches + num_prefix_tokens
                if num_new_tokens != self.pos_embed.shape[1]:
                    self.pos_embed = nn.Parameter(resample_abs_pos_embed(
                        self.pos_embed,
                        new_size=self.patch_embed.grid_size,
                        old_size=prev_grid_size,
                        num_prefix_tokens=num_prefix_tokens,
                        verbose=True,
                    ))

            if self.rope is not None:
                self.rope.update_feat_shape(self.patch_embed.grid_size)

        def _pos_embed(self, x) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            if self.dynamic_img_size:
                B, H, W, C = x.shape
                if self.pos_embed is not None:
                    prev_grid_size = self.patch_embed.grid_size
                    pos_embed = resample_abs_pos_embed(
                        self.pos_embed,
                        new_size=(H, W),
                        old_size=prev_grid_size,
                        num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
                    )
                else:
                    pos_embed = None
                x = x.view(B, -1, C)
                rot_pos_embed = self.rope.get_embed(shape=(H, W)) if self.rope is not None else None
            else:
                pos_embed = self.pos_embed
                rot_pos_embed = self.rope.get_embed() if self.rope is not None else None

            to_cat = []
            if self.cls_token is not None:
                to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
            if self.reg_token is not None:
                to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

            if self.no_embed_class:
                # position embedding does not overlap with class / reg token
                if pos_embed is not None:
                    x = x + pos_embed
                if to_cat:
                    x = torch.cat(to_cat + [x], dim=1)
            else:
                # pos_embed has entry for class / reg token, concat then add
                if to_cat:
                    x = torch.cat(to_cat + [x], dim=1)
                if pos_embed is not None:
                    x = x + pos_embed

            x = self.pos_drop(x)

            # apply patch dropout to patches and rotary position embedding
            if self.patch_drop is not None:
                x, keep_indices = self.patch_drop(x)
                if rot_pos_embed is not None and keep_indices is not None:
                    rot_pos_embed = apply_keep_indices_nlc(x, rot_pos_embed, keep_indices)
                    # After applying keep indices to rope embeds, batch dim is added
                    if getattr(self, 'rope_mixed', False):
                        # B, D, nH, N, dim -> D, B, nH, N, dim. For consistent iteration over depth at index 0.
                        rot_pos_embed = rot_pos_embed.transpose(0, 1)
                    else:
                        # B, N, dim -> B, 1, N, dim.  Need head dim singleton for correct dim alignment in axial mode.
                        rot_pos_embed = rot_pos_embed.unsqueeze(1)

            return x, rot_pos_embed

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
                indices: Take last n blocks if an int, if is a sequence, select by matching indices
                return_prefix_tokens: Return both prefix and spatial intermediate tokens
                norm: Apply norm layer to all intermediates
                stop_early: Stop iterating over blocks when last desired intermediate hit
                output_fmt: Shape of intermediate feature outputs
                intermediates_only: Only return intermediate features
            """
            assert output_fmt in ('NCHW', 'NLC'), 'Output format for EVA-ViT features must be one of NCHW or NLC.'
            reshape = output_fmt == 'NCHW'
            intermediates = []
            take_indices, max_index = feature_take_indices(len(self.blocks), indices)

            # forward pass
            B, _, height, width = x.shape
            x = self.patch_embed(x)
            x, rot_pos_embed = self._pos_embed(x)
            x = self.norm_pre(x)
            if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
                blocks = self.blocks
            else:
                blocks = self.blocks[:max_index + 1]
            shared_rel_pos = self.shared_rel_pos.get_bias() if self.shared_rel_pos is not None else None

            # Handle depth-dependent embeddings for mixed mode
            if getattr(self, 'rope_mixed', False) and rot_pos_embed is not None:
                for i, blk in enumerate(blocks):
                    if self.grad_checkpointing and not torch.jit.is_scripting():
                        x = checkpoint(blk, x, rope=rot_pos_embed[i], shared_rel_pos=shared_rel_pos)
                    else:
                        x = blk(x, rope=rot_pos_embed[i], shared_rel_pos=shared_rel_pos)
                    if i in take_indices:
                        intermediates.append(self.norm(x) if norm else x)
            else:
                for i, blk in enumerate(blocks):
                    if self.grad_checkpointing and not torch.jit.is_scripting():
                        x = checkpoint(blk, x, rope=rot_pos_embed, shared_rel_pos=shared_rel_pos)
                    else:
                        x = blk(x, rope=rot_pos_embed, shared_rel_pos=shared_rel_pos)
                    if i in take_indices:
                        intermediates.append(self.norm(x) if norm else x)

            # process intermediates
            if self.num_prefix_tokens:
                # split prefix (e.g. class, distill) and spatial feature tokens
                prefix_tokens = [y[:, 0:self.num_prefix_tokens] for y in intermediates]
                intermediates = [y[:, self.num_prefix_tokens:] for y in intermediates]
            if reshape:
                # reshape to BCHW output format
                H, W = self.patch_embed.dynamic_feat_size((height, width))
                intermediates = [y.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for y in intermediates]
            if not torch.jit.is_scripting() and return_prefix_tokens:
                # return_prefix not support in torchscript due to poor type handling
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
                self.attn_pool = None
                self.fc_norm = nn.Identity()
                self.reset_classifier(0, '')
            return take_indices

        def pool(self, x: torch.Tensor, pool_type: Optional[str] = None) -> torch.Tensor:
            if self.attn_pool is not None:
                x = self.attn_pool(x)
                return x
            pool_type = self.global_pool if pool_type is None else pool_type
            x = global_pool_nlc(x, pool_type=pool_type, num_prefix_tokens=self.num_prefix_tokens)
            return x

        def forward_features(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass through feature extraction layers.

            Args:
                x: Input tensor.

            Returns:
                Feature tensor.
            """
            x = self.patch_embed(x)
            x, rot_pos_embed = self._pos_embed(x)
            x = self.norm_pre(x)
            shared_rel_pos = self.shared_rel_pos.get_bias() if self.shared_rel_pos is not None else None

            if getattr(self, 'rope_mixed', False) and rot_pos_embed is not None:
                # Handle depth-dependent embeddings for mixed mode
                # pos embed has shape (depth, num_heads, H*W, dim) or (depth, batch_size, num_heads, H*W, dim)
                for i, blk in enumerate(self.blocks):
                    if self.grad_checkpointing and not torch.jit.is_scripting():
                        x = checkpoint(blk, x, rope=rot_pos_embed[i], shared_rel_pos=shared_rel_pos)
                    else:
                        x = blk(x, rope=rot_pos_embed[i], shared_rel_pos=shared_rel_pos)
            else:
                # Standard path for non-mixed mode
                for blk in self.blocks:
                    if self.grad_checkpointing and not torch.jit.is_scripting():
                        x = checkpoint(blk, x, rope=rot_pos_embed, shared_rel_pos=shared_rel_pos)
                    else:
                        x = blk(x, rope=rot_pos_embed, shared_rel_pos=shared_rel_pos)

            x = self.norm(x)
            return x

        def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
            """Forward pass through classifier head.

            Args:
                x: Feature tensor.
                pre_logits: Return pre-logits if True.

            Returns:
                Output tensor.
            """
            x = self.pool(x)
            x = self.fc_norm(x)
            x = self.head_drop(x)
            return x if pre_logits else self.head(x)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass.

            Args:
                x: Input tensor.

            Returns:
                Output tensor.
            """
            x = self.forward_features(x)
            x = self.forward_head(x)
            return x


    def _convert_pe(
        state_dict: Dict[str, torch.Tensor],
        model: nn.Module,
        prefix: str = 'visual.',
    ) -> Dict[str, torch.Tensor]:
        """Convert Perception Encoder weights.

        Args:
            state_dict: State dictionary to convert.
            model: Target model instance.
            prefix: Prefix to strip from keys.

        Returns:
            Converted state dictionary.
        """
        state_dict = state_dict.get('model', state_dict)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        out_dict = {}
        swaps = [
            ('conv1', 'patch_embed.proj'),
            ('positional_embedding', 'pos_embed'),
            ('transformer.resblocks.', 'blocks.'),
            ('ln_pre', 'norm_pre'),
            ('ln_post', 'norm'),
            ('ln_', 'norm'),
            ('ls_1.gamma', 'gamma_1'),
            ('ls_2.gamma', 'gamma_2'),
            ('in_proj_', 'qkv.'),
            ('out_proj', 'proj'),
            ('mlp.c_fc', 'mlp.fc1'),
            ('mlp.c_proj', 'mlp.fc2'),
        ]
        len_prefix = len(prefix)
        for k, v in state_dict.items():
            if prefix:
                if not k.startswith(prefix):
                    continue
                k = k[len_prefix:]

            for sp in swaps:
                k = k.replace(sp[0], sp[1])

            if k.startswith('attn_pool'):
                k = k.replace('attn_pool.attn', 'attn_pool')
                k = k.replace('attn_pool.layernorm', 'attn_pool.norm')
                k = k.replace('attn_pool.probe', 'attn_pool.latent')
                if k.startswith('attn_pool.qkv'):
                    dim = v.shape[0] // 3
                    if k.endswith('weight'):
                        out_dict['attn_pool.q.weight'] = v[:dim]
                        out_dict['attn_pool.kv.weight'] = v[dim:]
                    elif k.endswith('bias'):
                        out_dict['attn_pool.q.bias'] = v[:dim]
                        out_dict['attn_pool.kv.bias'] = v[dim:]
                    continue
            elif k == 'proj':
                k = 'head.weight'
                v = v.transpose(0, 1)
                out_dict['head.bias'] = torch.zeros(v.shape[0])
            elif k == 'class_embedding':
                k = 'cls_token'
                v = v.unsqueeze(0).unsqueeze(1)
            elif k == 'pos_embed':
                v = v.unsqueeze(0)
            out_dict[k] = v

        return out_dict


    def checkpoint_filter_fn(
            state_dict: Dict[str, torch.Tensor],
            model: nn.Module,
            interpolation: str = 'bicubic',
            antialias: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Convert patch embedding weight from manual patchify + linear proj to conv.

        Args:
            state_dict: Checkpoint state dictionary.
            model: Target model instance.
            interpolation: Interpolation method for resizing.
            antialias: Whether to use antialiasing when resizing.

        Returns:
            Filtered state dictionary.
        """
        out_dict = {}
        # Standard EVA checkpoint processing
        state_dict = state_dict.get('model_ema', state_dict)
        state_dict = state_dict.get('model', state_dict)
        state_dict = state_dict.get('module', state_dict)
        state_dict = state_dict.get('state_dict', state_dict)

        # Loading Meta PE (Perception Encoder) weights
        if 'visual.conv1.weight' in state_dict:
            return _convert_pe(state_dict, model)
        elif 'conv1.weight' in state_dict:
            return _convert_pe(state_dict, model, prefix='')

        # prefix for loading OpenCLIP compatible weights
        if 'visual.trunk.pos_embed' in state_dict:
            prefix = 'visual.trunk.'
        elif 'visual.pos_embed' in state_dict:
            prefix = 'visual.'
        else:
            prefix = ''

        dinov3_weights = 'storage_tokens' in state_dict
        mim_weights = not dinov3_weights and prefix + 'mask_token' in state_dict
        no_qkv = prefix + 'blocks.0.attn.q_proj.weight' in state_dict

        len_prefix = len(prefix)
        for k, v in state_dict.items():
            if prefix:
                if not k.startswith(prefix):
                    continue
                k = k[len_prefix:]

            if 'rope' in k and not k == 'rope.freqs':
                # fixed embedding no need to load buffer from checkpoint
                continue

            if dinov3_weights:
                if any([k.endswith(f) for f in ['.periods', '.bias_mask', 'mask_token']]):
                    # discard unused/non-persistent/pretrain only params
                    continue
                if k.startswith('local_cls_norm'):
                    # discard, only used for 7b dinov3 pretrain w/ local crops
                    continue
                if k.endswith('qkv.bias'):
                    q_bias_k = k.replace('qkv.bias', 'q_bias')
                    try:
                        # the distilled b,l,h models ended up with all zero biases, so timm
                        # has both qkv_bias=True and qkv_bias=False impl, test which
                        model.get_parameter(q_bias_k)
                    except Exception as e:
                        print(e)
                        # skip as target model has no bias parameter
                        continue
                    # split bias into components and skip the k as its supposed to be fixed at 0
                    qv, kv, vv = v.chunk(3, dim=-1)
                    out_dict[q_bias_k] = qv
                    out_dict[k.replace('qkv.bias', 'v_bias')] = vv
                    continue
                k = k.replace('ls1.gamma', 'gamma_1')  # match EVA ls naming
                k = k.replace('ls2.gamma', 'gamma_2')  # match EVA ls naming
                k = k.replace('storage_tokens', 'reg_token')  # rename storage to existing register naming

            elif mim_weights and k in ('mask_token', 'lm_head.weight', 'lm_head.bias', 'norm.weight', 'norm.bias'):
                if k == 'norm.weight' or k == 'norm.bias':
                    # try moving norm -> fc norm on fine-tune, probably a better starting point than new init
                    k = k.replace('norm', 'fc_norm')
                else:
                    # skip pretrain mask token & head weights
                    continue

            if 'patch_embed.proj.weight' in k:
                _, _, H, W = model.patch_embed.proj.weight.shape
                if v.shape[-1] != W or v.shape[-2] != H:
                    v = resample_patch_embed(
                        v,
                        (H, W),
                        interpolation=interpolation,
                        antialias=antialias,
                        verbose=True,
                    )
            elif k == 'pos_embed' and v.shape[1] != model.pos_embed.shape[1]:
                # To resize pos embedding when using model at different size from pretrained weights
                num_prefix_tokens = 0 if getattr(model, 'no_embed_class', False) else getattr(model, 'num_prefix_tokens', 1)
                v = resample_abs_pos_embed(
                    v,
                    new_size=model.patch_embed.grid_size,
                    num_prefix_tokens=num_prefix_tokens,
                    interpolation=interpolation,
                    antialias=antialias,
                    verbose=True,
                )

            k = k.replace('mlp.ffn_ln', 'mlp.norm')
            k = k.replace('attn.inner_attn_ln', 'attn.norm')
            k = k.replace('mlp.w12', 'mlp.fc1')
            k = k.replace('mlp.w1', 'mlp.fc1_g')
            k = k.replace('mlp.w2', 'mlp.fc1_x')
            k = k.replace('mlp.w3', 'mlp.fc2')
            if no_qkv:
                k = k.replace('q_bias', 'q_proj.bias')
                k = k.replace('v_bias', 'v_proj.bias')

            out_dict[k] = v

        return out_dict


    def _create_eva(variant: str, pretrained: bool = False, **kwargs) -> Eva:
        """Create an EVA model.

        Args:
            variant: Model variant name.
            pretrained: Load pretrained weights.
            **kwargs: Additional model arguments.

        Returns:
            Instantiated Eva model.
        """
        # Check if we should use NaFlexVit implementation
        use_naflex = kwargs.pop('use_naflex', None)
        _USE_NAFLEX_DEFAULT = os.environ.get('TIMM_USE_NAFLEX', '0') == '1'
        if use_naflex is None:
            use_naflex = _USE_NAFLEX_DEFAULT
        if use_naflex:
            # Import here to avoid circular import
            from .naflexvit import _create_naflexvit_from_eva
            return _create_naflexvit_from_eva(variant, pretrained, **kwargs)

        out_indices = kwargs.pop('out_indices', 3)
        model = build_model_with_cfg(
            Eva, variant, pretrained,
            pretrained_filter_fn=checkpoint_filter_fn,
            feature_cfg=dict(out_indices=out_indices, feature_cls='getter'),
            **kwargs,
        )
        return model


    def _cfg(url: str = '', **kwargs) -> Dict[str, Any]:
        """Generate default configuration for EVA models.

        Args:
            url: Model weights URL.
            **kwargs: Additional configuration parameters.

        Returns:
            Model configuration dictionary.
        """
        return {
            'url': url,
            'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
            'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
            'mean': OPENAI_CLIP_MEAN, 'std': OPENAI_CLIP_STD,
            'first_conv': 'patch_embed.proj', 'classifier': 'head',
            'license': 'mit', **kwargs
        }


    def _pe_cfg(url: str = '', **kwargs) -> Dict[str, Any]:
        """Generate default configuration for Perception Encoder models.

        Args:
            url: Model weights URL.
            **kwargs: Additional configuration parameters.

        Returns:
            Model configuration dictionary.
        """
        return {
            'url': url,
            'num_classes': 0, 'input_size': (3, 224, 224), 'pool_size': None,
            'crop_pct': 1.0, 'interpolation': 'bicubic', 'fixed_input_size': True,
            'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
            'first_conv': 'patch_embed.proj', 'classifier': 'head',
            'license': 'apache-2.0', **kwargs
        }


    def _dinov3_cfg(url: str = '', **kwargs) -> Dict[str, Any]:
        """Generate default configuration for DINOv3 models.

        Args:
            url: Model weights URL.
            **kwargs: Additional configuration parameters.

        Returns:
            Model configuration dictionary.
        """
        return {
            'url': url,
            'num_classes': 0, 'input_size': (3, 256, 256), 'pool_size': None,
            'crop_pct': 1.0, 'interpolation': 'bicubic', 'fixed_input_size': True,
            'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
            'first_conv': 'patch_embed.proj', 'classifier': 'head',
            'license': 'dinov3-license', **kwargs
        }

    default_cfgs = generate_default_cfgs({

        # EVA 01 CLIP fine-tuned on imagenet-1k
        'eva_giant_patch14_224.clip_ft_in1k': _cfg(
            # hf_hub_id='BAAI/EVA', hf_hub_filename='eva_clip_vis_enc_sz224_ftcls_89p1.pt',
            hf_hub_id='timm/',
        ),
        'eva_giant_patch14_336.clip_ft_in1k': _cfg(
            # hf_hub_id='BAAI/EVA', hf_hub_filename='eva_clip_vis_enc_sz336_ftcls_89p4.pt',
            hf_hub_id='timm/',
            input_size=(3, 336, 336), crop_pct=1.0, crop_mode='squash'),

        # MIM EVA 01 pretrain, ft on in22k -> in1k
        'eva_giant_patch14_336.m30m_ft_in22k_in1k': _cfg(
            # hf_hub_id='BAAI/EVA', hf_hub_filename='eva_21k_1k_336px_psz14_ema_89p6.pt',
            hf_hub_id='timm/',
            mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD,
            input_size=(3, 336, 336), crop_pct=1.0, crop_mode='squash'),
        'eva_giant_patch14_560.m30m_ft_in22k_in1k': _cfg(
            # hf_hub_id='BAAI/EVA', hf_hub_filename='eva_21k_1k_560px_psz14_ema_89p7.pt',
            hf_hub_id='timm/',
            mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD,
            input_size=(3, 560, 560), crop_pct=1.0, crop_mode='squash'),

        # in22k or m38m MIM pretrain w/ intermediate in22k fine-tune and final in1k fine-tune
        'eva02_base_patch14_448.mim_in22k_ft_in22k_in1k': _cfg(
            # hf_hub_id='Yuxin-CV/EVA-02', hf_hub_filename='eva02/cls/in21k_to_in1k/eva02_B_pt_in21k_medft_in21k_ft_in1k_p14.pt',
            hf_hub_id='timm/',
            input_size=(3, 448, 448), crop_pct=1.0, crop_mode='squash',
        ),
        'eva02_large_patch14_448.mim_in22k_ft_in22k_in1k': _cfg(
            # hf_hub_id='Yuxin-CV/EVA-02', hf_hub_filename='eva02/cls/in21k_to_in1k/eva02_L_pt_in21k_medft_in21k_ft_in1k_p14.pt',
            hf_hub_id='timm/',
            input_size=(3, 448, 448), crop_pct=1.0, crop_mode='squash',
        ),
        'eva02_large_patch14_448.mim_m38m_ft_in22k_in1k': _cfg(
            hf_hub_id='timm/',
            #hf_hub_id='Yuxin-CV/EVA-02', hf_hub_filename='eva02/cls/in21k_to_in1k/eva02_L_pt_m38m_medft_in21k_ft_in1k_p14.pt',
            input_size=(3, 448, 448), crop_pct=1.0, crop_mode='squash',
        ),

        # in22k or m3m MIM pretrain w/ in1k fine-tune
        'eva02_tiny_patch14_336.mim_in22k_ft_in1k': _cfg(
            #hf_hub_id='Yuxin-CV/EVA-02', hf_hub_filename='eva02/cls/in1k/eva02_Ti_pt_in21k_ft_in1k_p14.pt',
            hf_hub_id='timm/',
            input_size=(3, 336, 336), crop_pct=1.0,
        ),
        'eva02_small_patch14_336.mim_in22k_ft_in1k': _cfg(
            #hf_hub_id='Yuxin-CV/EVA-02', hf_hub_filename='eva02/cls/in1k/eva02_S_pt_in21k_ft_in1k_p14.pt',
            hf_hub_id='timm/',
            input_size=(3, 336, 336), crop_pct=1.0,
        ),
        'eva02_base_patch14_448.mim_in22k_ft_in1k': _cfg(
            #hf_hub_id='Yuxin-CV/EVA-02', hf_hub_filename='eva02/cls/in1k/eva02_B_pt_in21k_ft_in1k_p14.pt',
            hf_hub_id='timm/',
            input_size=(3, 448, 448), crop_pct=1.0,
        ),
        'eva02_large_patch14_448.mim_in22k_ft_in1k': _cfg(
            #hf_hub_id='Yuxin-CV/EVA-02', hf_hub_filename='eva02/cls/in1k/eva02_L_pt_in21k_ft_in1k_p14.pt',
            hf_hub_id='timm/',
            input_size=(3, 448, 448), crop_pct=1.0,
        ),
        'eva02_large_patch14_448.mim_m38m_ft_in1k': _cfg(
            #hf_hub_id='Yuxin-CV/EVA-02', hf_hub_filename='eva02/cls/in1k/eva02_L_pt_m38m_ft_in1k_p14.pt',
            hf_hub_id='timm/',
            input_size=(3, 448, 448), crop_pct=1.0,
        ),

        # in22k or m3m MIM pretrain w/ in22k fine-tune
        'eva02_base_patch14_448.mim_in22k_ft_in22k': _cfg(
            #hf_hub_id='Yuxin-CV/EVA-02', hf_hub_filename='eva02/cls/in21k/eva02_B_pt_in21k_medft_in21k_p14.pt',
            hf_hub_id='timm/',
            input_size=(3, 448, 448), crop_pct=1.0, crop_mode='squash', num_classes=21841,
        ),
        'eva02_large_patch14_448.mim_in22k_ft_in22k': _cfg(
            #hf_hub_id='Yuxin-CV/EVA-02', hf_hub_filename='eva02/cls/in21k/eva02_L_pt_in21k_medft_in21k_p14.pt',
            hf_hub_id='timm/',
            input_size=(3, 448, 448), crop_pct=1.0, crop_mode='squash', num_classes=21841,
        ),
        'eva02_large_patch14_448.mim_m38m_ft_in22k': _cfg(
            #hf_hub_id='Yuxin-CV/EVA-02', hf_hub_filename='eva02/cls/in21k/eva02_L_pt_m38m_medft_in21k_p14.pt',
            hf_hub_id='timm/',
            input_size=(3, 448, 448), crop_pct=1.0, crop_mode='squash', num_classes=21841,
        ),

        # in22k or m38m MIM pretrain
        'eva02_tiny_patch14_224.mim_in22k': _cfg(
            # hf_hub_id='Yuxin-CV/EVA-02', hf_hub_filename='eva02/pt/eva02_Ti_pt_in21k_p14.pt',
            hf_hub_id='timm/',
            num_classes=0,
        ),
        'eva02_small_patch14_224.mim_in22k': _cfg(
            #hf_hub_id='Yuxin-CV/EVA-02', hf_hub_filename='eva02/pt/eva02_S_pt_in21k_p14.pt',
            hf_hub_id='timm/',
            num_classes=0,
        ),
        'eva02_base_patch14_224.mim_in22k': _cfg(
            #hf_hub_id='Yuxin-CV/EVA-02', hf_hub_filename='eva02/pt/eva02_B_pt_in21k_p14.pt',
            hf_hub_id='timm/',
            num_classes=0,
        ),
        'eva02_large_patch14_224.mim_in22k': _cfg(
            #hf_hub_id='Yuxin-CV/EVA-02', hf_hub_filename='eva02/pt/eva02_L_pt_in21k_p14.pt',
            hf_hub_id='timm/',
            num_classes=0,
        ),
        'eva02_large_patch14_224.mim_m38m': _cfg(
            #hf_hub_id='Yuxin-CV/EVA-02', hf_hub_filename='eva02/pt/eva02_L_pt_m38m_p14.pt',
            hf_hub_id='timm/',
            num_classes=0,
        ),

        # EVA01 and EVA02 CLIP image towers
        'eva_giant_patch14_clip_224.laion400m': _cfg(
            # hf_hub_id='QuanSun/EVA-CLIP', hf_hub_filename='EVA01_CLIP_g_14_plus_psz14_s11B.pt',
            # hf_hub_id='timm/eva_giant_patch14_clip_224.laion400m_s11b_b41k',  # float16 weights
            # hf_hub_filename='open_clip_pytorch_model.bin',
            hf_hub_id='timm/',
            num_classes=1024,
        ),
        'eva_giant_patch14_clip_224.merged2b': _cfg(
            # hf_hub_id='QuanSun/EVA-CLIP', hf_hub_filename='EVA01_CLIP_g_14_plus_psz14_s11B.pt',
            # hf_hub_id='timm/eva_giant_patch14_plus_clip_224.merged2b_s11b_b114k',  # float16 weights
            # hf_hub_filename='open_clip_pytorch_model.bin',
            hf_hub_id='timm/',
            num_classes=1024,
        ),
        'eva02_base_patch16_clip_224.merged2b': _cfg(
            # hf_hub_id='QuanSun/EVA-CLIP', hf_hub_filename='EVA02_CLIP_L_psz14_s4B.pt',
            # hf_hub_id='timm/eva02_base_patch16_clip_224.merged2b_s8b_b131k',  # float16 weights
            # hf_hub_filename='open_clip_pytorch_model.bin',
            hf_hub_id='timm/',
            num_classes=512,
        ),
        'eva02_large_patch14_clip_224.merged2b': _cfg(
            # hf_hub_id='QuanSun/EVA-CLIP', hf_hub_filename='EVA02_CLIP_L_psz14_s4B.pt',
            # hf_hub_id='timm/eva02_large_patch14_clip_224.merged2b_s4b_b131k',  # float16 weights
            # hf_hub_filename='open_clip_pytorch_model.bin',
            hf_hub_id='timm/',
            num_classes=768,
        ),
        'eva02_large_patch14_clip_336.merged2b': _cfg(
            # hf_hub_id='QuanSun/EVA-CLIP', hf_hub_filename='EVA02_CLIP_L_psz14_s4B.pt',
            # hf_hub_id='timm/eva02_large_patch14_clip_336.merged2b_s6b_b61k',  # float16 weights
            # hf_hub_filename='open_clip_pytorch_model.bin',
            hf_hub_id='timm/',
            input_size=(3, 336, 336), crop_pct=1.0,
            num_classes=768,
        ),
        'eva02_enormous_patch14_clip_224.laion2b': _cfg(
            # hf_hub_id='QuanSun/EVA-CLIP', hf_hub_filename='EVA02_CLIP_E_psz14_plus_s9B.pt',
            # hf_hub_id='timm/eva02_enormous_patch14_clip_224.laion2b_s4b_b115k',  # float16 weights
            # hf_hub_filename='open_clip_pytorch_model.bin',
            hf_hub_id='timm/',
            num_classes=1024,
        ),
        'eva02_enormous_patch14_clip_224.laion2b_plus': _cfg(
            # hf_hub_id='QuanSun/EVA-CLIP', hf_hub_filename='EVA02_CLIP_E_psz14_plus_s9B.pt',
            # hf_hub_id='timm/eva02_enormous_patch14_plus_clip_224.laion2b_s9b_b144k',  # bfloat16 weights
            # hf_hub_filename='open_clip_pytorch_model.bin',
            hf_hub_id='timm/',
            num_classes=1024,
        ),
        'eva02_enormous_patch14_clip_224.pretrain': _cfg(
            # hf_hub_id='QuanSun/EVA-CLIP', hf_hub_filename='EVA02_E_psz14.pt',
            num_classes=0,
        ),

        'vit_medium_patch16_rope_reg1_gap_256.sbb_in1k': _cfg(
            hf_hub_id='timm/',
            input_size=(3, 256, 256), crop_pct=0.95,
            mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
        ),
        'vit_mediumd_patch16_rope_reg1_gap_256.sbb_in1k': _cfg(
            hf_hub_id='timm/',
            input_size=(3, 256, 256), crop_pct=0.95,
            mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
        ),
        'vit_betwixt_patch16_rope_reg4_gap_256.sbb_in1k': _cfg(
            hf_hub_id='timm/',
            input_size=(3, 256, 256), crop_pct=0.95,
        ),
        'vit_base_patch16_rope_reg1_gap_256.sbb_in1k': _cfg(
            hf_hub_id='timm/',
            input_size=(3, 256, 256), crop_pct=0.95,
            mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
        ),

        # Perception Encoder weights
        'vit_pe_core_tiny_patch16_384.fb': _pe_cfg(
            hf_hub_id='timm/',
            #hf_hub_id='facebook/PE-Core-T16-384',
            #hf_hub_filename='PE-Core-T16-384.pt',
            input_size=(3, 384, 384),
            num_classes=512,  # output proj dim
        ),
        'vit_pe_core_small_patch16_384.fb': _pe_cfg(
            hf_hub_id='timm/',
            #hf_hub_id='facebook/PE-Core-S16-384',
            #hf_hub_filename='PE-Core-S16-384.pt',
            input_size=(3, 384, 384),
            num_classes=512,  # output proj dim
        ),
        'vit_pe_core_base_patch16_224.fb': _pe_cfg(
            hf_hub_id='timm/',
            #hf_hub_id='facebook/PE-Core-B16-224',
            #hf_hub_filename='PE-Core-B16-224.pt',
            input_size=(3, 224, 224),
            num_classes=1024,  # output proj dim
        ),
        'vit_pe_core_large_patch14_336.fb': _pe_cfg(
            hf_hub_id='timm/',
            #hf_hub_id='facebook/PE-Core-L14-336',
            #hf_hub_filename='PE-Core-L14-336.pt',
            input_size=(3, 336, 336),
            num_classes=1024,  # output proj dim
        ),
        'vit_pe_core_gigantic_patch14_448.fb': _pe_cfg(
            hf_hub_id='timm/',
            #hf_hub_id='facebook/PE-Core-G14-448',
            #hf_hub_filename='PE-Core-G14-448.pt',
            input_size=(3, 448, 448),
            num_classes=1280,  # output proj dim
        ),

        'vit_pe_lang_large_patch14_448.fb': _pe_cfg(
            hf_hub_id='timm/',
            #hf_hub_id='facebook/PE-Lang-L14-448',
            #hf_hub_filename='PE-Lang-L14-448.pt',
            input_size=(3, 448, 448),
            num_classes=0,
        ),
        'vit_pe_lang_large_patch14_448.fb_tiling': _pe_cfg(
            hf_hub_id='timm/',
            #hf_hub_id='facebook/PE-Lang-L14-448-Tiling',
            #hf_hub_filename='PE-Lang-L14-448-Tiling.pt',
            input_size=(3, 448, 448),
            num_classes=0,
        ),
        'vit_pe_lang_gigantic_patch14_448.fb': _pe_cfg(
            hf_hub_id='timm/',
            #hf_hub_id='facebook/PE-Lang-G14-448',
            #hf_hub_filename='PE-Lang-G14-448.pt',
            input_size=(3, 448, 448),
            num_classes=0,
        ),
        'vit_pe_lang_gigantic_patch14_448.fb_tiling': _pe_cfg(
            hf_hub_id='timm/',
            #hf_hub_id='facebook/PE-Lang-G14-448-Tiling',
            #hf_hub_filename='PE-Lang-G14-448-Tiling.pt',
            input_size=(3, 448, 448),
            num_classes=0,
        ),

        'vit_pe_spatial_tiny_patch16_512.fb': _pe_cfg(
            hf_hub_id='timm/',
            #hf_hub_id='facebook/PE-Spatial-T16-512',
            #hf_hub_filename='PE-Spatial-T16-512.pt',
            input_size=(3, 512, 512),
            num_classes=0,
        ),
        'vit_pe_spatial_small_patch16_512.fb': _pe_cfg(
            hf_hub_id='timm/',
            #hf_hub_id='facebook/PE-Spatial-S16-512',
            #hf_hub_filename='PE-Spatial-S16-512.pt',
            input_size=(3, 512, 512),
            num_classes=0,
        ),
        'vit_pe_spatial_base_patch16_512.fb': _pe_cfg(
            hf_hub_id='timm/',
            #hf_hub_id='facebook/PE-Spatial-B16-512',
            #hf_hub_filename='PE-Spatial-B16-512.pt',
            input_size=(3, 512, 512),
            num_classes=0,
        ),
        'vit_pe_spatial_large_patch14_448.fb': _pe_cfg(
            hf_hub_id='timm/',
            #hf_hub_id='facebook/PE-Spatial-L14-448',
            #hf_hub_filename='PE-Spatial-L14-448.pt',
            input_size=(3, 448, 448),
            num_classes=0,
        ),
        'vit_pe_spatial_gigantic_patch14_448.fb': _pe_cfg(
            hf_hub_id='timm/',
            #hf_hub_id='facebook/PE-Spatial-G14-448',
            #hf_hub_filename='PE-Spatial-G14-448.pt',
            input_size=(3, 448, 448),
            num_classes=0,
        ),

        # RoPE-ViT models from Naver
        'vit_small_patch16_rope_224.naver_in1k': _cfg(
            hf_hub_id='timm/',
            mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD,
            license='apache-2.0',
        ),
        'vit_base_patch16_rope_224.naver_in1k': _cfg(
            hf_hub_id='timm/',
            mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD,
            license='apache-2.0',
        ),
        'vit_large_patch16_rope_224.naver_in1k': _cfg(
            hf_hub_id='timm/',
            mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD,
            license='apache-2.0',
        ),
        'vit_small_patch16_rope_mixed_224.naver_in1k': _cfg(
            hf_hub_id='timm/',
            mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD,
            license='apache-2.0',
        ),
        'vit_base_patch16_rope_mixed_224.naver_in1k': _cfg(
            hf_hub_id='timm/',
            mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD,
            license='apache-2.0',
        ),
        'vit_large_patch16_rope_mixed_224.naver_in1k': _cfg(
            hf_hub_id='timm/',
            mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD,
            license='apache-2.0',
        ),
        'vit_small_patch16_rope_ape_224.naver_in1k': _cfg(
            hf_hub_id='timm/',
            mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD,
            license='apache-2.0',
        ),
        'vit_base_patch16_rope_ape_224.naver_in1k': _cfg(
            hf_hub_id='timm/',
            mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD,
            license='apache-2.0',
        ),
        'vit_large_patch16_rope_ape_224.naver_in1k': _cfg(
            hf_hub_id='timm/',
            mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD,
            license='apache-2.0',
        ),
        'vit_small_patch16_rope_mixed_ape_224.naver_in1k': _cfg(
            hf_hub_id='timm/',
            mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD,
            license='apache-2.0',
        ),
        'vit_base_patch16_rope_mixed_ape_224.naver_in1k': _cfg(
            hf_hub_id='timm/',
            mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD,
            license='apache-2.0',
        ),
        'vit_large_patch16_rope_mixed_ape_224.naver_in1k': _cfg(
            hf_hub_id='timm/',
            mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD,
            license='apache-2.0',
        ),

        # DINOv3 weights are under a specific license with redistribution terms, please see
        # https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md
        'vit_small_patch16_dinov3.lvd1689m': _dinov3_cfg(
            hf_hub_id='timm/',
        ),
        'vit_small_patch16_dinov3_qkvb.lvd1689m': _dinov3_cfg(
            hf_hub_id='timm/',
        ),
        'vit_small_plus_patch16_dinov3.lvd1689m': _dinov3_cfg(
            hf_hub_id='timm/',
        ),
        'vit_small_plus_patch16_dinov3_qkvb.lvd1689m': _dinov3_cfg(
            hf_hub_id='timm/',
        ),
        'vit_base_patch16_dinov3.lvd1689m': _dinov3_cfg(
            hf_hub_id='timm/',
        ),
        'vit_base_patch16_dinov3_qkvb.lvd1689m': _dinov3_cfg(
            hf_hub_id='timm/',
        ),
        'vit_large_patch16_dinov3.lvd1689m': _dinov3_cfg(
            hf_hub_id='timm/',
        ),
        'vit_large_patch16_dinov3_qkvb.lvd1689m': _dinov3_cfg(
            hf_hub_id='timm/',
        ),
        'vit_large_patch16_dinov3.sat493m': _dinov3_cfg(
            hf_hub_id='timm/',
            mean=(0.430, 0.411, 0.296), std=(0.213, 0.156, 0.143),
        ),
        'vit_large_patch16_dinov3_qkvb.sat493m': _dinov3_cfg(
            hf_hub_id='timm/',
            mean=(0.430, 0.411, 0.296), std=(0.213, 0.156, 0.143),
        ),
        'vit_huge_plus_patch16_dinov3.lvd1689m': _dinov3_cfg(
            hf_hub_id='timm/',
        ),
        'vit_huge_plus_patch16_dinov3_qkvb.lvd1689m': _dinov3_cfg(
            hf_hub_id='timm/',
        ),
        'vit_7b_patch16_dinov3.lvd1689m': _dinov3_cfg(
            hf_hub_id='timm/',
        ),
        'vit_7b_patch16_dinov3.sat493m': _dinov3_cfg(
            hf_hub_id='timm/',
            mean=(0.430, 0.411, 0.296), std=(0.213, 0.156, 0.143),
        ),

    })


    # Restrict cfgs to relpos variants to avoid conflicts with base EVA registrations.
    default_cfgs = generate_default_cfgs({
        'vit_relpos_small_patch16_dinov3': _dinov3_cfg(),
        'vit_relpos_base_patch16_dinov3': _dinov3_cfg(),
        'vit_relpos_large_patch16_dinov3': _dinov3_cfg(),
    })





    @register_model
    def vit_relpos_small_patch16_dinov3(pretrained: bool = False, **kwargs) -> Eva:
        """DINOv3 S/16 with RelPos (no AbsPos, no RoPE)."""
        if pretrained:
            raise RuntimeError('No pretrained weights are provided for vit_relpos_small_patch16_dinov3.')
        model_args = dict(
            img_size=256,
            patch_size=16,
            dynamic_img_size=False,
            embed_dim=384,
            depth=12,
            num_heads=6,
            qkv_bias=False,
            init_values=1.0e-05,  # layer-scale
            use_rot_pos_emb=False,
            use_abs_pos_emb=False,
            use_relpos_pos_emb=True,
            num_reg_tokens=4,
            use_fc_norm=False,
            rel_pos_type='mlp',
            shared_rel_pos=True,
        )
        model = _create_eva('vit_relpos_small_patch16_dinov3', pretrained=pretrained, **dict(model_args, **kwargs))
        return model


    @register_model
    def vit_relpos_base_patch16_dinov3(pretrained: bool = False, **kwargs) -> Eva:
        """DINOv3 B/16 with RelPos (no AbsPos, no RoPE)."""
        if pretrained:
            raise RuntimeError('No pretrained weights are provided for vit_relpos_base_patch16_dinov3.')
        model_args = dict(
            img_size=256,
            patch_size=16,
            dynamic_img_size=False,
            embed_dim=768,
            depth=12,
            num_heads=12,
            qkv_bias=False,
            init_values=1.0e-05,  # layer-scale
            use_rot_pos_emb=False,
            use_abs_pos_emb=False,
            use_relpos_pos_emb=True,
            num_reg_tokens=4,
            use_fc_norm=False,
            rel_pos_type='mlp',
            shared_rel_pos=True,
        )
        model = _create_eva('vit_relpos_base_patch16_dinov3', pretrained=pretrained, **dict(model_args, **kwargs))
        return model


    @register_model
    def vit_relpos_large_patch16_dinov3(pretrained: bool = False, **kwargs) -> Eva:
        """DINOv3 L/16 with RelPos (no AbsPos, no RoPE)."""
        if pretrained:
            raise RuntimeError('No pretrained weights are provided for vit_relpos_large_patch16_dinov3.')
        model_args = dict(
            img_size=256,
            patch_size=16,
            dynamic_img_size=False,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            qkv_bias=False,
            init_values=1.0e-05,  # layer-scale
            use_rot_pos_emb=False,
            use_abs_pos_emb=False,
            use_relpos_pos_emb=True,
            num_reg_tokens=4,
            use_fc_norm=False,
            rel_pos_type='mlp',
            shared_rel_pos=True,
        )
        model = _create_eva('vit_relpos_large_patch16_dinov3', pretrained=pretrained, **dict(model_args, **kwargs))
        return model

if args.pos_type == 'alibi':
    import itertools
    import math
    import os
    from functools import partial
    from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
    from timm.layers import (
        PatchEmbed,
        Mlp,
        GluMlp,
        SwiGLU,
        LayerNorm,
        DropPath, calculate_drop_path_rates,
        PatchDropoutWithIndices,
        create_rope_embed,
        apply_rot_embed_cat,
        apply_keep_indices_nlc,
        trunc_normal_,
        resample_patch_embed,
        resample_abs_pos_embed,
        global_pool_nlc,
        to_2tuple,
        use_fused_attn,
        maybe_add_mask,
        AttentionRope,
        AttentionPoolLatent,
    )
    from timm.models._builder import build_model_with_cfg
    from timm.models._features import feature_take_indices
    from timm.models._manipulate import named_apply, checkpoint
    from timm.models._registry import generate_default_cfgs, register_model


    __all__ = ['Eva']


    class EvaAttention(nn.Module):
        """ EVA Attention with ROPE, no k-bias, and fused/unfused qkv options
        """
        fused_attn: torch.jit.Final[bool]

        def __init__(
                self,
                dim: int,
                num_heads: int = 8,
                qkv_bias: bool = True,
                qkv_fused: bool = True,
                qkv_bias_separate: bool = False,
                num_prefix_tokens: int = 1,
                attn_drop: float = 0.,
                proj_drop: float = 0.,
                attn_head_dim: Optional[int] = None,
                norm_layer: Optional[Callable] = None,
                qk_norm: bool = False,
                scale_norm: bool = True,
                rotate_half: bool = False,
                alibi: bool = False,
                device=None,
                dtype=None,
        ):
            """
            Args:
                dim: Input dimension of the token embeddings
                num_heads: Number of attention heads
                qkv_bias: Whether to add a bias term to the query, key, and value projections
                qkv_fused: Whether qkv projections are fused into one projection or separate
                qkv_bias_separate: Whether to apply bias to qkv as a separate addition or part of F.linear() call
                num_prefix_tokens: Number of reg/cls tokens at the beginning of the sequence that
                    should not have position embeddings applied
                attn_drop: Dropout rate for attention weights
                proj_drop: Dropout rate for the output projection
                attn_head_dim: Dimension of each attention head (if None, computed as dim // num_heads)
                norm_layer: Normalization layer constructor to use for QK and scale normalization
                qk_norm: Enable normalization of query (Q) and key (K) vectors with norm_layer
                scale_norm: Enable normalization (scaling) of attention output with norm_layer
                rotate_half: Use half rotation layout instead of interleaved
                alibi: Enable ALiBi bias (additive, non-trainable)
            """
            dd = {'device': device, 'dtype': dtype}
            super().__init__()
            if scale_norm or qk_norm:
                assert norm_layer is not None, 'norm_layer must be provided if qk_norm or scale_norm is True'
            self.num_heads = num_heads
            head_dim = dim // num_heads
            if attn_head_dim is not None:
                head_dim = attn_head_dim
            attn_dim = head_dim * self.num_heads
            self.scale = head_dim ** -0.5
            self.head_dim = head_dim
            self.num_prefix_tokens = num_prefix_tokens
            self.fused_attn = use_fused_attn()
            self.qkv_bias_separate = qkv_bias_separate
            self.rotate_half = rotate_half
            self.use_alibi = alibi

            if qkv_fused:
                self.qkv = nn.Linear(dim, attn_dim * 3, bias=False, **dd)
                self.q_proj = self.k_proj = self.v_proj = None
                if qkv_bias:
                    self.q_bias = nn.Parameter(torch.zeros(attn_dim, **dd))
                    self.register_buffer('k_bias', torch.zeros(attn_dim, **dd), persistent=False)
                    self.v_bias = nn.Parameter(torch.zeros(attn_dim, **dd))
                else:
                    self.q_bias = self.k_bias = self.v_bias = None
            else:
                self.q_proj = nn.Linear(dim, attn_dim, bias=qkv_bias, **dd)
                self.k_proj = nn.Linear(dim, attn_dim, bias=False, **dd)
                self.v_proj = nn.Linear(dim, attn_dim, bias=qkv_bias, **dd)
                self.qkv = None
                self.q_bias = self.k_bias = self.v_bias = None
            self.q_norm = norm_layer(self.head_dim, **dd) if qk_norm else nn.Identity()
            self.k_norm = norm_layer(self.head_dim, **dd) if qk_norm else nn.Identity()
            self.attn_drop = nn.Dropout(attn_drop)
            self.norm = norm_layer(attn_dim, **dd) if scale_norm else nn.Identity()
            self.proj = nn.Linear(attn_dim, dim, **dd)
            self.proj_drop = nn.Dropout(proj_drop)

        def forward(
                self,
                x,
                rope: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None,
                alibi_bias: Optional[torch.Tensor] = None,
        ):
            """Forward pass for the attention module.

            Args:
                x: Input tensor of shape (batch_size, sequence_length, embedding_dim)
                rope: Rotary position embeddings tensor for position-aware attention
                attn_mask: Optional attention mask to apply during attention computation

            Returns:
                Tensor of shape (batch_size, sequence_length, embedding_dim)
            """
            B, N, C = x.shape

            if self.qkv is not None:
                if self.q_bias is None:
                    qkv = self.qkv(x)
                else:
                    qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
                    if self.qkv_bias_separate:
                        qkv = self.qkv(x)
                        qkv += qkv_bias
                    else:
                        qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
                qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
            else:
                q = self.q_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)  # B, num_heads, N, C
                k = self.k_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)
                v = self.v_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)

            q, k = self.q_norm(q), self.k_norm(k)

            if rope is not None:
                npt = self.num_prefix_tokens
                half = getattr(self, 'rotate_half', False)
                q = torch.cat([q[:, :, :npt, :], apply_rot_embed_cat(q[:, :, npt:, :], rope, half=half)], dim=2).type_as(v)
                k = torch.cat([k[:, :, :npt, :], apply_rot_embed_cat(k[:, :, npt:, :], rope, half=half)], dim=2).type_as(v)

            attn_bias = alibi_bias
            if self.fused_attn:
                if attn_mask is not None:
                    if attn_mask.dtype == torch.bool:
                        attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype).masked_fill(attn_mask, float('-inf'))
                    elif attn_mask.dtype != q.dtype:
                        attn_mask = attn_mask.to(dtype=q.dtype)
                    attn_bias = attn_mask if attn_bias is None else attn_bias + attn_mask
                x = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_bias,
                    dropout_p=self.attn_drop.p if self.training else 0.,
                )
            else:
                q = q * self.scale
                attn = (q @ k.transpose(-2, -1))
                if attn_bias is not None:
                    attn = attn + attn_bias
                attn = maybe_add_mask(attn, attn_mask)
                attn = attn.softmax(dim=-1)

                attn = self.attn_drop(attn)
                x = attn @ v

            x = x.transpose(1, 2).reshape(B, N, C)
            x = self.norm(x)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x


    class EvaBlock(nn.Module):

        def __init__(
                self,
                dim: int,
                num_heads: int,
                qkv_bias: bool = True,
                qkv_fused: bool = True,
                mlp_ratio: float = 4.,
                swiglu_mlp: bool = False,
                swiglu_align_to: int = 0,
                scale_mlp: bool = False,
                scale_attn_inner: bool = False,
                num_prefix_tokens: int = 1,
                attn_type: str = 'eva',
                rotate_half: bool = False,
                proj_drop: float = 0.,
                attn_drop: float = 0.,
                drop_path: float = 0.,
                alibi: bool = False,
                init_values: Optional[float] = None,
                act_layer: Callable = nn.GELU,
                norm_layer: Callable = LayerNorm,
                attn_head_dim: Optional[int] = None,
                device=None,
                dtype=None,
                **kwargs,
        ):
            """ Initialize the EVA transformer block.

            Args:
            dim: Input dimension of the token embeddings
                num_heads: Number of attention heads
                qkv_bias: Whether to use bias terms in query, key, value projections
                qkv_fused: Whether to use a single projection for query, key, value
                mlp_ratio: Ratio of MLP hidden dimension to input dimension
                swiglu_mlp: Whether to use SwiGLU activation in the MLP
                scale_mlp: Whether to use normalization in the MLP
                scale_attn_inner: Whether to use normalization within the attention mechanism
                num_prefix_tokens: Number of tokens at the beginning of the sequence (class tokens, etc.)
                attn_type: Type of attention module to use ('eva' or 'rope')
                proj_drop: Dropout rate for projection layers
                attn_drop: Dropout rate for attention matrix
                drop_path: Stochastic depth rate
                init_values: Initial value for LayerScale, None = no LayerScale
                act_layer: Activation layer constructor
                norm_layer: Normalization layer constructor
                attn_head_dim: Dimension of each attention head (if None, computed as dim // num_heads)
            """
            dd = {'device': device, 'dtype': dtype}
            super().__init__()

            self.norm1 = norm_layer(dim, **dd)
            attn_cls = AttentionRope if attn_type == 'rope' else EvaAttention
            self.attn = attn_cls(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qkv_fused=qkv_fused,
                num_prefix_tokens=num_prefix_tokens,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                attn_head_dim=attn_head_dim,
                norm_layer=norm_layer,
                scale_norm=scale_attn_inner,
                rotate_half=rotate_half,
                alibi=alibi,
                **dd,
            )
            self.gamma_1 = nn.Parameter(init_values * torch.ones(dim, **dd)) if init_values is not None else None
            self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

            self.norm2 = norm_layer(dim, **dd)
            hidden_features = int(dim * mlp_ratio)
            if swiglu_mlp:
                if scale_mlp or swiglu_align_to:
                    # when norm in SwiGLU used or alignment enabled, an impl with separate fc for gate & x is used
                    self.mlp = SwiGLU(
                        in_features=dim,
                        hidden_features=hidden_features,
                        norm_layer=norm_layer if scale_mlp else None,
                        drop=proj_drop,
                        align_to=swiglu_align_to,
                        **dd,
                    )
                else:
                    # w/o any extra norm, an impl with packed weights is used
                    self.mlp = GluMlp(
                        in_features=dim,
                        hidden_features=hidden_features * 2,
                        norm_layer=norm_layer if scale_mlp else None,
                        act_layer=nn.SiLU,
                        gate_last=False,
                        drop=proj_drop,
                        **dd,
                    )
            else:
                self.mlp = Mlp(
                    in_features=dim,
                    hidden_features=hidden_features,
                    act_layer=act_layer,
                    norm_layer=norm_layer if scale_mlp else None,
                    drop=proj_drop,
                    **dd,
                )
            self.gamma_2 = nn.Parameter(init_values * torch.ones(dim, **dd)) if init_values is not None else None
            self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        def forward(
                self,
                x: torch.Tensor,
                rope: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None,
                alibi_bias: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            if self.gamma_1 is None:
                x = x + self.drop_path1(self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask, alibi_bias=alibi_bias))
                x = x + self.drop_path2(self.mlp(self.norm2(x)))
            else:
                x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask, alibi_bias=alibi_bias))
                x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
            return x


    class EvaBlockPostNorm(nn.Module):
        """ EVA block w/ post-norm and support for swiglu, MLP norm scale, ROPE. """
        def __init__(
                self,
                dim: int,
                num_heads: int,
                qkv_bias: bool = True,
                qkv_fused: bool = True,
                mlp_ratio: float = 4.,
                attn_type: str = 'eva',
                rotate_half: bool = False,
                swiglu_mlp: bool = False,
                swiglu_align_to: int = 0,
                scale_mlp: bool = False,
                scale_attn_inner: bool = False,
                num_prefix_tokens: int = 1,
                proj_drop: float = 0.,
                attn_drop: float = 0.,
                drop_path: float = 0.,
                alibi: bool = False,
                init_values: Optional[float] = None,  # ignore for post-norm
                act_layer: Callable = nn.GELU,
                norm_layer: Callable = nn.LayerNorm,
                attn_head_dim: Optional[int] = None,
                device=None,
                dtype=None,
        ):
            """ Initialize the post-norm EVA transformer block.

            Args:
            dim: Input dimension of the token embeddings
                num_heads: Number of attention heads
                qkv_bias: Whether to use bias terms in query, key, value projections
                qkv_fused: Whether to use a single projection for query, key, value
                mlp_ratio: Ratio of MLP hidden dimension to input dimension
                swiglu_mlp: Whether to use SwiGLU activation in the MLP
                scale_mlp: Whether to use normalization in the MLP
                scale_attn_inner: Whether to use normalization within the attention mechanism
                num_prefix_tokens: Number of tokens at the beginning of the sequence (class tokens, etc.)
                attn_type: Type of attention module to use ('eva' or 'rope')
                proj_drop: Dropout rate for projection layers
                attn_drop: Dropout rate for attention matrix
                drop_path: Stochastic depth rate
                init_values: Initial value for LayerScale, None = no LayerScale (NOTE: ignored for post-norm block)
                act_layer: Activation layer constructor
                norm_layer: Normalization layer constructor
                attn_head_dim: Dimension of each attention head (if None, computed as dim // num_heads)
            """
            dd = {'device': device, 'dtype': dtype}
            super().__init__()

            attn_cls = AttentionRope if attn_type == 'rope' else EvaAttention
            self.attn = attn_cls(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qkv_fused=qkv_fused,
                num_prefix_tokens=num_prefix_tokens,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                attn_head_dim=attn_head_dim,
                norm_layer=norm_layer,
                scale_norm=scale_attn_inner,
                rotate_half=rotate_half,
                alibi=alibi,
                **dd,
            )
            self.norm1 = norm_layer(dim, **dd)
            self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

            hidden_features = int(dim * mlp_ratio)
            if swiglu_mlp:
                if scale_mlp:
                    # when norm in SwiGLU used, an impl with separate fc for gate & x is used
                    self.mlp = SwiGLU(
                        in_features=dim,
                        hidden_features=hidden_features,
                        norm_layer=norm_layer if scale_mlp else None,
                        drop=proj_drop,
                        align_to=swiglu_align_to,
                        **dd,
                    )
                else:
                    # w/o any extra norm, an impl with packed fc1 weights is used, matches existing GluMLP
                    self.mlp = GluMlp(
                        in_features=dim,
                        hidden_features=hidden_features * 2,
                        norm_layer=norm_layer if scale_mlp else None,
                        act_layer=nn.SiLU,
                        gate_last=False,
                        drop=proj_drop,
                        **dd,
                    )
            else:
                self.mlp = Mlp(
                    in_features=dim,
                    hidden_features=hidden_features,
                    act_layer=act_layer,
                    norm_layer=norm_layer if scale_mlp else None,
                    drop=proj_drop,
                    **dd,
                )
            self.norm2 = norm_layer(dim, **dd)
            self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        def forward(
                self,
                x: torch.Tensor,
                rope: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None,
                alibi_bias: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            x = x + self.drop_path1(self.norm1(self.attn(x, rope=rope, attn_mask=attn_mask, alibi_bias=alibi_bias)))
            x = x + self.drop_path2(self.norm2(self.mlp(x)))
            return x


    class Eva(nn.Module):
        """ Eva Vision Transformer w/ Abs & Rotary Pos Embed

        This class implements the EVA and EVA02 models that were based on the BEiT ViT variant
        * EVA - abs pos embed, global avg pool
        * EVA02 - abs + rope pos embed, global avg pool, SwiGLU, scale Norm in MLP (ala normformer)
        """

        def __init__(
                self,
                img_size: Union[int, Tuple[int, int]] = 224,
                patch_size: Union[int, Tuple[int, int]] = 16,
                in_chans: int = 3,
                num_classes: int = 1000,
                global_pool: str = 'avg',
                embed_dim: int = 768,
                depth: int = 12,
                num_heads: int = 12,
                qkv_bias: bool = True,
                qkv_fused: bool = True,
                mlp_ratio: float = 4.,
                swiglu_mlp: bool = False,
                swiglu_align_to: int = 0,
                scale_mlp: bool = False,
                scale_attn_inner: bool = False,
                attn_type: str = 'eva',
                drop_rate: float = 0.,
                pos_drop_rate: float = 0.,
                patch_drop_rate: float = 0.,
                proj_drop_rate: float = 0.,
                attn_drop_rate: float = 0.,
                drop_path_rate: float = 0.,
                norm_layer: Callable = LayerNorm,
                init_values: Optional[float] = None,
                class_token: bool = True,
                num_reg_tokens: int = 0,
                no_embed_class: bool = False,
                use_abs_pos_emb: bool = True,
                use_rot_pos_emb: bool = False,
                use_alibi_pos_emb: bool = False,
                rope_type: Optional[str] = 'cat',
                rope_grid_offset: float = 0.,
                rope_grid_indexing: str = 'ij',
                rope_temperature: float = 10000.,
                rope_rotate_half: bool = False,
                use_post_norm: bool = False,
                use_pre_transformer_norm: bool = False,
                use_post_transformer_norm: Optional[bool] = None,
                use_fc_norm: Optional[bool] = None,
                attn_pool_num_heads: Optional[int] = None,
                attn_pool_mlp_ratio: Optional[float] = None,
                dynamic_img_size: bool = False,
                dynamic_img_pad: bool = False,
                ref_feat_shape: Optional[Union[Tuple[int, int], int]] = None,
                head_init_scale: float = 0.001,
                device=None,
                dtype=None,
        ):
            """Initialize the EVA Vision Transformer model.

            Args:
                img_size: Input image size (single int for square, or tuple for rectangular)
                patch_size: Patch size to divide image into tokens (single int for square, or tuple)
                in_chans: Number of input image channels
                num_classes: Number of classes (output dim) for classification head (final projection), 0 for pass-through
                global_pool: Type of global pooling for final sequence ('avg', 'token', 'map', etc.)
                embed_dim: Embedding dimension for tokens
                depth: Number of transformer blocks
                num_heads: Number of attention heads
                qkv_bias: Enable bias for query, key, value projections
                qkv_fused: Use a single projection for query, key, value
                mlp_ratio: Ratio of mlp hidden dim to embedding dim
                swiglu_mlp: Use SwiGLU activation in MLP
                scale_mlp: Apply scaling normalization in MLP (normformer style)
                scale_attn_inner: Apply scaling normalization inside attention
                attn_type: Type of attention module to use
                drop_rate: Dropout rate after final projection and pooling
                pos_drop_rate: Dropout rate for positional embeddings
                patch_drop_rate: Rate of dropping patches during training
                proj_drop_rate: Dropout rate for projections
                attn_drop_rate: Dropout rate for attention
                drop_path_rate: Stochastic depth rate
                norm_layer: Normalization layer constructor
                init_values: Initial layer-scale values
                class_token: Use class token
                num_reg_tokens: Number of additional learnable 'register' tokens to add to the sequence
                no_embed_class: Don't include position embeddings for class (or reg) tokens
                use_abs_pos_emb: Use absolute (learned) positional embeddings
                use_rot_pos_emb: Use rotary position embeddings
                use_alibi_pos_emb: Use ALiBi positional bias (adds bias to attention logits)
                rope_type: Type of RoPE to use ('cat', 'mixed', 'dinov3', etc.).
                rope_grid_offset: Offset for rotary position embedding grid
                rope_grid_indexing: Indexing mode for rotary position embeddings ('ij' or 'xy')
                rope_temperature: Temperature parameter for ROPE frequency computation
                rope_rotate_half: Use half rotation layout (rotate D/2 dims), else use interleaved rotation layout
                use_post_norm: Use post-norm transformer block type
                use_pre_transformer_norm: Use normalization layer before transformer blocks
                use_post_transformer_norm: Use normalization layer after transformer blocks
                use_fc_norm: Use normalization layer after pooling, before final classifier
                attn_pool_num_heads: Number of heads in attention pooling
                attn_pool_mlp_ratio: MLP ratio in attention pooling
                dynamic_img_size: Support dynamic image sizes in forward pass
                dynamic_img_pad: Apply dynamic padding for irregular image sizes
                ref_feat_shape: Reference feature shape for rotary position embedding scale
                head_init_scale: Initialization scale for classification head weights
            """
            super().__init__()
            dd = {'device': device, 'dtype': dtype}
            assert global_pool in ('', 'avg', 'avgmax', 'max', 'token', 'map')
            self.num_classes = num_classes
            self.global_pool = global_pool
            self.num_features = self.head_hidden_size = self.embed_dim = embed_dim  # for consistency with other models
            self.num_prefix_tokens = (1 if class_token else 0) + num_reg_tokens
            self.no_embed_class = no_embed_class
            self.dynamic_img_size = dynamic_img_size
            self.grad_checkpointing = False

            # resolve norm / pool usage
            activate_pre_norm = use_pre_transformer_norm
            if use_fc_norm is not None:
                activate_fc_norm = use_fc_norm  # pass through if explicit
            else:
                activate_fc_norm = global_pool == 'avg'  # default on if avg pool used
            if use_post_transformer_norm is not None:
                activate_post_norm = use_post_transformer_norm  # pass through if explicit
            else:
                activate_post_norm = not activate_fc_norm  # default on if fc_norm isn't active

            embed_args = {}
            if dynamic_img_size:
                # flatten deferred until after pos embed
                embed_args.update(dict(strict_img_size=False, output_fmt='NHWC'))
            self.patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
                dynamic_img_pad=dynamic_img_pad,
                bias=not use_pre_transformer_norm,
                **embed_args,
                **dd,
            )
            num_patches = self.patch_embed.num_patches
            r = self.patch_embed.feat_ratio() if hasattr(self.patch_embed, 'feat_ratio') else patch_size

            self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim, **dd)) if class_token else None
            self.reg_token = nn.Parameter(torch.empty(1, num_reg_tokens, embed_dim, **dd)) if num_reg_tokens else None
            self.cls_embed = class_token and self.reg_token is None

            num_pos_tokens = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
            self.pos_embed = nn.Parameter(torch.empty(1, num_pos_tokens, embed_dim, **dd)) if use_abs_pos_emb else None
            self.pos_drop = nn.Dropout(p=pos_drop_rate)
            if patch_drop_rate > 0:
                self.patch_drop = PatchDropoutWithIndices(patch_drop_rate, num_prefix_tokens=self.num_prefix_tokens)
            else:
                self.patch_drop = None

            self.alibi = None
            if use_alibi_pos_emb:
                assert not dynamic_img_size, 'ALiBi currently requires dynamic_img_size=False (fixed patch grid).'
                assert patch_drop_rate == 0., 'ALiBi currently requires patch_drop_rate=0 to keep dense patch grid.'
                assert not use_rot_pos_emb, 'ALiBi is mutually exclusive with RoPE in this implementation.'
                self.alibi = get_2dalibi(num_heads=num_heads, num_patches=num_patches)

            self.rope_mixed = False
            if use_rot_pos_emb:
                ref_feat_shape = to_2tuple(ref_feat_shape) if ref_feat_shape is not None else None

                # Setup RoPE kwargs
                rope_kwargs = dict(
                    dim=embed_dim,
                    num_heads=num_heads,
                    feat_shape=None if dynamic_img_size else self.patch_embed.grid_size,
                    temperature=rope_temperature,
                    grid_indexing=rope_grid_indexing,
                    **dd,
                )
                if rope_type == 'mixed':
                    rope_kwargs.update(dict(depth=depth))
                    self.rope_mixed = True
                elif rope_type == 'cat':
                    rope_kwargs.update(dict(
                        in_pixels=False,
                        grid_offset=rope_grid_offset,
                        ref_feat_shape=ref_feat_shape,
                    ))

                self.rope = create_rope_embed(rope_type=rope_type, **rope_kwargs)
            else:
                self.rope = None

            self.norm_pre = norm_layer(embed_dim, **dd) if activate_pre_norm else nn.Identity()

            dpr = calculate_drop_path_rates(drop_path_rate, depth)  # stochastic depth decay rule
            block_fn = EvaBlockPostNorm if use_post_norm else EvaBlock
            self.blocks = nn.ModuleList([
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    qkv_fused=qkv_fused,
                    mlp_ratio=mlp_ratio,
                    swiglu_mlp=swiglu_mlp,
                    swiglu_align_to=swiglu_align_to,
                    scale_mlp=scale_mlp,
                    scale_attn_inner=scale_attn_inner,
                    attn_type=attn_type,
                    rotate_half=rope_rotate_half,
                    num_prefix_tokens=self.num_prefix_tokens,
                    alibi=use_alibi_pos_emb,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    init_values=init_values,
                    **dd,
                )
                for i in range(depth)])
            self.feature_info = [
                dict(module=f'blocks.{i}', num_chs=embed_dim, reduction=r) for i in range(depth)]

            self.norm = norm_layer(embed_dim, **dd) if activate_post_norm else nn.Identity()

            if global_pool == 'map':
                self.attn_pool = AttentionPoolLatent(
                    self.embed_dim,
                    num_heads=attn_pool_num_heads or num_heads,
                    mlp_ratio=attn_pool_mlp_ratio or mlp_ratio,
                    norm_layer=norm_layer,
                    act_layer=nn.GELU,
                    **dd,
                )
            else:
                self.attn_pool = None
            self.fc_norm = norm_layer(embed_dim, **dd) if activate_fc_norm else nn.Identity()
            self.head_drop = nn.Dropout(drop_rate)
            self.head = nn.Linear(embed_dim, num_classes, **dd) if num_classes > 0 else nn.Identity()

            self.init_weights(head_init_scale=head_init_scale)

        def init_weights(self, head_init_scale=None):
            self.apply(self._init_weights)
            if self.pos_embed is not None:
                trunc_normal_(self.pos_embed, std=.02)
            if self.cls_token is not None:
                trunc_normal_(self.cls_token, std=.02)
            if self.reg_token is not None:
                trunc_normal_(self.reg_token, std=.02)
            self.fix_init_weight()
            if head_init_scale and isinstance(self.head, nn.Linear):
                trunc_normal_(self.head.weight, std=.02)
                self.head.weight.data.mul_(head_init_scale)
                self.head.bias.data.mul_(head_init_scale)

        def fix_init_weight(self) -> None:
            """Fix initialization weights by rescaling based on layer depth."""
            def rescale(param, layer_id):
                param.div_(math.sqrt(2.0 * layer_id))

            for layer_id, layer in enumerate(self.blocks):
                rescale(layer.attn.proj.weight.data, layer_id + 1)
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

        def _init_weights(self, m: nn.Module) -> None:
            """Initialize weights for Linear layers.

            Args:
                m: Module to initialize.
            """
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        @torch.jit.ignore
        def no_weight_decay(self) -> Set[str]:
            """Parameters to exclude from weight decay."""
            nwd = {'pos_embed', 'cls_token'}
            if (rope := getattr(self, "rope", None)) and hasattr(rope, "no_weight_decay"):
                return nwd | {f"rope.{p}" for p in rope.no_weight_decay()}
            return nwd

        @torch.jit.ignore
        def set_grad_checkpointing(self, enable: bool = True) -> None:
            """Enable or disable gradient checkpointing."""
            self.grad_checkpointing = enable

        @torch.jit.ignore
        def group_matcher(self, coarse: bool = False) -> Dict[str, Any]:
            """Create layer groupings for optimization."""
            matcher = dict(
                stem=r'^cls_token|pos_embed|patch_embed',  # stem and embed
                blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))],
            )
            return matcher

        @torch.jit.ignore
        def get_classifier(self) -> nn.Module:
            return self.head

        def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None) -> None:
            """Reset the classifier head.

            Args:
                num_classes: Number of output classes.
                global_pool: Global pooling type.
            """
            self.num_classes = num_classes
            if global_pool is not None:
                self.global_pool = global_pool
            self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        def set_input_size(
                self,
                img_size: Optional[Tuple[int, int]] = None,
                patch_size: Optional[Tuple[int, int]] = None,
        ) -> None:
            """Update the input image resolution and patch size.

            Args:
                img_size: New input resolution, if None current resolution is used.
                patch_size: New patch size, if None existing patch size is used.
            """
            prev_grid_size = self.patch_embed.grid_size
            self.patch_embed.set_input_size(img_size=img_size, patch_size=patch_size)

            if self.pos_embed is not None:
                num_prefix_tokens = 0 if self.no_embed_class else self.num_prefix_tokens
                num_new_tokens = self.patch_embed.num_patches + num_prefix_tokens
                if num_new_tokens != self.pos_embed.shape[1]:
                    self.pos_embed = nn.Parameter(resample_abs_pos_embed(
                        self.pos_embed,
                        new_size=self.patch_embed.grid_size,
                        old_size=prev_grid_size,
                        num_prefix_tokens=num_prefix_tokens,
                        verbose=True,
                    ))

            if self.rope is not None:
                self.rope.update_feat_shape(self.patch_embed.grid_size)

        def _pos_embed(self, x) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            if self.dynamic_img_size:
                B, H, W, C = x.shape
                if self.pos_embed is not None:
                    prev_grid_size = self.patch_embed.grid_size
                    pos_embed = resample_abs_pos_embed(
                        self.pos_embed,
                        new_size=(H, W),
                        old_size=prev_grid_size,
                        num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
                    )
                else:
                    pos_embed = None
                x = x.view(B, -1, C)
                rot_pos_embed = self.rope.get_embed(shape=(H, W)) if self.rope is not None else None
            else:
                pos_embed = self.pos_embed
                rot_pos_embed = self.rope.get_embed() if self.rope is not None else None

            to_cat = []
            if self.cls_token is not None:
                to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
            if self.reg_token is not None:
                to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

            if self.no_embed_class:
                # position embedding does not overlap with class / reg token
                if pos_embed is not None:
                    x = x + pos_embed
                if to_cat:
                    x = torch.cat(to_cat + [x], dim=1)
            else:
                # pos_embed has entry for class / reg token, concat then add
                if to_cat:
                    x = torch.cat(to_cat + [x], dim=1)
                if pos_embed is not None:
                    x = x + pos_embed

            x = self.pos_drop(x)

            # apply patch dropout to patches and rotary position embedding
            if self.patch_drop is not None:
                x, keep_indices = self.patch_drop(x)
                if rot_pos_embed is not None and keep_indices is not None:
                    rot_pos_embed = apply_keep_indices_nlc(x, rot_pos_embed, keep_indices)
                    # After applying keep indices to rope embeds, batch dim is added
                    if getattr(self, 'rope_mixed', False):
                        # B, D, nH, N, dim -> D, B, nH, N, dim. For consistent iteration over depth at index 0.
                        rot_pos_embed = rot_pos_embed.transpose(0, 1)
                    else:
                        # B, N, dim -> B, 1, N, dim.  Need head dim singleton for correct dim alignment in axial mode.
                        rot_pos_embed = rot_pos_embed.unsqueeze(1)

            return x, rot_pos_embed

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
                indices: Take last n blocks if an int, if is a sequence, select by matching indices
                return_prefix_tokens: Return both prefix and spatial intermediate tokens
                norm: Apply norm layer to all intermediates
                stop_early: Stop iterating over blocks when last desired intermediate hit
                output_fmt: Shape of intermediate feature outputs
                intermediates_only: Only return intermediate features
            """
            assert output_fmt in ('NCHW', 'NLC'), 'Output format for EVA-ViT features must be one of NCHW or NLC.'
            reshape = output_fmt == 'NCHW'
            intermediates = []
            take_indices, max_index = feature_take_indices(len(self.blocks), indices)

            # forward pass
            B, _, height, width = x.shape
            x = self.patch_embed(x)
            x, rot_pos_embed = self._pos_embed(x)
            x = self.norm_pre(x)
            if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
                blocks = self.blocks
            else:
                blocks = self.blocks[:max_index + 1]
            alibi_bias = None
            if self.alibi is not None:
                alibi_bias = self.alibi.to(x.device, dtype=x.dtype)
                if self.num_prefix_tokens:
                    alibi_bias = F.pad(alibi_bias, (self.num_prefix_tokens, 0, self.num_prefix_tokens, 0))

            # Handle depth-dependent embeddings for mixed mode
            if getattr(self, 'rope_mixed', False) and rot_pos_embed is not None:
                for i, blk in enumerate(blocks):
                    if self.grad_checkpointing and not torch.jit.is_scripting():
                        x = checkpoint(blk, x, rope=rot_pos_embed[i], alibi_bias=alibi_bias)
                    else:
                        x = blk(x, rope=rot_pos_embed[i], alibi_bias=alibi_bias)
                    if i in take_indices:
                        intermediates.append(self.norm(x) if norm else x)
            else:
                for i, blk in enumerate(blocks):
                    if self.grad_checkpointing and not torch.jit.is_scripting():
                        x = checkpoint(blk, x, rope=rot_pos_embed, alibi_bias=alibi_bias)
                    else:
                        x = blk(x, rope=rot_pos_embed, alibi_bias=alibi_bias)
                    if i in take_indices:
                        intermediates.append(self.norm(x) if norm else x)

            # process intermediates
            if self.num_prefix_tokens:
                # split prefix (e.g. class, distill) and spatial feature tokens
                prefix_tokens = [y[:, 0:self.num_prefix_tokens] for y in intermediates]
                intermediates = [y[:, self.num_prefix_tokens:] for y in intermediates]
            if reshape:
                # reshape to BCHW output format
                H, W = self.patch_embed.dynamic_feat_size((height, width))
                intermediates = [y.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for y in intermediates]
            if not torch.jit.is_scripting() and return_prefix_tokens:
                # return_prefix not support in torchscript due to poor type handling
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
                self.attn_pool = None
                self.fc_norm = nn.Identity()
                self.reset_classifier(0, '')
            return take_indices

        def pool(self, x: torch.Tensor, pool_type: Optional[str] = None) -> torch.Tensor:
            if self.attn_pool is not None:
                x = self.attn_pool(x)
                return x
            pool_type = self.global_pool if pool_type is None else pool_type
            x = global_pool_nlc(x, pool_type=pool_type, num_prefix_tokens=self.num_prefix_tokens)
            return x

        def forward_features(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass through feature extraction layers.

            Args:
                x: Input tensor.

            Returns:
                Feature tensor.
            """
            x = self.patch_embed(x)
            x, rot_pos_embed = self._pos_embed(x)
            x = self.norm_pre(x)
            alibi_bias = None
            if self.alibi is not None:
                alibi_bias = self.alibi.to(x.device, dtype=x.dtype)
                if self.num_prefix_tokens:
                    alibi_bias = F.pad(alibi_bias, (self.num_prefix_tokens, 0, self.num_prefix_tokens, 0))

            if getattr(self, 'rope_mixed', False) and rot_pos_embed is not None:
                # Handle depth-dependent embeddings for mixed mode
                # pos embed has shape (depth, num_heads, H*W, dim) or (depth, batch_size, num_heads, H*W, dim)
                for i, blk in enumerate(self.blocks):
                    if self.grad_checkpointing and not torch.jit.is_scripting():
                        x = checkpoint(blk, x, rope=rot_pos_embed[i], alibi_bias=alibi_bias)
                    else:
                        x = blk(x, rope=rot_pos_embed[i], alibi_bias=alibi_bias)
            else:
                # Standard path for non-mixed mode
                for blk in self.blocks:
                    if self.grad_checkpointing and not torch.jit.is_scripting():
                        x = checkpoint(blk, x, rope=rot_pos_embed, alibi_bias=alibi_bias)
                    else:
                        x = blk(x, rope=rot_pos_embed, alibi_bias=alibi_bias)

            x = self.norm(x)
            return x

        def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
            """Forward pass through classifier head.

            Args:
                x: Feature tensor.
                pre_logits: Return pre-logits if True.

            Returns:
                Output tensor.
            """
            x = self.pool(x)
            x = self.fc_norm(x)
            x = self.head_drop(x)
            return x if pre_logits else self.head(x)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass.

            Args:
                x: Input tensor.

            Returns:
                Output tensor.
            """
            x = self.forward_features(x)
            x = self.forward_head(x)
            return x


    def get_2dalibi(num_heads: int, num_patches: int) -> torch.Tensor:
        """Generate 2D ALiBi bias for a square patch grid (copy of vision_transformer_alibi)."""
        points = list(itertools.product(range(int(math.sqrt(num_patches))), range(int(math.sqrt(num_patches)))))

        def get_slopes(n: int):
            def get_slopes_power_of_2(n_inner: int):
                start = 2 ** (-2 ** -(math.log2(n_inner) - 3))
                ratio = start
                return [start * ratio ** i for i in range(n_inner)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)
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


    def _convert_pe(
        state_dict: Dict[str, torch.Tensor],
        model: nn.Module,
        prefix: str = 'visual.',
    ) -> Dict[str, torch.Tensor]:
        """Convert Perception Encoder weights.

        Args:
            state_dict: State dictionary to convert.
            model: Target model instance.
            prefix: Prefix to strip from keys.

        Returns:
            Converted state dictionary.
        """
        state_dict = state_dict.get('model', state_dict)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        out_dict = {}
        swaps = [
            ('conv1', 'patch_embed.proj'),
            ('positional_embedding', 'pos_embed'),
            ('transformer.resblocks.', 'blocks.'),
            ('ln_pre', 'norm_pre'),
            ('ln_post', 'norm'),
            ('ln_', 'norm'),
            ('ls_1.gamma', 'gamma_1'),
            ('ls_2.gamma', 'gamma_2'),
            ('in_proj_', 'qkv.'),
            ('out_proj', 'proj'),
            ('mlp.c_fc', 'mlp.fc1'),
            ('mlp.c_proj', 'mlp.fc2'),
        ]
        len_prefix = len(prefix)
        for k, v in state_dict.items():
            if prefix:
                if not k.startswith(prefix):
                    continue
                k = k[len_prefix:]

            for sp in swaps:
                k = k.replace(sp[0], sp[1])

            if k.startswith('attn_pool'):
                k = k.replace('attn_pool.attn', 'attn_pool')
                k = k.replace('attn_pool.layernorm', 'attn_pool.norm')
                k = k.replace('attn_pool.probe', 'attn_pool.latent')
                if k.startswith('attn_pool.qkv'):
                    dim = v.shape[0] // 3
                    if k.endswith('weight'):
                        out_dict['attn_pool.q.weight'] = v[:dim]
                        out_dict['attn_pool.kv.weight'] = v[dim:]
                    elif k.endswith('bias'):
                        out_dict['attn_pool.q.bias'] = v[:dim]
                        out_dict['attn_pool.kv.bias'] = v[dim:]
                    continue
            elif k == 'proj':
                k = 'head.weight'
                v = v.transpose(0, 1)
                out_dict['head.bias'] = torch.zeros(v.shape[0])
            elif k == 'class_embedding':
                k = 'cls_token'
                v = v.unsqueeze(0).unsqueeze(1)
            elif k == 'pos_embed':
                v = v.unsqueeze(0)
            out_dict[k] = v

        return out_dict


    def checkpoint_filter_fn(
            state_dict: Dict[str, torch.Tensor],
            model: nn.Module,
            interpolation: str = 'bicubic',
            antialias: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Convert patch embedding weight from manual patchify + linear proj to conv.

        Args:
            state_dict: Checkpoint state dictionary.
            model: Target model instance.
            interpolation: Interpolation method for resizing.
            antialias: Whether to use antialiasing when resizing.

        Returns:
            Filtered state dictionary.
        """
        out_dict = {}
        # Standard EVA checkpoint processing
        state_dict = state_dict.get('model_ema', state_dict)
        state_dict = state_dict.get('model', state_dict)
        state_dict = state_dict.get('module', state_dict)
        state_dict = state_dict.get('state_dict', state_dict)

        # Loading Meta PE (Perception Encoder) weights
        if 'visual.conv1.weight' in state_dict:
            return _convert_pe(state_dict, model)
        elif 'conv1.weight' in state_dict:
            return _convert_pe(state_dict, model, prefix='')

        # prefix for loading OpenCLIP compatible weights
        if 'visual.trunk.pos_embed' in state_dict:
            prefix = 'visual.trunk.'
        elif 'visual.pos_embed' in state_dict:
            prefix = 'visual.'
        else:
            prefix = ''

        dinov3_weights = 'storage_tokens' in state_dict
        mim_weights = not dinov3_weights and prefix + 'mask_token' in state_dict
        no_qkv = prefix + 'blocks.0.attn.q_proj.weight' in state_dict

        len_prefix = len(prefix)
        for k, v in state_dict.items():
            if prefix:
                if not k.startswith(prefix):
                    continue
                k = k[len_prefix:]

            if 'rope' in k and not k == 'rope.freqs':
                # fixed embedding no need to load buffer from checkpoint
                continue

            if dinov3_weights:
                if any([k.endswith(f) for f in ['.periods', '.bias_mask', 'mask_token']]):
                    # discard unused/non-persistent/pretrain only params
                    continue
                if k.startswith('local_cls_norm'):
                    # discard, only used for 7b dinov3 pretrain w/ local crops
                    continue
                if k.endswith('qkv.bias'):
                    q_bias_k = k.replace('qkv.bias', 'q_bias')
                    try:
                        # the distilled b,l,h models ended up with all zero biases, so timm
                        # has both qkv_bias=True and qkv_bias=False impl, test which
                        model.get_parameter(q_bias_k)
                    except Exception as e:
                        print(e)
                        # skip as target model has no bias parameter
                        continue
                    # split bias into components and skip the k as its supposed to be fixed at 0
                    qv, kv, vv = v.chunk(3, dim=-1)
                    out_dict[q_bias_k] = qv
                    out_dict[k.replace('qkv.bias', 'v_bias')] = vv
                    continue
                k = k.replace('ls1.gamma', 'gamma_1')  # match EVA ls naming
                k = k.replace('ls2.gamma', 'gamma_2')  # match EVA ls naming
                k = k.replace('storage_tokens', 'reg_token')  # rename storage to existing register naming

            elif mim_weights and k in ('mask_token', 'lm_head.weight', 'lm_head.bias', 'norm.weight', 'norm.bias'):
                if k == 'norm.weight' or k == 'norm.bias':
                    # try moving norm -> fc norm on fine-tune, probably a better starting point than new init
                    k = k.replace('norm', 'fc_norm')
                else:
                    # skip pretrain mask token & head weights
                    continue

            if 'patch_embed.proj.weight' in k:
                _, _, H, W = model.patch_embed.proj.weight.shape
                if v.shape[-1] != W or v.shape[-2] != H:
                    v = resample_patch_embed(
                        v,
                        (H, W),
                        interpolation=interpolation,
                        antialias=antialias,
                        verbose=True,
                    )
            elif k == 'pos_embed' and v.shape[1] != model.pos_embed.shape[1]:
                # To resize pos embedding when using model at different size from pretrained weights
                num_prefix_tokens = 0 if getattr(model, 'no_embed_class', False) else getattr(model, 'num_prefix_tokens', 1)
                v = resample_abs_pos_embed(
                    v,
                    new_size=model.patch_embed.grid_size,
                    num_prefix_tokens=num_prefix_tokens,
                    interpolation=interpolation,
                    antialias=antialias,
                    verbose=True,
                )

            k = k.replace('mlp.ffn_ln', 'mlp.norm')
            k = k.replace('attn.inner_attn_ln', 'attn.norm')
            k = k.replace('mlp.w12', 'mlp.fc1')
            k = k.replace('mlp.w1', 'mlp.fc1_g')
            k = k.replace('mlp.w2', 'mlp.fc1_x')
            k = k.replace('mlp.w3', 'mlp.fc2')
            if no_qkv:
                k = k.replace('q_bias', 'q_proj.bias')
                k = k.replace('v_bias', 'v_proj.bias')

            out_dict[k] = v

        return out_dict


    def _create_eva(variant: str, pretrained: bool = False, **kwargs) -> Eva:
        """Create an EVA model.

        Args:
            variant: Model variant name.
            pretrained: Load pretrained weights.
            **kwargs: Additional model arguments.

        Returns:
            Instantiated Eva model.
        """
        # Check if we should use NaFlexVit implementation
        use_naflex = kwargs.pop('use_naflex', None)
        _USE_NAFLEX_DEFAULT = os.environ.get('TIMM_USE_NAFLEX', '0') == '1'
        if use_naflex is None:
            use_naflex = _USE_NAFLEX_DEFAULT
        if use_naflex:
            # Import here to avoid circular import
            from .naflexvit import _create_naflexvit_from_eva
            return _create_naflexvit_from_eva(variant, pretrained, **kwargs)

        out_indices = kwargs.pop('out_indices', 3)
        model = build_model_with_cfg(
            Eva, variant, pretrained,
            pretrained_filter_fn=checkpoint_filter_fn,
            feature_cfg=dict(out_indices=out_indices, feature_cls='getter'),
            **kwargs,
        )
        return model


    def _cfg(url: str = '', **kwargs) -> Dict[str, Any]:
        """Generate default configuration for EVA models.

        Args:
            url: Model weights URL.
            **kwargs: Additional configuration parameters.

        Returns:
            Model configuration dictionary.
        """
        return {
            'url': url,
            'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
            'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
            'mean': OPENAI_CLIP_MEAN, 'std': OPENAI_CLIP_STD,
            'first_conv': 'patch_embed.proj', 'classifier': 'head',
            'license': 'mit', **kwargs
        }


    def _pe_cfg(url: str = '', **kwargs) -> Dict[str, Any]:
        """Generate default configuration for Perception Encoder models.

        Args:
            url: Model weights URL.
            **kwargs: Additional configuration parameters.

        Returns:
            Model configuration dictionary.
        """
        return {
            'url': url,
            'num_classes': 0, 'input_size': (3, 224, 224), 'pool_size': None,
            'crop_pct': 1.0, 'interpolation': 'bicubic', 'fixed_input_size': True,
            'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
            'first_conv': 'patch_embed.proj', 'classifier': 'head',
            'license': 'apache-2.0', **kwargs
        }


    def _dinov3_cfg(url: str = '', **kwargs) -> Dict[str, Any]:
        """Generate default configuration for DINOv3 models.

        Args:
            url: Model weights URL.
            **kwargs: Additional configuration parameters.

        Returns:
            Model configuration dictionary.
        """
        return {
            'url': url,
            'num_classes': 0, 'input_size': (3, 256, 256), 'pool_size': None,
            'crop_pct': 1.0, 'interpolation': 'bicubic', 'fixed_input_size': True,
            'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
            'first_conv': 'patch_embed.proj', 'classifier': 'head',
            'license': 'dinov3-license', **kwargs
        }

    # Override default cfgs to include only ALiBi variants defined here.
    default_cfgs = generate_default_cfgs({
        'vit_alibi_small_patch16_dinov3': _cfg(input_size=(3, 256, 256), crop_pct=1.0),
        'vit_alibi_base_patch16_dinov3': _cfg(input_size=(3, 256, 256), crop_pct=1.0),
        'vit_alibi_large_patch16_dinov3': _cfg(input_size=(3, 256, 256), crop_pct=1.0),
    })


    @register_model
    def vit_alibi_small_patch16_dinov3(pretrained: bool = False, **kwargs) -> Eva:
        """DINOv3 S/16 with ALiBi (no AbsPos, no RoPE)."""
        if pretrained:
            raise RuntimeError('No pretrained weights are provided for vit_alibi_small_patch16_dinov3.')
        model_args = dict(
            img_size=256,
            patch_size=16,
            dynamic_img_size=False,
            embed_dim=384,
            depth=12,
            num_heads=6,
            qkv_bias=False,
            init_values=1.0e-5,  # layer-scale
            use_rot_pos_emb=False,
            use_abs_pos_emb=False,
            use_alibi_pos_emb=True,
            num_reg_tokens=4,
            use_fc_norm=False,
            norm_layer=partial(LayerNorm, eps=1e-5),
        )
        model = _create_eva('vit_alibi_small_patch16_dinov3', pretrained=pretrained, **dict(model_args, **kwargs))
        return model


    @register_model
    def vit_alibi_base_patch16_dinov3(pretrained: bool = False, **kwargs) -> Eva:
        """DINOv3 B/16 with ALiBi (no AbsPos, no RoPE)."""
        if pretrained:
            raise RuntimeError('No pretrained weights are provided for vit_alibi_base_patch16_dinov3.')
        model_args = dict(
            img_size=256,
            patch_size=16,
            dynamic_img_size=False,
            embed_dim=768,
            depth=12,
            num_heads=12,
            qkv_bias=False,
            init_values=1.0e-5,  # layer-scale
            use_rot_pos_emb=False,
            use_abs_pos_emb=False,
            use_alibi_pos_emb=True,
            num_reg_tokens=4,
            use_fc_norm=False,
            norm_layer=partial(LayerNorm, eps=1e-5),
        )
        model = _create_eva('vit_alibi_base_patch16_dinov3', pretrained=pretrained, **dict(model_args, **kwargs))
        return model


    @register_model
    def vit_alibi_large_patch16_dinov3(pretrained: bool = False, **kwargs) -> Eva:
        """DINOv3 L/16 with ALiBi (no AbsPos, no RoPE)."""
        if pretrained:
            raise RuntimeError('No pretrained weights are provided for vit_alibi_large_patch16_dinov3.')
        model_args = dict(
            img_size=256,
            patch_size=16,
            dynamic_img_size=False,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            qkv_bias=False,
            init_values=1.0e-5,  # layer-scale
            use_rot_pos_emb=False,
            use_abs_pos_emb=False,
            use_alibi_pos_emb=True,
            num_reg_tokens=4,
            use_fc_norm=False,
            norm_layer=partial(LayerNorm, eps=1e-5),
        )
        model = _create_eva('vit_alibi_large_patch16_dinov3', pretrained=pretrained, **dict(model_args, **kwargs))
        return model

# if args.pos_type is not None:
#     sys.path.append(r".")
#     from timm_pe.eva_relpos import *
#     from timm_pe.eva_alibi import *
#     from timm_pe.eva_sin import *
    # from vision_transformer_rope import *
    # from vision_transformer_rope2d import *
    # from vision_transformer_rpe import *
    # from vision_transformer_relpos import *
    # from vision_transformer_alibi import *
    # from vision_transformer_sin import *

# %%
# timm.list_models("vit_*_dinov2")
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional, Dict, Tuple

LossType = Literal["mse", "smooth_l1", "l1"]

class PatchRowColRegressionCriterion(nn.Module):
    def __init__(self, feat_dim, grid_h, grid_w, normalize=True, huber_beta=None):
        """
        Predict row and column index of each patch via regression (single resolution).

        Args:
            feat_dim (int): Dimension of patch features (D)
            grid_h (int): Number of patch rows (fixed)
            grid_w (int): Number of patch columns (fixed)
            normalize (bool): If True, normalize row/col targets to [0, 1]
        """
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.normalize = normalize

        # Regression heads: scalar row / scalar col
        self.row_mlp = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)   # scalar row index
        )

        self.col_mlp = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)   # scalar col index
        )

        if huber_beta is None:
            self.loss_fn = nn.SmoothL1Loss()
        else:
            self.loss_fn = nn.SmoothL1Loss(beta=0.5/self.grid_h)

        # Precompute row/col targets once (N = grid_h * grid_w)
        rows_2d = torch.arange(grid_h, dtype=torch.float32).unsqueeze(1).repeat(1, grid_w)
        cols_2d = torch.arange(grid_w, dtype=torch.float32).unsqueeze(0).repeat(grid_h, 1)

        if normalize:
            rows_2d = rows_2d / (grid_h - 1)
            cols_2d = cols_2d / (grid_w - 1)

        # Flatten to 1D (N,)
        row_targets = rows_2d.flatten()
        col_targets = cols_2d.flatten()

        # Register as buffers so they move with .to(device)
        self.register_buffer("row_targets", row_targets, persistent=False)  # (N,)
        self.register_buffer("col_targets", col_targets, persistent=False)  # (N,)

    def forward(self, feats):
        """
        Args:
            feats: (B, N, D) patch features, N = grid_h * grid_w

        Returns:
            avg_loss: scalar, average of row and column regression losses
        """
        B, N, D = feats.shape
        assert N == self.grid_h * self.grid_w, f"Expected N = grid_h * grid_w = {self.grid_h * self.grid_w}, got N = {N}"

        # (B*N, D)
        x = feats.reshape(-1, D)

        # Repeat targets for batch: (N,) -> (B*N,)
        row_targets = self.row_targets.repeat(B)
        col_targets = self.col_targets.repeat(B)

        # Predict rows and columns: (B*N, 1) -> (B*N,)
        row_pred = self.row_mlp(x).squeeze(-1)
        col_pred = self.col_mlp(x).squeeze(-1)

        loss_row = self.loss_fn(row_pred, row_targets)
        loss_col = self.loss_fn(col_pred, col_targets)

        return (loss_row + loss_col) / 2.0

class PatchRowColRegressionCriterionDynamic(nn.Module):
    def __init__(self, feat_dim, grid_h, grid_w, normalize=True):
        """
        Predict row and column index of each patch via regression,
        supporting dynamic training resolutions.

        Args:
            feat_dim (int): Dimension of patch features (D)
            grid_h (int): Max number of patch rows (upper bound)
            grid_w (int): Max number of patch columns (upper bound)
            normalize (bool): If True, normalize row/col targets to [0, 1]
                              based on the *current* hp/wp for each batch.
        """
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.normalize = normalize

        # MLP for row regression (scalar output)
        self.row_mlp = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)   # scalar row index
        )

        # MLP for column regression (scalar output)
        self.col_mlp = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)   # scalar col index
        )

        self.loss_fn = nn.SmoothL1Loss()

        # Precompute integer row/col indices (max grid) as floats
        rows = torch.arange(grid_h, dtype=torch.float32).unsqueeze(1).repeat(1, grid_w)  # (grid_h, grid_w)
        cols = torch.arange(grid_w, dtype=torch.float32).unsqueeze(0).repeat(grid_h, 1)  # (grid_h, grid_w)

        self.register_buffer("row_index_full", rows, persistent=False)  # (grid_h, grid_w)
        self.register_buffer("col_index_full", cols, persistent=False)  # (grid_h, grid_w)

    def forward(self, feats, hp=None, wp=None):
        """
        Args:
            feats: (B, N, D) patch features, with N = hp * wp for this batch.
            hp, wp: number of patch rows / columns used for this batch
                    (single scalar each; one resolution per batch).

        Returns:
            avg_loss: scalar, average of row and column regression losses.
        """
        B, N, D = feats.shape

        # Fallback to maximum grid if hp/wp not given
        if hp is None:
            hp = self.grid_h
        if wp is None:
            wp = self.grid_w

        assert N == hp * wp, f"Expected N = hp * wp = {hp * wp}, got N = {N}"

        # Flatten features: (B*N, D)
        x = feats.reshape(-1, D)

        # Slice the index grids to current resolution: (hp, wp)
        row_idx_2d = self.row_index_full[:hp, :wp]  # [0..hp-1]
        col_idx_2d = self.col_index_full[:hp, :wp]  # [0..wp-1]

        if self.normalize:
            # Normalize to [0, 1] based on current hp/wp
            row_idx_2d = row_idx_2d / max(hp - 1, 1)
            col_idx_2d = col_idx_2d / max(wp - 1, 1)

        # Flatten to 1D and repeat for batch: (hp*wp,) -> (B*hp*wp,)
        row_targets = row_idx_2d.flatten().repeat(B)
        col_targets = col_idx_2d.flatten().repeat(B)

        # Predict rows and columns: (B*N, 1) -> (B*N,)
        row_pred = self.row_mlp(x).squeeze(-1)
        col_pred = self.col_mlp(x).squeeze(-1)

        # Regression losses
        loss_row = self.loss_fn(row_pred, row_targets)
        loss_col = self.loss_fn(col_pred, col_targets)

        return (loss_row + loss_col) / 2.0

class PatchRowColCriterionDynamic(nn.Module):
    def __init__(self, feat_dim, grid_h, grid_w):
        """
        Predict row and column of each patch independently.

        Args:
            feat_dim (int): Dimension of patch features (D)
            grid_h (int): Number of patch rows
            grid_w (int): Number of patch columns
        """
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w

        # MLP for row prediction
        self.row_mlp = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, grid_h)
        )

        # MLP for column prediction
        self.col_mlp = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, grid_w)
        )

        self.ce = nn.CrossEntropyLoss()

        # Precompute row/col labels
        rows = torch.arange(grid_h).unsqueeze(1).repeat(1, grid_w)
        cols = torch.arange(grid_w).unsqueeze(0).repeat(grid_h, 1)
        self.register_buffer("row_labels", rows)
        self.register_buffer("col_labels", cols)

    def forward(self, feats, hp=None, wp=None):
        """
        Args:
            feats: (B, N, D) patch features, N = grid_h * grid_w
            wp, hp: (B,) number of patches in each row/column
        Returns:
            avg_loss: scalar, sum of row and column classification losses
        """
        B, N, D = feats.shape
        # assert N == self.grid_h * self.grid_w, f"Expected {self.grid_h*self.grid_w} patches, got {N}"

        x = feats.reshape(-1, D)  # (B*N, D)

        if hp is None or wp is None:
            hp = self.grid_h
            wp = self.grid_w
        # Repeat labels for batch
        row_labels = self.row_labels[:hp, :wp].flatten().repeat(B)
        col_labels = self.col_labels[:hp, :wp].flatten().repeat(B)

        # Predict rows and columns
        row_logits = self.row_mlp(x)
        col_logits = self.col_mlp(x)

        # Compute cross-entropy loss for rows and columns
        loss_row = self.ce(row_logits, row_labels)
        loss_col = self.ce(col_logits, col_labels)

        return (loss_row + loss_col) / 2  # average

class PatchRowColCriterion(nn.Module):
    def __init__(self, feat_dim, grid_h, grid_w):
        """
        Predict row and column of each patch independently.

        Args:
            feat_dim (int): Dimension of patch features (D)
            grid_h (int): Number of patch rows
            grid_w (int): Number of patch columns
        """
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w

        # MLP for row prediction
        self.row_mlp = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, grid_h)
        )

        # MLP for column prediction
        self.col_mlp = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, grid_w)
        )

        self.ce = nn.CrossEntropyLoss()

        # Precompute row/col labels
        rows = torch.arange(grid_h).unsqueeze(1).repeat(1, grid_w).flatten()
        cols = torch.arange(grid_w).repeat(grid_h)
        self.register_buffer("row_labels", rows)
        self.register_buffer("col_labels", cols)

    def forward(self, feats):
        """
        Args:
            feats: (B, N, D) patch features, N = grid_h * grid_w
        Returns:
            avg_loss: scalar, sum of row and column classification losses
        """
        B, N, D = feats.shape
        assert N == self.grid_h * self.grid_w, f"Expected {self.grid_h*self.grid_w} patches, got {N}"

        x = feats.reshape(-1, D)  # (B*N, D)

        # Repeat labels for batch
        row_labels = self.row_labels.repeat(B)
        col_labels = self.col_labels.repeat(B)

        # Predict rows and columns
        row_logits = self.row_mlp(x)
        col_logits = self.col_mlp(x)

        # Compute cross-entropy loss for rows and columns
        loss_row = self.ce(row_logits, row_labels)
        loss_col = self.ce(col_logits, col_labels)

        return (loss_row + loss_col) / 2  # average


# if Use_Row_Col_Loss:
#     grid_h, grid_w = model.patch_embed.grid_size
#     rowcol_loss = PatchRowColCriterion(
#         feat_dim=model.embed_dim,
#         grid_h=grid_h,
#         grid_w=grid_w
#     ).to(DEVICE)
#     print("✅ Row-Column loss initialized.")

class PatchPositionCriterion(nn.Module):
    def __init__(self, feat_dim, hidden_dim=256, num_classes=None):
        """
        Args:
            feat_dim (int): Feature dimension of each patch (D)
            hidden_dim (int): Hidden layer size for MLP
            num_classes (int): Number of patches (grid_h * grid_w)
        """
        super().__init__()
        self.num_classes = num_classes
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
        self.ce = nn.CrossEntropyLoss()

        # Precompute patch position labels once
        self.register_buffer("patch_positions", torch.arange(num_classes), persistent=False)  # shape (num_patches,)
        
    def forward(self, feats):
        """
        Args:
            feats: (B, N, D) patch features
        Returns:
            avg_loss: scalar, mean cross-entropy over all patches
        """
        B, N, D = feats.shape
        assert N == self.num_classes, f"Expected {self.num_classes} patches, got {N}"

        # Flatten batch and patches: (B*N, D)
        x = feats.reshape(-1, D)
        # Repeat labels for all images in batch: (B*N,)
        labels = self.patch_positions.repeat(B)
        # Predict positions
        logits = self.mlp(x)
        # Compute CE loss
        loss = self.ce(logits, labels)
        return loss

class PatchPositionRegressionCriterion(nn.Module):
    def __init__(self, feat_dim, num_classes, normalize=True):
        """
        Predict patch position index via regression (single resolution).

        Args:
            feat_dim (int): Feature dimension of each patch (D)
            num_classes (int): Number of patches (grid_h * grid_w)
            normalize (bool): If True, normalize position targets to [0, 1]
        """
        super().__init__()
        self.num_classes = num_classes
        self.normalize = normalize

        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)   # scalar position index
        )

        self.loss_fn = nn.SmoothL1Loss()

        # Precompute patch position targets once
        position_targets = torch.arange(num_classes, dtype=torch.float32)
        if normalize:
            position_targets = position_targets / max(num_classes - 1, 1)
        self.register_buffer("position_targets", position_targets, persistent=False)  # (N,)

    def forward(self, feats):
        """
        Args:
            feats: (B, N, D) patch features
        Returns:
            loss: scalar, SmoothL1 loss over all patches
        """
        B, N, D = feats.shape
        assert N == self.num_classes, f"Expected {self.num_classes} patches, got {N}"

        # Flatten batch and patches: (B*N, D)
        x = feats.reshape(-1, D)
        # Repeat targets for batch: (N,) -> (B*N,)
        targets = self.position_targets.repeat(B)
        # Predict positions: (B*N, 1) -> (B*N,)
        pred = self.mlp(x).squeeze(-1)
        # Compute regression loss
        loss = self.loss_fn(pred, targets)
        return loss

class PatchPositionRegressionCriterionDynamic(nn.Module):
    def __init__(self, feat_dim, max_patch_count, normalize=True):
        """
        Predict patch position index via regression, supporting dynamic resolutions.

        Args:
            feat_dim (int): Feature dimension of each patch (D)
            max_patch_count (int): Max number of patches (upper bound)
            normalize (bool): If True, normalize position targets to [0, 1]
                              based on the *current* patch count for each batch.
        """
        super().__init__()
        self.max_patch_count = max_patch_count
        self.normalize = normalize

        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)   # scalar position index
        )

        self.loss_fn = nn.SmoothL1Loss()

        positions = torch.arange(max_patch_count, dtype=torch.float32)
        self.register_buffer("position_index_full", positions, persistent=False) 

    def forward(self, feats):
        """
        Args:
            feats: (B, N, D) patch features.
        Returns:
            loss: scalar, SmoothL1 loss over all patches.
        """
        B, N, D = feats.shape
        if N > self.max_patch_count:
            raise ValueError(f"Expected N <= max_patch_count={self.max_patch_count}, got N={N}")

        # Flatten features: (B*N, D)
        x = feats.reshape(-1, D)

        # Slice position indices to current patch count: (N,)
        pos_idx = self.position_index_full[:N]

        if self.normalize:
            pos_idx = pos_idx / max(N - 1, 1)

        # Repeat for batch: (N,) -> (B*N,)
        targets = pos_idx.repeat(B)

        # Predict positions: (B*N, 1) -> (B*N,)
        pred = self.mlp(x).squeeze(-1)

        loss = self.loss_fn(pred, targets)
        return loss
        
# if Use_Patch_Position_Loss:
#     position_loss = PatchPositionCriterion(
#         feat_dim=model.embed_dim,
#         num_classes=model.patch_embed.num_patches
#     ).to(DEVICE)


from torch.utils.data import Dataset
from PIL import Image

class MultiScaleImageDataset(Dataset):
    def __init__(self, samples, size_to_transform):
        """
        samples: list of (path, target)
        size_to_transform: dict[int, torchvision.transforms.Compose]
        """
        self.samples = samples
        self.size_to_transform = size_to_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, key):
        # key comes from the batch sampler: (idx, size)
        idx, size = key
        path, target = self.samples[idx]

        with open(path, "rb") as f:
            img = Image.open(f).convert("RGB")

        transform = self.size_to_transform[size]
        img = transform(img)

        return img, target
        
class CustomImageDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def set_transform(self, transform):
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        if self.transform:
            sample = self.transform(sample)
        return sample, target
    
import math
import random

class DynamicResolutionBatchSampler:
    """
    Yields batches of (idx, size) with dynamic batch size so that
    batch_size * size^2 ≈ base_batch_size * base_img_size^2.
    """

    def __init__(
        self,
        dataset,
        image_sizes,
        base_batch_size,
        base_img_size,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 0,
    ):
        self.dataset_len = len(dataset)
        self.image_sizes = list(image_sizes)
        self.base_batch_size = base_batch_size
        self.base_img_size = base_img_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        # pixel budget based on reference configuration
        self.pixel_budget = base_batch_size * (base_img_size ** 2)

        # for __len__ (approximate)
        avg_size_sq = sum(s * s for s in self.image_sizes) / len(self.image_sizes)
        self.avg_batch_size = self.pixel_budget / avg_size_sq

    def __len__(self):
        # approximate number of batches per epoch
        return math.ceil(self.dataset_len / self.avg_batch_size)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)

        indices = list(range(self.dataset_len))
        if self.shuffle:
            rng.shuffle(indices)

        ptr = 0
        n = len(indices)

        while ptr < n:
            # 1) choose resolution for this batch
            size = rng.choice(self.image_sizes)

            # 2) compute batch size from pixel budget
            pixels_per_sample = size * size
            if pixels_per_sample > 0:
                batch_size = max(1, self.pixel_budget // pixels_per_sample)
            else:
                batch_size = self.base_batch_size

            # 3) adjust for remaining samples
            remaining = n - ptr
            if remaining < batch_size:
                if self.drop_last:
                    break
                else:
                    batch_size = remaining

            batch_indices = indices[ptr: ptr + batch_size]
            ptr += batch_size

            # 4) yield (idx, size) pairs
            yield [(idx, size) for idx in batch_indices]

# %%
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import collections

# =================================================================================
# Step 1: Configuration
# =================================================================================

# =================================================================================
# Step 2: Custom Dataset Class
# =================================================================================
# This simple Dataset class will load images from a pre-made list of file paths.

# =================================================================================
# Step 3: Efficiently Find and Load Data for Only 10 Classes
# =================================================================================

# --- Discover and select the first 10 class folders ---
# This is a fast filesystem operation. We only scan one directory to get the names.
all_class_dirs = [
    d
    for train_path in TRAIN_PATHS
    for d in os.listdir(train_path)
    if os.path.isdir(os.path.join(train_path, d))
]
selected_class_dirs = sorted(list(set(all_class_dirs)))[offset:args.num_classes+offset]
class_to_idx = {cls_name: i for i, cls_name in enumerate(selected_class_dirs)}

logger.info(f"✅ Efficiently loading the following {len(selected_class_dirs)} classes: {selected_class_dirs}")
args.num_classes = len(selected_class_dirs)
# --- Manually build the list of training samples (images, labels) ---
train_samples = []
for train_path_part in TRAIN_PATHS:
    for class_name in selected_class_dirs:
        class_idx = class_to_idx[class_name]
        class_dir = os.path.join(train_path_part, class_name)
        if os.path.isdir(class_dir):
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    path = os.path.join(class_dir, fname)
                    item = (path, class_idx)
                    train_samples.append(item)

# --- Manually build the list of validation samples ---
valid_samples = []
for class_name in selected_class_dirs:
    class_idx = class_to_idx[class_name]
    class_dir = os.path.join(VALID_PATH, class_name)
    if os.path.isdir(class_dir):
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(class_dir, fname)
                item = (path, class_idx)
                valid_samples.append(item)

# =================================================================================
# Step 4: Create Datasets and DataLoaders
# =================================================================================
#%%
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

img_mean = [0.485, 0.456, 0.406]
img_std  = [0.229, 0.224, 0.225]


def make_train_transform(size: int):
    t_list = [
        T.RandomResizedCrop(size, interpolation=InterpolationMode.BICUBIC, antialias=True),
        T.RandomHorizontalFlip(),
    ]
    if args.randaugment:
        t_list.append(
            T.RandAugment(
                num_ops=args.randaugment_n,
                magnitude=args.randaugment_m,
                interpolation=InterpolationMode.BICUBIC,
                fill=(128, 128, 128),
            )
        )
    t_list.extend([
        T.ToTensor(),
        T.Normalize(mean=img_mean, std=img_std),
    ])
    if args.random_erasing:
        t_list.append(T.RandomErasing(p=args.re_prob))
    return T.Compose(t_list)

size_to_transform = {
    s: make_train_transform(s) for s in args.img_sizes
}

def make_valid_transform(img_size):
    return transforms.Compose([
        transforms.Resize(
            size=int(img_size * 1.143),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=img_mean, std=img_std),
    ])

valid_transforms = make_valid_transform(args.img_sizes[0])
# --- Create the final datasets from the filtered samples ---

valid_dataset = CustomImageDataset(valid_samples, transform=valid_transforms)

logger.info(f"Total validation images ({args.num_classes} classes): {len(valid_dataset)}")

# --- Create DataLoaders ---
batch_sampler = None
prefetch_kwargs = {"prefetch_factor": 2} if args.workers > 0 else {}
train_generator = torch.Generator()
train_generator.manual_seed(args.seed)
if len(args.img_sizes) == 1:
    train_dataset = CustomImageDataset(train_samples, transform=size_to_transform[args.img_sizes[0]])
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0),
        **prefetch_kwargs,
    )
else:
    train_dataset = MultiScaleImageDataset(
        samples=train_samples,              # list of (path, target)
        size_to_transform=size_to_transform
    )
    batch_sampler = DynamicResolutionBatchSampler(
        dataset=train_dataset,
        image_sizes=args.img_sizes,
        base_batch_size=args.batch_size,    # your “reference” batch size
        base_img_size=224, #args.img_sizes[0],       # your “reference” resolution (e.g. 224)
        shuffle=True,
        drop_last=True,
        seed=42,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_sampler=batch_sampler,
        num_workers=args.workers,           # now workers do the transforms
        pin_memory=True,
        persistent_workers=(args.workers > 0),
        **prefetch_kwargs,
    )
logger.info(f"Total training images ({args.num_classes} classes): {len(train_dataset)}")
valid_loader = DataLoader(
    dataset=valid_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.workers,
    pin_memory=True,
    persistent_workers=(args.workers > 0),
    **prefetch_kwargs,
)
steps_per_epoch = len(train_loader)
accum_steps = max(1, int(getattr(args, "grad_accum_steps", 1)))
optimizer_steps_per_epoch = math.ceil(steps_per_epoch / accum_steps)
logger.info(f"✅ DataLoaders for {args.num_classes} classes created successfully.")
logger.info(f"{steps_per_epoch=}, val_steps: {len(valid_loader)}")
logger.info(f"Effective batch size: {args.batch_size * accum_steps}")

# %%
# %% [code]
# =================================================================================
# Step 3.5: Visualize a Batch of Training Data
# =================================================================================
import matplotlib.pyplot as plt
import numpy as np
import torchvision

def imshow(inp, title=None):
    """A helper function to denormalize and display an image tensor."""
    # Define the same mean and std used for normalization
    mean = np.array(img_mean)
    std = np.array(img_std)
    
    # Transpose from (C, H, W) to (H, W, C)
    inp = inp.numpy().transpose((1, 2, 0))
    # Denormalize
    inp = std * inp + mean
    # Clip values to be between 0 and 1
    inp = np.clip(inp, 0, 1)
    
    plt.imshow(inp)
    if title is not None:
        plt.title(title, fontsize=10)
    plt.axis('off')

# Get one batch of training images
# try:
#     inputs, classes = next(iter(train_loader))
    
#     # Get the class names from the dataset object
#     # class_names = meta_dict['fine_label_names']

#     # Create a grid of images
#     fig = plt.figure(figsize=(16, 8))
#     plt.suptitle("Sample Images from CIFAR-100 Dataset", fontsize=16)
    
#     # Display the first 16 images from the batch
#     for i in range(16):
#         ax = plt.subplot(4, 8, i + 1)
#         class_name = classes[i]
#         imshow(inputs[i], title=class_name)
        
#     plt.tight_layout(rect=[0, 0, 1, 0.96])
#     plt.show()

# except NameError:
#     logger.info("Could not display images. Please ensure the previous cells have been run to create 'train_loader'.")



# %%
# =================================================================================
# Step 4: Initialize the Model, Loss Function, and Optimizer
# =================================================================================
# --- Model ---
logger.info(f"🤖 Initializing model: {MODEL_NAME} for {args.num_classes} classes...")
model = timm.create_model(
    MODEL_NAME,
    pretrained=False, # As requested: trains the model from scratch
    use_abs_pos_emb=args.use_abs_pos_emb,
    use_rot_pos_emb=args.use_rot_pos_emb,
    num_classes=args.num_classes, # Set the classifier head to 100 classes
    dynamic_img_size=args.dynamic_img_size,
    img_size=args.img_sizes[0],
).to(DEVICE)
# feature_layers = [2, 5, 8, 11]
# dummy_input = torch.randn(2, 3, args.img_size, args.img_size).to(DEVICE)
# with torch.no_grad():
#     feats = model.forward_features(dummy_input)
#     multi_feats = model.forward_intermediates(dummy_input, indices=feature_layers, intermediates_only=True)


# logger.info(f"Model created successfully!")
# logger.info(f"Input shape: {dummy_input.shape}")
# logger.info(f"Output shape: {feats.shape}") 
# logger.info(f"multi_feats shape: {multi_feats[-1].shape} X {len(multi_feats)}")
# del feats, multi_feats, dummy_input
# gc.collect()

# %%
logger.info(f'model.patch_embed.proj{model.patch_embed.proj}')
# if args.overlap > 0:
#     # Customize patch embedding for overlap (e.g., patch_size=15, stride=14)
#     original_patch_size = model.patch_embed.proj.kernel_size[0]
#     new_patch_size = original_patch_size + args.overlap  # Or 15, 16, 17, etc., as desired
#     stride = original_patch_size
#     original_grid_size = args.img_sizes[0] // stride  # 16 for 224//14
#     padding = ((original_grid_size - 1) * stride + new_patch_size - args.img_sizes[0] + 1) // 2  # +1 for ceiling effect; yields 1 for patch_size=15
    
#     # Override the PatchEmbed projection (Conv2d layer)
#     in_chans = model.patch_embed.proj.in_channels  # Typically 3 for RGB
#     embed_dim = model.patch_embed.proj.out_channels  # e.g., 768 for base
#     model.patch_embed.proj = nn.Conv2d(
#         in_chans, embed_dim,
#         kernel_size=(new_patch_size, new_patch_size),
#         stride=(stride, stride),
#         padding=padding  # Updated to ensure full coverage and original grid size
#     ).to(DEVICE)
    
    # Recompute grid size and num_patches
    # grid_size_h = ((args.img_size + 2 * padding - new_patch_size) // stride) + 1
    # grid_size_w = grid_size_h  # Assuming square input
    # logger.info(new_patch_size, padding, grid_size_h, model.patch_embed.grid_size)
    # model.patch_embed.grid_size = (grid_size_h, grid_size_w)
    # model.patch_embed.num_patches = grid_size_h * grid_size_w
    # logger.info(f"Updated to patch_size={new_patch_size}, stride={stride}, padding={padding}, num_patches={model.patch_embed.num_patches}")

# if not args.has_pos and hasattr(model, 'pos_embed') and model.pos_embed is not None:
#     model.pos_embed.data.zero_()
#     model.pos_embed.requires_grad = False
#     logger.info("✅ Positional embedding has been disabled.")

# if not args.has_pos or args.pos_type is not None:
#     if hasattr(model, 'pos_embed') and model.pos_embed is not None:
#         model.pos_embed.data.zero_()
#         model.pos_embed.requires_grad = False
#         logger.info("✅ Positional embedding has been disabled.")
#     if hasattr(model, 'rope'):
#         model.rope = None

# if args.pretrained is not None:
#     state_dicts = torch.load(args.pretrained, map_location=DEVICE)
#     IncompatibleKeys = model.load_state_dict(state_dicts)
#     logger.info(IncompatibleKeys)
# %%
if args.compile_model and len(args.img_sizes)==1:
    if hasattr(torch, "compile"):
        logger.info("Compiling model with torch.compile (mode='reduce-overhead').")
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
    else:
        logger.warning("torch.compile not available; skipping compilation.")

dynamic = True
training_parameters = list(model.parameters()) 
param_groups = []
lr_aux = getattr(args, "lr_aux", args.lr)
if args.use_rc_loss:
    if len(args.img_sizes)==1:
        grid_h, grid_w = model.patch_embed.grid_size
        dynamic = False
        # from core.patch_pos import PatchRowColRegressionCriterion
        rowcol_loss = PatchRowColRegressionCriterion(
            feat_dim=model.embed_dim,
            grid_h=grid_h,
            grid_w=grid_w,
            # loss_type=args.loss_type,
            # huber_beta=args.huber_beta,
        ).to(DEVICE)
    else:
        grid_h = grid_w = max(args.img_sizes)//args.patch_size
        # from core.patch_pos import PatchRowColRegressionCriterionDynamic
        rowcol_loss = PatchRowColRegressionCriterionDynamic(
            feat_dim=model.embed_dim,
            grid_h=grid_h,
            grid_w=grid_w,
            # loss_type=args.loss_type,
            # huber_beta=args.huber_beta,
        ).to(DEVICE)
    training_parameters += list(rowcol_loss.parameters())
    param_groups.append({"params": rowcol_loss.parameters(), "weight_decay": 0.0, "lr": lr_aux})
if args.use_patch_position_loss:
    if len(args.img_sizes)==1:
        # from core.patch_pos import PatchPositionRegressionCriterion
        position_loss = PatchPositionRegressionCriterion(
            feat_dim=model.embed_dim,
            num_classes=model.patch_embed.num_patches
        ).to(DEVICE)
    else:
        max_grid = max(args.img_sizes)//args.patch_size
        max_patch_count = max_grid * max_grid
        # from core.patch_pos import PatchPositionRegressionCriterionDynamic
        position_loss = PatchPositionRegressionCriterionDynamic(
            feat_dim=model.embed_dim,
            max_patch_count=max_patch_count
        ).to(DEVICE)
    training_parameters += list(position_loss.parameters())
    param_groups.append({"params": position_loss.parameters(), "weight_decay": 0.0, "lr": lr_aux})

decay_params = []
no_decay_params = []

for n, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if n.endswith(".bias") or ("norm" in n.lower()):
        no_decay_params.append(p)
    else:
        decay_params.append(p)

param_groups.append({
    "params": decay_params,
    "lr": args.lr,
    "weight_decay": args.weight_decay,
})
param_groups.append({
    "params": no_decay_params,
    "lr": args.lr,
    "weight_decay": 0.0,
})
# --- Loss Function & Optimizer ---
criterion = nn.CrossEntropyLoss()
if args.composite_lr:
    # optimizer = torch.optim.AdamW(training_parameters, lr=args.lr, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)

    total = sum(p.numel() for p in model.parameters())
    opt_total = sum(p.numel() for g in optimizer.param_groups for p in g["params"])
    print("model params:", total, "optimizer params:", opt_total)

    # Ensure no parameter appears in multiple groups
    seen = set()
    dups = 0
    for g in optimizer.param_groups:
        for p in g["params"]:
            pid = id(p)
            if pid in seen:
                dups += 1
            seen.add(pid)
    print("duplicate params in groups:", dups)

    total_steps = args.epochs * optimizer_steps_per_epoch
    # warmup_steps = 100 #int(0.01 * total_steps)

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-7 / args.lr,   # warmup start lr = 1e-7, weight_decay=0.05
        end_factor=1.0,                # warmup end lr = base_lr
        total_iters=args.warmup_steps
    )

    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps - args.warmup_steps,
        eta_min=1e-8
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[args.warmup_steps]
    )
else:
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = optim.AdamW(training_parameters, lr=args.lr, weight_decay=args.weight_decay)
    logger.info("✅ Model, Loss Function, and Optimizer are ready.")

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    # logger.info("✅ Model, Loss, Optimizer, and LR Scheduler are ready.")

    total_steps = args.epochs * optimizer_steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.eta_min)
    logger.info("✅ Step-based LR Scheduler is ready.")


# %%
# dummy_input = torch.randn(2, 3, args.img_sizes[0], args.img_sizes[0]).to(DEVICE)
# with torch.no_grad():
#     feats = model.forward_features(dummy_input)
# logger.info(f"Model created successfully!")
# logger.info(f"Input shape: {dummy_input.shape}")
# logger.info(f"Output shape: {feats.shape}") 
    
sys.stdout.flush()
# %%
#%%
def get_patch_numbers(img_size, patch_size):
    """
    Calculate the number of patches in an image.

    Args:
        img_size (int or tuple): Size of the input image (H, W)
        patch_size (int): Size of the patch

    Returns:
        tuple: Number of patches in the image (H, W)
    """
    if isinstance(img_size, int):
        img_size = (img_size, img_size)
    assert 2 == len(img_size)
    hp, wp = img_size[0] // patch_size, img_size[1] // patch_size  
    return hp, wp


# %%
import csv

ckpt_path = None
if args.train:
    # FP16: Initialize the Gradient Scaler
    use_scaler = use_amp and (autocast_dtype == torch.float16)
    scaler = torch.amp.GradScaler(enabled=use_scaler)
    start_epoch = 0
    step = 0
    best_acc = 0.0
    if args.resume_full_ckpt and args.resume_ckpt_path:
        # resume_ckpt = ckpt
        # torch.load(args.resume_ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(resume_ckpt["model"])
        if args.resume_optimizer:
            if "optimizer" in resume_ckpt:
                optimizer.load_state_dict(resume_ckpt["optimizer"])
        else:
            logger.info("Skipping optimizer state load (resume_optimizer=False).")
        if args.resume_scheduler:
            start_epoch = resume_ckpt.get("epoch", 0)
            step = resume_ckpt.get("step", 0)
            if resume_ckpt.get("scheduler") is not None:
                scheduler.load_state_dict(resume_ckpt["scheduler"])
        else:
            logger.info("Skipping scheduler state load (resume_scheduler=False).")
        if resume_ckpt.get("scaler") is not None:
            scaler.load_state_dict(resume_ckpt["scaler"])
        if args.use_rc_loss and resume_ckpt.get("rowcol_loss") is not None:
            for k in ["row_targets", "col_targets", "row_index_full", "col_index_full"]:
                if k in resume_ckpt["rowcol_loss"]:
                    resume_ckpt["rowcol_loss"].pop(k)
            rowcol_loss.load_state_dict(resume_ckpt["rowcol_loss"])
        if args.use_patch_position_loss and resume_ckpt.get("position_loss") is not None:
            position_loss.load_state_dict(resume_ckpt["position_loss"])
        best_acc = resume_ckpt.get("best_acc", 0.0)
        logger.info(f"Resumed full checkpoint from '{args.resume_ckpt_path}' at epoch={start_epoch}, step={step}")
    # =================================================================================
    # Step 5: Training and Validation Loop
    # =================================================================================
    logger.info(f"\n🚀 Starting training for {MODEL_NAME}...")

    # ✅ Initialize training_history as a dictionary of lists
    if args.use_rc_loss or args.use_patch_position_loss:
        training_history = {
            'train_loss': [],
            'train_acc': [],
            'valid_acc': [],
            'train_time': [],
            'val_time': [],
            'epoch': [],
            'step': [],
            'base_loss': [],
            'aux_loss': [],
        }
    else:
        training_history = {
            'train_loss': [],
            'train_acc': [],
            'valid_acc': [],
            'train_time': [],
            'val_time': [],
            'epoch': [],
            'step': [],
        }
    if resume_ckpt is not None and resume_ckpt.get("training_history") is not None:
        training_history = resume_ckpt["training_history"]
    training_history.setdefault('train_time', [])
    training_history.setdefault('val_time', [])
    def _pad_history(hist, fill_value=None):
        keys = [k for k, v in hist.items() if isinstance(v, list)]
        if not keys:
            return
        max_len = max(len(hist[k]) for k in keys)
        for k in keys:
            if len(hist[k]) < max_len:
                hist[k].extend([fill_value] * (max_len - len(hist[k])))
    if args.resume_full_ckpt:
        _pad_history(training_history)
    log_interval = getattr(args, "log_interval", 50)
    csv_interval = getattr(args, "csv_interval", 1) 
    # train_epoch_times = []
    for epoch in range(start_epoch, args.epochs):
        epoch_train_start = time.time()
        # --- Training Phase ---
        model.train()
        # epoch_train_start = time.perf_counter()

        aux_loss = None

        running_loss_t = torch.zeros((), device=DEVICE)   # scalar tensor
        aux_loss_sum_t = torch.zeros((), device=DEVICE)
        base_loss_t = torch.zeros((), device=DEVICE)
        train_correct_t = torch.zeros((), device=DEVICE)
        train_total = 0

        # running_loss = 0.0
        # train_correct = 0
        train_total = 0
        # aux_loss_sum = 0.0
        # train_pbar = train_loader
        if batch_sampler is not None:
            batch_sampler.set_epoch(epoch)
        
        # FP16: Use autocast for the forward pass
        optimizer.zero_grad(set_to_none=True)
        for step_in_epoch, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            bs = inputs.size(0)
            if args.show_peak_gpu_mem and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            aux_loss = None
            with torch.amp.autocast(device_type=DEVICE.type, dtype=autocast_dtype, enabled=use_amp):
                feats = model.forward_features(inputs)
                outputs = model.forward_head(feats)
                # outputs = model(inputs)
                loss = criterion(outputs, labels)
                if args.use_rc_loss:
                    base_loss_t += loss.detach() * bs
                    if dynamic:
                        hp, wp = get_patch_numbers(inputs.shape[-2:], model.patch_embed.patch_size[0])
                        aux_loss = rowcol_loss(feats[:, model.num_prefix_tokens:, :], hp, wp)
                    else:
                        aux_loss = rowcol_loss(feats[:, model.num_prefix_tokens:, :])
                    
                    # logger.info(f"grid={model.patch_embed.grid_size}, {dynamic=} num_prefix={model.num_prefix_tokens}")
                    # # once after a forward:
                    # logger.info(f"feats={feats.shape}, patch_tokens={feats[:, model.num_prefix_tokens:, :].shape[1]}")

                    aux_loss_sum_t += aux_loss.detach() * bs
                    # warmup_steps_for_aux = 100
                    # alpha_t = args.rc_alpha * min(1.0, (step + 1) / args.warmup_steps_for_aux)
                    loss = loss + args.rc_alpha * aux_loss
                
                if args.use_patch_position_loss:
                    base_loss_t += loss.detach() * bs
                    aux_loss = position_loss(feats[:, model.num_prefix_tokens:, :])
                    aux_loss_sum_t += aux_loss.detach() * bs
                    loss = loss + args.rc_alpha * aux_loss
            
            # FP16: Scale, backward, and step (with grad accumulation)
            loss_scaled = loss / accum_steps
            scaler.scale(loss_scaled).backward()

            do_step = ((step_in_epoch + 1) % accum_steps == 0) or (step_in_epoch + 1 == len(train_loader))
            if do_step:
                if args.clip_value is not None:
                    scaler.unscale_(optimizer)
                    # log_grads(logger, model, rowcol_loss=rowcol_loss if args.use_rc_loss else None,
            #   every=331, step=step)
                    torch.nn.utils.clip_grad_norm_(training_parameters, max_norm=args.clip_value)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss_t += loss.detach() * bs
            train_total += bs

            with torch.no_grad():
                pred = outputs.detach().argmax(dim=1)
                train_correct_t += (pred == labels).sum()

            # only log every N steps (minimize sync + formatting)
            if (step + 1) % log_interval == 0:
                # now pay the sync cost, but only occasionally
                avg_loss = (running_loss_t / train_total).float().item()
                avg_acc = (train_correct_t / train_total).float().item()
                peak_mb = None
                if args.show_peak_gpu_mem and torch.cuda.is_available():
                    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                msg = f"Epoch {epoch+1}/{args.epochs} step {step+1}: loss={avg_loss:.4f} acc={avg_acc:.3f}"
                if aux_loss is not None:
                    avg_aux = (aux_loss_sum_t / train_total).float().item()
                    msg += f" aux={avg_aux:.4f}"
                if peak_mb is not None:
                    msg += f" peak_mem={peak_mb:.0f}MB"
                logger.info(msg)

            step += 1

        train_time = time.time() - epoch_train_start
        # if (step + 1) % VAL_STEPS == 0:
        # epoch_train_end = time.perf_counter()
        # train_epoch_times.append(epoch_train_end - epoch_train_start)
        # --- Validation Phase ---
        model.eval()
        val_correct_t = torch.zeros((), device=DEVICE)
        val_total = 0
        # val_pbar = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Validation]")
        val_start = time.time()
        
        with torch.inference_mode():
            for inputs, labels in valid_loader:
                inputs = inputs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                with torch.amp.autocast(device_type=DEVICE.type, dtype=autocast_dtype, enabled=use_amp):
                    outputs = model(inputs)
                pred = outputs.argmax(dim=1)
                val_correct_t += (pred == labels).sum()
                val_total += labels.size(0)

        val_time = time.time() - val_start
        epoch_val_acc = (val_correct_t / val_total).item()
        is_best = False
        if best_acc < epoch_val_acc:
            best_acc = epoch_val_acc
            is_best = True

        epoch_train_acc  = (train_correct_t / train_total).item()
        epoch_train_loss = (running_loss_t / train_total).item()
        logger.info(f"\nEpoch {epoch+1}/{args.epochs} Summary:")
        logger.info(f"\nStep {step} Summary:")

        if aux_loss is not None:
            epoch_aux_loss   = (aux_loss_sum_t / train_total).item()
            epoch_base_loss  = (base_loss_t / train_total).item()
            training_history['aux_loss'].append(epoch_aux_loss)
            training_history['base_loss'].append(epoch_base_loss)
            logger.info(
                f"  Train Loss: {epoch_train_loss:.4f} | Aux Loss: {epoch_aux_loss:.4f} | Base Loss: {epoch_base_loss:.4f} | "
                f"Train Acc: {epoch_train_acc:.4f} | Valid Acc: {epoch_val_acc:.4f} | "
                f"train_time: {train_time:.1f}s | val_time: {val_time:.1f}s\n"
            )
        else:
            logger.info(
                f"  Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | "
                f"Valid Acc: {epoch_val_acc:.4f} | train_time: {train_time:.1f}s | val_time: {val_time:.1f}s\n"
            )
        

        # ✅ Append the results to the correct lists within the dictionary
        
        training_history['train_loss'].append(epoch_train_loss)
        training_history['train_acc'].append(epoch_train_acc)
        training_history['valid_acc'].append(epoch_val_acc)  
        training_history['train_time'].append(train_time)
        training_history['val_time'].append(val_time)
        training_history['epoch'].append(epoch+1)
        training_history['step'].append(step+1)
        if (epoch + 1) % csv_interval == 0:
            pd.DataFrame(training_history).to_csv(os.path.join(output_dir, f'{subdir_name}.csv'), index=False)
        if args.save_full_ckpt:
            ckpt = {
                "epoch": epoch + 1,
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "scaler": scaler.state_dict() if scaler is not None else None,
                "rowcol_loss": rowcol_loss.state_dict() if args.use_rc_loss else None,
                "position_loss": position_loss.state_dict() if args.use_patch_position_loss else None,
                "training_history": training_history,
                "args": args,
                "best_acc": best_acc,
            }
            torch.save(ckpt, last_ckpt_path)
            logger.info(f"Saved full checkpoint to '{last_ckpt_path}'")

        if args.total_run_time_hr is not None:
            elapsed = time.time() - train_start_time
            max_run_time_sec = args.total_run_time_hr * 3600
            if elapsed + (train_time + val_time) + 300 >= max_run_time_sec:
                logger.info(
                    "Stopping training: elapsed time exceeded %.2fh.",
                    args.total_run_time_hr,
                )
                break
        # gc.collect()
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()

        # Update the learning rate scheduler
        # if 'scheduler' in locals():
        #     scheduler.step()

    logger.info("🏁 Training complete.")
    logger.info(f"Best Accuracy: {best_acc:.4f}")
    logger.info(output_dir)

    # =================================================================================
    # Step 6: Save the Results and Model
    # =================================================================================

    # ✅ Step 1: Convert the dictionary directly into a pandas DataFrame

    # ✅ Step 2: Add the 'epoch' column at the beginning
    # Create the list of epochs where validation was actually performed
    # epochs_validated = range(5, EPOCHS + 1, 5) 
    # history_df.insert(0, 'epoch', epochs_validated)

    # ✅ Step 3: Save the DataFrame to a CSV file
    pd.DataFrame(training_history).to_csv(os.path.join(output_dir, f'{subdir_name}.csv'), index=False)
    # times_csv_path = os.path.join(output_dir, f'{subdir_name}_train_epoch_times.csv')
    # logger.info(f"{train_epoch_times=}")
    # with open(times_csv_path, "w", newline="") as csv_file:
    #     writer = csv.writer(csv_file)
    #     for epoch_time in train_epoch_times:
    #         writer.writerow([epoch_time])
    # if args.save_ckpt:
    #     # Save the model's state dictionary
    #     ckpt_path = os.path.join(ckpt_output_dir,  f'{subdir_name}{MODEL_NAME}_final.pth')
    #     torch.save(model.state_dict(), ckpt_path)
    #     logger.info(f"✅ Model saved to '{ckpt_path}'")

if args.val:    
    val_results = {
        'img_size': [],
        'valid_acc': []
    }

    if not args.train:
        if ckpt_path is None:
            ckpt_path = f"{args.root_dir}/{args.ckpt_path}"
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False))
    model.to(DEVICE)
    model.eval()
    for img_size in args.val_img_sizes:
        valid_dataset.set_transform(make_valid_transform(img_size))
        batch_size = max(1, int((args.batch_size * 0.8 * 224 * 224) / (img_size * img_size)))
        valid_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=False,
            **prefetch_kwargs,
        )
        val_correct = 0
        val_total = 0
        with torch.inference_mode():
            for inputs, labels in valid_loader:
                inputs = inputs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                with torch.amp.autocast(device_type=DEVICE.type, dtype=autocast_dtype, enabled=use_amp):
                    outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        epoch_val_acc = val_correct / val_total
        val_results['img_size'].append(img_size)
        val_results['valid_acc'].append(epoch_val_acc)
        val_df = pd.DataFrame(val_results)
        val_df.to_csv(os.path.join(output_dir, f'{subdir_name}_eval.csv'), index=False)
        logger.info(f"{img_size=}: {epoch_val_acc=}")

# del model
# gc.collect()
# if torch.cuda.is_available():
#     torch.cuda.empty_cache()

# if gpu_lock and gpu_lock.is_locked:
#     logger.info("Manually releasing lock.")
#     gpu_lock.release()
# %%
# import matplotlib.pyplot as plt
# import pandas as pd

# if history_df is None:
#     logger.info("Training history is empty. Please run the training loop first.")
# else:
#     # --- Create a single figure and axis for the plot ---
#     fig, ax = plt.subplots(figsize=(12, 7))
#     plt.title('Training and Validation Accuracy Over Epochs', fontsize=16)
    
#     # --- Plot Training & Validation Accuracy ---
#     ax.plot(history_df['epoch'], history_df['train_acc'], 's--', color='tab:green', label='Training Accuracy')
#     ax.plot(history_df['epoch'], history_df['valid_acc'], '^-', color='tab:blue', label='Validation Accuracy')
    
#     # --- Set labels and legend ---
#     ax.set_xlabel('Epochs')
#     ax.set_ylabel('Accuracy')
#     ax.legend()
#     ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
#     # Set the y-axis to be formatted as percentages
#     ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
#     ax.set_ylim(0, 1) # Set y-axis limits from 0 to 1 for accuracy

#     # Set the x-axis to show integer epoch numbers
#     ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

#     plt.tight_layout()
#     plt.show()
# %%
# import matplotlib.pyplot as plt
# import pandas as pd

# if history_df is None:
#     logger.info("Training history is empty. Please run the training loop first.")
# else:
#     # --- Create a single figure and axis for the plot ---
#     fig, ax = plt.subplots(figsize=(12, 7))
#     plt.title('Training and Validation Accuracy Over Epochs', fontsize=16)
    
#     # --- Plot Training & Validation Accuracy ---
#     ax.plot(history_df['step'], history_df['train_acc'], 's--', color='tab:green', label='Training Accuracy')
#     ax.plot(history_df['step'], history_df['valid_acc'], '^-', color='tab:blue', label='Validation Accuracy')
    
#     # --- Set labels and legend ---
#     ax.set_xlabel('Steps')
#     ax.set_ylabel('Accuracy')
#     ax.legend()
#     ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
#     # Set the y-axis to be formatted as percentages
#     ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
#     ax.set_ylim(0, 1) # Set y-axis limits from 0 to 1 for accuracy

#     # Set the x-axis to show integer epoch numbers
#     ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

#     plt.tight_layout()
#     plt.show()
