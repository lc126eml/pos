# %%
# =================================================================================
# DINOv3 segmentation training (single-file, Kaggle-friendly)
# =================================================================================
import math
import os
import glob
import sys
import time
import logging
import random
import gc
import subprocess
import shutil
import urllib.request
import zipfile
from types import SimpleNamespace
from typing import Tuple, Optional, Dict, Union, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import csv
from PIL import Image

import torchvision.transforms.functional as TF
from torchvision.transforms import ColorJitter
# =============================================================================
# Kaggle environment setup
# =============================================================================
_IS_KAGGLE = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.path.exists("/kaggle/working"))
train_start_time = time.time()
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

def _download_with_retries(url, dst, retries=3, timeout=30):
    tmp_path = dst + ".part"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/zip",
    }

    def _cleanup_tmp():
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

    def _finalize():
        if not zipfile.is_zipfile(tmp_path):
            raise RuntimeError("Downloaded file is not a zip.")
        os.replace(tmp_path, dst)

    def _try_curl():
        curl = shutil.which("curl")
        if not curl:
            print("download: curl not found", flush=True)
            return False
        print("download: trying curl", flush=True)
        cmd = [
            curl,
            "-L",
            "--retry",
            "3",
            "--retry-all-errors",
            "--max-redirs",
            "20",
            "--connect-timeout",
            str(timeout),
            "-H",
            f"User-Agent: {headers['User-Agent']}",
            "-H",
            f"Accept: {headers['Accept']}",
            "-o",
            tmp_path,
            url,
        ]
        subprocess.check_call(cmd)
        _finalize()
        print("download: curl ok", flush=True)
        return True

    def _try_requests():
        try:
            import requests  # type: ignore
        except Exception:
            print("download: requests not available", flush=True)
            return False
        print("download: trying requests", flush=True)
        resp = requests.get(url, stream=True, timeout=timeout, headers=headers, allow_redirects=True)
        resp.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        _finalize()
        print("download: requests ok", flush=True)
        return True

    def _try_urllib():
        print("download: trying urllib", flush=True)
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            with open(tmp_path, mode="wb") as f:
                shutil.copyfileobj(resp, f)
        _finalize()
        print("download: urllib ok", flush=True)
        return True

    methods = [_try_curl, _try_requests, _try_urllib]
    last_exc = None
    for attempt in range(1, retries + 1):
        for method in methods:
            try:
                ok = method()
                if ok:
                    return
            except Exception as exc:
                last_exc = exc
                print(f"download: method failed ({exc})", flush=True)
                _cleanup_tmp()
        if attempt < retries:
            time.sleep(2 * attempt)
    raise RuntimeError(f"Downloaded file is not a zip. Last error: {last_exc}")


def _ensure_pos_repo():
    if not _IS_KAGGLE:
        return None
    def _find_repo_root():
        if os.path.isdir("/kaggle/working/pos"):
            return "/kaggle/working/pos"
        for name in os.listdir("/kaggle/working"):
            cand = os.path.join("/kaggle/working", name)
            if os.path.isdir(os.path.join(cand, "core")) and os.path.isdir(os.path.join(cand, "data")):
                return cand
        return None

    url = "https://github.com/lc126eml/pos/archive/refs/heads/master.zip"
    zip_path = "/kaggle/working/pos.zip"
    if os.path.exists(zip_path) and not zipfile.is_zipfile(zip_path):
        os.remove(zip_path)
    if not os.path.exists(zip_path):
        _download_with_retries(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall("/kaggle/working")
    if os.path.exists(zip_path):
        os.remove(zip_path)

    repo_root = _find_repo_root()
    if not repo_root:
        raise RuntimeError("POS repo not found after unzip; expected /kaggle/working/pos or a repo with core/ and data/.")
    os.environ["POS_REPO_ROOT"] = repo_root
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    print(f"POS repo ready: {repo_root}", flush=True)
    return repo_root


_ensure_pos_repo()
from core.patch_pos import PatchRowColRegressionCriterion
# =============================================================================
# Configuration
# =============================================================================
if _IS_KAGGLE:
    root_dir = "/kaggle/working"
    base_path_default =  "/kaggle/input/ade20k-dataset/ADEChallengeData2016"
args = SimpleNamespace(
    model_type="dinov3",
    use_abs_pos_emb=True,
    use_rot_pos_emb=False,
    model_size='base',
    num_classes=150,
    batch_size=16,
    grad_accum_steps=1,
    train_img_size=336,
    eval_img_size=368,
    use_ms_flip_eval=False,
    scale_jitter=(1.0, 1.3),
    use_cat_max_ratio=True,
    cat_max_ratio=0.70,
    cat_max_ratio_tries=10,
    ms_scales=(0.90, 1.0, 1.15),
    eval_crop_mode="crop_or_pad",
    final_ms_flip_eval=True,
    lr=7e-05,
    lr_aux=1e-5,
    eta_min=1e-7,
    composite_lr=True,
    warmup_steps=500,
    weight_decay=0.01,
    epochs=130,
    overlap=0,
    start_epoch=0,
    seed=50,
    use_rc_loss=False,
    use_patch_position_loss=False,
    huber_beta=0.1,
    rc_alpha=70.0,
    seg_head="upernet",  # "ppmlite", "upernet", "fcn", "linear"
    feature_layers=[2, 5, 8, 11],
    workers=2 if _IS_KAGGLE else 5,
    color_jitter={"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.05},
    color_jitter_prob=0.1,
    train=True,
    val=False,
    ckpt_path=None,
    lock=False if _IS_KAGGLE else True,
    clip_value=1.0,
    output_dir=root_dir,
    log_interval=300,
    csv_interval=3,
    show_peak_gpu_mem=True,
    compile_model=False,
    save_full_ckpt=True,
    resume_full_ckpt=True,
    resume_ckpt_path='/kaggle/input/seg-base-abs-d-350/ckpt/last.pth', #seg/base_abs_pos_rc_False_lr50
    resume_scheduler=True,
    resume_optimizer=True,
    resume_bs=True,
    total_run_time_hr=12.0,
    base_path=base_path_default,
    pos_type=None,
)

ckpt = None
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
        "output_dir",
    ]
    if not args.resume_scheduler:
        skip_keys.extend([
            "epochs",
            "warmup_steps",
            "warmup_ratio",
            "eta_min",
            "composite_lr",
        ])
    if not args.resume_bs:
        skip_keys.extend(["batch_size", "grad_accum_steps"])
    ckpt = torch.load(args.resume_ckpt_path, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args", None)
    if ckpt_args is not None:
        for k, v in vars(ckpt_args).items():
            if k not in skip_keys:
                setattr(args, k, v)

if args.use_abs_pos_emb or args.use_rot_pos_emb:
    args.overlap = 0
    args.use_rc_loss = False

# if args.eval_img_size != args.train_img_size:
#     print("Best practice is to keep eval_img_size == train_img_size; overriding.", flush=True)
#     args.eval_img_size = args.train_img_size
if hasattr(args, "seg_head"):
    args.seg_head = str(args.seg_head).lower()

# =============================================================================
# Segmentation augmentations
# =============================================================================
ImageLike = Union[Image.Image, np.ndarray, torch.Tensor]
MaskLike = Union[Image.Image, np.ndarray, torch.Tensor]

from seg.seg_aug import TrainSegAug, EvalSegPreprocess, EvalSegPreprocessMSFlip

# Segmentation heads and losses
# =============================================================================
from seg.seg_head import PPMliteFCNHead, UPerNetTokenHead, FCNSegHead, LinearSegHead

from seg.seg_loss import MMSegCrossEntropyLoss

MODEL_NAME = f"vit_{args.model_size}_patch16_{args.model_type}"
TRAIN_IMAGE_PATH = os.path.join(args.base_path, "images", "training")
TRAIN_ANNOTATION_PATH = os.path.join(args.base_path, "annotations", "training")
VALID_IMAGE_PATH = os.path.join(args.base_path, "images", "validation")
VALID_ANNOTATION_PATH = os.path.join(args.base_path, "annotations", "validation")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

use_amp = False
use_bf16 = False
autocast_dtype = None

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def _seed_worker(worker_id):
    worker_seed = args.seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

data_rng = torch.Generator()
data_rng.manual_seed(args.seed)
run_tag = time.strftime("%Y%m%d_%H%M%S")

subdir_name = (
    f"{args.model_size}"
    f"{'_abs_pos' if args.use_abs_pos_emb else ''}"
    f"{'_rot_pos' if args.use_rot_pos_emb else ''}_rc_{args.use_rc_loss}_lr{int(args.lr/1e-5)}_s{args.seed}"
)
if args.use_rc_loss:
    subdir_name += f"_overlap_{args.overlap}_alpha_{int(args.rc_alpha)}"

output_dir = args.output_dir
ckpt_output_dir = os.path.join(args.output_dir, "ckpt")
# output_dir = os.path.join(args.output_dir, subdir_name)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(ckpt_output_dir, exist_ok=True)
last_ckpt_path = os.path.join(ckpt_output_dir, "last.pth")
if args.resume_full_ckpt and args.resume_ckpt_path is None:
    args.resume_ckpt_path = last_ckpt_path

log_file_path = os.path.join(output_dir, f"{subdir_name}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()],
)
logger = logging.getLogger()

logger.info("Using device: %s", DEVICE)
logger.info("Using mixed precision: disabled (fp32)")
logger.info("Arguments: %s", args)
logger.info("Output dir: %s", output_dir)
logger.info("Subdir name: %s", subdir_name)

if not os.path.isdir(TRAIN_IMAGE_PATH):
    logger.error("Missing training images at %s", TRAIN_IMAGE_PATH)
    if _IS_KAGGLE and os.path.isdir("/kaggle/input"):
        logger.error("Available /kaggle/input entries: %s", os.listdir("/kaggle/input"))
    raise FileNotFoundError(f"Training images not found: {TRAIN_IMAGE_PATH}")

# =============================================================================
# Dataset and dataloaders
# =============================================================================
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, pair_transform):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])
        self.pair_transform = pair_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        ann_path = os.path.join(self.annotation_dir, img_name.replace(".jpg", ".png"))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(ann_path).convert("L")

        out = self.pair_transform(image, mask)
        if isinstance(out, tuple) and len(out) == 3:
            image_t, mask_t, _ = out
        else:
            image_t, mask_t = out
        mask_t = mask_t.long() - 1
        return image_t, mask_t

train_dataset = SegmentationDataset(
    TRAIN_IMAGE_PATH,
    TRAIN_ANNOTATION_PATH,
    pair_transform=TrainSegAug(
        target_size=(args.train_img_size, args.train_img_size),
        scale_jitter=args.scale_jitter,
        cat_max_ratio=(args.cat_max_ratio if args.use_cat_max_ratio else None),
        cat_max_ratio_tries=args.cat_max_ratio_tries,
        ignore_index=0 if args.use_cat_max_ratio else None,
        color_jitter=args.color_jitter,
        color_jitter_prob=args.color_jitter_prob,
        normalize=True,
    ),
)
valid_dataset = SegmentationDataset(
    VALID_IMAGE_PATH,
    VALID_ANNOTATION_PATH,
    pair_transform=EvalSegPreprocess(
        target_size=(args.eval_img_size, args.eval_img_size),
        target_by="shorter",
        eval_crop_mode=args.eval_crop_mode,
        normalize=True,
    ),
)

loader_kwargs = dict(
    num_workers=args.workers,
    pin_memory=True,
    worker_init_fn=_seed_worker,
    generator=data_rng,
    persistent_workers=(args.workers > 0),
)
if args.workers > 0:
    loader_kwargs["prefetch_factor"] = 2

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
    **loader_kwargs,
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    drop_last=False,
    **loader_kwargs,
)

steps_per_epoch = len(train_loader)
accum_steps = max(1, int(getattr(args, "grad_accum_steps", 1)))
optimizer_steps_per_epoch = math.ceil(steps_per_epoch / accum_steps)
logger.info("DataLoaders created: train=%s, val=%s", len(train_dataset), len(valid_dataset))

# =============================================================================
# Model, head, optimizer
# =============================================================================
logger.info("Initializing model: %s for %s classes", MODEL_NAME, args.num_classes)
model = timm.create_model(
    MODEL_NAME,
    pretrained=False,
    use_abs_pos_emb=args.use_abs_pos_emb,
    use_rot_pos_emb=args.use_rot_pos_emb,
    num_classes=0,
    dynamic_img_size=True,
    img_size=args.train_img_size,
).to(DEVICE)

grid_h, grid_w = model.patch_embed.grid_size
decoder_type = getattr(args, "seg_head", "ppmlite").lower()
if decoder_type == "ppmlite":
    decoder = PPMliteFCNHead(
        embed_dim=model.embed_dim,
        num_classes=args.num_classes,
        grid_size=(grid_h, grid_w),
        out_size=(args.train_img_size, args.train_img_size),
        mid_channels=256,
        ppm_bins=(1, 2, 3),
        ppm_channels=64,
        dropout=0.1,
        norm="gn",
    ).to(DEVICE)
elif decoder_type == "upernet":
    embed_dims = [model.embed_dim] * len(args.feature_layers)
    decoder = UPerNetTokenHead(
        embed_dims=embed_dims,
        num_classes=args.num_classes,
        grid_size=(grid_h, grid_w),
        out_size=(args.train_img_size, args.train_img_size),
        fpn_channels=256,
        ppm_bins=(1, 2, 3, 6),
        dropout=0.1,
        norm="gn",
    ).to(DEVICE)
elif decoder_type == "fcn":
    decoder = FCNSegHead(
        embed_dim=model.embed_dim,
        num_classes=args.num_classes,
        grid_size=(grid_h, grid_w),
        out_size=(args.train_img_size, args.train_img_size),
        mid_channels=256,
        dropout=0.1,
        norm="gn",
    ).to(DEVICE)
elif decoder_type == "linear":
    decoder = LinearSegHead(
        embed_dim=model.embed_dim,
        num_classes=args.num_classes,
        grid_size=(grid_h, grid_w),
        out_size=(args.train_img_size, args.train_img_size),
        dropout=0.1,
    ).to(DEVICE)
else:
    raise ValueError(f"Unsupported seg_head='{decoder_type}'. Use 'ppmlite', 'upernet', 'fcn', or 'linear'.")

logger.info("model.patch_embed.proj %s", model.patch_embed.proj)
if args.overlap > 0:
    original_patch_size = model.patch_embed.proj.kernel_size[0]
    new_patch_size = original_patch_size + args.overlap
    stride = original_patch_size
    original_grid_size = args.train_img_size // stride
    padding = ((original_grid_size - 1) * stride + new_patch_size - args.train_img_size + 1) // 2
    in_chans = model.patch_embed.proj.in_channels
    embed_dim = model.patch_embed.proj.out_channels
    model.patch_embed.proj = nn.Conv2d(
        in_chans, embed_dim,
        kernel_size=(new_patch_size, new_patch_size),
        stride=(stride, stride),
        padding=padding,
    ).to(DEVICE)

if args.compile_model:
    if hasattr(torch, "compile"):
        logger.info("Compiling model with torch.compile (mode='reduce-overhead').")
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
        decoder = torch.compile(decoder)
    else:
        logger.warning("torch.compile not available; skipping compilation.")

dynamic = True
training_parameters = list(model.parameters()) + list(decoder.parameters())
param_groups = []
lr_aux = getattr(args, "lr_aux", args.lr)
if args.use_rc_loss:
    grid_h, grid_w = model.patch_embed.grid_size
    dynamic = False
    rowcol_loss = PatchRowColRegressionCriterion(
        feat_dim=model.embed_dim,
        grid_h=grid_h,
        grid_w=grid_w,
        huber_beta=args.huber_beta,
    ).to(DEVICE)
    training_parameters += list(rowcol_loss.parameters())
    param_groups.append({"params": rowcol_loss.parameters(), "weight_decay": 0.0, "lr": lr_aux})

decay_params = []
no_decay_params = []
for n, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if n.endswith(".bias") or ("norm" in n.lower()):
        no_decay_params.append(p)
    else:
        decay_params.append(p)
for n, p in decoder.named_parameters():
    if not p.requires_grad:
        continue
    if n.endswith(".bias") or ("norm" in n.lower()):
        no_decay_params.append(p)
    else:
        decay_params.append(p)

param_groups.append({"params": decay_params, "lr": args.lr, "weight_decay": args.weight_decay})
param_groups.append({"params": no_decay_params, "lr": args.lr, "weight_decay": 0.0})

ce_criterion = MMSegCrossEntropyLoss(ignore_index=-1, avg_non_ignore=True)
optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
total_steps = args.epochs * optimizer_steps_per_epoch
if args.composite_lr:
    warmup_steps = min(args.warmup_steps, max(1, total_steps - 1))
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-7 / args.lr,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=1e-8,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )
else:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.eta_min
    )
logger.info("Initialized loss, optimizer, and scheduler.")

# =============================================================================
# Helpers
# =============================================================================
def _infer_grid_hw(model, inputs):
    patch_size = model.patch_embed.patch_size
    if isinstance(patch_size, tuple):
        ph, pw = patch_size
    else:
        ph = pw = patch_size
    return (inputs.shape[-2] // ph, inputs.shape[-1] // pw)

def _round_to_multiple(x: int, m: int) -> int:
    return max(m, int(round(x / m) * m))

def _strip_prefix_tokens(features, grid_hw, num_prefix_tokens):
    if num_prefix_tokens <= 0:
        return features
    tokens_needed = grid_hw[0] * grid_hw[1]
    stripped = []
    for feat in features:
        if feat.dim() == 3 and feat.shape[1] == tokens_needed + num_prefix_tokens:
            stripped.append(feat[:, num_prefix_tokens:, :])
        else:
            stripped.append(feat)
    return stripped

def _forward_upernet(model, decoder, inputs, feature_layers):
    if not feature_layers:
        raise ValueError("feature_layers must be set when using seg_head='upernet'.")
    grid_hw = _infer_grid_hw(model, inputs)
    features = model.forward_intermediates(
        inputs,
        indices=feature_layers,
        norm=False,
        intermediates_only=True,
        output_fmt="NLC",
    )
    features = _strip_prefix_tokens(features, grid_hw, model.num_prefix_tokens)
    outputs = decoder(features, grid_sizes=[grid_hw] * len(features), out_size=inputs.shape[-2:])
    return outputs, features, grid_hw

def _ms_flip_predict(model, decoder, inputs, num_classes, scales, flip, patch_size, *, feature_layers=None):
    if isinstance(patch_size, tuple):
        ph, pw = patch_size
    else:
        ph = pw = patch_size
    b, _, h0, w0 = inputs.shape
    logits_sum = torch.zeros((b, num_classes, h0, w0), device=inputs.device, dtype=inputs.dtype)
    count = 0
    for s in scales:
        hs = _round_to_multiple(int(round(h0 * s)), ph)
        ws = _round_to_multiple(int(round(w0 * s)), pw)
        x_s = F.interpolate(inputs, size=(hs, ws), mode="bilinear", align_corners=False)
        grid_hw = _infer_grid_hw(model, x_s)
        if args.seg_head == "upernet":
            if not feature_layers:
                raise ValueError("feature_layers must be set when using seg_head='upernet'.")
            feats = model.forward_intermediates(
                x_s,
                indices=feature_layers,
                norm=False,
                intermediates_only=True,
                output_fmt="NLC",
            )
            feats = _strip_prefix_tokens(feats, grid_hw, model.num_prefix_tokens)
            logits = decoder(feats, grid_sizes=[grid_hw] * len(feats), out_size=(hs, ws))
        else:
            feats = model.forward_features(x_s)
            logits = decoder(feats[:, model.num_prefix_tokens:, :], grid_size=grid_hw, out_size=(hs, ws))
        logits = F.interpolate(logits, size=(h0, w0), mode="bilinear", align_corners=False)
        logits_sum += logits
        count += 1
        if flip:
            x_f = torch.flip(x_s, dims=[3])
            if args.seg_head == "upernet":
                if not feature_layers:
                    raise ValueError("feature_layers must be set when using seg_head='upernet'.")
                feats_f = model.forward_intermediates(
                    x_f,
                    indices=feature_layers,
                    norm=False,
                    intermediates_only=True,
                    output_fmt="NLC",
                )
                feats_f = _strip_prefix_tokens(feats_f, grid_hw, model.num_prefix_tokens)
                logits_f = decoder(feats_f, grid_sizes=[grid_hw] * len(feats_f), out_size=(hs, ws))
            else:
                feats_f = model.forward_features(x_f)
                logits_f = decoder(feats_f[:, model.num_prefix_tokens:, :], grid_size=grid_hw, out_size=(hs, ws))
            logits_f = torch.flip(logits_f, dims=[3])
            logits_f = F.interpolate(logits_f, size=(h0, w0), mode="bilinear", align_corners=False)
            logits_sum += logits_f
            count += 1
    return logits_sum / max(count, 1)

@torch.no_grad()
def fast_confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int = -1):
    pred = pred.view(-1).to(torch.int64)
    target = target.view(-1).to(torch.int64)
    valid = target != ignore_index
    pred = pred[valid]
    target = target[valid]
    idx = target * num_classes + pred
    conf = torch.bincount(idx, minlength=num_classes * num_classes)
    return conf.view(num_classes, num_classes)

# =============================================================================
# Train / validation
# =============================================================================
ckpt_path = None
if args.train:
    logger.info("Starting training for %s", MODEL_NAME)
    start_epoch = 0
    training_history = None
    if args.resume_full_ckpt and args.resume_ckpt_path and ckpt is not None:
        model.load_state_dict(ckpt.get("model", {}), strict=False)
        decoder.load_state_dict(ckpt.get("decoder", {}), strict=False)
        if args.resume_optimizer:
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
        else:
            logger.info("Skipping optimizer state load (resume_optimizer=False).")
        if args.resume_scheduler:
            start_epoch = int(ckpt.get("epoch", 0))
            if "scheduler" in ckpt and ckpt["scheduler"] is not None:
                scheduler.load_state_dict(ckpt["scheduler"])
        else:
            logger.info("Skipping scheduler state load (resume_scheduler=False).")
        if args.use_rc_loss and "rowcol_loss" in ckpt and ckpt["rowcol_loss"] is not None:
            for k in ["row_targets", "col_targets", "row_index_full", "col_index_full"]:
                if k in ckpt["rowcol_loss"]:
                    ckpt["rowcol_loss"].pop(k)
            rowcol_loss.load_state_dict(ckpt["rowcol_loss"])
        logger.info("Resumed full checkpoint from %s at epoch %s", args.resume_ckpt_path, start_epoch)
        training_history = ckpt.get("training_history", None)

    if not isinstance(training_history, dict):
        if args.use_rc_loss:
            training_history = {
                "train_loss": [],
                "train_acc": [],
                "valid_acc": [],
                "valid_miou": [],
                "train_time": [],
                "val_time": [],
                "epoch": [],
                "step": [],
                "base_loss": [],
                "aux_loss": [],
            }
        else:
            training_history = {
                "train_loss": [],
                "train_acc": [],
                "valid_acc": [],
                "valid_miou": [],
                "train_time": [],
                "val_time": [],
                "epoch": [],
                "step": [],
            }
    training_history.setdefault("train_time", [])
    training_history.setdefault("val_time", [])

    def _pad_history(hist, fill_value=None):
        keys = [k for k, v in hist.items() if isinstance(v, list)]
        if not keys:
            return
        max_len = max(len(hist[k]) for k in keys)
        for k in keys:
            if len(hist[k]) < max_len:
                hist[k].extend([fill_value] * (max_len - len(hist[k])))

    def _append_eval_log(log_path, row):
        if not row:
            return
        df = pd.DataFrame([row])
        header = not os.path.exists(log_path)
        df.to_csv(log_path, mode="a", header=header, index=False)

    if args.resume_full_ckpt:
        _pad_history(training_history)

    step = int(training_history.get("step", [0])[-1]) if training_history.get("step") else 0
    best_acc = 0.0
    best_miou = 0.0
    best_ckpt_path = os.path.join(ckpt_output_dir, "best.pth")
    if training_history.get("valid_miou"):
        best_miou = max(training_history.get("valid_miou", [0.0]) or [0.0])
    if training_history.get("valid_acc"):
        best_acc = max(training_history.get("valid_acc", [0.0]) or [0.0])
    log_interval = getattr(args, "log_interval", 50)
    csv_interval = getattr(args, "csv_interval", 1)
    last_trained_epoch = int(start_epoch)

    for epoch in range(start_epoch, args.epochs):
        epoch_train_start = time.time()
        model.train()
        decoder.train()
        running_loss_t = torch.zeros((), device=DEVICE)
        base_loss_t = torch.zeros((), device=DEVICE)
        aux_loss_sum_t = torch.zeros((), device=DEVICE)
        train_correct_t = torch.zeros((), device=DEVICE)
        train_total_t = torch.zeros((), device=DEVICE)
        train_samples_t = torch.zeros((), device=DEVICE)
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            bs = inputs.size(0)
            aux_loss = None

            if args.seg_head == "upernet":
                outputs, features, grid_hw = _forward_upernet(model, decoder, inputs, args.feature_layers)
                last_tokens = features[-1]
            else:
                feats = model.forward_features(inputs)
                grid_hw = _infer_grid_hw(model, inputs)
                outputs = decoder(
                    feats[:, model.num_prefix_tokens:, :],
                    grid_size=grid_hw,
                    out_size=inputs.shape[-2:],
                )
                last_tokens = feats[:, model.num_prefix_tokens:, :]
            loss = ce_criterion(outputs, labels)
            base_loss = loss

            if args.use_rc_loss:
                aux_loss = rowcol_loss(last_tokens)
                aux_loss_sum_t += aux_loss.detach() * bs
                loss = base_loss + args.rc_alpha * aux_loss

            loss_scaled = loss / accum_steps
            loss_scaled.backward()

            do_step = ((batch_idx + 1) % accum_steps == 0) or (batch_idx + 1 == len(train_loader))
            if do_step:
                if args.clip_value is not None:
                    torch.nn.utils.clip_grad_norm_(training_parameters, max_norm=args.clip_value)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                pred = outputs.detach().argmax(dim=1)
                mask = (labels >= 0)
                valid_pixels = mask.sum()
                train_correct_t += ((pred == labels) & mask).sum()
                train_total_t += valid_pixels
                train_samples_t += bs

            running_loss_t += loss.detach() * valid_pixels
            if args.use_rc_loss:
                base_loss_t += base_loss.detach() * valid_pixels

            if (step + 1) % log_interval == 0:
                avg_loss = (running_loss_t / train_total_t.clamp_min(1)).float().item()
                avg_acc = (train_correct_t / train_total_t.clamp_min(1)).float().item()
                peak_mb = None
                if args.show_peak_gpu_mem and torch.cuda.is_available():
                    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                msg = f"Epoch {epoch+1}/{args.epochs} step {step+1}: loss={avg_loss:.4f} acc={avg_acc:.3f}"
                if args.use_rc_loss:
                    avg_aux = (aux_loss_sum_t / train_samples_t.clamp_min(1)).float().item()
                    msg += f" aux={avg_aux:.4f}"
                if peak_mb is not None:
                    msg += f" peak_mem={peak_mb:.0f}MB"
                logger.info(msg)

            step += 1

        train_time = time.time() - epoch_train_start
        model.eval()
        decoder.eval()
        val_correct_t = torch.zeros((), device=DEVICE)
        val_total_t = torch.zeros((), device=DEVICE)
        confmat = torch.zeros((args.num_classes, args.num_classes), device=DEVICE, dtype=torch.int64)

        val_start = time.time()
        with torch.inference_mode():
            for inputs, labels in valid_loader:
                inputs = inputs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                if args.use_ms_flip_eval:
                    outputs = _ms_flip_predict(
                        model,
                        decoder,
                        inputs,
                        args.num_classes,
                        args.ms_scales,
                        True,
                        model.patch_embed.patch_size,
                        feature_layers=args.feature_layers,
                    )
                else:
                    if args.seg_head == "upernet":
                        outputs, _, grid_hw = _forward_upernet(model, decoder, inputs, args.feature_layers)
                    else:
                        feats = model.forward_features(inputs)
                        grid_hw = _infer_grid_hw(model, inputs)
                        outputs = decoder(
                            feats[:, model.num_prefix_tokens:, :],
                            grid_size=grid_hw,
                            out_size=inputs.shape[-2:],
                        )
                pred = outputs.argmax(dim=1)
                mask = (labels >= 0)
                val_correct_t += ((pred == labels) & mask).sum()
                val_total_t += mask.sum()
                confmat += fast_confusion_matrix(pred, labels, args.num_classes, ignore_index=-1)

        val_time = time.time() - val_start
        confmat_f = confmat.to(torch.float32)
        intersection = torch.diag(confmat_f)
        union = confmat_f.sum(dim=1) + confmat_f.sum(dim=0) - intersection
        valid = union > 0
        epoch_val_miou = (intersection[valid] / union[valid]).mean().item() if valid.any() else 0.0

        epoch_val_acc = (val_correct_t / val_total_t.clamp_min(1)).float().item()
        epoch_train_acc = (train_correct_t / train_total_t.clamp_min(1)).float().item()
        denom_pixels = train_total_t.clamp_min(1).float()
        denom_samples = train_samples_t.clamp_min(1).float()
        epoch_train_loss = (running_loss_t / denom_pixels).float().item()
        if best_acc < epoch_val_acc:
            best_acc = epoch_val_acc
        improved_miou = epoch_val_miou > best_miou
        if improved_miou:
            best_miou = epoch_val_miou

        logger.info("Epoch %s/%s Summary:", epoch + 1 + args.start_epoch, args.epochs)
        logger.info("Step %s Summary:", step)

        if args.use_rc_loss:
            epoch_aux_loss = (aux_loss_sum_t / denom_samples).float().item()
            epoch_base_loss = (base_loss_t / denom_pixels).float().item()
            logger.info(
                "  Train Loss: %.4f | Aux Loss: %.4f | Base Loss: %.4f | Train Acc: %.4f | "
                "Valid Acc: %.4f | Valid mIoU: %.4f | train_time: %.1fs | val_time: %.1fs",
                epoch_train_loss, epoch_aux_loss, epoch_base_loss, epoch_train_acc, epoch_val_acc,
                epoch_val_miou, train_time, val_time,
            )
            training_history["aux_loss"].append(epoch_aux_loss)
            training_history["base_loss"].append(epoch_base_loss)
        else:
            logger.info(
                "  Train Loss: %.4f | Train Acc: %.4f | Valid Acc: %.4f | Valid mIoU: %.4f | "
                "train_time: %.1fs | val_time: %.1fs",
                epoch_train_loss, epoch_train_acc, epoch_val_acc, epoch_val_miou, train_time, val_time,
            )

        training_history["train_loss"].append(epoch_train_loss)
        training_history["train_acc"].append(epoch_train_acc)
        training_history["valid_acc"].append(epoch_val_acc)
        training_history["valid_miou"].append(epoch_val_miou)
        training_history["train_time"].append(train_time)
        training_history["val_time"].append(val_time)
        training_history["epoch"].append(epoch + 1)
        training_history["step"].append(step)
        last_trained_epoch = epoch + 1

        if (epoch + 1) % csv_interval == 0:
            pd.DataFrame(training_history).to_csv(os.path.join(output_dir, f"{subdir_name}.csv"), index=False)
        if improved_miou:
            best_ckpt = {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "decoder": decoder.state_dict(),
                "metric": {
                    "valid_miou": epoch_val_miou,
                    "valid_acc": epoch_val_acc,
                },
            }
            torch.save(best_ckpt, best_ckpt_path)
            logger.info("Saved best checkpoint (weights only) to %s", best_ckpt_path)
        if args.save_full_ckpt:
            ckpt = {
                "epoch": epoch + 1,
                "step": step,
                "model": model.state_dict(),
                "decoder": decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "scaler": None,
                "rowcol_loss": rowcol_loss.state_dict() if args.use_rc_loss else None,
                "training_history": training_history,
                "args": args,
            }
            torch.save(ckpt, last_ckpt_path)
            logger.info("Saved full checkpoint to %s", last_ckpt_path)

        if args.total_run_time_hr is not None:
            elapsed = time.time() - train_start_time
            max_run_time_sec = args.total_run_time_hr * 3600            
            if elapsed + (train_time + val_time) + 900 >= max_run_time_sec:
                logger.info(
                    "Stopping training: elapsed %.0fs reached limit %.2fh.",
                    elapsed,
                    args.total_run_time_hr,
                )
                break
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("Training complete.")
    logger.info("Best Accuracy: %.4f", best_acc)
    logger.info(output_dir)

    if args.final_ms_flip_eval and not args.use_ms_flip_eval:
        def _run_ms_flip_eval(tag: str):
            model.eval()
            decoder.eval()
            val_correct_t = torch.zeros((), device=DEVICE)
            val_total_t = torch.zeros((), device=DEVICE)
            confmat = torch.zeros((args.num_classes, args.num_classes), device=DEVICE, dtype=torch.int64)
            with torch.inference_mode():
                for inputs, labels in valid_loader:
                    inputs = inputs.to(DEVICE, non_blocking=True)
                    labels = labels.to(DEVICE, non_blocking=True)
                    outputs = _ms_flip_predict(
                        model,
                        decoder,
                        inputs,
                        args.num_classes,
                        args.ms_scales,
                        True,
                        model.patch_embed.patch_size,
                        feature_layers=args.feature_layers,
                    )
                    pred = outputs.argmax(dim=1)
                    mask = (labels >= 0)
                    val_correct_t += ((pred == labels) & mask).sum()
                    val_total_t += mask.sum()
                    confmat += fast_confusion_matrix(pred, labels, args.num_classes, ignore_index=-1)

            confmat_f = confmat.to(torch.float32)
            intersection = torch.diag(confmat_f)
            union = confmat_f.sum(dim=1) + confmat_f.sum(dim=0) - intersection
            valid = union > 0
            ms_miou = (intersection[valid] / union[valid]).mean().item() if valid.any() else 0.0
            ms_acc = (val_correct_t / val_total_t.clamp_min(1)).float().item()
            logger.info("%s MS+Flip Acc: %.4f | %s MS+Flip mIoU: %.4f", tag, ms_acc, tag, ms_miou)
            return ms_acc, ms_miou

        logger.info("Running final multi-scale + flip evaluation (final checkpoint)...")
        final_eval_row = {
            "run_tag": run_tag,
            "subdir_name": subdir_name,
            "output_dir": output_dir,
            "epoch": int(last_trained_epoch),
        }
        final_ms_acc, final_ms_miou = _run_ms_flip_eval("Final")
        final_eval_row["final_ms_flip_acc"] = final_ms_acc
        final_eval_row["final_ms_flip_miou"] = final_ms_miou

        if os.path.exists(best_ckpt_path):
            best_ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(best_ckpt.get("model", {}), strict=False)
            decoder.load_state_dict(best_ckpt.get("decoder", {}), strict=False)
            logger.info("Loaded best checkpoint for final MS+Flip evaluation.")
            best_ms_acc, best_ms_miou = _run_ms_flip_eval("Best")
            final_eval_row["best_ms_flip_acc"] = best_ms_acc
            final_eval_row["best_ms_flip_miou"] = best_ms_miou
        else:
            logger.info("Best checkpoint not found; skipping best MS+Flip evaluation.")

        final_eval_log = os.path.join(output_dir, f"{subdir_name}_final_eval.csv")
        _append_eval_log(final_eval_log, final_eval_row)
    history_df = pd.DataFrame(training_history)
    history_df.to_csv(os.path.join(output_dir, f'{subdir_name}.csv'), index=False)
    # save_checkpoint(model, decoder, ckpt_output_dir, "final")

    best_miou = history_df['valid_miou'].max()
    best_epoch = history_df.loc[history_df['valid_miou'].idxmax(), 'epoch']
    logger.info(f"Best miou: {best_miou:.4f} at epoch {best_epoch}")

    # Find the epoch with the best validation a1 score
    best_miou_row = history_df.loc[history_df['valid_miou'].idxmax()]
    best_miou_epoch = int(best_miou_row['epoch'])
    best_miou_val = best_miou_row['valid_miou']

    # Find the epoch with the best validation abs_rel
    best_acc_row = history_df.loc[history_df['valid_acc'].idxmax()]
    best_acc_epoch = int(best_acc_row['epoch'])
    best_acc_val = best_acc_row['valid_acc']

    logger.info("\n--- Best Validation Metrics from History ---")
    logger.info(f"  Best miou:      {best_miou_val:.4f} (Epoch {best_miou_epoch})")
    logger.info(f"  Best acc:  {best_acc_val:.4f} (Epoch {best_acc_epoch})")
    logger.info("------------------------------------------")
