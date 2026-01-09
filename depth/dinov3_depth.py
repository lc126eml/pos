import gc
import os
import torch
import torch.nn as nn
from functools import partial
from types import SimpleNamespace
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
import warnings
import timm  # Imported for DINOv2
from torch.nn import functional as F
import logging
import warnings
from typing import List, Tuple, Union
try:
    from filelock import FileLock
except ImportError:
    FileLock = None

if os.path.exists('/home/sshuser'):
    root_dir = '/home/sshuser'
    BASE_PATH = f'{root_dir}/Data/imagenet100/'
elif os.path.exists('/lc'):
    root_dir = '/lc/logs'
    BASE_PATH = f'/lc/data/imagenet100/'
elif os.path.exists("/home/liucong"):
    root_dir = '/home/liucong/codes/pos/logs'
    BASE_PATH = f'/home/liucong/data/3d'
else:
    root_dir = '/linux'
    BASE_PATH = f'{root_dir}/Data/imagenet100/'
# Add a3R project to path
import sys
sys.path.insert(0, f'/lc/code/pos')
# sys.path.insert(0, '/lc/code/3D/a3R/src')

# from utils import wait_for_python_gpu_processes

from hypersim_simple_dataset import HyperSim_Simple
from aug import train_aug_depth_ar_resize_random_crop, eval_preprocess_depth_keep_ar

warnings.filterwarnings('ignore')

args = SimpleNamespace(
    # data_root="/lc/data/3D",
    data_root="/home/liucong/data/3d",
    model_type= "dinov3",
    use_abs_pos_emb=False,
    use_rot_pos_emb=False,
    model_size='base',
    img_sizes=[224],
    batch_size=256,
    # batch_size=6,
    patch_size=16,
    lr=5e-4,
    lr_aux=1e-5,
    epochs=80,
    has_pos=False,
    weight_decay=0.01,
    overlap=0,
    seed=55,
    val_steps=None,
    use_rc_loss=False,
    rc_alpha=20.0,
    # warmup_steps_for_aux=100,
    workers=5,
    warmup_steps=20,
    clip_value=1.0,
    lock=True,
    depth_decoder="lite4",  # "simple" or "lite4"
    log_interval=100,
    depth_eval_mode="scale_invariant",  # "metric" or "scale_invariant"
    depth_norm="median",  # "mean" or "median" (scale-invariant alignment)
    ssim_norm_mode="per_image",  # "fixed_range" or "per_image"
    ssim_percentiles=(5.0, 95.0),
    debug_dataset=False,
    output_dir=f'{root_dir}/output/depth_alpha',
    csv_interval=5,
    # prefetch_factor=2,
    compile_model=False,
)

if args.use_abs_pos_emb or args.use_rot_pos_emb:
    args.overlap = 0
    # args.use_patch_position_loss=False
    args.use_rc_loss = False
print(args)

MODEL_NAME = f"vit_{args.model_size}_patch16_{args.model_type}"
NUM_CLASSES = 1
BATCH_SIZE = args.batch_size
IMG_SIZE = args.img_sizes[0]
EPOCHS = args.epochs
HAS_POS = args.has_pos
OVERLAP = args.overlap
SEED = args.seed
VAL_STEPS = args.val_steps
Use_Row_Col_Loss = args.use_rc_loss
RC_ALPHA = args.rc_alpha

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# torch.backends.cudnn.benchmark = True

use_amp = torch.cuda.is_available()
use_bf16 = use_amp and torch.cuda.is_bf16_supported()
autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

# --- Setup Logging ---
output_dir = args.output_dir

subdir_name = (
    f"{args.model_size}"
    f"{'_abs_pos' if args.use_abs_pos_emb else ''}"
    f"{'_rot_pos' if args.use_rot_pos_emb else ''}_rc_{args.use_rc_loss}_lr{int(args.lr/1e-5)}"
)
if args.use_rc_loss:
    subdir_name += f"_overlap_{args.overlap}_alpha_{int(args.rc_alpha)}"

output_dir = os.path.join(args.output_dir, subdir_name)
ckpt_output_dir = output_dir.replace("/output/", "/output/ckpt/")
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

logger.info(f"Arguments: {args}")
logger.info(f"Using device: {DEVICE}")
logger.info(f"Using mixed precision: {'disabled' if not use_amp else ('bfloat16' if use_bf16 else 'float16')}")
logger.info(args)
logger.info(output_dir)
logger.info(subdir_name)
# wait_for_python_gpu_processes(poll_interval_minutes=5, logger=logger)
# --- Acquire a file lock to ensure exclusive GPU usage ---
if args.lock:
    if FileLock:
        lock_path = "/tmp/gpu.lock"
        gpu_lock = FileLock(lock_path)
        logger.info(f"Attempting to acquire lock on '{lock_path}'...")
        gpu_lock.acquire()
        logger.info("Lock acquired. It is safe to proceed.")
        # The lock will be automatically released when the script exits.
    else:
        logger.warning("`filelock` library not found, skipping lock. Run `pip install filelock`.")

logger.info(args)
# %%
# =================================================================================
# Step 2: Dataset and DataLoader
# =================================================================================
logger.info("Creating datasets...")
try:
    train_dataset = HyperSim_Simple(
        split='train',
        ROOT=f'{args.data_root}/hypersim_processed/train',
        resolution=IMG_SIZE,
        num_views=1,
        pair_transform=partial(
            train_aug_depth_ar_resize_random_crop,
            target_size=(IMG_SIZE, IMG_SIZE),
            normalize=True,
        ),
    )
    valid_dataset = HyperSim_Simple(
        split='test',
        ROOT=f'{args.data_root}/hypersim_processed/test',
        resolution=IMG_SIZE,
        num_views=1,
        seed=777,
        pair_transform=partial(
            eval_preprocess_depth_keep_ar,
            target_size=(IMG_SIZE, IMG_SIZE),
            target_by="height",
            ensure_multiple_of=args.patch_size,
            normalize=True,
        ),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=args.workers,
        pin_memory=torch.cuda.is_available(), drop_last=True,
        persistent_workers=(args.workers > 0), 
        # prefetch_factor=args.prefetch_factor,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=5,
        pin_memory=torch.cuda.is_available(), drop_last=False,
        persistent_workers=True, 
        # prefetch_factor=args.prefetch_factor,
    )
    steps_per_epoch = len(train_loader)
    logger.info(f"✅ DataLoaders created successfully.")
    logger.info(f"   - Training samples: {len(train_dataset)}, Batches per epoch: {len(train_loader)}")
    logger.info(f"   - Validation samples: {len(valid_dataset)}, Batches per epoch: {len(valid_loader)}")
    
    if args.debug_dataset:
        # Test sample loading and display stats
        logger.info("\n🔍 Dataset validation:")
        batch_imgs, batch_depths = next(iter(train_loader))
        logger.info(f"   - Batch shapes: images {batch_imgs.shape}, depths {batch_depths.shape}")
        logger.info(f"   - Depth range: {batch_depths.min().item():.2f}m to {batch_depths.max().item():.2f}m")
        logger.info(f"   - Image stats: mean={batch_imgs.mean():.3f}, std={batch_imgs.std():.3f}")

        logger.info("\n🔍 Valid Dataset validation:")
        batch_imgs, batch_depths = next(iter(valid_loader))
        logger.info(f"   - Batch shapes: images {batch_imgs.shape}, depths {batch_depths.shape}")
        logger.info(f"   - Depth range: {batch_depths.min().item():.2f}m to {batch_depths.max().item():.2f}m")
        logger.info(f"   - Image stats: mean={batch_imgs.mean():.3f}, std={batch_imgs.std():.3f}")
    
except Exception as e:
    logger.error(f"❌ Error creating datasets: {e}")
    logger.error("   Please ensure 'data_root' is configured correctly and the HyperSim dataset exists.")
    import traceback
    traceback.print_exc()
    raise


def setup_model(img_size, device):
    logger.info(f"Creating {MODEL_NAME} via timm...")
    model = timm.create_model(MODEL_NAME, 
                            pretrained=False, # As requested: trains the model from scratch
                            use_abs_pos_emb=args.use_abs_pos_emb,
                            use_rot_pos_emb=args.use_rot_pos_emb,
                            num_classes=0, # Set the classifier head to 100 classes
                            dynamic_img_size=True,
                              img_size=img_size)
    model = model.to(device)
    
    for param in model.parameters():
        param.requires_grad = True
    
    from depth_head import Lite4LayerDepthHead, SimpleDepthDecoderV2
    decoder_type = getattr(args, "depth_decoder", "simple")
    if decoder_type == "lite4":
        decoder = Lite4LayerDepthHead(
            embed_dim=model.embed_dim,
        ).to(device)
    elif decoder_type == "simple":
        decoder = SimpleDepthDecoderV2(embed_dim=model.embed_dim).to(device)
    else:
        raise ValueError(f"Unsupported depth_decoder='{decoder_type}'. Use 'simple' or 'lite4'.")

    
    # Sanity check
    feature_layers = [2, 5, 8, 11]
    # dummy_input = torch.randn(2, 3, img_size, img_size).to(device)
    # with torch.no_grad():
    #     # features = model.forward_features(dummy_input)  # (B, N+1, 768)
    #     # dummy_output = decoder(features)
    #     features = model.get_intermediate_layers(dummy_input, n=feature_layers, norm=False)
    #     dummy_output, _ = decoder(features, dummy_input.unsqueeze(1), patch_start_idx=0)
    #     dummy_output = dummy_output.squeeze(1)
    # logger.info(f"Model created successfully!")
    # logger.info(f"Input shape: {dummy_input.shape}")
    # logger.info(f"Number of feature maps extracted: {len(features)}")
    # logger.info(f"Output shape: {dummy_output.shape}")
    # assert dummy_output.shape == (2, 1, img_size, img_size), f"Expected output shape (2, 1, {img_size}, {img_size})"
    # logger.info("✅ Output shape is correct.")
    # encoder_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # decoder_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    # logger.info(f"Encoder params: {encoder_params/1e6:.2f}M, Decoder: {decoder_params/1e6:.2f}M")
    
    return model, decoder, feature_layers

model, decoder, feature_layers = setup_model(IMG_SIZE, DEVICE)
if args.compile_model:
    try:
        model = torch.compile(model)
        decoder = torch.compile(decoder)
        logger.info("✅ torch.compile enabled for model.")
    except Exception as e:
        logger.warning(f"torch.compile failed; continuing without it. Error: {e}")


def _compute_scale_from_gt(gt, mask, mode):
    if mode == "mean":
        denom = mask.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1)
        scale = (gt * mask).sum(dim=(1, 2, 3), keepdim=True) / denom
        return scale.clamp_min(1e-8)
    if mode == "median":
        out = []
        for b in range(gt.shape[0]):
            gb = gt[b, 0]
            mb = mask[b, 0] > 0.5
            vals = gb[mb]
            if vals.numel() == 0:
                out.append(torch.tensor(1.0, device=gt.device, dtype=gt.dtype))
            else:
                out.append(vals.median())
        return torch.stack(out, dim=0).view(gt.shape[0], 1, 1, 1).clamp_min(1e-8)
    raise ValueError(f"Unsupported depth_norm='{mode}'. Use 'mean' or 'median'.")


def _compute_scale_align_pred(gt, pred, mask, mode):
    if mode == "mean":
        denom = mask.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1)
        gt_mean = (gt * mask).sum(dim=(1, 2, 3), keepdim=True) / denom
        pred_mean = (pred * mask).sum(dim=(1, 2, 3), keepdim=True) / denom
        scale = gt_mean / pred_mean.clamp_min(1e-8)
        return scale.clamp_min(1e-8)
    if mode == "median":
        out = []
        for b in range(gt.shape[0]):
            mb = mask[b, 0] > 0.5
            gt_vals = gt[b, 0][mb]
            pred_vals = pred[b, 0][mb]
            if gt_vals.numel() == 0 or pred_vals.numel() == 0:
                out.append(torch.tensor(1.0, device=gt.device, dtype=gt.dtype))
            else:
                out.append(gt_vals.median() / pred_vals.median().clamp_min(1e-8))
        return torch.stack(out, dim=0).view(gt.shape[0], 1, 1, 1).clamp_min(1e-8)
    raise ValueError(f"Unsupported depth_norm='{mode}'. Use 'mean' or 'median'.")


def compute_depth_metrics(pred, target, mask=None):
    """
    Computes depth estimation metrics.
    This optimized version performs all calculations on the GPU and transfers
    results to the CPU only once at the end.
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    if target.dim() == 3:
        target = target.unsqueeze(1)
    if pred.dim() != 4 or target.dim() != 4:
        raise ValueError(f"Expected (B,1,H,W) or (B,H,W); got pred={pred.shape}, target={target.shape}")

    # Create a mask for valid pixels (finite, positive depth)
    valid_mask = (target > 0) & (pred > 0) & torch.isfinite(pred) & torch.isfinite(target)
    if mask is not None:
        valid_mask = valid_mask & mask.bool()

    valid_mask_f = valid_mask.float()
    denom = valid_mask_f.sum(dim=(1, 2, 3))
    valid_img = denom > 0
    if not valid_img.any():
        return {}
    denom = denom.clamp_min(1)

    if args.depth_eval_mode == "scale_invariant":
        scale = _compute_scale_align_pred(target, pred, valid_mask_f, args.depth_norm)
        pred_cmp = pred * scale
        target_cmp = target
    else:
        pred_cmp = pred
        target_cmp = target

    diff = pred_cmp - target_cmp
    pred_c = pred_cmp.clamp_min(1e-8)
    target_c = target_cmp.clamp_min(1e-8)
    log_diff = torch.log(pred_c) - torch.log(target_c)
    ratio = torch.maximum(pred_c / target_c, target_c / pred_c)

    def masked_mean_per_image(x):
        return (x * valid_mask_f).sum(dim=(1, 2, 3)) / denom

    abs_rel = masked_mean_per_image(torch.abs(diff) / target_c)
    sq_rel = masked_mean_per_image((diff ** 2) / target_c)
    rmse = torch.sqrt(masked_mean_per_image(diff ** 2))
    rmse_log = torch.sqrt(masked_mean_per_image(log_diff ** 2))
    a1 = masked_mean_per_image((ratio < 1.25).float())
    a2 = masked_mean_per_image((ratio < 1.25 ** 2).float())
    a3 = masked_mean_per_image((ratio < 1.25 ** 3).float())

    metrics = {
        'abs_rel': abs_rel[valid_img].mean(),
        'sq_rel': sq_rel[valid_img].mean(),
        'rmse': rmse[valid_img].mean(),
        'rmse_log': rmse_log[valid_img].mean(),
        'a1': a1[valid_img].mean(),
        'a2': a2[valid_img].mean(),
        'a3': a3[valid_img].mean(),
    }

    return {k: v.item() for k, v in metrics.items()}

training_parameters = list(model.parameters()) + list(decoder.parameters())
param_groups = []
lr_aux = getattr(args, "lr_aux", args.lr)
if args.use_rc_loss:
    if len(args.img_sizes)==1:
        grid_h, grid_w = model.patch_embed.grid_size
        dynamic = False
        from core.patch_pos import PatchRowColRegressionCriterion
        rowcol_loss = PatchRowColRegressionCriterion(
            feat_dim=model.embed_dim,
            grid_h=grid_h,
            grid_w=grid_w,
            # loss_type=args.loss_type,
            # huber_beta=args.huber_beta,
        ).to(DEVICE)
    else:
        grid_h = grid_w = max(args.img_sizes)//args.patch_size
        from core.patch_pos import PatchRowColRegressionCriterionDynamic
        rowcol_loss = PatchRowColRegressionCriterionDynamic(
            feat_dim=model.embed_dim,
            grid_h=grid_h,
            grid_w=grid_w,
            # loss_type=args.loss_type,
            # huber_beta=args.huber_beta,
        ).to(DEVICE)
    training_parameters += list(rowcol_loss.parameters())
    param_groups.append({"params": rowcol_loss.parameters(), "weight_decay": 0.0, "lr": lr_aux})
# if args.use_patch_position_loss:
#     from core.patch_pos import PatchPositionCriterion
#     position_loss = PatchPositionCriterion(
#         feat_dim=model.embed_dim,
#         num_classes=model.patch_embed.num_patches
#     ).to(DEVICE)
#     training_parameters += list(position_loss.parameters())
#     param_groups.append({"params": position_loss.parameters(), "weight_decay": 0.0, "lr": lr_aux})

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

from depth.depth_loss import MonocularDepthLoss
depth_norm_mode = getattr(args, "depth_norm", "mean")
depth_eval_mode = getattr(args, "depth_eval_mode", "scale_invariant")
if depth_eval_mode not in ("metric", "scale_invariant"):
    raise ValueError(f"Unsupported depth_eval_mode='{depth_eval_mode}'. Use 'metric' or 'scale_invariant'.")
if depth_eval_mode == "metric":
    ssim_norm_mode = "per_image"
else:
    ssim_norm_mode = "per_image"
if depth_eval_mode == "scale_invariant":
    if depth_norm_mode == "mean":
        scale_mode = "gt_mean"
    elif depth_norm_mode == "median":
        scale_mode = "gt_median"
    else:
        raise ValueError(f"Unsupported depth_norm='{depth_norm_mode}'. Use 'mean' or 'median'.")
else:
    scale_mode = "none"

if depth_eval_mode == "metric":
    silog_w = 0.0
    l1_w = 1.0
    grad_w = 0.5
    ssim_w = 0.2
    lambda_var = 0.0
    grad_use_log = True
else:
    silog_w = 1.0
    l1_w = 0.0
    grad_w = 0.5
    ssim_w = 0.2
    lambda_var = 1.0
    grad_use_log = True

criterion = MonocularDepthLoss(
    silog_w=silog_w,
    l1_w=l1_w,
    grad_w=grad_w,
    ssim_w=ssim_w,
    lambda_var=lambda_var,
    scale_mode=scale_mode,   # internal per-image normalization
    ssim_log_range=4.0,     # SSIM compares within ~[1/4, 4] multiplicative band
    grad_use_log=grad_use_log,
    ssim_norm_mode=ssim_norm_mode,
    ssim_percentiles=getattr(args, "ssim_percentiles", (5.0, 95.0)),
)


# criterion = MonocularDepthLossSimple()
optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
# optimizer = optim.AdamW(list(model.parameters()) + list(decoder.parameters()), lr=args.lr)
total_steps = EPOCHS * steps_per_epoch
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
logger.info("✅ Loss, Optimizer, and Scheduler are ready.")

# %%
# =================================================================================
# Step 5: Training and Validation Loop
# =================================================================================

def _infer_grid_hw(model, inputs):
    patch_size = model.patch_embed.patch_size
    if isinstance(patch_size, tuple):
        ph, pw = patch_size
    else:
        ph = pw = patch_size
    return (inputs.shape[-2] // ph, inputs.shape[-1] // pw)

def predict_depth(model, decoder, inputs, feature_layers, grid_hw=None):
    if grid_hw is None:
        grid_hw = _infer_grid_hw(model, inputs)
    if args.depth_decoder == "lite4":
        features = model.forward_intermediates(
            inputs,
            indices=feature_layers,
            norm=False,
            intermediates_only=True,
            output_fmt="NLC",
        )
        pred_depths = decoder(features, grid_hw=grid_hw, out_hw=inputs.shape[-2:])
    else:
        features = model.forward_features(inputs)
        pred_depths = decoder(features, grid_hw=grid_hw, out_hw=inputs.shape[-2:])
    return pred_depths, features

def train_one_epoch(model, decoder, loader, criterion, optimizer, scheduler, scaler, feature_layers, epoch, total_epochs):
    """Trains the model for one epoch."""
    model.train()
    decoder.train()
    running_loss_t = torch.zeros((), device=DEVICE)
    base_loss_t = torch.zeros((), device=DEVICE)
    aux_loss_sum_t = torch.zeros((), device=DEVICE)
    total_samples = 0
    log_interval = getattr(args, "log_interval", 50)
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs} [Train]")
    
    for i, (inputs, gt_depths) in enumerate(pbar):
        inputs = inputs.to(DEVICE, non_blocking=True)
        gt_depths = gt_depths.to(DEVICE, non_blocking=True)
        bs = inputs.size(0)
        optimizer.zero_grad(set_to_none=True)
        aux_loss = None
        with torch.amp.autocast(device_type=DEVICE.type, dtype=autocast_dtype, enabled=use_amp):
            pred_depths, features = predict_depth(model, decoder, inputs, feature_layers)
            valid = (gt_depths > 0) & torch.isfinite(gt_depths) & torch.isfinite(pred_depths)
            base_loss, loss_dict = criterion(pred_depths, gt_depths, valid_mask=valid)
            loss = base_loss
        
        if Use_Row_Col_Loss:
            last_feat = features[-1] if isinstance(features, (list, tuple)) else features
            if args.depth_decoder == "lite4":
                aux_loss = rowcol_loss(last_feat)
            else:
                aux_loss = rowcol_loss(last_feat[:, model.num_prefix_tokens:, :])
            # alpha_t = args.rc_alpha
            # if epoch == 0:
            #     alpha_t = args.rc_alpha * min(1.0, (i + 1) / args.warmup_steps_for_aux)
            loss = base_loss + args.rc_alpha * aux_loss
            base_loss_t += base_loss.detach() * bs
            aux_loss_sum_t += aux_loss.detach() * bs
        scaler.scale(loss).backward()
        if args.clip_value is not None:
            scaler.unscale_(optimizer)
            # log_grads(logger, model, rowcol_loss=rowcol_loss if args.use_rc_loss else None,
    #   every=331, step=step)
            
            torch.nn.utils.clip_grad_norm_(training_parameters, max_norm=args.clip_value)
        
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        running_loss_t += loss.detach() * bs
        total_samples += bs            

        if (i + 1) % log_interval == 0:
            avg_loss = (running_loss_t / max(total_samples, 1)).float().item()
            if aux_loss is not None:
                avg_aux = (aux_loss_sum_t / max(total_samples, 1)).float().item()
                pbar.set_postfix_str(f"loss={avg_loss:.4f} aux={avg_aux:.4f}")
            else:
                pbar.set_postfix_str(f"loss={avg_loss:.4f}")
    
    denom = max(total_samples, 1)
    avg_loss = (running_loss_t / denom).float().item()
    avg_aux = (aux_loss_sum_t / denom).float().item()
    avg_base = (base_loss_t / denom).float().item()
    return avg_loss, avg_aux, avg_base
# {k: v / len(loader) for k, v in train_metrics.items()}

def validate(model, decoder, loader, criterion, feature_layers, max_steps=None):
    """Validates the model."""
    model.eval()
    decoder.eval()
    val_loss = 0.0
    val_metrics = {'abs_rel': 0, 'sq_rel': 0, 'rmse': 0, 'rmse_log': 0, 'a1': 0, 'a2': 0, 'a3': 0}
    steps = 0

    with torch.inference_mode():
        for val_inputs, gt_depths in loader:
            val_inputs = val_inputs.to(DEVICE, non_blocking=True)
            gt_depths = gt_depths.to(DEVICE, non_blocking=True)
            with torch.amp.autocast(device_type=DEVICE.type, dtype=autocast_dtype, enabled=use_amp):
                val_pred_depths, _ = predict_depth(model, decoder, val_inputs, feature_layers)
                # v_loss, _ = criterion(val_pred_depths, gt_depths)
            # val_loss += v_loss.item()
            batch_metrics = compute_depth_metrics(val_pred_depths, gt_depths)
            for k in val_metrics:
                val_metrics[k] += batch_metrics.get(k, 0)
            steps += 1
            if max_steps and steps >= max_steps:
                break
    # val_loss / len(loader)
    denom = max(steps, 1)
    return 0.0, {k: v / denom for k, v in val_metrics.items()}

def save_checkpoint(model, decoder, output_dir, suffix):
    encoder_path = os.path.join(output_dir, f'encoder_{suffix}.pth')
    decoder_path = os.path.join(output_dir, f'decoder_{suffix}.pth')
    torch.save(model.state_dict(), encoder_path)
    torch.save(decoder.state_dict(), decoder_path)
    logger.info(f"Checkpoint saved: {suffix}")

scaler = torch.amp.GradScaler(DEVICE.type, enabled=use_amp)
logger.info(f"\n🚀 Starting training for {MODEL_NAME}...")
# 'valid_loss': [], 
training_history = {
    'train_loss': [], 'base_loss': [], 'aux_loss': [],
    'valid_abs_rel': [], 'valid_rmse': [], 'valid_a1': [],
    'epoch': []
}
best_val_abs_rel = float('inf')

logger.info("Starting training...")
for epoch in range(EPOCHS):
    avg_train_loss, avg_aux_loss, base_loss = train_one_epoch(
        model, decoder, train_loader, criterion, optimizer, scheduler, scaler, feature_layers, epoch, EPOCHS
    )
    # , avg_train_metrics
    avg_val_loss, avg_val_metrics = validate(
        model, decoder, valid_loader, criterion, feature_layers, max_steps=VAL_STEPS
    )

    logger.info(f"\n--- Epoch {epoch+1} Validation Summary ---")
    logger.info(f"  Train Loss: {avg_train_loss:.4f} | aux_loss: {avg_aux_loss:.4f} | base_loss: {base_loss:.4f}")
    logger.info(f" Valid AbsRel: {avg_val_metrics['abs_rel']:.4f} | Valid RMSE: {avg_val_metrics['rmse']:.4f} | Valid a1: {avg_val_metrics['a1']:.4f}\n")
    #   Valid Loss: {avg_val_loss:.4f} |
    training_history['train_loss'].append(avg_train_loss)
    training_history['base_loss'].append(base_loss)
    training_history['aux_loss'].append(avg_aux_loss)
    # training_history['train_abs_rel'].append(avg_train_metrics['abs_rel'])
    # training_history['train_rmse'].append(avg_train_metrics['rmse'])
    # training_history['train_a1'].append(avg_train_metrics['a1'])
    training_history['valid_abs_rel'].append(avg_val_metrics['abs_rel'])
    training_history['valid_rmse'].append(avg_val_metrics['rmse'])
    training_history['valid_a1'].append(avg_val_metrics['a1'])
    training_history['epoch'].append(epoch + 1)
    
    # if avg_val_metrics['abs_rel'] < best_val_abs_rel:
    #     best_val_abs_rel = avg_val_metrics['abs_rel']
    if args.csv_interval and (epoch + 1) % args.csv_interval == 0:
        history_df = pd.DataFrame(training_history)
        history_df.to_csv(os.path.join(output_dir, f'{subdir_name}.csv'), index=False)
        # save_checkpoint(model, decoder, ckpt_output_dir, "best")
    # if epoch == 3:
    #     break

logger.info("Training complete.")

history_df = pd.DataFrame(training_history)
history_df.to_csv(os.path.join(output_dir, f'{subdir_name}.csv'), index=False)
# save_checkpoint(model, decoder, ckpt_output_dir, "final")

if not history_df.empty:
    best_a1 = history_df['valid_a1'].max()
    best_epoch = history_df.loc[history_df['valid_a1'].idxmax(), 'epoch']
    logger.info(f"Best a1: {best_a1:.4f} at epoch {best_epoch}")

if not history_df.empty:
    # Find the epoch with the best validation loss
    # best_loss_row = history_df.loc[history_df['valid_loss'].idxmin()]
    # best_loss_epoch = int(best_loss_row['epoch'])
    # best_loss_val = best_loss_row['valid_loss']

    # Find the epoch with the best validation a1 score
    best_a1_row = history_df.loc[history_df['valid_a1'].idxmax()]
    best_a1_epoch = int(best_a1_row['epoch'])
    best_a1_val = best_a1_row['valid_a1']

    # Find the epoch with the best validation abs_rel
    best_abs_rel_row = history_df.loc[history_df['valid_abs_rel'].idxmin()]
    best_abs_rel_epoch = int(best_abs_rel_row['epoch'])
    best_abs_rel_val = best_abs_rel_row['valid_abs_rel']

    # Find the epoch with the best validation rmse
    best_rmse_row = history_df.loc[history_df['valid_rmse'].idxmin()]
    best_rmse_epoch = int(best_rmse_row['epoch'])
    best_rmse_val = best_rmse_row['valid_rmse']

    logger.info("\n--- Best Validation Metrics from History ---")
    # logger.info(f"  Best Loss:    {best_loss_val:.4f} (Epoch {best_loss_epoch})")
    logger.info(f"  Best a1:      {best_a1_val:.4f} (Epoch {best_a1_epoch})")
    logger.info(f"  Best AbsRel:  {best_abs_rel_val:.4f} (Epoch {best_abs_rel_epoch})")
    logger.info(f"  Best RMSE:    {best_rmse_val:.4f} (Epoch {best_rmse_epoch})")
    logger.info("------------------------------------------")

logger.info(output_dir)
logger.info(subdir_name)

del model, decoder
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    
if args.lock and gpu_lock and gpu_lock.is_locked:
    logger.info("Manually releasing lock.")
    gpu_lock.release()
