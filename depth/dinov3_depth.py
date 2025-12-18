import os
import torch
import torch.nn as nn
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
import torchvision.transforms.functional as TF
import logging
import cv2
import h5py
from scipy import ndimage
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
else:
    root_dir = '/linux'
    BASE_PATH = f'{root_dir}/Data/imagenet100/'
# Add a3R project to path
import sys
sys.path.insert(0, f'/lc/code/pos')
# sys.path.insert(0, '/lc/code/3D/a3R/src')

# from utils import wait_for_python_gpu_processes

from hypersim_simple_dataset import HyperSim_Simple
from transforms import SeqColorJitter
import torchvision.transforms as tvf

warnings.filterwarnings('ignore')

args = SimpleNamespace(
    data_root="/lc/data/3D",
    model_type= "dinov3",
    use_abs_pos_emb=False,
    use_rot_pos_emb=False,
    model_size='base',
    img_sizes=[224],
    batch_size=196,
    # batch_size=6,
    patch_size=16,
    lr=5e-4,
    epochs=100,
    has_pos=False,
    weight_decay=0.01,
    overlap=0,
    seed=55,
    val_steps=500,
    use_rc_loss=True,
    rc_alpha=300.0,
    warmup_steps_for_aux=100,
    workers=5,
    warmup_steps=20,
    clip_value=1.0,
    lock=False,
    output_dir=f'{root_dir}/output/depth',
)

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

use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
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
    f"{MODEL_NAME}{'_pos' if args.has_pos else ''}_overlap_{args.overlap}_rc_{args.use_rc_loss}"
)
output_dir = os.path.join(output_dir, subdir_name)
os.makedirs(output_dir, exist_ok=True)

log_file_path = os.path.join(output_dir, 'training.log')
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
logger.info(f"Using mixed precision: {'bfloat16' if use_bf16 else 'float16'}")
logger.info(args)
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
def collate_fn(batch):
    images = []
    depths = []
    for sample in batch:
        view = sample[0]
        img_tensor = view['img']
        depth = view['depthmap']
        depth_tensor = torch.from_numpy(np.ascontiguousarray(depth)).unsqueeze(0)
        images.append(img_tensor)
        depths.append(depth_tensor)
    return torch.stack(images), torch.stack(depths)

logger.info("Creating datasets...")
try:
    train_dataset = HyperSim_Simple(
        split='train',
        ROOT=f'{args.data_root}/hypersim_processed/train',
        resolution=IMG_SIZE,
        num_views=1,
        useImgnet=True,
    )
    valid_dataset = HyperSim_Simple(
        split='test',
        ROOT=f'{args.data_root}/hypersim_processed/test',
        resolution=IMG_SIZE,
        num_views=1,
        seed=777,
        useImgnet=True,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=args.workers,
        pin_memory=True, drop_last=True, persistent_workers=True, collate_fn=collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
        pin_memory=True, drop_last=True, persistent_workers=True, collate_fn=collate_fn
    )
    steps_per_epoch = len(train_loader)
    logger.info(f"✅ DataLoaders created successfully.")
    logger.info(f"   - Training samples: {len(train_dataset)}, Batches per epoch: {len(train_loader)}")
    logger.info(f"   - Validation samples: {len(valid_dataset)}, Batches per epoch: {len(valid_loader)}")
    
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
    
    # decoder = SimpleDepthDecoder(embed_dim=768, patch_size=14, img_size=img_size).to(device)

    decoder = DPTHead(
        dim_in=model.embed_dim,
        patch_size=model.patch_embed.patch_size[0],
        output_dim=1, # Monocular depth
        pos_embed=False
    ).to(device)
    
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


def compute_depth_metrics(pred, target, mask=None):
    """
    Computes depth estimation metrics.
    This optimized version performs all calculations on the GPU and transfers
    results to the CPU only once at the end.
    """
    if mask is not None:
        pred, target = pred[mask], target[mask]
    
    # Ensure tensors are flat
    pred = pred.flatten()
    target = target.flatten()

    # Create a mask for valid pixels (finite, positive depth)
    valid_mask = (target > 0) & (pred > 0) & torch.isfinite(pred) & torch.isfinite(target)
    if valid_mask.sum() == 0:
        return {}

    pred = pred[valid_mask]
    target = target[valid_mask]

    # --- All calculations below are on GPU ---
    diff = pred - target
    log_diff = torch.log(pred) - torch.log(target)
    ratio = torch.maximum(pred / target, target / pred)

    metrics = {
        'abs_rel': (torch.abs(diff) / target).mean(),
        'sq_rel': (((diff) ** 2) / target).mean(),
        'rmse': torch.sqrt((diff ** 2).mean()),
        'rmse_log': torch.sqrt((log_diff ** 2).mean()),
        'a1': (ratio < 1.25).float().mean(),
        'a2': (ratio < 1.25 ** 2).float().mean(),
        'a3': (ratio < 1.25 ** 3).float().mean(),
    }

    # Transfer all results to CPU at once
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
criterion = MonocularDepthLoss(
    silog_w=0.0,     # start without SILog; add later if needed
    l1_w=1.0,
    grad_w=0.5,
    ssim_w=0.2,
    lambda_var=0.0,
    scale_mode="gt_mean",   # internal per-image normalization
    ssim_log_range=4.0,     # SSIM compares within ~[1/4, 4] multiplicative band
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

def train_one_epoch(model, decoder, loader, criterion, optimizer, scheduler, scaler, feature_layers, epoch, total_epochs):
    """Trains the model for one epoch."""
    model.train()
    decoder.train()
    epoch_loss = 0.0
    train_metrics = {'abs_rel': 0, 'sq_rel': 0, 'rmse': 0, 'rmse_log': 0, 'a1': 0, 'a2': 0, 'a3': 0}
    base_loss = 0.0
    aux_loss_sum = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs} [Train]")
    
    for i, (inputs, gt_depths) in enumerate(pbar):
        inputs, gt_depths = inputs.to(DEVICE), gt_depths.to(DEVICE)
        optimizer.zero_grad()
        aux_loss = None
        with torch.amp.autocast('cuda', dtype=autocast_dtype):
            # features = model.forward_features(inputs)
            # pred_depths = decoder(features)
            # loss, _ = criterion(pred_depths, gt_depths)
            features = model.forward_intermediates(inputs, indices=feature_layers, norm=False, intermediates_only=True, output_fmt="NLC")
            pred_depths, _ = decoder(features, inputs.unsqueeze(1), patch_start_idx=0)
            pred_depths = pred_depths.squeeze(1)
            loss, loss_dict = criterion(pred_depths, gt_depths)
        
        if Use_Row_Col_Loss:
            base_loss += loss.item() 
            aux_loss = rowcol_loss(features[-1])
            alpha_t = args.rc_alpha
            if epoch == 0:
                alpha_t = args.rc_alpha * min(1.0, (i + 1) / args.warmup_steps_for_aux)
            loss = loss + alpha_t * aux_loss
            aux_loss_sum += aux_loss.item()
        scaler.scale(loss).backward()
        if args.clip_value is not None:
            scaler.unscale_(optimizer)
            # log_grads(logger, model, rowcol_loss=rowcol_loss if args.use_rc_loss else None,
    #   every=331, step=step)
            
            torch.nn.utils.clip_grad_norm_(training_parameters, max_norm=args.clip_value)
        
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        epoch_loss += loss.item()
        # batch_metrics = compute_depth_metrics(pred_depths.detach(), gt_depths.detach())
        # for k in train_metrics:
        #     train_metrics[k] += batch_metrics.get(k, 0)
        # , 'grad_norm': f'{total_norm:.2f}'
        pbar_dict = {'loss': f'{loss.item():.4f}'}
        if aux_loss is not None:
            pbar_dict['aux'] = aux_loss.item()
        # pbar_dict.update(train_metrics)
        # pbar_dict.update({k: f'{v / (i + 1):.4f}' for k, v in train_metrics.items()})
        pbar.set_postfix(pbar_dict)
    
    # if args.use_rc_loss:
    #     train_metrics["aux_loss"] = aux_loss_sum
    #     train_metrics["base_loss"] = base_loss
    
    return epoch_loss / len(loader), aux_loss_sum / len(loader), base_loss / len(loader)
# {k: v / len(loader) for k, v in train_metrics.items()}

def validate(model, decoder, loader, criterion, feature_layers):
    """Validates the model."""
    model.eval()
    decoder.eval()
    val_loss = 0.0
    val_metrics = {'abs_rel': 0, 'sq_rel': 0, 'rmse': 0, 'rmse_log': 0, 'a1': 0, 'a2': 0, 'a3': 0}

    with torch.no_grad():
        for val_inputs, gt_depths in loader:
            val_inputs, gt_depths = val_inputs.to(DEVICE), gt_depths.to(DEVICE)
            with torch.amp.autocast('cuda', dtype=autocast_dtype):
                # features = model.forward_features(inputs)
                # pred_depths = decoder(features)
                # v_loss, _ = criterion(pred_depths, gt_depths)

                features = model.forward_intermediates(val_inputs, indices=feature_layers, norm=False, intermediates_only=True, output_fmt="NLC")
                val_pred_depths, _ = decoder(features, val_inputs.unsqueeze(1), patch_start_idx=0)
                val_pred_depths = val_pred_depths.squeeze(1)
                # v_loss, _ = criterion(val_pred_depths, gt_depths)
            # val_loss += v_loss.item()
            batch_metrics = compute_depth_metrics(val_pred_depths, gt_depths)
            for k in val_metrics:
                val_metrics[k] += batch_metrics.get(k, 0)
    # val_loss / len(loader)
    return 0.0, {k: v / len(loader) for k, v in val_metrics.items()}

def save_checkpoint(model, decoder, output_dir, suffix):
    encoder_path = os.path.join(output_dir, f'encoder_{suffix}.pth')
    decoder_path = os.path.join(output_dir, f'decoder_{suffix}.pth')
    torch.save(model.state_dict(), encoder_path)
    torch.save(decoder.state_dict(), decoder_path)
    logger.info(f"Checkpoint saved: {suffix}")

scaler = torch.amp.GradScaler('cuda')
logger.info(f"\n🚀 Starting training for {MODEL_NAME}...")
# 'valid_loss': [], 
training_history = {
    'train_loss': [], 'base_loss': [], 'aux_loss': [], 'train_abs_rel': [], 'train_rmse': [], 'train_a1': [],
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
        model, decoder, valid_loader, criterion, feature_layers
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
    history_df = pd.DataFrame(training_history)
    history_df.to_csv(os.path.join(output_dir, f'{subdir_name}.csv'), index=False)
        # save_checkpoint(model, decoder, output_dir, "best")

logger.info("Training complete.")

history_df = pd.DataFrame(training_history)
history_df.to_csv(os.path.join(output_dir, f'{subdir_name}.csv'), index=False)
# save_checkpoint(model, decoder, output_dir, "final")

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

if args.lock and gpu_lock and gpu_lock.is_locked:
    logger.info("Manually releasing lock.")
    gpu_lock.release()