import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
import warnings
import argparse
import timm  # Imported for DINOv2
from torch.nn import functional as F
import torchvision.transforms.functional as TF

# Add a3R project to path
import sys
sys.path.insert(0, '/lc/code/3D/a3R')
sys.path.insert(0, '/lc/code/3D/a3R/src')

from hypersim_simple_dataset import HyperSim_Simple
from src.dust3r.datasets.utils.transforms import SeqColorJitter

warnings.filterwarnings('ignore')

def get_args():
    parser = argparse.ArgumentParser(description="Simplified monocular depth estimation with DINOv2.")
    parser.add_argument('--data_root', type=str, default="/lc/data/3D", help="Dataset root.")
    parser.add_argument('--resolution', type=int, default=224, help="Image resolution.")
    parser.add_argument('--batch_size', type=int, default=80, help="Batch size.")
    parser.add_argument('--model_name', type=str, default='vit_base_patch14_dinov2', help="Name of the model to use.")
    parser.add_argument('--learning_rate', type=float, default=1e-5, help="Learning rate.")  # Lowered for stability
    parser.add_argument('--epochs', type=int, default=120, help="Epochs.")
    parser.add_argument('--has_pos', action='store_true', help="Enable positional embedding.")
    parser.add_argument('--overlap', type=int, default=0, help="Overlap parameter.")
    parser.add_argument('--seed', type=int, default=55, help="Seed.")
    parser.add_argument('--val_steps', type=int, default=500, help="Validation frequency in steps.")
    parser.add_argument('--use_row_col_loss', action='store_true', help="Use row and column loss.")
    parser.add_argument('--rc_alpha', type=float, default=30.0, help="Alpha for row and column loss.")
    parser.add_argument('--output_dir', type=str, default="/lc/code/3D/pos/output", help="Output dir.")
    return parser.parse_args()

args = get_args()
print(args)

MODEL_NAME = args.model_name
NUM_CLASSES = 1
BATCH_SIZE = args.batch_size
IMG_SIZE = args.resolution
LEARNING_RATE = args.learning_rate
EPOCHS = args.epochs
HAS_POS = args.has_pos
OVERLAP = args.overlap
SEED = args.seed
VAL_STEPS = args.val_steps
Use_Row_Col_Loss = args.use_row_col_loss
RC_ALPHA = args.rc_alpha

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
print(f"Using mixed precision: {'bfloat16' if use_bf16 else 'float16'}")

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

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

print("Creating datasets...")
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
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
        pin_memory=True, drop_last=True, persistent_workers=True, collate_fn=collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
        pin_memory=True, drop_last=True, persistent_workers=True, collate_fn=collate_fn
    )
    steps_per_epoch = len(train_loader)
    print(f"Training samples: {len(train_dataset)}, Batches: {len(train_loader)}")
    print(f"Validation samples: {len(valid_dataset)}, Batches: {len(valid_loader)}")
except Exception as e:
    print(f"Error creating datasets: {e}")
    raise

class SimpleDepthDecoder(nn.Module):
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

def setup_model(img_size, device):
    print("Creating DINOv2 ViT-B/14 via timm...")
    model = timm.create_model('vit_base_patch14_dinov2', pretrained=False, img_size=img_size)
    model = model.to(device)
    
    for param in model.parameters():
        param.requires_grad = True
    
    decoder = SimpleDepthDecoder(embed_dim=768, patch_size=14, img_size=img_size).to(device)
    
    # Sanity check
    dummy_input = torch.randn(2, 3, img_size, img_size).to(device)
    with torch.no_grad():
        features = model.forward_features(dummy_input)  # (B, N+1, 768)
        dummy_output = decoder(features)
    print(f"Model setup successful. Input: {dummy_input.shape}, Features: {features.shape}, Output: {dummy_output.shape}")
    assert dummy_output.shape == (2, 1, img_size, img_size), "Output shape mismatch"
    
    encoder_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    decoder_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"Encoder params: {encoder_params/1e6:.2f}M, Decoder: {decoder_params/1e6:.2f}M")
    
    return model, decoder

model, decoder = setup_model(IMG_SIZE, DEVICE)

def ssim(pred, target, max_val=1.0):
    """Computes the structural similarity index measure (SSIM) between two images."""
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    mu_p = F.avg_pool2d(pred, 3, 1, 1)
    mu_t = F.avg_pool2d(target, 3, 1, 1)
    sigma_p = F.avg_pool2d(pred ** 2, 3, 1, 1) - mu_p ** 2
    sigma_t = F.avg_pool2d(target ** 2, 3, 1, 1) - mu_t ** 2
    sigma_pt = F.avg_pool2d(pred * target, 3, 1, 1) - mu_p * mu_t

    ssim_map = ((2 * mu_p * mu_t + C1) * (2 * sigma_pt + C2)) / ((mu_p ** 2 + mu_t ** 2 + C1) * (sigma_p + sigma_t + C2))
    return torch.clamp((1 - ssim_map) / 2, 0, 1)


class MonocularDepthLoss(nn.Module):
    def __init__(self, silog_w=10.0, l1_w=1.0, grad_w=1.0, ssim_w=1.0, l_inf_w=0.0, lambda_var=0.5, valid_mask=True):
        super(MonocularDepthLoss, self).__init__()
        self.silog_w = silog_w
        self.l1_w = l1_w
        self.grad_w = grad_w
        self.ssim_w = ssim_w
        self.l_inf_w = l_inf_w
        self.lambda_var = lambda_var
        self.valid_mask = valid_mask

    def forward(self, pred_depth, gt_depth, valid_mask=None):
        if pred_depth.dim() == 3:
            pred_depth = pred_depth.unsqueeze(1)
        if gt_depth.dim() == 3:
            gt_depth = gt_depth.unsqueeze(1)

        if valid_mask is None and self.valid_mask:
            valid_mask = (gt_depth > 1e-8).float()
        elif valid_mask is not None:
            if valid_mask.dim() == 3:
                valid_mask = valid_mask.unsqueeze(1)
            valid_mask = valid_mask.float()
        else:
            valid_mask = torch.ones_like(gt_depth)

        loss_dict = {}
        total_loss = 0

        # SILog loss
        if self.silog_w > 0:
            silog_loss = self._silog_loss(pred_depth, gt_depth, valid_mask)
            loss_dict['silog'] = silog_loss.item()
            total_loss += self.silog_w * silog_loss

        # L1 loss
        if self.l1_w > 0:
            l1_loss = self._l1_loss(pred_depth, gt_depth, valid_mask)
            loss_dict['l1'] = l1_loss.item()
            total_loss += self.l1_w * l1_loss

        # Gradient loss
        if self.grad_w > 0:
            grad_loss = self._gradient_loss(pred_depth, gt_depth, valid_mask)
            loss_dict['grad'] = grad_loss.item()
            total_loss += self.grad_w * grad_loss

        # SSIM loss
        if self.ssim_w > 0:
            ssim_loss_val = ssim(pred_depth * valid_mask, gt_depth * valid_mask, max_val=gt_depth.max()).mean()
            loss_dict['ssim'] = ssim_loss_val.item()
            total_loss += self.ssim_w * ssim_loss_val

        # L-infinity loss
        if self.l_inf_w > 0:
            l_inf_loss_val = torch.max(torch.abs(pred_depth - gt_depth) * valid_mask)
            loss_dict['l_inf'] = l_inf_loss_val.item()
            total_loss += self.l_inf_w * l_inf_loss_val
        
        return total_loss, loss_dict

    def _silog_loss(self, pred_depth, gt_depth, valid_mask):
        valid_pixels = valid_mask.sum(dim=[1, 2, 3], keepdim=True) + 1e-8
        log_pred = torch.log(torch.clamp(pred_depth, min=1e-8))
        log_gt = torch.log(torch.clamp(gt_depth, min=1e-8))
        log_diff = (log_pred - log_gt) * valid_mask
        silog_term = torch.sum(log_diff**2, dim=[1, 2, 3], keepdim=True) / valid_pixels
        log_diff_mean = torch.sum(log_diff, dim=[1, 2, 3], keepdim=True) / valid_pixels
        variance_term = (log_diff_mean ** 2) * self.lambda_var
        return (silog_term - variance_term).mean()

    def _l1_loss(self, pred_depth, gt_depth, valid_mask):
        valid_pixels = valid_mask.sum(dim=[1, 2, 3], keepdim=True) + 1e-8
        l1_diff = torch.abs(pred_depth - gt_depth) * valid_mask
        return (torch.sum(l1_diff, dim=[1, 2, 3], keepdim=True) / valid_pixels).mean()

    def _gradient_loss(self, pred_depth, gt_depth, valid_mask):
        def compute_gradients(depth):
            grad_x = depth[:, :, :, 1:] - depth[:, :, :, :-1]
            grad_y = depth[:, :, 1:, :] - depth[:, :, :-1, :]
            return grad_x, grad_y

        pred_grad_x, pred_grad_y = compute_gradients(pred_depth)
        gt_grad_x, gt_grad_y = compute_gradients(gt_depth)

        valid_mask_x = valid_mask[:, :, :, 1:] * valid_mask[:, :, :, :-1]
        valid_mask_y = valid_mask[:, :, 1:, :] * valid_mask[:, :, :-1, :]

        grad_diff_x = torch.abs(pred_grad_x - gt_grad_x) * valid_mask_x
        grad_diff_y = torch.abs(pred_grad_y - gt_grad_y) * valid_mask_y

        grad_loss = grad_diff_x.mean() + grad_diff_y.mean()
        return grad_loss
    
class MonocularDepthLossSimple(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_depth, gt_depth):
        if pred_depth.dim() == 3:
            pred_depth = pred_depth.unsqueeze(1)
        if gt_depth.dim() == 3:
            gt_depth = gt_depth.unsqueeze(1)
        
        valid_mask = (gt_depth > 1e-8).float()
        valid_pixels = valid_mask.sum(dim=[1,2,3], keepdim=True) + 1e-8
        
        # L1 Loss
        l1_loss = (torch.abs(pred_depth - gt_depth) * valid_mask).sum(dim=[1,2,3]) / valid_pixels
        return l1_loss.mean()
def compute_depth_metrics(pred, target, mask=None):
    if mask is not None:
        pred, target = pred[mask], target[mask]
    pred, target = pred.flatten(), target.flatten()
    valid = (target > 0) & (pred > 0) & torch.isfinite(pred) & torch.isfinite(target)
    if valid.sum() == 0: return {}
    pred, target = pred[valid], target[valid]
    abs_rel = torch.mean(torch.abs(pred - target) / target)
    sq_rel = torch.mean(((pred - target) ** 2) / target)
    rmse = torch.sqrt(torch.mean((pred - target) ** 2))
    rmse_log = torch.sqrt(torch.mean((torch.log(pred) - torch.log(target)) ** 2))
    ratio = torch.maximum(pred / target, target / pred)
    a1 = (ratio < 1.25).float().mean()
    a2 = (ratio < 1.25 ** 2).float().mean()
    a3 = (ratio < 1.25 ** 3).float().mean()
    return {'abs_rel': abs_rel.item(), 'sq_rel': sq_rel.item(), 'rmse': rmse.item(), 'rmse_log': rmse_log.item(), 'a1': a1.item(), 'a2': a2.item(), 'a3': a3.item()}

criterion = MonocularDepthLoss(silog_w=10.0, l1_w=1.0, grad_w=1.0, ssim_w=3.0)
# criterion = MonocularDepthLossSimple()
optimizer = optim.AdamW(list(model.parameters()) + list(decoder.parameters()), lr=LEARNING_RATE)
total_steps = EPOCHS * steps_per_epoch
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
print("✅ Loss, Optimizer, and Scheduler are ready.")

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

# %%

if Use_Row_Col_Loss:
    grid_h, grid_w = model.patch_embed.grid_size
    rowcol_loss = PatchRowColCriterion(
        feat_dim=model.embed_dim,
        grid_h=grid_h,
        grid_w=grid_w
    ).to(DEVICE)
    print("✅ Row-Column loss initialized.")

# %%
# =================================================================================
# Step 5: Training and Validation Loop
# =================================================================================

def train_one_epoch(model, decoder, loader, criterion, optimizer, scheduler, scaler, epoch, total_epochs):
    """Trains the model for one epoch."""
    model.train()
    decoder.train()
    epoch_loss = 0.0
    train_metrics = {'abs_rel': 0, 'sq_rel': 0, 'rmse': 0, 'rmse_log': 0, 'a1': 0, 'a2': 0, 'a3': 0}
    
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs} [Train]")
    
    for i, (inputs, gt_depths) in enumerate(pbar):
        inputs, gt_depths = inputs.to(DEVICE), gt_depths.to(DEVICE)
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda', dtype=autocast_dtype):
            features = model.forward_features(inputs)
            pred_depths = decoder(features)
            loss, _ = criterion(pred_depths, gt_depths)
        
        if Use_Row_Col_Loss:
            aux_loss = rowcol_loss(features[:, 1:, :])
            loss = loss + RC_ALPHA * aux_loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
        
        total_norm = sum(p.grad.norm(2).item() ** 2 for p in list(model.parameters()) + list(decoder.parameters()) if p.grad is not None) ** 0.5
        
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        epoch_loss += loss.item()
        batch_metrics = compute_depth_metrics(pred_depths.detach(), gt_depths.detach())
        for k in train_metrics:
            train_metrics[k] += batch_metrics.get(k, 0)
        
        pbar_dict = {'loss': f'{loss.item():.4f}', 'grad_norm': f'{total_norm:.2f}'}
        pbar_dict.update(train_metrics)
        pbar_dict.update({k: f'{v / (i + 1):.4f}' for k, v in train_metrics.items()})
        pbar.set_postfix(pbar_dict)
    
    return epoch_loss / len(loader), {k: v / len(loader) for k, v in train_metrics.items()}

def validate(model, decoder, loader, criterion):
    model.eval()
    decoder.eval()
    val_loss = 0.0
    val_metrics = {'abs_rel': 0, 'sq_rel': 0, 'rmse': 0, 'rmse_log': 0, 'a1': 0, 'a2': 0, 'a3': 0}

    with torch.no_grad():
        for inputs, gt_depths in loader:
            inputs, gt_depths = inputs.to(DEVICE), gt_depths.to(DEVICE)
            with torch.amp.autocast('cuda', dtype=autocast_dtype):
                features = model.forward_features(inputs)
                pred_depths = decoder(features)
                v_loss, _ = criterion(pred_depths, gt_depths)
            val_loss += v_loss.item()
            batch_metrics = compute_depth_metrics(pred_depths, gt_depths)
            for k in val_metrics:
                val_metrics[k] += batch_metrics.get(k, 0)
    
    return val_loss / len(loader), {k: v / len(loader) for k, v in val_metrics.items()}

def save_checkpoint(model, decoder, output_dir, suffix):
    encoder_path = os.path.join(output_dir, f'encoder_{suffix}.pth')
    decoder_path = os.path.join(output_dir, f'decoder_{suffix}.pth')
    torch.save(model.state_dict(), encoder_path)
    torch.save(decoder.state_dict(), decoder_path)
    print(f"Checkpoint saved: {suffix}")

# Create output directory before training starts
output_dir = args.output_dir
subdir_name = (
    f"has_pos_{HAS_POS}_overlap_{OVERLAP}_"
    f"use_rc_loss_{Use_Row_Col_Loss}_rc_alpha_{RC_ALPHA}"
)
output_dir = os.path.join(output_dir, subdir_name)
os.makedirs(output_dir, exist_ok=True)

scaler = torch.amp.GradScaler('cuda')
print(f"\n🚀 Starting training for {MODEL_NAME}...")

training_history = {
    'train_loss': [], 'train_abs_rel': [], 'train_rmse': [], 'train_a1': [],
    'valid_loss': [], 'valid_abs_rel': [], 'valid_rmse': [], 'valid_a1': [],
    'epoch': []
}
best_val_loss = float('inf')

print("Starting training...")
for epoch in range(EPOCHS):
    avg_train_loss, avg_train_metrics = train_one_epoch(
        model, decoder, train_loader, criterion, optimizer, scheduler, scaler, epoch, EPOCHS
    )
    avg_val_loss, avg_val_metrics = validate(model, decoder, valid_loader, criterion)
    
    print(f"\n--- Epoch {epoch+1} Validation Summary ---")
    print(f"  Train Loss: {avg_train_loss:.4f} | Train AbsRel: {avg_train_metrics['abs_rel']:.4f} | Train RMSE: {avg_train_metrics['rmse']:.4f} | Train a1: {avg_train_metrics['a1']:.4f}")
    print(f"  Valid Loss: {avg_val_loss:.4f} | Valid AbsRel: {avg_val_metrics['abs_rel']:.4f} | Valid RMSE: {avg_val_metrics['rmse']:.4f} | Valid a1: {avg_val_metrics['a1']:.4f}\n")
    
    training_history['train_loss'].append(avg_train_loss)
    training_history['train_abs_rel'].append(avg_train_metrics['abs_rel'])
    training_history['train_a1'].append(avg_train_metrics['a1'])
    training_history['valid_loss'].append(avg_val_loss)
    training_history['valid_abs_rel'].append(avg_val_metrics['abs_rel'])
    training_history['valid_a1'].append(avg_val_metrics['a1'])
    training_history['epoch'].append(epoch + 1)
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        save_checkpoint(model, decoder, output_dir, "best")

print("Training complete.")

history_df = pd.DataFrame(training_history)
history_df.to_csv(os.path.join(output_dir, 'training_history.csv'), index=False)
save_checkpoint(model, decoder, output_dir, "final")

if not history_df.empty:
    best_a1 = history_df['valid_a1'].max()
    best_epoch = history_df.loc[history_df['valid_a1'].idxmax(), 'epoch']
    print(f"Best a1: {best_a1:.4f} at epoch {best_epoch}")

if not history_df.empty:
    # Find the epoch with the best validation loss
    best_loss_row = history_df.loc[history_df['valid_loss'].idxmin()]
    best_loss_epoch = int(best_loss_row['epoch'])
    best_loss_val = best_loss_row['valid_loss']

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

    print("\n--- Best Validation Metrics from History ---")
    print(f"  Best Loss:    {best_loss_val:.4f} (Epoch {best_loss_epoch})")
    print(f"  Best a1:      {best_a1_val:.4f} (Epoch {best_a1_epoch})")
    print(f"  Best AbsRel:  {best_abs_rel_val:.4f} (Epoch {best_abs_rel_epoch})")
    print(f"  Best RMSE:    {best_rmse_val:.4f} (Epoch {best_rmse_epoch})")
    print("------------------------------------------")
