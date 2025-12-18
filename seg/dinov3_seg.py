# %%
# =================================================================================
# Step 1: Install and Import Necessary Libraries
# =================================================================================
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset,TensorDataset, DataLoader

from tqdm import tqdm
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
import timm
from types import SimpleNamespace
import gc
import argparse
import logging
try:
    from filelock import FileLock
except ImportError:
    FileLock = None
# %%
# =================================================================================
# Step 2: Configuration
# =================================================================================
sys.path.append("/lc/code/pos")
# --- Dynamically set root directory ---
if os.path.exists('/home/sshuser'):
    root_dir = '/home/sshuser'
    BASE_PATH = f'{root_dir}/Data/ADEChallengeData2016'
elif os.path.exists('/lc'):
    root_dir = '/lc/logs'
    BASE_PATH = f'/ubuntu/Data/ADEChallengeData2016'
else:
    root_dir = '/linux'
    BASE_PATH = f'{root_dir}/Data/ADEChallengeData2016'
# --- Configuration via SimpleNamespace for easy interactive use ---
args = SimpleNamespace(
    # --- Model & Training Settings ---
    model_type= "dinov3",
    use_abs_pos_emb=False,
    use_rot_pos_emb=False,
    model_size='base',
    num_classes=150,  # For ADE20K
    batch_size=168,
    # batch_size=24,
    img_size=256,
    # img_size=384,
    # lr=3e-4,
    # lr=8e-4,
    lr=5e-4,
    lr_aux=1e-5,
    # lr=1e-3,
    # lr=8e-4,
    eta_min=0.0,
    weight_decay=0.01, 
    epochs=130,
    # has_pos=True,
    overlap=1,
    start_epoch=0,
    seed=55,
    use_rc_loss=True,
    # loss_type="l1",
    # huber_beta=0.1,
    rc_alpha=600.0,
    dice_weight=0.0,
    workers=5,
    train=True,
    val=False,
    ckpt_path=None,
    lock=True,
    clip_value=1.0,
    output_dir=f"{root_dir}/output/seg_redo",
    # --- Dataset Paths ---
    base_path=f"{BASE_PATH}",
)
if args.use_abs_pos_emb or args.use_rot_pos_emb:
    args.overlap = 0
    args.use_rc_loss = False

MODEL_NAME = f"vit_{args.model_size}_patch16_{args.model_type}"
TRAIN_IMAGE_PATH = os.path.join(args.base_path, 'images', 'training')
TRAIN_ANNOTATION_PATH = os.path.join(args.base_path, 'annotations', 'training')
VALID_IMAGE_PATH = os.path.join(args.base_path, 'images', 'validation')
VALID_ANNOTATION_PATH = os.path.join(args.base_path, 'annotations', 'validation')

# --- Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
# %%
# Set seeds
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
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

logger.info(f"Using device: {DEVICE}")
logger.info(f"Using mixed precision: {'bfloat16' if use_bf16 else 'float16'}")
logger.info(f"Arguments: {args}")
logger.info(output_dir)
logger.info(subdir_name)
# wait_for_python_gpu_processes(poll_interval_minutes=5)
# --- Acquire a file lock to ensure exclusive GPU usage ---
gpu_lock = None
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
# %%
# =================================================================================
# Custom Dataset for Segmentation (Image + Mask)
# =================================================================================
import torchvision.transforms.functional as TF
class SegmentationDataset(Dataset):
    """
    Custom PyTorch Dataset for semantic segmentation.

    Reads images and their corresponding segmentation masks, and applies
    appropriate data augmentation for training and validation.
    """
    def __init__(self, image_dir, annotation_dir, img_size, is_train, mean, std):
        """
        Args:
            image_dir (str): Directory with all the images.
            annotation_dir (str): Directory with all the segmentation masks.
            img_size (int): The target size for the images and masks.
            is_train (bool): If true, applies training augmentations.
            mean (list): Mean for normalization.
            std (list): Standard deviation for normalization.
        """
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        self.img_size = img_size
        self.is_train = is_train
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        # Assumes annotation has the same name but with a .png extension
        ann_path = os.path.join(self.annotation_dir, img_name.replace('.jpg', '.png'))

        image = Image.open(img_path).convert('RGB')
        # Masks contain class labels, so they are typically opened in grayscale ('L') mode
        mask = Image.open(ann_path).convert('L')

        # --- Apply Synchronized Transformations ---

        # 1. Spatial transformations (applied to both image and mask)
        if self.is_train:
            # Random horizontal flip (50% chance)
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # Random resized crop
            # Get parameters for crop and apply it to both image and mask
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                image, scale=(0.08, 1.0), ratio=(0.75, 1.33)
            )
            image = TF.resized_crop(image, i, j, h, w, [self.img_size, self.img_size], TF.InterpolationMode.BILINEAR)
            # Use NEAREST interpolation for masks to avoid creating new class labels
            mask = TF.resized_crop(mask, i, j, h, w, [self.img_size, self.img_size], TF.InterpolationMode.NEAREST)

        else:  # Validation/Testing
            # Resize to a size larger than the crop size
            image = TF.resize(image, [int(self.img_size * 1.14), int(self.img_size * 1.14)], TF.InterpolationMode.BILINEAR)
            mask = TF.resize(mask, [int(self.img_size * 1.14), int(self.img_size * 1.14)], TF.InterpolationMode.NEAREST)
            # Center crop to the final size
            image = TF.center_crop(image, [self.img_size, self.img_size])
            mask = TF.center_crop(mask, [self.img_size, self.img_size])

        # 2. Pixel-level transformations
        # Image: convert to tensor and normalize
        image = TF.to_tensor(image)
        image = TF.normalize(image, self.mean, self.std)

        # Mask: convert to numpy array, then to a long tensor
        # The values should be class indices (0, 1, 2, ...), not floats.
        mask = torch.from_numpy(np.array(mask)).long() - 1 # ADE20K labels are 1-150, background 0. Map to 0-149, ignore -1.
        # mask = torch.clamp(mask, min=0, max=NUM_CLASSES-1)

        return image, mask
#%%
# Data Augmentation
img_mean = [0.485, 0.456, 0.406]
img_std = [0.229, 0.224, 0.225]

# --- Create Datasets and DataLoaders ---
train_dataset = SegmentationDataset(
    TRAIN_IMAGE_PATH,
    TRAIN_ANNOTATION_PATH,
    img_size=args.img_size,
    is_train=True,
    mean=img_mean,
    std=img_std
)
valid_dataset = SegmentationDataset(
    VALID_IMAGE_PATH,
    VALID_ANNOTATION_PATH,
    img_size=args.img_size,
    is_train=False,
    mean=img_mean,
    std=img_std
)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=False)

steps_per_epoch = len(train_loader)
logger.info(f"✅ DataLoaders created successfully.")
logger.info(f"   - Training samples: {len(train_dataset)}, Batches per epoch: {len(train_loader)}")
logger.info(f"   - Validation samples: {len(valid_dataset)}, Batches per epoch: {len(valid_loader)}")
#%%
# =================================================================================
# Step 3.5: Visualize a Batch of Training Data
# =================================================================================
def imshow(img, mask, title=None):
    """Display an image and its mask side by side."""
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array(img_mean)
    std = np.array(img_std)
    img = std * img + mean
    img = np.clip(img, 0, 1)
    
    mask = mask.numpy().squeeze()  # Remove channel dimension
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(img)
    ax1.set_title('Image' if title is None else f'Image - {title}')
    ax1.axis('off')
    ax2.imshow(mask, cmap='jet')  # Use jet colormap for mask visualization
    ax2.set_title('Mask' if title is None else f'Mask - {title}')
    ax2.axis('off')
    plt.show()

# Get one batch of training images and masks
# try:
#     inputs, masks = next(iter(train_loader))
#     inputs = inputs[:8]  # Limit to 8 for display
#     masks = masks[:8]
    
#     # Denormalize and display
#     fig = plt.figure(figsize=(16, 8))
#     plt.suptitle("Sample Images and Masks from ADE20K Dataset", fontsize=16)
    
#     for i in range(min(8, len(inputs))):
#         imshow(inputs[i].cpu(), masks[i].cpu(), title=str(i))
        
#     plt.tight_layout(rect=[0, 0, 1, 0.96])
#     plt.show()

# except Exception as e:
#     logger.info(f"Could not display images. Error: {str(e)}. Ensure previous cells have been run to create 'train_loader'.")
#%%

#%%
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
    num_classes=0, 
    img_size=args.img_size,
).to(DEVICE)


from seg_head import PPMliteFCNHead, FCNSegHead
grid_h, grid_w = model.patch_embed.grid_size
# decoder = FCNSegHead(
#     embed_dim=model.embed_dim,
#     num_classes=args.num_classes,
#     grid_size=(grid_h, grid_w),
#     out_size=(args.img_size, args.img_size),
#     mid_channels=256,
#     dropout=0.1,
#     norm='gn',   # or 'gn' if your effective batch per GPU is small
# ).to(DEVICE)

decoder = PPMliteFCNHead(
    embed_dim=model.embed_dim,
    num_classes=args.num_classes,
    grid_size=(grid_h, grid_w),
    out_size=(args.img_size, args.img_size),
    mid_channels=256,
    ppm_bins=(1, 2, 3),     # lite; try (1,2,3,6) if you can afford a bit more
    ppm_channels=64,
    dropout=0.1,
    norm="gn",              # "bn" if your per-GPU batch is large/stable
).to(DEVICE)

# --- Test with a dummy input ---
# dummy_input = torch.randn(2, 3, args.img_size, args.img_size).to(DEVICE)
# with torch.no_grad():
#     feats = model.forward_features(dummy_input)
#     output = decoder(feats[:, model.num_prefix_tokens:, :])

# logger.info(f"Model created successfully!")
# logger.info(f"Input shape: {dummy_input.shape}")
# logger.info(f"Output shape: {output.shape}")
# assert output.shape == (2, args.num_classes, args.img_size, args.img_size)
# logger.info("✅ Output shape is correct.")
# del feats, output, dummy_input

# %%
logger.info(f'model.patch_embed.proj {model.patch_embed.proj}')
if args.overlap > 0:
    # Customize patch embedding for overlap (e.g., patch_size=15, stride=14)
    original_patch_size = model.patch_embed.proj.kernel_size[0]
    new_patch_size = original_patch_size + args.overlap  # Or 15, 16, 17, etc., as desired
    stride = original_patch_size
    original_grid_size = args.img_size // stride  # 16 for 224//14
    padding = ((original_grid_size - 1) * stride + new_patch_size - args.img_size + 1) // 2  # +1 for ceiling effect; yields 1 for patch_size=15
    
    # Override the PatchEmbed projection (Conv2d layer)
    in_chans = model.patch_embed.proj.in_channels  # Typically 3 for RGB
    embed_dim = model.patch_embed.proj.out_channels  # e.g., 768 for base
    model.patch_embed.proj = nn.Conv2d(
        in_chans, embed_dim,
        kernel_size=(new_patch_size, new_patch_size),
        stride=(stride, stride),
        padding=padding  # Updated to ensure full coverage and original grid size
    ).to(DEVICE)
    
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
dynamic = True
training_parameters = list(model.parameters()) + list(decoder.parameters())
param_groups = []
lr_aux = getattr(args, "lr_aux", args.lr)
if args.use_rc_loss:
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

# Loss and Optimizer
from seg_loss import MMSegCrossEntropyLoss
# ce_criterion = nn.CrossEntropyLoss(ignore_index=-1)  # Standard Cross-Entropy
ce_criterion = MMSegCrossEntropyLoss(ignore_index=-1, avg_non_ignore=True)
# dice_criterion = GatherDiceLoss(ignore_index=-1)  # Our new Dice Loss

optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
# optimizer = optim.AdamW(list(model.parameters()) + list(decoder.parameters()), lr=args.lr)
total_steps = args.epochs * steps_per_epoch
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.eta_min)
logger.info("✅ Initialized Loss, Optimizer, and LR Scheduler.")

# %%
# dummy_input = torch.randn(2, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
# with torch.no_grad():
#     feats = model.forward_features(dummy_input)
# logger.info(f"Model created successfully!")
# logger.info(f"Input shape: {dummy_input.shape}")
# logger.info(f"Output shape: {feats.shape}") 

# %%



#%%
def compute_miou(preds, labels, num_classes, ignore_index=-1):
    """Compute mean IoU with proper handling of ignore index"""
    mask = (labels != ignore_index)
    preds = preds[mask]
    labels = labels[mask]
    
    if len(preds) == 0:
        return 0.0
    
    iou_list = []
    for c in range(num_classes):
        pred_c = (preds == c)
        label_c = (labels == c)
        intersection = (pred_c & label_c).sum().float()
        union = (pred_c | label_c).sum().float()
        
        if union > 0:
            iou_list.append((intersection / union).item())
        else:
            # If no pixels of this class, don't count it
            continue
    
    return np.mean(iou_list) if iou_list else 0.0
#%%
def save_checkpoint(model, decoder, output_dir, suffix):
    encoder_path = os.path.join(output_dir, f'encoder_{suffix}.pth')
    decoder_path = os.path.join(output_dir, f'decoder_{suffix}.pth')
    torch.save(model.state_dict(), encoder_path)
    torch.save(decoder.state_dict(), decoder_path)
    logger.info(f"Checkpoint saved: {suffix}")
# %%
import csv

ckpt_path = None
if args.train:
    # FP16: Initialize the Gradient Scaler
    scaler = torch.amp.GradScaler('cuda', enabled=(autocast_dtype == torch.float16))

    # =================================================================================
    # Step 5: Training and Validation Loop
    # =================================================================================
    logger.info(f"\n🚀 Starting training for {MODEL_NAME}...")

    # ✅ Initialize training_history as a dictionary of lists
    if args.use_rc_loss:
        training_history = {
            'train_loss': [],
            'train_acc': [],
            'valid_acc': [],
            'valid_miou': [],
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
            'valid_miou': [],
            'epoch': [],
            'step': [],
        }
    step = 0
    best_acc = 0.0

    for epoch in range(args.epochs):
        # --- Training Phase ---
        model.train()
        decoder.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        base_loss = 0.0
        aux_loss_sum = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Training]")
        # train_pbar = train_loader
        
        # FP16: Use autocast for the forward pass
        for batch_idx, (inputs, labels) in enumerate(train_pbar):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad(set_to_none=True)
            aux_loss = None
            with torch.amp.autocast('cuda', dtype=autocast_dtype):
                feats = model.forward_features(inputs)
                outputs = decoder(feats[:, model.num_prefix_tokens:, :])            

                loss = ce_criterion(outputs, labels)               

                if args.use_rc_loss:
                    base_loss += loss.item() * inputs.size(0)
                    aux_loss = rowcol_loss(feats[:, model.num_prefix_tokens:, :])
                    aux_loss_sum += aux_loss.item() * inputs.size(0)
                    # logger.info(loss, aux_loss)
                    loss = loss + args.rc_alpha * aux_loss
            
            # FP16: Scale, backward, and step
            scaler.scale(loss).backward()

            if args.clip_value is not None:
                scaler.unscale_(optimizer)
                clip_value = args.clip_value  # typical default for Transformer/ViT-style training
                torch.nn.utils.clip_grad_norm_(training_parameters, max_norm=clip_value)

            scaler.step(optimizer)
            scaler.update()

            scheduler.step()
            
            running_loss += loss.item() * inputs.size(0)
            predicted = torch.argmax(outputs, dim=1)
            mask = (labels >= 0)
            train_correct += ((predicted == labels) & mask).sum().item()
            train_total += mask.sum().item()

            batch_pixel_acc = train_correct / train_total if train_total > 0 else 0.0
            bar_msg = {
                    'loss': f'{loss.item():.4f}', 
                    'acc': f'{batch_pixel_acc:.3f}'
                }
            if aux_loss is not None:
                bar_msg['aux'] = f'{aux_loss.item():.4f}'
            train_pbar.set_postfix(bar_msg)

            step += 1

            # if (step) % VAL_STEPS == 0:
        
        model.eval()
        decoder.eval()
        val_correct = 0
        val_total = 0
        val_intersection = torch.zeros(args.num_classes).to(DEVICE)
        val_union = torch.zeros(args.num_classes).to(DEVICE)
        val_pbar = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Validation]")
        
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                with torch.amp.autocast('cuda', dtype=autocast_dtype):
                    feats = model.forward_features(inputs)
                    outputs = decoder(feats[:, model.num_prefix_tokens:, :])
                predicted = torch.argmax(outputs, dim=1)
                mask = (labels >= 0)
                val_correct += ((predicted == labels) & mask).sum().item()
                val_total += mask.sum().item()

                # Compute IoU for the batch
                for c in range(args.num_classes):
                    pred_c = (predicted == c) & mask
                    label_c = (labels == c) & mask
                    val_intersection[c] += (pred_c & label_c).sum()
                    val_union[c] += (pred_c | label_c).sum()

        # Compute validation mIoU
        iou = torch.zeros(args.num_classes).to(DEVICE)
        valid = val_union > 0
        iou[valid] = val_intersection[valid] / val_union[valid]
        epoch_val_miou = iou[valid].mean().item()
        
        epoch_val_acc = val_correct / val_total if val_total > 0 else 0.0
        epoch_train_acc = train_correct / train_total if train_total > 0 else 0.0
        epoch_train_loss = running_loss / len(train_dataset)
        if best_acc < epoch_val_acc:
            best_acc = epoch_val_acc

        if args.use_rc_loss:
            epoch_aux_loss = aux_loss_sum / len(train_loader.dataset)
            epoch_base_loss = base_loss / len(train_loader.dataset)
            training_history['aux_loss'].append(epoch_aux_loss)
            training_history['base_loss'].append(epoch_base_loss)

        logger.info(f"\nEpoch {epoch+1+args.start_epoch}/{args.epochs} Summary:")
        logger.info(f"Step {step} Summary:")
        if args.use_rc_loss:
            logger.info(f"  Train Loss: {epoch_train_loss:.4f} | Aux Loss: {epoch_aux_loss:.4f} | Base Loss: {epoch_base_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | Valid Acc: {epoch_val_acc:.4f} | "
                f"Valid mIoU: {epoch_val_miou:.4f}\n")
        else:
            logger.info(f"  Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | Valid Acc: {epoch_val_acc:.4f} | "
                f"Valid mIoU: {epoch_val_miou:.4f}\n")
        
        # ✅ Append the results to the correct lists within the dictionary
        training_history['train_loss'].append(epoch_train_loss)
        training_history['train_acc'].append(epoch_train_acc)
        training_history['valid_acc'].append(epoch_val_acc)  
        training_history['valid_miou'].append(epoch_val_miou)
        training_history['epoch'].append(epoch+1)
        training_history['step'].append(step+1)
        history_df = pd.DataFrame(training_history)
        history_df.to_csv(os.path.join(output_dir, f'{subdir_name}.csv'), index=False)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    logger.info("🏁 Training complete.")
    logger.info(f"Best Accuracy: {best_acc:.4f}")
    logger.info(output_dir)

    # =================================================================================
    # Step 6: Save the Results and Model
    # =================================================================================

    # ✅ Step 1: Convert the dictionary directly into a pandas DataFrame
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