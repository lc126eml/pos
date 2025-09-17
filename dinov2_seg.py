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
import logging

#%%
sys.path.append(r".")
from vision_transformer_rope import *
from vision_transformer_rpe import *
from vision_transformer_relpos import *
from vision_transformer_alibi import *
from vision_transformer_sin import *

# %%
# =================================================================================
# Step 2: Configuration
# =================================================================================
# %%
# =================================================================================
# Step 2: Configuration
# =================================================================================

# --- Model & Training Settings ---
MODEL_TYPE = "small"
MODEL_NAME = f'vit_{MODEL_TYPE}_patch14_dinov2'
NUM_CLASSES = 150  # For ADE20K (150 classes)
BATCH_SIZE = 200  # Adjusted for segmentation (memory-intensive)
IMG_SIZE = 224  # ViT fixed size; adjust if needed
LEARNING_RATE = 5e-4
EPOCHS = 130  # Reduced for segmentation
HAS_POS = True
OVERLAP = 0
pretrained = None
START_EPOCH = 0
WANDB = False
SEED = 55
hid = 'ubuntu'
VAL_STEPS = 500
ALPHA = 3.0
Use_Row_Col_Loss = False
RC_ALPHA = 30.0
WORKERS = 4
output_dir = "/home/sshuser/Codes/pos/output"

# --- Dataset Paths ---
BASE_PATH = "/home/sshuser/Data/ADEChallengeData2016"
TRAIN_IMAGE_PATH = os.path.join(BASE_PATH, 'images', 'training')
TRAIN_ANNOTATION_PATH = os.path.join(BASE_PATH, 'annotations', 'training')
VALID_IMAGE_PATH = os.path.join(BASE_PATH, 'images', 'validation')
VALID_ANNOTATION_PATH = os.path.join(BASE_PATH, 'annotations', 'validation')

# --- Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16

# %%
# Set seeds
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
subdir_name = (
    f"has_pos_{HAS_POS}_overlap_{OVERLAP}_"
    f"use_rc_loss_{Use_Row_Col_Loss}_rc_alpha_{RC_ALPHA}"
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

logger.info(f"Using device: {DEVICE}")
logger.info(f"Using mixed precision: {'bfloat16' if use_bf16 else 'float16'}")
# %%
if WANDB:
    import wandb
    # from wandb.keras import WandbCallback
    wandb.login(key='bb050692d5a8ea8b20a38ddcd72a9eb06f497aff')

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
        mask = torch.from_numpy(np.array(mask)).long() - 1
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
    img_size=IMG_SIZE,
    is_train=True,
    mean=img_mean,
    std=img_std
)
valid_dataset = SegmentationDataset(
    VALID_IMAGE_PATH,
    VALID_ANNOTATION_PATH,
    img_size=IMG_SIZE,
    is_train=False,
    mean=img_mean,
    std=img_std
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, pin_memory=True, drop_last=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS, pin_memory=True, drop_last=True)

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
# %%
if WANDB:        
    params = {}
    params['NUM_CLASSES'] = NUM_CLASSES
    params['BATCH_SIZE'] = BATCH_SIZE
    params['IMG_SIZE'] = IMG_SIZE
    params['EPOCHS'] = EPOCHS
    # params['START_EPOCH'] = START_EPOCH
    params['HAS_POS'] = HAS_POS
    params['OVERLAP'] = OVERLAP
    params["ALPHA"] = ALPHA
    params["MODEL_NAME"] = MODEL_NAME
    params["Row_Col_Loss"] = Use_Row_Col_Loss
    params["RC_ALPHA"] = RC_ALPHA
    params['lr'] = LEARNING_RATE
    params['train_imgs'] = len(train_dataset)
    params['hid'] = hid
    params['seed'] = SEED
    wandb.init(
#             reinit=True,
        # set the wandb project where this run will be logged
        project=f"dinov2_{MODEL_TYPE}_seg",
        # track hyperparameters and run metadata
        config=params,
        group='ade20k',
        job_type='val'
    )

#%%
# =================================================================================
# An Improved, Progressive Decoder
# =================================================================================
class ProgressiveSegDecoder(nn.Module):
    """
    A more robust decoder that progressively upsamples features.
    This is a common pattern inspired by architectures like U-Net and FPN.
    """
    def __init__(self, in_channels, num_classes, grid_size):
        super().__init__()
        self.grid_size = grid_size
        
        # The embedding dimension from the ViT
        embed_dim = in_channels
        
        # A series of upsampling blocks
        # Each block consists of Upsample -> Conv -> BatchNorm -> ReLU
        # This allows the model to learn to refine features at increasing resolutions.
        self.decoder = nn.Sequential(
            # First, project the flattened patches into a channel-rich 2D grid
            nn.Conv2d(embed_dim, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # Upsample x2 (e.g., 16x16 -> 28x28)
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Upsample x2 (e.g., 28x28 -> 56x56)
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Upsample x4 to get to a higher resolution (e.g., 56x56 -> 224x224)
            # Another option is to continue with x2 upsampling for more refinement
            nn.Upsample(size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Final 1x1 convolution to map to the number of classes
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # x has shape (B, N, C) where N = (grid_size*grid_size)
        B, N, C = x.shape
        
        # Reshape to a 2D grid: (B, C, H, W)
        x = x.permute(0, 2, 1).view(B, C, self.grid_size, self.grid_size)
        
        # Pass through the progressive decoder
        return self.decoder(x)
# %%
# =================================================================================
# Step 4: Initialize the Model, Loss Function, and Optimizer
# =================================================================================
# --- Model ---
logger.info(f"🤖 Initializing model: {MODEL_NAME} ...")
model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=0, img_size=IMG_SIZE).to(DEVICE)
grid_h, grid_w = model.patch_embed.grid_size
decoder = ProgressiveSegDecoder(model.embed_dim, NUM_CLASSES, grid_h).to(DEVICE)

# --- Test with a dummy input ---
dummy_input = torch.randn(2, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
with torch.no_grad():
    feats = model.forward_features(dummy_input)
    output = decoder(feats[:, 1:, :])

logger.info(f"Model created successfully!")
logger.info(f"Input shape: {dummy_input.shape}")
logger.info(f"Output shape: {output.shape}") 
assert output.shape == (2, NUM_CLASSES, IMG_SIZE, IMG_SIZE)
logger.info("✅ Output shape is correct.")
del feats, output, dummy_input

# %%
logger.info(f'model.patch_embed.proj {model.patch_embed.proj}')
if OVERLAP>0:
    # Customize patch embedding for overlap (e.g., patch_size=15, stride=14)
    original_patch_size = model.patch_embed.proj.kernel_size[0]
    new_patch_size = original_patch_size + OVERLAP  # Or 15, 16, 17, etc., as desired
    stride = original_patch_size
    original_grid_size = IMG_SIZE // stride  # 16 for 224//14
    padding = ((original_grid_size - 1) * stride + new_patch_size - IMG_SIZE + 1) // 2  # +1 for ceiling effect; yields 1 for patch_size=15
    
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
    # grid_size_h = ((IMG_SIZE + 2 * padding - new_patch_size) // stride) + 1
    # grid_size_w = grid_size_h  # Assuming square input
    # logger.info(new_patch_size, padding, grid_size_h, model.patch_embed.grid_size)
    # model.patch_embed.grid_size = (grid_size_h, grid_size_w)
    # model.patch_embed.num_patches = grid_size_h * grid_size_w
    # logger.info(f"Updated to patch_size={new_patch_size}, stride={stride}, padding={padding}, num_patches={model.patch_embed.num_patches}")

if not HAS_POS and hasattr(model, 'pos_embed') and model.pos_embed is not None:
    model.pos_embed.data.zero_()
    model.pos_embed.requires_grad = False
    logger.info("✅ Positional embedding has been disabled.")
if pretrained is not None:
    state_dicts = torch.load(pretrained, map_location=DEVICE)
    IncompatibleKeys = model.load_state_dict(state_dicts)
    logger.info(IncompatibleKeys)
# --- Loss Function & Optimizer ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
logger.info("✅ Model, Loss Function, and Optimizer are ready.")

# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
# logger.info("✅ Model, Loss, Optimizer, and LR Scheduler are ready.")

# Loss and Optimizer
criterion = nn.CrossEntropyLoss(ignore_index=-1)  # Ignore background if index 0
optimizer = optim.AdamW(list(model.parameters()) + list(decoder.parameters()), lr=LEARNING_RATE)
total_steps = EPOCHS * steps_per_epoch
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
logger.info("✅ Step-based LR Scheduler is ready.")

# %%
# dummy_input = torch.randn(2, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
# with torch.no_grad():
#     feats = model.forward_features(dummy_input)
# logger.info(f"Model created successfully!")
# logger.info(f"Input shape: {dummy_input.shape}")
# logger.info(f"Output shape: {feats.shape}") 

# %%
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
        feat_dim=model.get_classifier().in_features,
        grid_h=grid_h,
        grid_w=grid_w
    ).to(DEVICE)
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

# FP16: Initialize the Gradient Scaler
scaler = torch.amp.GradScaler('cuda')
# =================================================================================
# Step 5: Training and Validation Loop
# =================================================================================
logger.info(f"\n🚀 Starting training for {MODEL_NAME}...")

# ✅ Initialize training_history as a dictionary of lists
training_history = {
    'train_loss': [],
    'train_acc': [],
    'valid_acc': [],
    'train_miou': [],
    'valid_miou': [],
    'epoch': [],
    'step': [],
}
step = 0

for epoch in range(EPOCHS):
    # --- Training Phase ---
    model.train()
    decoder.train()
    running_loss = 0.0
    train_correct = 0
    train_total = 0
    train_intersection = torch.zeros(NUM_CLASSES).to(DEVICE)
    train_union = torch.zeros(NUM_CLASSES).to(DEVICE)
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Training]")
    # train_pbar = train_loader
    
    # FP16: Use autocast for the forward pass
    for batch_idx, (inputs, labels) in enumerate(train_pbar):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', dtype=autocast_dtype):
            feats = model.forward_features(inputs)
            outputs = decoder(feats[:, 1:, :])            
            loss = criterion(outputs, labels)
            if Use_Row_Col_Loss:
                aux_loss = rowcol_loss(feats[:, 1:, :])
                # logger.info(loss, aux_loss)
                loss = loss + RC_ALPHA * aux_loss
        
        # FP16: Scale, backward, and step
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()
        
        running_loss += loss.item() * inputs.size(0)
        predicted = torch.argmax(outputs, dim=1)
        mask = (labels >= 0)
        train_correct += ((predicted == labels) & mask).sum().item()
        train_total += mask.sum().item()

        # Accumulate for epoch mIoU
        for c in range(NUM_CLASSES):
            pred_c = (predicted == c) & mask
            label_c = (labels == c) & mask
            train_intersection[c] += (pred_c & label_c).sum().item()
            train_union[c] += (pred_c | label_c).sum().item()

        batch_pixel_acc = train_correct / train_total if train_total > 0 else 0.0
        batch_miou = compute_miou(predicted.cpu(), labels.cpu(), NUM_CLASSES)
        bar_msg = {
                'loss': f'{loss.item():.4f}', 
                'acc': f'{batch_pixel_acc:.3f}', 
                'miou': f'{batch_miou:.3f}'
            }
        if 'aux_loss' in locals():
            bar_msg['aux'] = f'{aux_loss.item():.4f}'
        train_pbar.set_postfix(bar_msg)

        step += 1

        # if (step) % VAL_STEPS == 0:
    # --- Validation Phase ---
    iou = torch.zeros(NUM_CLASSES).to(DEVICE)
    valid = train_union > 0
    iou[valid] = train_intersection[valid] / train_union[valid]
    epoch_train_miou = iou.mean().item()
    
    model.eval()
    decoder.eval()
    val_correct = 0
    val_total = 0
    val_intersection = torch.zeros(NUM_CLASSES).to(DEVICE)
    val_union = torch.zeros(NUM_CLASSES).to(DEVICE)
    val_pbar = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Validation]")
    
    with torch.no_grad():
        for inputs, labels in val_pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            with torch.amp.autocast('cuda', dtype=autocast_dtype):
                feats = model.forward_features(inputs)
                outputs = decoder(feats[:, 1:, :])
            predicted = torch.argmax(outputs, dim=1)
            mask = (labels >= 0)
            val_correct += ((predicted == labels) & mask).sum().item()
            val_total += mask.sum().item()

            # Compute IoU for the batch
            for c in range(NUM_CLASSES):
                pred_c = (predicted == c) & mask
                label_c = (labels == c) & mask
                val_intersection[c] += (pred_c & label_c).sum().item()
                val_union[c] += (pred_c | label_c).sum().item()

    # Compute validation mIoU
    iou = torch.zeros(NUM_CLASSES).to(DEVICE)
    valid = val_union > 0
    iou[valid] = val_intersection[valid] / val_union[valid]
    epoch_val_miou = iou.mean().item()
    
    epoch_val_acc = val_correct / val_total if val_total > 0 else 0.0
    epoch_train_acc = train_correct / train_total if train_total > 0 else 0.0
    epoch_train_loss = running_loss / (step % steps_per_epoch + 1)

    logger.info(f"\nEpoch {epoch+1+START_EPOCH}/{EPOCHS} Summary:")
    logger.info(f"\nStep {step+1+START_EPOCH * steps_per_epoch} Summary:")
    logger.info(f"  Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | "
            f"Train mIoU: {epoch_train_miou:.4f} | Valid Acc: {epoch_val_acc:.4f} | "
            f"Valid mIoU: {epoch_val_miou:.4f}\n")
    
    # ✅ Append the results to the correct lists within the dictionary
    training_history['train_loss'].append(epoch_train_loss)
    training_history['train_acc'].append(epoch_train_acc)
    training_history['train_miou'].append(epoch_train_miou)
    training_history['valid_acc'].append(epoch_val_acc)  
    training_history['valid_miou'].append(epoch_val_miou)
    training_history['epoch'].append(epoch+1)
    training_history['step'].append(step+1)

    model.train()
    decoder.train()
            
    
    # Update the learning rate scheduler
    # if 'scheduler' in locals():
    #     scheduler.step()

logger.info("🏁 Training complete.")

# =================================================================================
# Step 6: Save the Results and Model
# =================================================================================

# ✅ Step 1: Convert the dictionary directly into a pandas DataFrame
history_df = pd.DataFrame(training_history)

history_df.to_csv(os.path.join(output_dir, 'training_history.csv'), index=False)
save_checkpoint(model, decoder, output_dir, "final")

# %%
if WANDB:
    best_index = max(range(len(training_history['valid_acc'])), key=lambda i: training_history['valid_acc'][i])
    best_accuracy = training_history['valid_acc'][best_index]
    best_miou = training_history['valid_miou'][best_index]
    best_epoch = training_history['epoch'][best_index]
    best_step = training_history['step'][best_index]
    train_accuracy = max(training_history['train_acc'])
    wandb.log({"best_acc": best_accuracy, "best_miou": best_miou, "best_epoch": best_epoch, "best_step": best_step, "train_acc": train_accuracy})

if not history_df.empty:
    best_miou = history_df['valid_miou'].max()
    best_epoch = history_df.loc[history_df['valid_miou'].idxmax(), 'epoch']
    logger.info(f"Best miou: {best_miou:.4f} at epoch {best_epoch}")

if not history_df.empty:
    # Find the epoch with the best validation a1 score
    best_miou_row = history_df.loc[history_df['valid_miou'].idxmax()]
    best_miou_epoch = int(best_miou_row['epoch'])
    best_miou_val = best_miou_row['valid_miou']

    # Find the epoch with the best validation abs_rel
    best_acc_row = history_df.loc[history_df['valid_acc'].idxmin()]
    best_acc_epoch = int(best_acc_row['epoch'])
    best_acc_val = best_acc_row['valid_acc']

    logger.info("\n--- Best Validation Metrics from History ---")
    logger.info(f"  Best a1:      {best_miou_val:.4f} (Epoch {best_miou_epoch})")
    logger.info(f"  Best AbsRel:  {best_acc_val:.4f} (Epoch {best_acc_epoch})")
    logger.info("------------------------------------------")
# %%
# import matplotlib.pyplot as plt
# import pandas as pd
    
# # First, ensure the training_history list is not empty
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

# %%
if WANDB:
    wandb.finish()


