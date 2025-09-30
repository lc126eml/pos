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

# --- Dynamically set root directory ---
root_dir = '/home/sshuser' if os.path.exists('/home/sshuser') else '/linux'

# --- Configuration via SimpleNamespace for easy interactive use ---
args = SimpleNamespace(
    # --- Model & Training Settings ---
    pos_type = None, # 'sin', 'alibi', 'relpos',  'rpe', 'rope', 
    model_type='base',
    num_classes=10,
    # Adjust based on your GPU memory. BATCH_SIZE = 120, 128, 136, 392, 768, etc.
    batch_size=392,
    # ViT models have a fixed input size
    img_size=224,
    lr=5e-4,
    epochs=130,
    has_pos=False, # Set to True or False directly
    overlap=5,
    pretrained=None,
    seed=55,
    use_patch_position_loss=False,
    use_rc_loss=False,
    rc_alpha=30.0,
    workers=5,

    # --- Dataset Paths ---
    root_dir=root_dir,
)
if args.pos_type is not None:
    args.has_pos = True
    args.overlap = 0
    args.use_rc_loss=False
    args.use_patch_position_loss=False


MODEL_NAME = f"vit_{f'{args.pos_type}_' if args.pos_type is not None else ""}{args.model_type}_patch14_dinov2"
output_dir = f"{args.root_dir}/Codes/pos/output/imagenet/{args.model_type}b{args.batch_size}s{args.seed}"
BASE_PATH = f'{args.root_dir}/Data/imagenet100/'

# List of all the partial training directories
TRAIN_PATHS = [
    os.path.join(BASE_PATH, 'train.X1'),
    os.path.join(BASE_PATH, 'train.X2'),
    os.path.join(BASE_PATH, 'train.X3'),
    os.path.join(BASE_PATH, 'train.X4'),
]
offset = 0
VALID_PATH = os.path.join(BASE_PATH, 'val.X')
LABEL_PATH = os.path.join(BASE_PATH, 'Labels.json')

# --- Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16

#%%

if args.pos_type is not None:
    sys.path.append(r".")
    from vision_transformer_rope import *
    from vision_transformer_rpe import *
    from vision_transformer_relpos import *
    from vision_transformer_alibi import *
    from vision_transformer_sin import *

# %%
# timm.list_models("vit_*_dinov2")

# %%
# torch.backends.cudnn.deterministic=True
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
subdir_name = (
    f"{f'{args.pos_type}_' if args.pos_type is not None else ""}{args.model_type}{'_pos' if args.has_pos else ''}_overlap_{args.overlap}_"
    f"rc_{args.use_rc_loss}{'_patch_pos' if args.use_patch_position_loss else ''}_classes_{args.num_classes}"
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
logger.info(args)
logger.info(subdir_name)

# --- Acquire a file lock to ensure exclusive GPU usage ---
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
class CustomImageDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
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

# --- Data Augmentation ---
img_mean=[0.485, 0.456, 0.406]
img_std=[0.229, 0.224, 0.225]
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(args.img_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=img_mean, std=img_std),
])

valid_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(args.img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=img_mean, std=img_std),
])


# --- Create the final datasets from the filtered samples ---
train_dataset = CustomImageDataset(train_samples, transform=train_transforms)
valid_dataset = CustomImageDataset(valid_samples, transform=valid_transforms)

logger.info(f"Total training images ({args.num_classes} classes): {len(train_dataset)}")
logger.info(f"Total validation images ({args.num_classes} classes): {len(valid_dataset)}")

# --- Create DataLoaders ---
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=args.batch_size,
    shuffle=True, 
    num_workers=args.workers,
    pin_memory=True
)

valid_loader = DataLoader(
    dataset=valid_dataset,
    batch_size=args.batch_size,
    shuffle=False, 
    num_workers=2,
    pin_memory=True
)
steps_per_epoch = len(train_loader)
logger.info(f"✅ DataLoaders for {args.num_classes} classes created successfully.")
logger.info(f"{steps_per_epoch=}, val_steps: {len(valid_loader)}")

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
    num_classes=args.num_classes, # Set the classifier head to 100 classes
    img_size=args.img_size,
).to(DEVICE)

# feature_layers = [2, 5, 8, 11]
# dummy_input = torch.randn(2, 3, args.img_size, args.img_size).to(DEVICE)
# with torch.no_grad():
#     feats = model.forward_features(dummy_input)
# #     multi_feats = model.forward_intermediates(dummy_input, indices=feature_layers, intermediates_only=True)


# logger.info(f"Model created successfully!")
# logger.info(f"Input shape: {dummy_input.shape}")
# logger.info(f"Output shape: {feats.shape}") 
# logger.info(f"multi_feats shape: {multi_feats[-1].shape} X {len(multi_feats)}")
# del feats, multi_feats, dummy_input
# gc.collect()

# %%
logger.info(f'model.patch_embed.proj{model.patch_embed.proj}')
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

if not args.has_pos and hasattr(model, 'pos_embed') and model.pos_embed is not None:
    model.pos_embed.data.zero_()
    model.pos_embed.requires_grad = False
    logger.info("✅ Positional embedding has been disabled.")
if args.pretrained is not None:
    state_dicts = torch.load(args.pretrained, map_location=DEVICE)
    IncompatibleKeys = model.load_state_dict(state_dicts)
    logger.info(IncompatibleKeys)
# --- Loss Function & Optimizer ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=args.lr)
logger.info("✅ Model, Loss Function, and Optimizer are ready.")

# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
# logger.info("✅ Model, Loss, Optimizer, and LR Scheduler are ready.")

total_steps = args.epochs * steps_per_epoch
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
logger.info("✅ Step-based LR Scheduler is ready.")

# %%
# dummy_input = torch.randn(2, 3, args.img_size, args.img_size).to(DEVICE)
# with torch.no_grad():
#     feats = model.forward_features(dummy_input)
# logger.info(f"Model created successfully!")
# logger.info(f"Input shape: {dummy_input.shape}")
# logger.info(f"Output shape: {feats.shape}") 
    
sys.stdout.flush()
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

if args.use_rc_loss:
    grid_h, grid_w = model.patch_embed.grid_size
    rowcol_loss = PatchRowColCriterion(
        feat_dim=model.embed_dim,
        grid_h=grid_h,
        grid_w=grid_w
    ).to(DEVICE)
if args.use_patch_position_loss:
    from patch_pos import PatchPositionCriterion
    position_loss = PatchPositionCriterion(
        feat_dim=model.embed_dim,
        num_classes=model.patch_embed.num_patches
    ).to(DEVICE)

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
    'epoch': [],
    'step': [],
}
step = 0

for epoch in range(args.epochs):
    # --- Training Phase ---
    model.train()
    running_loss = 0.0
    train_correct = 0
    train_total = 0
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Training]")
    # train_pbar = train_loader
    
    # FP16: Use autocast for the forward pass
    for inputs, labels in train_pbar:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        aux_loss = None
        with torch.amp.autocast('cuda', dtype=autocast_dtype):
            feats = model.forward_features(inputs)
            outputs = model.forward_head(feats)
            # outputs = model(inputs)
            loss = criterion(outputs, labels)
            if args.use_rc_loss:
                aux_loss = rowcol_loss(feats[:, 1:, :])
                # logger.info(loss, aux_loss)
                loss = loss + args.rc_alpha * aux_loss
            
            if args.use_patch_position_loss:
                aux_loss = position_loss(feats[:, 1:, :])
                # logger.info(f"{loss}, {aux_loss}")
                loss = loss + args.rc_alpha * aux_loss
        
        # FP16: Scale, backward, and step
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        
        batch_acc = (predicted == labels).sum().item() / labels.size(0)
        bar_msg={'loss': loss.item(), 'acc': f'{batch_acc:.2f}'}
        if aux_loss is not None:
            bar_msg['aux'] = aux_loss.item()
        train_pbar.set_postfix(bar_msg)

        step += 1

        # if (step + 1) % VAL_STEPS == 0:
    # --- Validation Phase ---
    model.eval()
    val_correct = 0
    val_total = 0
    # val_pbar = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Validation]")
    
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            with torch.amp.autocast('cuda', dtype=autocast_dtype):
                outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    epoch_val_acc = val_correct / val_total

    epoch_train_acc = train_correct / train_total
    epoch_train_loss = running_loss / len(train_loader.dataset)
    
    logger.info(f"\nEpoch {epoch+1}/{args.epochs} Summary:")
    logger.info(f"\nStep {step} Summary:")
    logger.info(f"  Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | Valid Acc: {epoch_val_acc:.4f}\n")

    # ✅ Append the results to the correct lists within the dictionary
    training_history['train_loss'].append(epoch_train_loss)
    training_history['train_acc'].append(epoch_train_acc)
    training_history['valid_acc'].append(epoch_val_acc)  
    training_history['epoch'].append(epoch+1)
    training_history['step'].append(step+1)
    history_df = pd.DataFrame(training_history)
    history_df.to_csv(os.path.join(output_dir, f'{subdir_name}.csv'), index=False)

    # Update the learning rate scheduler
    # if 'scheduler' in locals():
    #     scheduler.step()

logger.info("🏁 Training complete.")

# =================================================================================
# Step 6: Save the Results and Model
# =================================================================================

# ✅ Step 1: Convert the dictionary directly into a pandas DataFrame
history_df = pd.DataFrame(training_history)

# ✅ Step 2: Add the 'epoch' column at the beginning
# Create the list of epochs where validation was actually performed
# epochs_validated = range(5, EPOCHS + 1, 5) 
# history_df.insert(0, 'epoch', epochs_validated)

# ✅ Step 3: Save the DataFrame to a CSV file
history_df.to_csv(os.path.join(output_dir, f'{subdir_name}.csv'), index=False)
# Save the model's state dictionary
# torch.save(model.state_dict(), f'{MODEL_NAME}_final.pth')
# logger.info(f"✅ Model saved to '{MODEL_NAME}_final.pth'")
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
