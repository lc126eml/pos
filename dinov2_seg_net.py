# %%
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
from torch.utils.data import Dataset, DataLoader
import timm
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

# %%
# %%
# =================================================================================
# Step 2: Configuration
# =================================================================================

# --- Model & Training Settings ---
MODEL_NAME = 'vit_base_patch14_dinov2'
NUM_CLASSES = 150  # For ADE20K (150 classes)
BATCH_SIZE = 56  # Adjusted for segmentation (memory-intensive)
IMG_SIZE = 224  # ViT fixed size; adjust if needed
LEARNING_RATE = 5e-4
EPOCHS = 38  # Reduced for segmentation
HAS_POS = False
OVERLAP = 2
pretrained = "/kaggle/input/dinov2-seg-ds/vit_base_patch14_dinov2_final.pth"
START_EPOCH = 85
WANDB = True
SEED = 55
hid = 16
VAL_STEPS = 500
ALPHA = 3.0
Use_Relative_Direction_Loss = 0
SAMPLE_RATE = 0.05
Neg_Ratio = 5
Discard_FAR = 0
Use_Patch_Position_Loss = False
Use_Row_Col_Loss = True
RC_ALPHA = 30.0

# --- Dataset Paths ---
BASE_PATH = '/kaggle/input/ade20k-dataset/ADEChallengeData2016'  # Updated to match Kaggle dataset name
TRAIN_IMAGE_PATH = os.path.join(BASE_PATH, 'images', 'training')
TRAIN_ANNOTATION_PATH = os.path.join(BASE_PATH, 'annotations', 'training')
VALID_IMAGE_PATH = os.path.join(BASE_PATH, 'images', 'validation')
VALID_ANNOTATION_PATH = os.path.join(BASE_PATH, 'annotations', 'validation')

# --- Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {DEVICE}")

# %%
# Set seeds
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

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

# %%
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

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=3, pin_memory=True, drop_last=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=3, pin_memory=True, drop_last=True)

steps_per_epoch = len(train_loader)
print(f"✅ DataLoaders created successfully.")
print(f"   - Training samples: {len(train_dataset)}, Batches per epoch: {len(train_loader)}")
print(f"   - Validation samples: {len(valid_dataset)}, Batches per epoch: {len(valid_loader)}")

# %%
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
#     print(f"Could not display images. Error: {str(e)}. Ensure previous cells have been run to create 'train_loader'.")

# %%
if WANDB:        
    params = {}
    params['NUM_CLASSES'] = NUM_CLASSES
    params['BATCH_SIZE'] = BATCH_SIZE
    params['IMG_SIZE'] = IMG_SIZE
    params['EPOCHS'] = EPOCHS
    params['START_EPOCH'] = START_EPOCH
    params['HAS_POS'] = HAS_POS
    params['OVERLAP'] = OVERLAP
    params["ALPHA"] = ALPHA
    params['Relative_Direction_Loss'] = Use_Relative_Direction_Loss
    params["Discard_FAR"] = Discard_FAR
    params["SAMPLE_RATE"] = SAMPLE_RATE
    params["Neg_Ratio"] = Neg_Ratio
    params["Patch_Position_Loss"] = Use_Patch_Position_Loss
    params["Row_Col_Loss"] = Use_Row_Col_Loss
    params["RC_ALPHA"] = RC_ALPHA
    params['lr'] = LEARNING_RATE
    params['train_imgs'] = len(train_dataset)
    params['hid'] = hid
    params['seed'] = SEED
    wandb.init(
#             reinit=True,
        # set the wandb project where this run will be logged
        project="dinov2_pos_seg",
        # track hyperparameters and run metadata
        config=params,
        group='ade20k',
        job_type='val'
    )

# %%
# # %%
# # =================================================================================
# # Simple Decoder for Segmentation
# # =================================================================================
# class SimpleSegDecoder(nn.Module):
#     def __init__(self, in_channels, num_classes, grid_size):
#         super().__init__()
#         self.grid_size = grid_size
#         self.fc = nn.Linear(in_channels, in_channels // 2)
#         self.up = nn.Upsample(scale_factor=grid_size, mode='nearest')  # Upsample to image size
#         self.conv = nn.Conv2d(in_channels // 2, num_classes, kernel_size=1)

#     def forward(self, x):
#         B, N, C = x.shape
#         x = self.fc(x)  # (B, N, C/2)
#         x = x.permute(0, 2, 1).view(B, -1, self.grid_size, self.grid_size)  # Reshape to grid
#         x = self.up(x)  # Upsample
#         x = self.conv(x)
#         return x

# # Model Setup
# model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=0, img_size=IMG_SIZE).to(DEVICE)  # Backbone only (num_classes=0 for features)
# grid_h, grid_w = model.patch_embed.grid_size
# decoder = SimpleSegDecoder(model.embed_dim, NUM_CLASSES, grid_h).to(DEVICE)

# %%
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


# =================================================================================
# Full Segmentation Model - Combining Backbone and Decoder
# =================================================================================
# class ViTSegmentationModel(nn.Module):
#     def __init__(self, model_name, num_classes, pretrained=True):
#         super().__init__()
#         # Use num_classes=0 to get a feature extractor (no classification head)
#         self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=IMG_SIZE)
        
#         # Get grid size and embedding dimension from the backbone
#         grid_size = self.backbone.patch_embed.grid_size
#         embed_dim = self.backbone.embed_dim
        
#         self.decoder = ProgressiveSegDecoder(embed_dim, num_classes, grid_size[0])

#     def forward(self, x):
#         # Pass input through the backbone to get patch embeddings
#         # NOTE: forward_features is a timm-specific method
#         patch_embeddings = self.backbone.forward_features(x) # (B, N+1, C)

#         # **CRITICAL FIX:** Remove the [CLS] token before passing to the decoder
#         # The CLS token is at index 0
#         patch_embeddings = patch_embeddings[:, 1:, :]
        
#         # Get the segmentation map from the decoder
#         segmentation_map = self.decoder(patch_embeddings)
        
#         return segmentation_map


# --- Model Setup Example 
# model = ViTSegmentationModel(MODEL_NAME, NUM_CLASSES, pretrained=False).to(DEVICE)

# # Model Setup
model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=0, img_size=IMG_SIZE).to(DEVICE)
grid_h, grid_w = model.patch_embed.grid_size
decoder = ProgressiveSegDecoder(model.embed_dim, NUM_CLASSES, grid_h).to(DEVICE)

# --- Test with a dummy input ---
dummy_input = torch.randn(2, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
with torch.no_grad():
    feats = model.forward_features(dummy_input)
    output = decoder(feats[:, 1:, :])

print(f"Model created successfully!")
print(f"Input shape: {dummy_input.shape}")
print(f"Output shape: {output.shape}") 
assert output.shape == (2, NUM_CLASSES, IMG_SIZE, IMG_SIZE)
print("✅ Output shape is correct.")

# %%
if OVERLAP > 0:
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
    print(f"✅ Updated patch embedding: patch_size={new_patch_size}, stride={stride}, padding={padding}")
    
    # Recompute grid size and num_patches
    # grid_size_h = ((IMG_SIZE + 2 * padding - new_patch_size) // stride) + 1
    # grid_size_w = grid_size_h  # Assuming square input
    # print(new_patch_size, padding, grid_size_h, model.patch_embed.grid_size)
    # model.patch_embed.grid_size = (grid_size_h, grid_size_w)
    # model.patch_embed.num_patches = grid_size_h * grid_size_w
    # print(f"Updated to patch_size={new_patch_size}, stride={stride}, padding={padding}, num_patches={model.patch_embed.num_patches}")

if not HAS_POS:
    model.pos_embed.data.zero_()
    model.pos_embed.requires_grad = False
    print("✅ Positional embedding has been disabled.")
if pretrained is not None:
    state_dicts = torch.load(pretrained, map_location=DEVICE)
    IncompatibleKeys = model.load_state_dict(state_dicts)
    print(IncompatibleKeys)
# --- Loss Function & Optimizer ---
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
# print("✅ Model, Loss, Optimizer, and LR Scheduler are ready.")

# Loss and Optimizer
criterion = nn.CrossEntropyLoss(ignore_index=-1)  # Ignore background if index 0
optimizer = optim.AdamW(list(model.parameters()) + list(decoder.parameters()), lr=LEARNING_RATE)
total_steps = EPOCHS * steps_per_epoch
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
print("✅ Step-based LR Scheduler is ready.")

# %%
if Use_Relative_Direction_Loss==8:
    direction_map = {
                # (0, 0): 0,   # same patch (center)
                (0, 1): 1,   # right
                (0, -1): 2,  # left
                (1, 0): 3,   # down
                (-1, 0): 4,  # up
                (1, 1): 5,   # down-right
                (1, -1): 6,  # down-left
                (-1, 1): 7,  # up-right
                (-1, -1): 8  # up-left
            }
    rc_classes = 9
if Use_Relative_Direction_Loss==4:
    direction_map = {
                # (0, 0): 0,   # same patch (center)
                (0, 1): 1,   # right
                (0, -1): 2,  # left
                (1, 0): 3,   # down
                (-1, 0): 4,  # up
                # (1, 1): 5,   # down-right
                # (1, -1): 6,  # down-left
                # (-1, 1): 7,  # up-right
                # (-1, -1): 8  # up-left
            }
    rc_classes = 5
if Use_Relative_Direction_Loss>0:
    def get_relative_direction(p1, p2):
        """
        Example placeholder function.
        Define mapping from relative (dy, dx) to class id (0–8).
        """
        dy = p2[0] - p1[0]
        dx = p2[1] - p1[1]   
        if Discard_FAR>0:
            if abs(dy) > Discard_FAR or abs(dx) > Discard_FAR:
                return -1
        return direction_map.get((dy,dx), 0)  # default self if outside range
            
    def precompute_patch_pairs(grid_h, grid_w):
        pos = [(i, j) for i in range(grid_h) for j in range(grid_w)]
        pairs, labels = [], []
    
        for idx1, p1 in enumerate(pos):
            for idx2, p2 in enumerate(pos):
                if idx1 == idx2:
                    continue
                label = get_relative_direction(p1, p2)  # 0–8 classes
                if label < 0:
                    continue
                pairs.append((idx1, idx2))
                labels.append(label)
    
        return torch.tensor(pairs), torch.tensor(labels)
    
    # Example: 14x14 patches
    grid_h, grid_w = model.patch_embed.grid_size
    pairs, labels = precompute_patch_pairs(grid_h, grid_w)
    labels = labels.to(DEVICE)
    pairs = pairs.to(DEVICE)
    # pairs.shape = (num_pairs, 2), labels.shape = (num_pairs,)

    pos_mask = labels > 0
    pos_indices = torch.nonzero(pos_mask, as_tuple=False).squeeze()
    
    # Sample positive indices with Bernoulli probability
    num_pos = len(pos_indices)

     # Identify negative pairs (labels == 0)
    neg_mask = labels == 0
    neg_indices = torch.nonzero(neg_mask, as_tuple=False).squeeze()
    num_neg = len(neg_indices)

    def random_sample(pairs, labels, sample_rate=0.5):
        """
        Samples pairs such that the number of negative pairs (label 0) equals the number of 
        sampled positive pairs (labels > 0), where positive pairs are sampled with the given rate.
        
        Args:
            pairs (torch.Tensor): Tensor of shape (num_pairs, 2) containing pair indices.
            labels (torch.Tensor): Tensor of shape (num_pairs,) containing labels.
            sample_rate (float): Probability for sampling each positive pair.
        
        Returns:
            sampled_pairs (torch.Tensor): Sampled pairs.
            sampled_labels (torch.Tensor): Corresponding sampled labels.
        """
                
        sample_mask_pos = torch.rand(num_pos, device=pos_indices.device) < sample_rate
        sampled_pos_indices = pos_indices[sample_mask_pos]
        num_sampled_neg = min(round(len(sampled_pos_indices) * Neg_Ratio), num_neg)
        
        perm = torch.randperm(num_neg, device=neg_indices.device)[:num_sampled_neg]
        sampled_neg_indices = neg_indices[perm]
        
        # Combine and shuffle indices
        all_sampled_indices = torch.cat([sampled_pos_indices, sampled_neg_indices])
        perm = torch.randperm(len(all_sampled_indices), device=all_sampled_indices.device)
        all_sampled_indices = all_sampled_indices[perm]
        
        # Extract sampled pairs and labels
        sampled_pairs = pairs[all_sampled_indices]
        sampled_labels = labels[all_sampled_indices]
        del perm, all_sampled_indices, sample_mask_pos, sampled_pos_indices, sampled_neg_indices        
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()
        
        return sampled_pairs, sampled_labels
    
    class RelativeDirectionCriterion(nn.Module):
        def __init__(self, feat_dim, hidden_dim=256, num_classes=rc_classes, sample_rate=1.0):
            """
            Args:
                feat_dim (int): Dimension of input features (D).
                hidden_dim (int): Hidden layer size for MLP.
                num_classes (int): Number of relative direction classes (default 9 for 8-neighbor + self).
            """
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(2 * feat_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes)
            )
            self.ce = nn.CrossEntropyLoss()
            self.sample_rate = sample_rate
            
        def forward(self, feats, pairs=pairs, labels=labels, chunk_size=196):
            # feats: (B, N, D)
            sampled_pairs, sampled_labels = random_sample(pairs, labels, sample_rate=self.sample_rate)
            idx1, idx2 = sampled_pairs[:, 0], sampled_pairs[:, 1]
            total_loss = 0.0
            count = 0
        
            for start in range(0, len(sampled_pairs), chunk_size):
                end = min(len(sampled_pairs), start + chunk_size)
                p1, p2 = idx1[start:end], idx2[start:end]
                lbl = sampled_labels[start:end]
                actual_chunk_size = len(p1)
        
                f1 = feats[:, p1, :]  # (B, chunk, D)
                f2 = feats[:, p2, :]
                labels_expanded = lbl.repeat(feats.size(0))

                f1 = f1.reshape(-1, feats.size(-1))
                f2 = f2.reshape(-1, feats.size(-1))
                x = torch.cat([f1, f2], dim=-1)
                logits = self.mlp(x)
                loss = self.ce(logits, labels_expanded)
                total_loss += loss * actual_chunk_size
                count += actual_chunk_size
            del sampled_pairs, sampled_labels, idx1, idx2, p1, p2, lbl           
        
            return total_loss / count
            
    direction_loss = RelativeDirectionCriterion(feat_dim=model.embed_dim, num_classes=rc_classes, sample_rate=SAMPLE_RATE).to(DEVICE)

# %%
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
        self.register_buffer("patch_positions", torch.arange(num_classes))  # shape (num_patches,)
        
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
        
if Use_Patch_Position_Loss:
    position_loss = PatchPositionCriterion(
        feat_dim=model.embed_dim,
        num_classes=model.patch_embed.num_patches
    ).to(DEVICE)

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
        feat_dim=model.embed_dim,
        grid_h=grid_h,
        grid_w=grid_w
    ).to(DEVICE)
    print("✅ Row-Column loss initialized.")

# %%
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

# %%
import csv

# FP16: Initialize the Gradient Scaler
scaler = torch.amp.GradScaler('cuda')
# =================================================================================
# Step 5: Training and Validation Loop
# =================================================================================
print(f"\n🚀 Starting training for {MODEL_NAME}...")

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
        with torch.amp.autocast('cuda'):
            feats = model.forward_features(inputs)
            outputs = decoder(feats[:, 1:, :])            
            loss = criterion(outputs, labels)
            if Use_Relative_Direction_Loss:
                aux_loss = direction_loss(feats[:, 1:, :])
                # print(loss, dloss)
                loss = loss + ALPHA * aux_loss

            if Use_Patch_Position_Loss:
                aux_loss = position_loss(feats[:, 1:, :])
                # print(loss, aux_loss)
                loss = loss + ALPHA * aux_loss

            if Use_Row_Col_Loss:
                aux_loss = rowcol_loss(feats[:, 1:, :])
                # print(loss, aux_loss)
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

        if (step + 1) % VAL_STEPS == 0:
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
                    with torch.amp.autocast('cuda'):
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

            print(f"\nEpoch {epoch+1+START_EPOCH}/{EPOCHS} Summary:")
            print(f"\nStep {step+1+START_EPOCH * steps_per_epoch} Summary:")
            print(f"  Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | "
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
            
        step += 1
    
    # Update the learning rate scheduler
    # if 'scheduler' in locals():
    #     scheduler.step()

print("🏁 Training complete.")

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
csv_file = 'training_history.csv'
history_df.to_csv(csv_file, index=False) # index=False prevents pandas from writing row numbe

print(f"✅ Training history saved to '{csv_file}'")

# Save the model's state dictionary
torch.save(model.state_dict(), f'{MODEL_NAME}_final.pth')
print(f"✅ Model saved to '{MODEL_NAME}_final.pth'")

# %%
if WANDB:
    best_index = max(range(len(training_history['valid_acc'])), key=lambda i: training_history['valid_acc'][i])
    best_accuracy = training_history['valid_acc'][best_index]
    best_epoch = training_history['epoch'][best_index]
    best_step = training_history['step'][best_index]
    # best_accuracy = max(training_history['valid_acc'])
    # best_index = training_history['valid_acc'].index(best_accuracy)
    # best_epoch = (best_index + 1) * VAL_STEPS + START_EPOCH * steps_per_epoch
    train_accuracy = max(training_history['train_acc'])
    wandb.log({"best_acc": best_accuracy, "best_epoch": best_epoch, "best_step": best_step, "train_acc": train_accuracy})

# %%
import matplotlib.pyplot as plt
import pandas as pd

# try:
#     # Use pandas to read the CSV file into a DataFrame
#     # history_df = pd.read_csv(csv_file)
    
#     # print(f"✅ Successfully loaded data from '{csv_file}':")

#     # # Convert the list of dictionaries to a pandas DataFrame for easy plotting
#     history_df = pd.DataFrame(training_history)    

# except FileNotFoundError:
#     history_df = None
#     print(f"❌ Error: The file '{csv_file}' was not found.")
#     print("Please make sure you have run the training loop to save the file first.")
    
# First, ensure the training_history list is not empty
if history_df is None:
    print("Training history is empty. Please run the training loop first.")
else:
    # --- Create a single figure and axis for the plot ---
    fig, ax = plt.subplots(figsize=(12, 7))
    plt.title('Training and Validation Accuracy Over Epochs', fontsize=16)
    
    # --- Plot Training & Validation Accuracy ---
    ax.plot(history_df['step'], history_df['train_acc'], 's--', color='tab:green', label='Training Accuracy')
    ax.plot(history_df['step'], history_df['valid_acc'], '^-', color='tab:blue', label='Validation Accuracy')
    
    # --- Set labels and legend ---
    ax.set_xlabel('Steps')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Set the y-axis to be formatted as percentages
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.set_ylim(0, 1) # Set y-axis limits from 0 to 1 for accuracy

    # Set the x-axis to show integer epoch numbers
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout()
    plt.show()

# %%
if WANDB:
    wandb.finish()


