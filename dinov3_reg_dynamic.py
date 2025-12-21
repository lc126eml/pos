# %%
# =================================================================================
# Step 1: Install and Import Necessary Libraries
# =================================================================================
import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset,TensorDataset, DataLoader
from data.MultiScaleImageDataset import MultiScaleImageDataset, CustomImageDataset
from data.DynamicResolutionBatchSampler import DynamicResolutionBatchSampler

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

from core.utils import log_grads
# Enable faster matmul/conv kernels on Ampere+ without extra memory cost
# if torch.cuda.is_available():
#     torch.backends.cuda.matmul.allow_tf32 = True
#     torch.backends.cudnn.allow_tf32 = True
#     # Prefer faster matmul kernels when available (Torch 2.0+)
#     if hasattr(torch, "set_float32_matmul_precision"):
#         torch.set_float32_matmul_precision("high")
# %%
# =================================================================================
# Step 2: Configuration
# =================================================================================

# --- Dynamically set root directory ---
if os.path.exists('/home/sshuser'):
    root_dir = '/home/sshuser'
    BASE_PATH = f'{root_dir}/Data/imagenet100/'
elif os.path.exists('/lc'):
    root_dir = '/lc/logs'
    BASE_PATH = f'/lc/data/imagenet100/'
else:
    root_dir = '/linux'
    BASE_PATH = f'{root_dir}/Data/imagenet100/'
# --- Configuration via SimpleNamespace for easy interactive use ---
args = SimpleNamespace(
    # --- Model & Training Settings ---
    pos_type = None, #"alibi", # 'sin', 'alibi', 'relpos', None #,  'rpe', 'rope', 
    dynamic_img_size=True,
    model_type= "dinov3",
    use_abs_pos_emb=False,
    use_rot_pos_emb=False,
    model_size='small',
    num_classes=100,
    patch_size = 16,
    # Adjust based on your GPU memory. BATCH_SIZE = 120, 128, 136, 392, 768, etc.
    # batch_size=256, #rpe
    # batch_size=320, #rope
    batch_size=392, 
    # batch_size=512, 
    # ViT models have a fixed input size
    # img_sizes=[224, 192, 288],
    img_sizes=[224],
    val_img_sizes=[160, 176, 192, 208,224, 256, 272, 288, 320, 336, 352, 368, 384, 400, 416],
    # val_img_sizes=[224],
    lr=1e-3, #small
    # lr=5e-4, #base
    lr_aux=1e-5,
    eta_min=0.0,
    weight_decay=0.01,
    epochs=130,
    # has_pos=True, # Set to True or False directly
    overlap=0,
    pretrained=None,
    seed=55,
    use_patch_position_loss=False,
    use_rc_loss=True,
    # loss_type="smooth_l1", # "mse", "smooth_l1"
    # huber_beta=None,
    rc_alpha=400.0,
    warmup_steps_for_aux=1,
    workers=5,
    train=True,
    val=True,
    ckpt_path=None,
    lock=True,
    composite_lr=False,
    warmup_steps=20,
    clip_value=1.0,
    log_interval=50,
    csv_interval=3,
    compile_model=False,
    # --- Dataset Paths ---
    root_dir=root_dir,
)
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
offset = 0
MODEL_NAME = f"vit_{f'{args.pos_type}_' if args.pos_type is not None else ""}{args.model_size}_patch16_{args.model_type}"
output_dir = f"{args.root_dir}/output/imagenet{args.num_classes}redo_ablation3/{args.model_type}b{args.batch_size}s{args.seed}of{offset}dynamic{args.img_sizes}".replace(',', '_').replace('[', '_').replace(']', '_').replace(' ', '')
ckpt_output_dir = output_dir.replace("/output/", "/output/ckpt/")

# List of all the partial training directories
TRAIN_PATHS = [
    os.path.join(BASE_PATH, 'train.X1'),
    os.path.join(BASE_PATH, 'train.X2'),
    os.path.join(BASE_PATH, 'train.X3'),
    os.path.join(BASE_PATH, 'train.X4'),
]

VALID_PATH = os.path.join(BASE_PATH, 'val.X')
LABEL_PATH = os.path.join(BASE_PATH, 'Labels.json')

# --- Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

use_amp = torch.cuda.is_available()
use_bf16 = use_amp and torch.cuda.is_bf16_supported()
autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16

print(f"Using device: {DEVICE}", use_bf16, autocast_dtype)
# sys.exit(0)

#%%

if args.pos_type is not None:
    sys.path.append(r".")
    from timm_pe.eva_relpos import *
    from timm_pe.eva_alibi import *
    from timm_pe.eva_sin import *
    # from vision_transformer_rope import *
    # from vision_transformer_rope2d import *
    # from vision_transformer_rpe import *
    # from vision_transformer_relpos import *
    # from vision_transformer_alibi import *
    # from vision_transformer_sin import *

# %%
# timm.list_models("vit_*_dinov2")

# %%
# torch.backends.cudnn.deterministic=True
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
# if torch.cuda.is_available() and len(args.img_sizes) == 1:
#     torch.backends.cudnn.benchmark = True
subdir_name = (
    f"{f'{args.pos_type}_' if args.pos_type is not None else ""}{args.model_size}{f'_abs_pos' if args.use_abs_pos_emb else ""}{f'_rot_pos' if args.use_rot_pos_emb else ""}_overlap_{args.overlap}_"
    f"rc_{args.use_rc_loss}{'_patch_pos' if args.use_patch_position_loss else ''}_alpha_{int(args.rc_alpha)}lr{int(args.lr/1e-5)}"
).replace(',', '_').replace('[', '_').replace(']', '_').replace(' ', '')
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

logger.info("Cleaning up memory...")
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
logger.info("Memory cleanup complete.")

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

img_mean = [0.485, 0.456, 0.406]
img_std  = [0.229, 0.224, 0.225]


def make_train_transform(size: int):
    return T.Compose([
        T.RandomResizedCrop(size),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=img_mean, std=img_std),
    ])

size_to_transform = {
    s: make_train_transform(s) for s in args.img_sizes
}

def make_valid_transform(img_size):
    return transforms.Compose([
        transforms.Resize(size=int(img_size * 1.143)),
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
if len(args.img_sizes) == 1:
    train_dataset = CustomImageDataset(train_samples, transform=size_to_transform[args.img_sizes[0]])
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0),
        # **prefetch_kwargs,
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
        # **prefetch_kwargs,
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
if args.overlap > 0:
    # Customize patch embedding for overlap (e.g., patch_size=15, stride=14)
    original_patch_size = model.patch_embed.proj.kernel_size[0]
    new_patch_size = original_patch_size + args.overlap  # Or 15, 16, 17, etc., as desired
    stride = original_patch_size
    original_grid_size = args.img_sizes[0] // stride  # 16 for 224//14
    padding = ((original_grid_size - 1) * stride + new_patch_size - args.img_sizes[0] + 1) // 2  # +1 for ceiling effect; yields 1 for patch_size=15
    
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
if args.use_patch_position_loss:
    if len(args.img_sizes)==1:
        from core.patch_pos import PatchPositionRegressionCriterion
        position_loss = PatchPositionRegressionCriterion(
            feat_dim=model.embed_dim,
            num_classes=model.patch_embed.num_patches
        ).to(DEVICE)
    else:
        max_grid = max(args.img_sizes)//args.patch_size
        max_patch_count = max_grid * max_grid
        from core.patch_pos import PatchPositionRegressionCriterionDynamic
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

    total_steps = args.epochs * steps_per_epoch
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

    total_steps = args.epochs * steps_per_epoch
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
    scaler = torch.amp.GradScaler(enabled=use_amp)
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
            'epoch': [],
            'step': [],
        }
    step = 0
    best_acc = 0.0
    log_interval = getattr(args, "log_interval", 50)
    csv_interval = getattr(args, "csv_interval", 1) 
    for epoch in range(args.epochs):
        # --- Training Phase ---
        model.train()

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
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Training]")
        # train_pbar = train_loader
        if batch_sampler is not None:
            batch_sampler.set_epoch(epoch)
        
        # FP16: Use autocast for the forward pass
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            bs = inputs.size(0)
            
            optimizer.zero_grad(set_to_none=True)
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
            
            # FP16: Scale, backward, and step
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
            train_total += bs

            with torch.no_grad():
                pred = outputs.detach().argmax(dim=1)
                train_correct_t += (pred == labels).sum()

            # only log every N steps (minimize sync + formatting)
            if (step + 1) % log_interval == 0:
                # now pay the sync cost, but only occasionally
                avg_loss = (running_loss_t / train_total).float().item()
                avg_acc = (train_correct_t / train_total).float().item()
                if aux_loss is not None:
                    avg_aux = (aux_loss_sum_t / train_total).float().item()
                    train_pbar.set_postfix_str(f"loss={avg_loss:.4f} acc={avg_acc:.3f} aux={avg_aux:.4f}")
                else:
                    train_pbar.set_postfix_str(f"loss={avg_loss:.4f} acc={avg_acc:.3f}")

            step += 1

            # if (step + 1) % VAL_STEPS == 0:
        # --- Validation Phase ---
        model.eval()
        val_correct_t = torch.zeros((), device=DEVICE)
        val_total = 0
        # val_pbar = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Validation]")
        
        with torch.inference_mode():
            for inputs, labels in valid_loader:
                inputs = inputs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                with torch.amp.autocast(device_type=DEVICE.type, dtype=autocast_dtype, enabled=use_amp):
                    outputs = model(inputs)
                pred = outputs.argmax(dim=1)
                val_correct_t += (pred == labels).sum()
                val_total += labels.size(0)

        epoch_val_acc = (val_correct_t / val_total).item()
        if best_acc < epoch_val_acc:
            best_acc = epoch_val_acc

        epoch_train_acc  = (train_correct_t / train_total).item()
        epoch_train_loss = (running_loss_t / train_total).item()
        logger.info(f"\nEpoch {epoch+1}/{args.epochs} Summary:")
        logger.info(f"\nStep {step} Summary:")

        if aux_loss is not None:
            epoch_aux_loss   = (aux_loss_sum_t / train_total).item()
            epoch_base_loss  = (base_loss_t / train_total).item()
            training_history['aux_loss'].append(epoch_aux_loss)
            training_history['base_loss'].append(epoch_base_loss)
            logger.info(f"  Train Loss: {epoch_train_loss:.4f} | Aux Loss: {epoch_aux_loss:.4f} | Base Loss: {epoch_base_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | Valid Acc: {epoch_val_acc:.4f}\n")
        else:
            logger.info(f"  Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | Valid Acc: {epoch_val_acc:.4f}\n")
        

        # ✅ Append the results to the correct lists within the dictionary
        
        training_history['train_loss'].append(epoch_train_loss)
        training_history['train_acc'].append(epoch_train_acc)
        training_history['valid_acc'].append(epoch_val_acc)  
        training_history['epoch'].append(epoch+1)
        training_history['step'].append(step+1)
        if (epoch + 1) % csv_interval == 0:
            pd.DataFrame(training_history).to_csv(os.path.join(output_dir, f'{subdir_name}.csv'), index=False)
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
    # Save the model's state dictionary
    ckpt_path = os.path.join(ckpt_output_dir,  f'{subdir_name}{MODEL_NAME}_final.pth')
    torch.save(model.state_dict(), ckpt_path)
    logger.info(f"✅ Model saved to '{ckpt_path}'")

if args.val:    
    val_results = {
        'img_size': [],
        'valid_acc': []
    }

    if ckpt_path is None:
        ckpt_path = f"{args.root_dir}/{args.ckpt_path}"
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
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
            persistent_workers=(args.workers > 0),
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

del model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

if gpu_lock and gpu_lock.is_locked:
    logger.info("Manually releasing lock.")
    gpu_lock.release()
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
