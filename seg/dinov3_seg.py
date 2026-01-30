# %%
# =================================================================================
# Step 1: Install and Import Necessary Libraries
# =================================================================================
import math
import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
import pandas as pd
import numpy as np
import random
from PIL import Image
from torch.nn import functional as F
from types import SimpleNamespace
import gc
import logging
from seg.seg_aug import TrainSegAug, EvalSegPreprocess
from seg.seg_head import PPMliteFCNHead, FCNSegHead, LinearSegHead, UPerNetTokenHead
from core.priority_lock import PriorityLock
# Enable faster matmul/conv kernels on Ampere+ without extra memory cost
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Prefer faster matmul kernels when available (Torch 2.0+)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
# %%
# =================================================================================
# Step 2: Configuration
# =================================================================================
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

LOCAL_TIMM = os.environ.get("LOCAL_TIMM_DIR", "/home/liucong/codes/pos/timm/pytorch-image-models-main")
if os.path.isdir(LOCAL_TIMM):
    sys.path.insert(0, LOCAL_TIMM)

import timm

sys.path.append("/lc/code/pos")
# --- Dynamically set root directory ---
if os.path.exists('/home/sshuser'):
    root_dir = '/home/sshuser'
    BASE_PATH = f'{root_dir}/Data/ADEChallengeData2016'
elif os.path.exists('/lc'):
    root_dir = '/lc/logs'
    BASE_PATH = f'/ubuntu/Data/ADEChallengeData2016'
elif os.path.exists("/home/liucong"):
    root_dir = '/home/liucong/codes/pos/seglogs'
    BASE_PATH = f'/home/liucong/data/ADEChallengeData2016'
else:
    root_dir = '/linux'
    BASE_PATH = f'{root_dir}/Data/ADEChallengeData2016'
root_dir = os.environ.get("OUTPUT_ROOT", os.path.join(REPO_ROOT, "outputs"))
data_root_default = BASE_PATH
base_path_default = data_root_default
# --- Configuration via SimpleNamespace for easy interactive use ---
args = SimpleNamespace(
    # --- Model & Training Settings ---
    model_type= "dinov3",
    use_abs_pos_emb=False,
    use_rot_pos_emb=True,
    model_size='base',
    num_classes=150,  # For ADE20K
    batch_size=16,
    grad_accum_steps=4,
    # batch_size=80,
    train_img_size=512,
    eval_img_size=512,
    use_ms_flip_eval=False,  # False or True
    scale_jitter=(1.0, None),
    use_cat_max_ratio=False,
    cat_max_ratio=0.75,
    cat_max_ratio_tries=10,
    ms_scales=(0.75, 1.0, 1.25, 1.5),  # e.g. (0.75, 1.0, 1.25, 1.5)
    eval_crop_mode="crop_or_pad",  # "crop_or_pad" (best practice), "pad", or "crop"
    final_ms_flip_eval=True,
    # lr=3e-4,
    # lr=8e-4,
    lr=7e-4,
    lr_aux=1e-5,
    # lr=1e-3,
    # lr=8e-4,
    eta_min=1e-8,  # cosine LR floor; 1e-8 is a common stable default
    composite_lr=True,
    warmup_steps=3000,
    weight_decay=0.01,  # typical 0.01-0.05 for ViT+AdamW
    epochs=130,
    # has_pos=True,
    overlap=0,
    start_epoch=0,
    seed=55,
    use_rc_loss=False,
    # loss_type="l1",
    huber_beta=0.1,
    rc_alpha=30.0,
    seg_head="ppmlite",  # "ppmlite", "upernet", "fcn", "linear"
    feature_layers=[2, 5, 8, 11],
    # dice_weight=0.0,
    workers=5,
    color_jitter={"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.05},
    color_jitter_prob=0.1,
    train=True,
    # val=False,
    ckpt_path=None,
    lock=True,
    lock_prio=6,
    clip_value=1.0,
    output_dir=os.path.join(root_dir, "seg"),
    log_interval=50,
    csv_interval=3,
    save_ckpt=True,
    compile_model=False,
    save_full_ckpt=True,
    resume_full_ckpt=False,
    resume_ckpt_path=None,
    resume_scheduler=True,
    resume_optimizer=True,
    resume_bs=False,
    total_run_time_hr=None,
    # --- Dataset Paths ---
    base_path=base_path_default,
)
ckpt = None
if args.resume_full_ckpt and args.resume_ckpt_path:
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
    # args.use_patch_position_loss=False
    args.use_rc_loss = False
if args.eval_img_size != args.train_img_size:
    print("Best practice is to keep eval_img_size == train_img_size; overriding.", flush=True)
    args.eval_img_size = args.train_img_size
if hasattr(args, "seg_head"):
    args.seg_head = str(args.seg_head).lower()

MODEL_NAME = f"vit_{args.model_size}_patch16_{args.model_type}"
TRAIN_IMAGE_PATH = os.path.join(args.base_path, 'images', 'training')
TRAIN_ANNOTATION_PATH = os.path.join(args.base_path, 'annotations', 'training')
VALID_IMAGE_PATH = os.path.join(args.base_path, 'images', 'validation')
VALID_ANNOTATION_PATH = os.path.join(args.base_path, 'annotations', 'validation')

# --- Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

use_amp = torch.cuda.is_available()
use_bf16 = use_amp and torch.cuda.is_bf16_supported(including_emulation=False)
autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
# Speed tweaks for Ampere+
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
# %%
# Set seeds
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
    f"{'_rot_pos' if args.use_rot_pos_emb else ''}_rc_{args.use_rc_loss}_lr{int(args.lr/1e-5)}"
)
if args.use_rc_loss:
    subdir_name += f"_overlap_{args.overlap}_alpha_{int(args.rc_alpha)}"

output_dir = os.path.join(args.output_dir, subdir_name)
ckpt_output_dir = os.path.join(output_dir, "ckpt")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(ckpt_output_dir, exist_ok=True)
last_ckpt_path = os.path.join(ckpt_output_dir, "last.pth")
if args.resume_full_ckpt and args.resume_ckpt_path is None:
    args.resume_ckpt_path = last_ckpt_path

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
logger.info(f"Using mixed precision: {'disabled' if not use_amp else ('bfloat16' if use_bf16 else 'float16')}")
logger.info(f"Arguments: {args}")
logger.info(output_dir)
logger.info(subdir_name)
# wait_for_python_gpu_processes(poll_interval_minutes=5)
# --- Acquire a file lock to ensure exclusive GPU usage ---
gpu_lock = None
if args.lock:
    lock_path = "/tmp/gpu.lock"
    lock_priority = int(os.environ.get("GPU_LOCK_PRIORITY", "10"))
    gpu_lock = PriorityLock(lock_dir=lock_path, priority=lock_priority)
    print(f"Attempting to acquire lock on '{lock_path}' (priority={lock_priority})...")
    gpu_lock.acquire()
    print("Lock acquired. It is safe to proceed.")

logger.info(output_dir)
# %%
# %%
# =================================================================================
# Custom Dataset for Segmentation (Image + Mask)
# =================================================================================
class SegmentationDataset(Dataset):
    """
    Custom PyTorch Dataset for semantic segmentation.

    Reads images and their corresponding segmentation masks, and applies
    appropriate data augmentation for training and validation.
    """
    def __init__(self, image_dir, annotation_dir, pair_transform):
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
        self.pair_transform = pair_transform

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

        image_t, mask_t = self.pair_transform(image, mask)
        mask_t = mask_t.long() - 1  # ADE20K labels are 1-150, background 0
        return image_t, mask_t
#%%
img_mean = [0.485, 0.456, 0.406]
img_std = [0.229, 0.224, 0.225]

# --- Create Datasets and DataLoaders ---
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
logger.info(f"✅ DataLoaders created successfully.")
logger.info(f"   - Training samples: {len(train_dataset)}, Batches per epoch: {len(train_loader)}")
logger.info(f"   - Validation samples: {len(valid_dataset)}, Batches per epoch: {len(valid_loader)}")
#%%
# =================================================================================
# Step 3.5: Visualize a Batch of Training Data
# =================================================================================
# def imshow(img, mask, title=None):
#     """Display an image and its mask side by side."""
#     img = img.numpy().transpose((1, 2, 0))
#     mean = np.array(img_mean)
#     std = np.array(img_std)
#     img = std * img + mean
#     img = np.clip(img, 0, 1)
    
#     mask = mask.numpy().squeeze()  # Remove channel dimension
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
#     ax1.imshow(img)
#     ax1.set_title('Image' if title is None else f'Image - {title}')
#     ax1.axis('off')
#     ax2.imshow(mask, cmap='jet')  # Use jet colormap for mask visualization
#     ax2.set_title('Mask' if title is None else f'Mask - {title}')
#     ax2.axis('off')
#     plt.show()

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

# --- Test with a dummy input ---
# dummy_input = torch.randn(2, 3, args.train_img_size, args.train_img_size).to(DEVICE)
# with torch.no_grad():
#     feats = model.forward_features(dummy_input)
#     output = decoder(feats[:, model.num_prefix_tokens:, :])

# logger.info(f"Model created successfully!")
# logger.info(f"Input shape: {dummy_input.shape}")
# logger.info(f"Output shape: {output.shape}")
# assert output.shape == (2, args.num_classes, args.train_img_size, args.train_img_size)
# logger.info("✅ Output shape is correct.")
# del feats, output, dummy_input

# %%
logger.info(f'model.patch_embed.proj {model.patch_embed.proj}')
if args.overlap > 0:
    # Customize patch embedding for overlap (e.g., patch_size=15, stride=14)
    original_patch_size = model.patch_embed.proj.kernel_size[0]
    new_patch_size = original_patch_size + args.overlap  # Or 15, 16, 17, etc., as desired
    stride = original_patch_size
    original_grid_size = args.train_img_size // stride  # 16 for 224//14
    padding = ((original_grid_size - 1) * stride + new_patch_size - args.train_img_size + 1) // 2  # +1 for ceiling effect; yields 1 for patch_size=15
    
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
    # grid_size_h = ((args.train_img_size + 2 * padding - new_patch_size) // stride) + 1
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

if args.compile_model:
    if hasattr(torch, "compile"):
        logger.info("Compiling model with torch.compile (mode='reduce-overhead').")
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
        decoder = torch.compile(decoder)
    else:
        logger.warning("torch.compile not available; skipping compilation.")

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
logger.info("✅ Initialized Loss, Optimizer, and LR Scheduler.")

# %%
# dummy_input = torch.randn(2, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
# with torch.no_grad():
#     feats = model.forward_features(dummy_input)
# logger.info(f"Model created successfully!")
# logger.info(f"Input shape: {dummy_input.shape}")
# logger.info(f"Output shape: {feats.shape}") 

# %%


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
    """
    pred/target: (B, H, W) or any same-shape tensors.
    Returns: confmat [C, C] on pred.device
    """
    pred = pred.view(-1).to(torch.int64)
    target = target.view(-1).to(torch.int64)

    valid = target != ignore_index
    pred = pred[valid]
    target = target[valid]

    # idx = target * C + pred
    idx = target * num_classes + pred
    conf = torch.bincount(idx, minlength=num_classes * num_classes)
    return conf.view(num_classes, num_classes)

#%%
ckpt_path = None
if args.train:
    # FP16: Initialize the Gradient Scaler
    use_scaler = use_amp and (autocast_dtype == torch.float16)
    scaler = torch.amp.GradScaler(DEVICE.type, enabled=use_scaler)

    # =================================================================================
    # Step 5: Training and Validation Loop
    # =================================================================================
    logger.info(f"\n🚀 Starting training for {MODEL_NAME}...")
    train_start_time = time.time()
    start_epoch = 0
    if args.resume_full_ckpt and args.resume_ckpt_path:
        if ckpt is not None:
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
            if "scaler" in ckpt and ckpt["scaler"] is not None:
                scaler.load_state_dict(ckpt["scaler"])
            if args.use_rc_loss and "rowcol_loss" in ckpt and ckpt["rowcol_loss"] is not None:
                for k in ["row_targets", "col_targets", "row_index_full", "col_index_full"]:
                    if k in ckpt["rowcol_loss"]:
                        ckpt["rowcol_loss"].pop(k)
                rowcol_loss.load_state_dict(ckpt["rowcol_loss"])
            logger.info(f"Resumed full checkpoint from '{args.resume_ckpt_path}' at epoch {start_epoch}")
            training_history = ckpt.get("training_history", None)

    # ✅ Initialize training_history as a dictionary of lists
    if not isinstance(locals().get("training_history", None), dict):
        if args.use_rc_loss:
            training_history = {
                'train_loss': [],
                'train_acc': [],
                'valid_acc': [],
                'valid_miou': [],
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
                'valid_miou': [],
                'train_time': [],
                'val_time': [],
                'epoch': [],
                'step': [],
            }
    if isinstance(locals().get("training_history", None), dict):
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
    def _append_eval_log(log_path, row):
        if not row:
            return
        df = pd.DataFrame([row])
        header = not os.path.exists(log_path)
        df.to_csv(log_path, mode="a", header=header, index=False)
    if args.resume_full_ckpt and isinstance(locals().get("training_history", None), dict):
        _pad_history(training_history)
    step = 0
    if isinstance(locals().get("training_history", None), dict):
        step = int(training_history.get("step", [0])[-1]) if training_history.get("step") else 0
    best_acc = 0.0
    log_interval = getattr(args, "log_interval", 50)
    csv_interval = getattr(args, "csv_interval", 1) 
    last_trained_epoch = int(start_epoch)
    for epoch in range(start_epoch, args.epochs):
        epoch_train_start = time.time()
        # --- Training Phase ---
        model.train()
        decoder.train()
        running_loss_t = torch.zeros((), device=DEVICE)
        base_loss_t    = torch.zeros((), device=DEVICE)
        aux_loss_sum_t = torch.zeros((), device=DEVICE)
        train_correct_t = torch.zeros((), device=DEVICE)
        train_total_t   = torch.zeros((), device=DEVICE)
        train_samples_t = torch.zeros((), device=DEVICE)
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Training]", mininterval=0.5)
        # train_pbar = train_loader
        
        # FP16: Use autocast for the forward pass
        optimizer.zero_grad(set_to_none=True)
        total_batches = len(train_loader)
        for batch_idx, (inputs, labels) in enumerate(train_pbar):
            if (batch_idx % accum_steps) == 0 and (total_batches - batch_idx) < accum_steps:
                break
            inputs = inputs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            bs = inputs.size(0)
            aux_loss = None
            with torch.amp.autocast(device_type=DEVICE.type, dtype=autocast_dtype, enabled=use_amp):
                if args.seg_head == "upernet":
                    outputs, features, grid_hw = _forward_upernet(
                        model, decoder, inputs, args.feature_layers
                    )
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
                    # logger.info(loss, aux_loss)
                    loss = base_loss + args.rc_alpha * aux_loss
            
            # FP16: Scale, backward, and step (with grad accumulation)
            loss_scaled = loss / accum_steps
            scaler.scale(loss_scaled).backward()

            do_step = ((batch_idx + 1) % accum_steps == 0) or (batch_idx + 1 == len(train_loader))
            if do_step:
                if args.clip_value is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(training_parameters, max_norm=args.clip_value)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            
            with torch.no_grad():
                pred = outputs.detach().argmax(dim=1)  # (B,H,W)
                mask = (labels >= 0)
                valid_pixels = mask.sum()
                train_correct_t += ((pred == labels) & mask).sum()
                train_total_t   += valid_pixels
                train_samples_t += bs

            running_loss_t += loss.detach() * valid_pixels
            if args.use_rc_loss:
                base_loss_t += base_loss.detach() * valid_pixels

            # Throttled logging/postfix: only sync every log_interval
            if (step + 1) % log_interval == 0:
                avg_loss = (running_loss_t / (train_total_t.clamp_min(1))).float().item()
                avg_acc  = (train_correct_t / train_total_t.clamp_min(1)).float().item()

                if args.use_rc_loss:
                    avg_aux  = (aux_loss_sum_t / train_samples_t.clamp_min(1)).float().item()
                    train_pbar.set_postfix_str(f"loss={avg_loss:.4f} acc={avg_acc:.3f} aux={avg_aux:.4f}")
                else:
                    train_pbar.set_postfix_str(f"loss={avg_loss:.4f} acc={avg_acc:.3f}")

            step += 1

            # if (step) % VAL_STEPS == 0:
        
        train_time = time.time() - epoch_train_start
        model.eval()
        decoder.eval()
        val_correct_t = torch.zeros((), device=DEVICE)
        val_total_t   = torch.zeros((), device=DEVICE)
        confmat = torch.zeros((args.num_classes, args.num_classes), device=DEVICE, dtype=torch.int64)

        val_pbar = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Validation]", mininterval=0.5)
        val_start = time.time()

        with torch.inference_mode():
            for inputs, labels in val_pbar:
                inputs = inputs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                with torch.amp.autocast(device_type=DEVICE.type, dtype=autocast_dtype, enabled=use_amp):
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
                            outputs, _, grid_hw = _forward_upernet(
                                model, decoder, inputs, args.feature_layers
                            )
                        else:
                            feats = model.forward_features(inputs)
                            grid_hw = _infer_grid_hw(model, inputs)
                            outputs = decoder(
                                feats[:, model.num_prefix_tokens:, :],
                                grid_size=grid_hw,
                                out_size=inputs.shape[-2:],
                            )

                pred = outputs.argmax(dim=1)  # (B,H,W)
                mask = (labels >= 0)

                val_correct_t += ((pred == labels) & mask).sum()
                val_total_t   += mask.sum()

                # mIoU via confusion matrix (vectorized)
                confmat += fast_confusion_matrix(pred, labels, args.num_classes, ignore_index=-1)

        val_time = time.time() - val_start
        # Compute IoU from confmat
        confmat_f = confmat.to(torch.float32)
        intersection = torch.diag(confmat_f)
        union = confmat_f.sum(dim=1) + confmat_f.sum(dim=0) - intersection
        valid = union > 0
        epoch_val_miou = (intersection[valid] / union[valid]).mean().item() if valid.any() else 0.0

        epoch_val_acc = (val_correct_t / val_total_t.clamp_min(1)).float().item()
        epoch_train_acc = (train_correct_t / train_total_t.clamp_min(1)).float().item()

        # Epoch losses (single sync each)
        # If you want dataset-normalized loss like before: divide by len(train_loader.dataset)
        denom_pixels = train_total_t.clamp_min(1).float()
        denom_samples = train_samples_t.clamp_min(1).float()
        epoch_train_loss = (running_loss_t / denom_pixels).float().item()
        if best_acc < epoch_val_acc:
            best_acc = epoch_val_acc

        logger.info(f"\nEpoch {epoch+1+args.start_epoch}/{args.epochs} Summary:")
        logger.info(f"Step {step} Summary:")

        if args.use_rc_loss:
            epoch_aux_loss  = (aux_loss_sum_t / denom_samples).float().item()
            epoch_base_loss = (base_loss_t / denom_pixels).float().item()
            logger.info(
                f"  Train Loss: {epoch_train_loss:.4f} | Aux Loss: {epoch_aux_loss:.4f} | Base Loss: {epoch_base_loss:.4f} | "
                f"Train Acc: {epoch_train_acc:.4f} | Valid Acc: {epoch_val_acc:.4f} | Valid mIoU: {epoch_val_miou:.4f} | "
                f"train_time: {train_time:.1f}s | val_time: {val_time:.1f}s\n"
            )
            training_history["aux_loss"].append(epoch_aux_loss)
            training_history["base_loss"].append(epoch_base_loss)
        else:
            logger.info(
                f"  Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | "
                f"Valid Acc: {epoch_val_acc:.4f} | Valid mIoU: {epoch_val_miou:.4f} | "
                f"train_time: {train_time:.1f}s | val_time: {val_time:.1f}s\n"
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
            pd.DataFrame(training_history).to_csv(os.path.join(output_dir, f'{subdir_name}.csv'), index=False)
        if args.save_full_ckpt:
            ckpt = {
                "epoch": epoch + 1,
                "step": step,
                "model": model.state_dict(),
                "decoder": decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "scaler": scaler.state_dict() if scaler is not None else None,
                "rowcol_loss": rowcol_loss.state_dict() if args.use_rc_loss else None,
                "training_history": training_history,
                "args": args,
            }
            torch.save(ckpt, last_ckpt_path)
            logger.info(f"Saved full checkpoint to '{last_ckpt_path}'")
        if args.total_run_time_hr is not None:
            elapsed = time.time() - train_start_time
            max_run_time_sec = args.total_run_time_hr * 3600
            if elapsed >= max_run_time_sec:
                logger.info(
                    "Stopping training: elapsed %.0fs reached limit %.2fh.",
                    elapsed,
                    args.total_run_time_hr,
                )
                break
        # gc.collect()
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()
        # if epoch == 3:
        #     break


    logger.info("🏁 Training complete.")
    logger.info(f"Best Accuracy: {best_acc:.4f}")
    logger.info(output_dir)
    if args.final_ms_flip_eval and not args.use_ms_flip_eval:
        logger.info("Running final multi-scale + flip evaluation...")
        final_eval_row = {
            "run_tag": run_tag,
            "subdir_name": subdir_name,
            "output_dir": output_dir,
            "epoch": int(last_trained_epoch),
        }
        model.eval()
        decoder.eval()
        val_correct_t = torch.zeros((), device=DEVICE)
        val_total_t   = torch.zeros((), device=DEVICE)
        confmat = torch.zeros((args.num_classes, args.num_classes), device=DEVICE, dtype=torch.int64)

        val_pbar = tqdm(valid_loader, desc="Final MS+Flip [Validation]", mininterval=0.5)
        with torch.inference_mode():
            for inputs, labels in val_pbar:
                inputs = inputs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                with torch.amp.autocast(device_type=DEVICE.type, dtype=autocast_dtype, enabled=use_amp):
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
                val_total_t   += mask.sum()
                confmat += fast_confusion_matrix(pred, labels, args.num_classes, ignore_index=-1)

        confmat_f = confmat.to(torch.float32)
        intersection = torch.diag(confmat_f)
        union = confmat_f.sum(dim=1) + confmat_f.sum(dim=0) - intersection
        valid = union > 0
        final_ms_miou = (intersection[valid] / union[valid]).mean().item() if valid.any() else 0.0
        final_ms_acc = (val_correct_t / val_total_t.clamp_min(1)).float().item()
        logger.info(f"Final MS+Flip Acc: {final_ms_acc:.4f} | Final MS+Flip mIoU: {final_ms_miou:.4f}")
        final_eval_row["final_ms_flip_acc"] = final_ms_acc
        final_eval_row["final_ms_flip_miou"] = final_ms_miou
        final_eval_log = os.path.join(output_dir, "final_eval_log.csv")
        _append_eval_log(final_eval_log, final_eval_row)
    if args.lock and gpu_lock and gpu_lock.is_locked:
        logger.info("Manually releasing lock.")
        gpu_lock.release()

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

del model, decoder
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

if args.lock and gpu_lock and gpu_lock.is_locked:
    logger.info("Manually releasing lock.")
    gpu_lock.release()
