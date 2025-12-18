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

# --- Dynamically set root directory ---
if os.path.exists('/ubuntu'):
    root_dir = '/ubuntu'
elif os.path.exists('/home/sshuser'):
    root_dir = '/home/sshuser'
else:
    root_dir = '/linux'
# --- Configuration via SimpleNamespace for easy interactive use ---
args = SimpleNamespace(
    # --- Model & Training Settings ---
    pos_type = None, # 'sin', 'alibi', 'relpos',  'rpe', 'rope', None, 
    model_type= "dinov3",
    use_abs_pos_emb=True,
    use_rot_pos_emb=False,
    model_size='small',
    num_classes=100,
    patch_size = 16,
    # Adjust based on your GPU memory. BATCH_SIZE = 528, etc.
    batch_size=512,
    # ViT models have a fixed input size
    img_size=32,
    lr=1e-3,
    epochs=30,
    has_pos=False, # Set to True or False directly
    overlap=0,
    pretrained=None,
    seed=55,
    wdecay=0.1,
    use_patch_position_loss=False,
    use_rc_loss=True,
    loss_type="l1",
    huber_beta=0.1,
    rc_alpha=50.0,
    workers=5,
    train=True,
    val=False,
    ckpt_path="",
    lock=False,
    clip_value=1.0,

    # --- Dataset Paths ---
    root_dir=root_dir,
)

MODEL_NAME = f"vit_{f'{args.pos_type}_' if args.pos_type is not None else ""}{args.model_size}_patch16_{args.model_type}"
output_dir = f"{args.root_dir}/output/cifar100redo/{args.model_size}b{args.batch_size}s{args.seed}_{args.img_size}"
BASE_PATH = f'{args.root_dir}/Data/cifar100/'

# --- Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16

# %%
# torch.backends.cudnn.deterministic=True
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
subdir_name = (
    f"{f'{args.pos_type}_' if args.pos_type is not None else ""}{args.model_size}{f'_abs_pos' if args.use_abs_pos_emb else ""}{f'_rot_pos' if args.use_rot_pos_emb else ""}_overlap_{args.overlap}_"
    f"rc_{args.use_rc_loss}{'_patch_pos' if args.use_patch_position_loss else ''}_alpha_{int(args.rc_alpha)}lr{int(args.lr/1e-5)}"
).replace(',', '_').replace('[', '_').replace(']', '_').replace(' ', '')
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
if torch.cuda.is_available():
    torch.cuda.empty_cache()
logger.info("Memory cleanup complete.")

logger.info(args)
# %%
def unpickle(file):
    """
    A function to load the CIFAR-100 data from a pickled file.
    """
    with open(file, 'rb') as fo:
        # The 'latin1' encoding is required for compatibility with the original dataset files.
        dict = pickle.load(fo, encoding='latin1')
    return dict

# --- Step 1: Load the data from the files ---
# Adjust these paths to where you have saved the dataset. In a Kaggle environment,
# this path is typically f'{BASE_PATH}/'.
try:
    train_dict = unpickle(f'{BASE_PATH}/train')
    test_dict = unpickle(f'{BASE_PATH}/test')
    meta_dict = unpickle(f'{BASE_PATH}/meta')
except FileNotFoundError:
    logger.info("Please adjust the file paths to point to your local CIFAR-100 directory.")
    # Use dummy dicts to allow the rest of the code to be checked
    train_dict = {'data': np.zeros((1, 3072)), 'fine_labels': [0]}
    test_dict = {'data': np.zeros((1, 3072)), 'fine_labels': [0]}

# %%
from PIL import Image
# --- Step 2: Define a Custom Dataset Class ---
class CustomCIFAR100(Dataset):
    def __init__(self, data_dict, transform=None):
        self.data = data_dict['data'].reshape(-1, 3, 32, 32)
        # Transpose from (N, 3, 32, 32) to (N, 32, 32, 3) for PIL
        self.data = self.data.transpose((0, 2, 3, 1))
        self.labels = data_dict['fine_labels']
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Get image and label at the index
        image = self.data[idx]
        label = self.labels[idx]

        # Convert numpy array to PIL Image
        image = Image.fromarray(image)

        # ✨ Apply transforms here!
        if self.transform:
            image = self.transform(image)

        return image, label

# --- Step 3: Define Your Transformations ---
# Now you can include data augmentation
train_transforms = transforms.Compose([
    transforms.RandomCrop(args.img_size, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

# --- Step 4: Create Datasets and DataLoaders ---
train_dataset = CustomCIFAR100(data_dict=train_dict, transform=train_transforms)
test_dataset = CustomCIFAR100(data_dict=test_dict, transform=test_transforms)

logger.info(f"Total training images ({args.num_classes} classes): {len(train_dataset)}")
logger.info(f"Total validation images ({args.num_classes} classes): {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=args.batch_size,num_workers=args.workers, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size,num_workers=2, shuffle=False)

# Now it works as expected!
logger.info("Successfully created DataLoaders with transforms.")
# images, labels = next(iter(train_loader))
# logger.info(f"Batch of images shape: {images.shape}")

# %%
# print(train_dict.keys(), test_dict.keys(), meta_dict.keys())
# %% [code]
# =================================================================================
# Step 3.5: Visualize a Batch of Training Data
# =================================================================================
import matplotlib.pyplot as plt
import numpy as np
import torchvision

img_mean, img_std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
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
#     class_names = meta_dict['fine_label_names']

#     # Create a grid of images
#     fig = plt.figure(figsize=(16, 8))
#     plt.suptitle("Sample Images from CIFAR-100 Dataset", fontsize=16)
    
#     # Display the first 16 images from the batch
#     for i in range(16):
#         ax = plt.subplot(4, 8, i + 1)
#         class_name = class_names[classes[i]]
#         imshow(inputs[i], title=class_name)
        
#     plt.tight_layout(rect=[0, 0, 1, 0.96])
#     plt.show()

# except NameError as e:
#     logger.info(e, "Could not display images. Please ensure the previous cells have been run to create 'train_loader'.")

# %%
# MODEL_NAME = "vit_rope_small_patch14_dinov2"
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
    img_size=args.img_size,
).to(DEVICE)

# feature_layers = [2, 5, 8, 11]
# dummy_input = torch.randn(2, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
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
    # grid_size_h = ((IMG_SIZE + 2 * padding - new_patch_size) // stride) + 1
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

if args.pretrained is not None:
    state_dicts = torch.load(args.pretrained, map_location=DEVICE)
    IncompatibleKeys = model.load_state_dict(state_dicts)
    logger.info(IncompatibleKeys)
# %%
dynamic = True
training_parameters = list(model.parameters()) 
if args.use_rc_loss:
    if len(args.img_sizes)==1:
        grid_h, grid_w = model.patch_embed.grid_size
        dynamic = False
        from core.patch_pos import PatchRowColRegressionCriterionFast
        rowcol_loss = PatchRowColRegressionCriterionFast(
            feat_dim=model.embed_dim,
            grid_h=grid_h,
            grid_w=grid_w,
            loss_type=args.loss_type,
            huber_beta=args.huber_beta,
        ).to(DEVICE)
    else:
        grid_h = grid_w = max(args.img_sizes)//args.patch_size
        from core.patch_pos import PatchRowColRegressionCriterionDynamicFast
        rowcol_loss = PatchRowColRegressionCriterionDynamicFast(
            feat_dim=model.embed_dim,
            grid_h=grid_h,
            grid_w=grid_w,
            loss_type=args.loss_type,
            huber_beta=args.huber_beta,
        ).to(DEVICE)
    training_parameters += list(rowcol_loss.parameters())
if args.use_patch_position_loss:
    from core.patch_pos import PatchPositionCriterion
    position_loss = PatchPositionCriterion(
        feat_dim=model.embed_dim,
        num_classes=model.patch_embed.num_patches
    ).to(DEVICE)
    training_parameters += list(position_loss.parameters())
# --- Loss Function & Optimizer ---
criterion = nn.CrossEntropyLoss()
# optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
logger.info("✅ Model, Loss Function, and Optimizer are ready.")

# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
# logger.info("✅ Model, Loss, Optimizer, and LR Scheduler are ready.")
steps_per_epoch = len(train_loader)
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


# %%
import csv

ckpt_path = None
if args.train:
    # FP16: Initialize the Gradient Scaler
    scaler = torch.amp.GradScaler('cuda')
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

    for epoch in range(args.epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        base_loss = 0.0
        aux_loss_sum = 0.0
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
                    base_loss += loss.item() * inputs.size(0)
                    aux_loss = rowcol_loss(feats[:, model.num_prefix_tokens:, :])
                    # logger.info(loss, aux_loss)
                    aux_loss_sum += aux_loss.item() * inputs.size(0)
                    loss = loss + args.rc_alpha * aux_loss
                
                if args.use_patch_position_loss:
                    aux_loss = position_loss(feats[:, model.num_prefix_tokens:, :])
                    # logger.info(f"{loss}, {aux_loss}")
                    loss = loss + args.rc_alpha * aux_loss
            
            # FP16: Scale, backward, and step
            scaler.scale(loss).backward()

            if args.clip_value is not None:
                scaler.unscale_(optimizer)
                clip_value = args.clip_value  # typical default for Transformer/ViT-style training
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)

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
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                with torch.amp.autocast('cuda', dtype=autocast_dtype):
                    outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        epoch_val_acc = val_correct / val_total
        if best_acc < epoch_val_acc:
            best_acc = epoch_val_acc

        epoch_train_acc = train_correct / train_total
        epoch_train_loss = running_loss / len(train_loader.dataset)
        if args.use_rc_loss:
            epoch_aux_loss = aux_loss_sum / len(train_loader.dataset)
            epoch_base_loss = base_loss / len(train_loader.dataset)
            training_history['aux_loss'].append(epoch_aux_loss)
            training_history['base_loss'].append(epoch_base_loss)
        
        logger.info(f"\nEpoch {epoch+1}/{args.epochs} Summary:")
        logger.info(f"\nStep {step} Summary:")
        if args.use_rc_loss:
            logger.info(f"  Train Loss: {epoch_train_loss:.4f} | Aux Loss: {epoch_aux_loss:.4f} | Base Loss: {epoch_base_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | Valid Acc: {epoch_val_acc:.4f}\n")
        else:
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
    logger.info(f"Best Accuracy: {best_acc:.4f}")
    logger.info(output_dir)
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
    # ckpt_path = os.path.join(output_dir, f'{MODEL_NAME}_final.pth')
    # torch.save(model.state_dict(), ckpt_path)
    # logger.info(f"✅ Model saved to '{ckpt_path}'")


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
