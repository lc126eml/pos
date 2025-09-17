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
import logging
import cv2
import h5py
from scipy import ndimage
import warnings
from typing import List, Tuple, Union
import argparse

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
    parser.add_argument('--epochs', type=int, default=100, help="Epochs.")
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

logger.info(f"Arguments: {args}")
logger.info(f"Using device: {DEVICE}")
logger.info(f"Using mixed precision: {'bfloat16' if use_bf16 else 'float16'}")

# %%
# =================================================================================
# Step 1: DPT Head Implementation (from reference file)
# =================================================================================

def activate_head(out, activation="inv_log", conf_activation="expp1"):
    """
A compatible activation head function for DPTHead.
    Since output_dim=1, it primarily ensures the depth prediction is positive.
    """
    if out.shape[1] > 1:
        preds = out[:, 0:1, :, :]
        conf = out[:, 1:2, :, :]
    else:
        preds = out
        conf = torch.ones_like(preds)

    # Ensure depth predictions are positive, as loss function uses log
    preds = F.relu(preds) + 1e-6 
    
    return preds, conf


def _make_scratch(in_shape: List[int], out_shape: int, groups: int = 1, expand: bool = False) -> nn.Module:
    scratch = nn.Module()
    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(
            in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
        )
    return scratch


def _make_fusion_block(features: int, size: int = None, has_residual: bool = True, groups: int = 1) -> nn.Module:
    """A helper function to create a FeatureFusionBlock."""
    return FeatureFusionBlock(
        features,
        nn.ReLU(inplace=True),
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=size,
        has_residual=has_residual,
        groups=groups,
    )


class ResidualConvUnit(nn.Module):
    """Residual convolution module."""
    def __init__(self, features, activation, bn, groups=1):
        super().__init__()
        self.bn = bn
        self.groups = groups
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        self.norm1 = None
        self.norm2 = None
        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        if self.norm1 is not None:
            out = self.norm1(out)
        out = self.activation(out)
        out = self.conv2(out)
        if self.norm2 is not None:
            out = self.norm2(out)
        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""
    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True, size=None, has_residual=True, groups=1):
        super(FeatureFusionBlock, self).__init__()
        self.deconv = deconv
        self.align_corners = align_corners
        self.groups = groups
        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=self.groups)
        if has_residual:
            self.resConfUnit1 = ResidualConvUnit(features, activation, bn, groups=self.groups)
        self.has_residual = has_residual
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn, groups=self.groups)
        self.skip_add = nn.quantized.FloatFunctional()
        self.size = size

    def forward(self, *xs, size=None):
        output = xs[0]
        if self.has_residual:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)
        output = self.resConfUnit2(output)
        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}
        output = custom_interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        output = self.out_conv(output)
        return output


def custom_interpolate(x: torch.Tensor, size: Tuple[int, int] = None, scale_factor: float = None, mode: str = "bilinear", align_corners: bool = True) -> torch.Tensor:
    if size is None:
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))
    INT_MAX = 1610612736
    input_elements = size[0] * size[1] * x.shape[0] * x.shape[1]
    if input_elements > INT_MAX:
        chunks = torch.chunk(x, chunks=(input_elements // INT_MAX) + 1, dim=0)
        interpolated_chunks = [nn.functional.interpolate(chunk, size=size, mode=mode, align_corners=align_corners) for chunk in chunks]
        x = torch.cat(interpolated_chunks, dim=0)
        return x.contiguous()
    else:
        return nn.functional.interpolate(x, size=size, mode=mode, align_corners=align_corners)


class DPTHead(nn.Module):
    def __init__(
        self,
        dim_in: int,
        patch_size: int = 14,
        output_dim: int = 1,
        activation: str = "inv_log",
        conf_activation: str = "expp1",
        features: int = 256,
        out_channels: List[int] = [256, 512, 1024, 1024],
        intermediate_layer_idx: List[int] = [0, 1, 2, 3], # Use indices for the feature list
        pos_embed: bool = False, # Disabled to avoid dependency issues
        feature_only: bool = False,
        down_ratio: int = 1,
    ) -> None:
        super(DPTHead, self).__init__()
        self.patch_size = patch_size
        self.activation = activation
        self.conf_activation = conf_activation
        self.pos_embed = pos_embed
        self.feature_only = feature_only
        self.down_ratio = down_ratio
        self.intermediate_layer_idx = intermediate_layer_idx
        
        self.norm = nn.LayerNorm(dim_in)
        self.projects = nn.ModuleList([nn.Conv2d(in_channels=dim_in, out_channels=oc, kernel_size=1, stride=1, padding=0) for oc in out_channels])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(in_channels=out_channels[0], out_channels=out_channels[0], kernel_size=4, stride=4, padding=0),
            nn.ConvTranspose2d(in_channels=out_channels[1], out_channels=out_channels[1], kernel_size=2, stride=2, padding=0),
            nn.Identity(),
            nn.Conv2d(in_channels=out_channels[3], out_channels=out_channels[3], kernel_size=3, stride=2, padding=1),
        ])
        
        self.scratch = _make_scratch(out_channels, features, expand=False)
        self.scratch.stem_transpose = None
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)
        head_features_1 = features
        head_features_2 = 32
        
        if feature_only:
            self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1, kernel_size=3, stride=1, padding=1)
        else:
            self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
            conv2_in_channels = head_features_1 // 2
            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(conv2_in_channels, head_features_2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_features_2, output_dim, kernel_size=1, stride=1, padding=0),
            )

    def forward(self, features: List[torch.Tensor], images: torch.Tensor, patch_start_idx: int, frames_chunk_size: int = 8) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, S, _, H, W = images.shape
        
        if frames_chunk_size is None or frames_chunk_size >= S:
            return self._forward_impl(features, images, patch_start_idx)
            
        assert frames_chunk_size > 0
        all_preds, all_conf = [], []
        for frames_start_idx in range(0, S, frames_chunk_size):
            frames_end_idx = min(frames_start_idx + frames_chunk_size, S)
            if self.feature_only:
                chunk_output = self._forward_impl(features, images, patch_start_idx, frames_start_idx, frames_end_idx)
                all_preds.append(chunk_output)
            else:
                chunk_preds, chunk_conf = self._forward_impl(features, images, patch_start_idx, frames_start_idx, frames_end_idx)
                all_preds.append(chunk_preds)
                all_conf.append(chunk_conf)
        if self.feature_only:
            return torch.cat(all_preds, dim=1)
        else:
            return torch.cat(all_preds, dim=1), torch.cat(all_conf, dim=1)

    def _forward_impl(self, features: List[torch.Tensor], images: torch.Tensor, patch_start_idx: int, frames_start_idx: int = None, frames_end_idx: int = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if frames_start_idx is not None and frames_end_idx is not None:
            images = images[:, frames_start_idx:frames_end_idx].contiguous()
            
        B, S, _, H, W = images.shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size
        out = []
        dpt_idx = 0
        
        # The `features` is the list of features from the backbone's intermediate layers
        # We iterate through the indices [0, 1, 2, 3] to get the 4 feature maps
        for layer_idx in self.intermediate_layer_idx:
            x = features[layer_idx][:, patch_start_idx:] # Use features directly, remove CLS/pose tokens
            x = x.unsqueeze(1) # Add sequence dimension for compatibility
            if frames_start_idx is not None and frames_end_idx is not None:
                x = x[:, frames_start_idx:frames_end_idx]
                
            x = x.view(B * S, -1, x.shape[-1])
            x = self.norm(x)
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            x = self.projects[dpt_idx](x)
            if self.pos_embed:
                x = self._apply_pos_embed(x, W, H)
            x = self.resize_layers[dpt_idx](x)
            out.append(x)
            dpt_idx += 1
            
        out = self.scratch_forward(out)
        out = custom_interpolate(out, (int(patch_h * self.patch_size / self.down_ratio), int(patch_w * self.patch_size / self.down_ratio)), mode="bilinear", align_corners=True)
        if self.pos_embed:
            out = self._apply_pos_embed(out, W, H)
        if self.feature_only:
            return out.view(B, S, *out.shape[1:])
            
        out = self.scratch.output_conv2(out)
        preds, conf = activate_head(out, activation=self.activation, conf_activation=self.conf_activation)
        preds = preds.view(B, S, *preds.shape[1:])
        conf = conf.view(B, S, *conf.shape[1:])
        return preds, conf

    def _apply_pos_embed(self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
        # This method is not used if pos_embed is False
        pass

    def scratch_forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        layer_1, layer_2, layer_3, layer_4 = features
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        out = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        del layer_4_rn, layer_4
        out = self.scratch.refinenet3(out, layer_3_rn, size=layer_2_rn.shape[2:])
        del layer_3_rn, layer_3
        out = self.scratch.refinenet2(out, layer_2_rn, size=layer_1_rn.shape[2:])
        del layer_2_rn, layer_2
        out = self.scratch.refinenet1(out, layer_1_rn)
        del layer_1_rn, layer_1
        out = self.scratch.output_conv1(out)
        return out

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
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
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
    logger.info("Creating DINOv2 ViT-B/14 via timm...")
    model = timm.create_model('vit_base_patch14_dinov2', pretrained=False, img_size=img_size)
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
    dummy_input = torch.randn(2, 3, img_size, img_size).to(device)
    with torch.no_grad():
        # features = model.forward_features(dummy_input)  # (B, N+1, 768)
        # dummy_output = decoder(features)
        features = model.get_intermediate_layers(dummy_input, n=feature_layers, norm=False)
        dummy_output, _ = decoder(features, dummy_input.unsqueeze(1), patch_start_idx=0)
        dummy_output = dummy_output.squeeze(1)
    logger.info(f"Model created successfully!")
    logger.info(f"Input shape: {dummy_input.shape}")
    logger.info(f"Number of feature maps extracted: {len(features)}")
    logger.info(f"Output shape: {dummy_output.shape}")
    assert dummy_output.shape == (2, 1, img_size, img_size), f"Expected output shape (2, 1, {img_size}, {img_size})"
    logger.info("✅ Output shape is correct.")
    encoder_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    decoder_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    logger.info(f"Encoder params: {encoder_params/1e6:.2f}M, Decoder: {decoder_params/1e6:.2f}M")
    
    return model, decoder, feature_layers

model, decoder, feature_layers = setup_model(IMG_SIZE, DEVICE)

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

if not HAS_POS:
    model.pos_embed.data.zero_()
    model.pos_embed.requires_grad = False
    logger.info("✅ Positional embedding has been disabled.")
criterion = MonocularDepthLoss(silog_w=10.0, l1_w=1.0, grad_w=1.0, ssim_w=3.0)
# criterion = MonocularDepthLossSimple()
optimizer = optim.AdamW(list(model.parameters()) + list(decoder.parameters()), lr=LEARNING_RATE)
total_steps = EPOCHS * steps_per_epoch
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
logger.info("✅ Loss, Optimizer, and Scheduler are ready.")

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
    logger.info("✅ Row-Column loss initialized.")

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
    
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs} [Train]")
    
    for i, (inputs, gt_depths) in enumerate(pbar):
        inputs, gt_depths = inputs.to(DEVICE), gt_depths.to(DEVICE)
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda', dtype=autocast_dtype):
            # features = model.forward_features(inputs)
            # pred_depths = decoder(features)
            # loss, _ = criterion(pred_depths, gt_depths)
            features = model.get_intermediate_layers(inputs, n=feature_layers, norm=False)
            pred_depths, _ = decoder(features, inputs.unsqueeze(1), patch_start_idx=0)
            pred_depths = pred_depths.squeeze(1)
            loss, loss_dict = criterion(pred_depths, gt_depths)
        
        if Use_Row_Col_Loss:
            aux_loss = rowcol_loss(features[-1])
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

                features = model.get_intermediate_layers(val_inputs, n=feature_layers, norm=False)
                val_pred_depths, _ = decoder(features, val_inputs.unsqueeze(1), patch_start_idx=0)
                val_pred_depths = val_pred_depths.squeeze(1)
                v_loss, _ = criterion(val_pred_depths, gt_depths)
            val_loss += v_loss.item()
            batch_metrics = compute_depth_metrics(val_pred_depths, gt_depths)
            for k in val_metrics:
                val_metrics[k] += batch_metrics.get(k, 0)
    
    return val_loss / len(loader), {k: v / len(loader) for k, v in val_metrics.items()}

def save_checkpoint(model, decoder, output_dir, suffix):
    encoder_path = os.path.join(output_dir, f'encoder_{suffix}.pth')
    decoder_path = os.path.join(output_dir, f'decoder_{suffix}.pth')
    torch.save(model.state_dict(), encoder_path)
    torch.save(decoder.state_dict(), decoder_path)
    logger.info(f"Checkpoint saved: {suffix}")

scaler = torch.amp.GradScaler('cuda')
logger.info(f"\n🚀 Starting training for {MODEL_NAME}...")

training_history = {
    'train_loss': [], 'train_abs_rel': [], 'train_rmse': [], 'train_a1': [],
    'valid_loss': [], 'valid_abs_rel': [], 'valid_rmse': [], 'valid_a1': [],
    'epoch': []
}
best_val_abs_rel = float('inf')

logger.info("Starting training...")
for epoch in range(EPOCHS):
    avg_train_loss, avg_train_metrics = train_one_epoch(
        model, decoder, train_loader, criterion, optimizer, scheduler, scaler, feature_layers, epoch, EPOCHS
    )
    
    avg_val_loss, avg_val_metrics = validate(
        model, decoder, valid_loader, criterion, feature_layers
    )

    logger.info(f"\n--- Epoch {epoch+1} Validation Summary ---")
    logger.info(f"  Train Loss: {avg_train_loss:.4f} | Train AbsRel: {avg_train_metrics['abs_rel']:.4f} | Train RMSE: {avg_train_metrics['rmse']:.4f} | Train a1: {avg_train_metrics['a1']:.4f}")
    logger.info(f"  Valid Loss: {avg_val_loss:.4f} | Valid AbsRel: {avg_val_metrics['abs_rel']:.4f} | Valid RMSE: {avg_val_metrics['rmse']:.4f} | Valid a1: {avg_val_metrics['a1']:.4f}\n")
    
    training_history['train_loss'].append(avg_train_loss)
    training_history['train_abs_rel'].append(avg_train_metrics['abs_rel'])
    training_history['train_rmse'].append(avg_train_metrics['rmse'])
    training_history['train_a1'].append(avg_train_metrics['a1'])
    training_history['valid_loss'].append(avg_val_loss)
    training_history['valid_abs_rel'].append(avg_val_metrics['abs_rel'])
    training_history['valid_rmse'].append(avg_val_metrics['rmse'])
    training_history['valid_a1'].append(avg_val_metrics['a1'])
    training_history['epoch'].append(epoch + 1)
    
    if avg_val_metrics['abs_rel'] < best_val_abs_rel:
        best_val_abs_rel = avg_val_metrics['abs_rel']
        history_df = pd.DataFrame(training_history)
        history_df.to_csv(os.path.join(output_dir, 'training_history.csv'), index=False)
        save_checkpoint(model, decoder, output_dir, "best")

logger.info("Training complete.")

history_df = pd.DataFrame(training_history)
history_df.to_csv(os.path.join(output_dir, 'training_history.csv'), index=False)
save_checkpoint(model, decoder, output_dir, "final")

if not history_df.empty:
    best_a1 = history_df['valid_a1'].max()
    best_epoch = history_df.loc[history_df['valid_a1'].idxmax(), 'epoch']
    logger.info(f"Best a1: {best_a1:.4f} at epoch {best_epoch}")

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

    logger.info("\n--- Best Validation Metrics from History ---")
    logger.info(f"  Best Loss:    {best_loss_val:.4f} (Epoch {best_loss_epoch})")
    logger.info(f"  Best a1:      {best_a1_val:.4f} (Epoch {best_a1_epoch})")
    logger.info(f"  Best AbsRel:  {best_abs_rel_val:.4f} (Epoch {best_abs_rel_epoch})")
    logger.info(f"  Best RMSE:    {best_rmse_val:.4f} (Epoch {best_rmse_epoch})")
    logger.info("------------------------------------------")
