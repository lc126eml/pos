#%%
from functools import partial
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
from torch.nn import functional as F
from typing import Tuple
import sys
# sys.path.append(r"D:\codes\working\pos\pytorch-image-models")
import timm
#%%
timm.list_models("*dinov2*")

#%%
MODEL_NAME = "vit_base_patch14_dinov2"
model = timm.create_model(
    MODEL_NAME,
    pretrained=False, # As requested: trains the model from scratch
    num_classes=80, # Set the classifier head to 100 classes
    img_size=224,
)
#%%
print(model.num_prefix_tokens)
#%%
sys.path.append(r".")
from vision_transformer_rope import *
from vision_transformer_rpe import *
from vision_transformer_relpos import *
from vision_transformer_alibi import *
from vision_transformer_sin import *
 #%%
MODEL_NAME = 'vit_base_patch14_dinov2'
MODEL_NAME = "vit_rope_small_patch14_224"
MODEL_NAME = "vit_alibi_base_patch14_dinov2"
# MODEL_NAME = "vit_rpe_base_patch14_dinov2"
# MODEL_NAME = "vit_relpos_base_patch14_dinov2"
# MODEL_NAME = "vit_sin_base_patch14_dinov2"

IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%

model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=0, img_size=IMG_SIZE).to(DEVICE)
# print(model)
# %%
feature_layers = [2, 5, 8, 11]
dummy_input = torch.randn(2, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
with torch.no_grad():
    feats = model.forward_features(dummy_input)
    # multi_feats = model.forward_intermediates(dummy_input, indices=feature_layers, intermediates_only=True)


print(f"Model created successfully!")
print(f"Input shape: {dummy_input.shape}")
print(f"Output shape: {feats.shape}") 
# print(f"multi_feats shape: {multi_feats[-1].shape} X {len(multi_feats)}")
# {multi_feats[-1].shape} 
# assert output.shape == (2, NUM_CLASSES, IMG_SIZE, IMG_SIZE)
# print("✅ Output shape is correct.")
# %%
