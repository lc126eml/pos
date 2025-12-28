import math
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
import time
import argparse
import logging

if os.path.exists('/kaggle/input/imagenet100'):
    print("kaggle")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

use_amp = torch.cuda.is_available()
use_bf16 = use_amp and torch.cuda.is_bf16_supported()
autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16

print(f"Using device: {DEVICE}", use_bf16, autocast_dtype)

def gpu_info_pytorch():
    try:
        import torch
    except ImportError:
        print("PyTorch not installed.")
        return

    if not torch.cuda.is_available():
        print("CUDA not available.")
        return

    n = torch.cuda.device_count()
    print(f"GPUs: {n}")
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        total_gb = props.total_memory / (1024**3)
        print(f"[{i}] {props.name} | total VRAM: {total_gb:.2f} GiB")

gpu_info_pytorch()
