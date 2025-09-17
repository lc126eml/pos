import os
import os.path as osp
import numpy as np
import pandas as pd
import glob
import cv2
import random
from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from src.dust3r.utils.image import imread_cv2, ImgnetNorm
from src.dust3r.datasets.utils import cropping


class HyperSim_Simple(Dataset):
    """
    A simplified, standalone dataset for Hypersim that enumerates all individual
    image/depth pairs by searching for '_rgb.png' files. It does not inherit
    from HyperSim_Multi, making it much simpler and more direct.

    It's useful for tasks like monocular depth estimation where each image-depth
    pair is an independent sample.
    """

    def __init__(self, ROOT, resolution, split=None, useImgnet=True, transform=None, **kwargs):
        super().__init__()
        self.ROOT = ROOT
        self.resolution = resolution
        self._setup_resolution()
        self.dataset_label = "HyperSim_Simple"
        self.is_train = split == 'train'

        # Set up image normalization
        if useImgnet:
            self.img_norm = ImgnetNorm
        else:
            # Fallback to default normalization if not using ImageNet stats
            from src.dust3r.utils.image import ImgNorm
            self.img_norm = ImgNorm

        # Set up data augmentation
        self.transform = None # Color Jitter
        if self.is_train:
            self.transform = T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)

        print(f"Searching for images in {self.ROOT}...")
        # Recursively find all files ending with '_rgb.png'
        self.image_paths = glob.glob(osp.join(self.ROOT, '**', '*_rgb.png'), recursive=True)
        
        if not self.image_paths:
            raise FileNotFoundError(f"No '_rgb.png' files found in {self.ROOT}")
        
        print(f"Found {len(self.image_paths)} images.")

    def _setup_resolution(self):
        """Ensures self.resolution is a (width, height) tuple."""
        if isinstance(self.resolution, int):
            self.resolution = (self.resolution, self.resolution)
        elif isinstance(self.resolution, (list, tuple)):
            assert len(self.resolution) == 2, "Resolution must be an int or a (width, height) tuple."

    def __len__(self):
        """Returns the total number of individual frames in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        if idx >= len(self.image_paths):
            raise IndexError("Index out of range")

        impath = self.image_paths[idx]
        depthpath = impath.replace("_rgb.png", "_depth.npy")

        # Load image and depth
        rgb_image = imread_cv2(impath)
        pil_img = Image.fromarray(rgb_image)
        depthmap = np.load(depthpath)
        depthmap[~np.isfinite(depthmap)] = 0.0
        depthmap = depthmap.astype(np.float32)/100.0

        # Apply augmentations if in training mode
        if self.is_train:
            # Random horizontal flip
            if random.random() > 0.5:
                pil_img = TF.hflip(pil_img)
                depthmap = np.fliplr(depthmap)

            if random.random() > 0.5:
                i, j, h, w = T.RandomResizedCrop.get_params(
                    pil_img, scale=(0.9, 1.0), ratio=(0.9, 1.1)  # Less aggressive
                )
                pil_img = TF.resized_crop(pil_img, i, j, h, w, list(self.resolution))
                depthmap = depthmap[i:i+h, j:j+w]
                # FIX: Use bilinear for depth (better than nearest)
                depthmap = cv2.resize(depthmap, self.resolution, 
                                 interpolation=cv2.INTER_LINEAR)
            else:
                pil_img = TF.resize(pil_img, list(self.resolution))
                depthmap = cv2.resize(depthmap, self.resolution, 
                                 interpolation=cv2.INTER_LINEAR)
            
            # Random color jitter
            if self.transform is not None and random.random() > 0.7:
                pil_img = self.transform(pil_img)

        else:
            # For validation/testing, just resize
            pil_img = TF.resize(pil_img, list(self.resolution))
            depthmap = cv2.resize(depthmap, self.resolution, 
                             interpolation=cv2.INTER_LINEAR)

        # The collate_fn expects a list of views, so we wrap the view in a list.
        view = dict(
            img=self.img_norm(pil_img), # ToTensor and Normalize
            depthmap=torch.from_numpy(np.ascontiguousarray(depthmap)),
            dataset=self.dataset_label,
            label=osp.relpath(osp.dirname(impath), self.ROOT),
            instance=osp.split(impath)[1],
        )
        return [view]


if __name__ == '__main__':
    dataset = HyperSim_Simple(
        ROOT='/lc/data/3D/hypersim_processed/train',
        split='train',
        resolution=224,
    )
    print(f"Found {len(dataset)} total frames.")

    sample = dataset[0]
    view = sample[0]
    print("\nSample view keys:", view.keys())
    print("Image shape:", view['img'].shape)
    print("Depthmap shape:", view['depthmap'].shape)
