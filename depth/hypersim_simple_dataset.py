import os.path as osp
import numpy as np
import glob
import cv2
import random
from PIL import Image
from torch.utils.data import Dataset
from aug import TrainDepthAug, EvalDepthPreprocess
def imread_cv2(path, options=cv2.IMREAD_COLOR):
    """Open an image or a depthmap with opencv-python."""
    if path.endswith((".exr", "EXR")):
        options = cv2.IMREAD_ANYDEPTH
    img = cv2.imread(path, options)
    if img is None:
        raise IOError(f"Could not load image={path} with {options=}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

class HyperSim_Simple(Dataset):
    """
    A simplified, standalone dataset for Hypersim that enumerates all individual
    image/depth pairs by searching for '_rgb.png' files. It does not inherit
    from HyperSim_Multi, making it much simpler and more direct.

    It's useful for tasks like monocular depth estimation where each image-depth
    pair is an independent sample.
    """

    def __init__(
        self,
        ROOT,
        resolution,
        split=None,
        sample_rate=1.0,
        pair_transform=None,
        **kwargs
    ):
        super().__init__()
        self.ROOT = ROOT
        self.resolution = resolution
        self._setup_resolution()
        self.dataset_label = "HyperSim_Simple"
        self.is_train = split == 'train'
        if pair_transform is None:
            target_size = (self.resolution[1], self.resolution[0]) if isinstance(self.resolution, (list, tuple)) else (self.resolution, self.resolution)
            if self.is_train:
                self.pair_transform = TrainDepthAug(
                    target_size=target_size,
                    normalize=True,
                )
            else:
                self.pair_transform = EvalDepthPreprocess(
                    target_size=target_size,
                    target_by="height",
                    ensure_multiple_of=32,
                    normalize=True,
                )
        else:
            self.pair_transform = pair_transform

        print(f"Searching for images in {self.ROOT}...")
        # Recursively find all files ending with '_rgb.png'
        self.image_paths = glob.glob(osp.join(self.ROOT, '**', '*_rgb.png'), recursive=True)
        
        if not self.image_paths:
            raise FileNotFoundError(f"No '_rgb.png' files found in {self.ROOT}")
        if sample_rate < 1.0:
            num_samples = int(len(self.image_paths) * sample_rate)
            # Take a random subset of the images
            self.image_paths = random.sample(self.image_paths, num_samples)

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
        depthmap = depthmap.astype(np.float32)

        out = self.pair_transform(pil_img, depthmap)
        if isinstance(out, tuple) and len(out) == 3:
            img_t, depth_t, meta = out
        else:
            img_t, depth_t = out
            h, w = depth_t.shape[-2], depth_t.shape[-1]
            meta = {
                "orig_h": float(h),
                "orig_w": float(w),
                "resized_h": float(h),
                "resized_w": float(w),
                "scale_h": 1.0,
                "scale_w": 1.0,
                "pad_h": 0.0,
                "pad_w": 0.0,
                "padded_h": float(h),
                "padded_w": float(w),
            }
        return img_t, depth_t, meta


if __name__ == '__main__':
    dataset = HyperSim_Simple(
        ROOT='/lc/data/3D/hypersim_processed/train',
        split='train',
        resolution=224,
    )
    print(f"Found {len(dataset)} total frames.")

    img_t, depth_t, _ = dataset[0]
    print("\nSample shapes:")
    print("Image shape:", img_t.shape)
    print("Depthmap shape:", depth_t.shape)
