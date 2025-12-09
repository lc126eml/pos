import math
import random

class DynamicResolutionBatchSampler:
    """
    Yields batches of (idx, size) with dynamic batch size so that
    batch_size * size^2 ≈ base_batch_size * base_img_size^2.
    """

    def __init__(
        self,
        dataset,
        image_sizes,
        base_batch_size,
        base_img_size,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 0,
    ):
        self.dataset_len = len(dataset)
        self.image_sizes = list(image_sizes)
        self.base_batch_size = base_batch_size
        self.base_img_size = base_img_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        # pixel budget based on reference configuration
        self.pixel_budget = base_batch_size * (base_img_size ** 2)

        # for __len__ (approximate)
        avg_size_sq = sum(s * s for s in self.image_sizes) / len(self.image_sizes)
        self.avg_batch_size = self.pixel_budget / avg_size_sq

    def __len__(self):
        # approximate number of batches per epoch
        return math.ceil(self.dataset_len / self.avg_batch_size)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)

        indices = list(range(self.dataset_len))
        if self.shuffle:
            rng.shuffle(indices)

        ptr = 0
        n = len(indices)

        while ptr < n:
            # 1) choose resolution for this batch
            size = rng.choice(self.image_sizes)

            # 2) compute batch size from pixel budget
            pixels_per_sample = size * size
            if pixels_per_sample > 0:
                batch_size = max(1, self.pixel_budget // pixels_per_sample)
            else:
                batch_size = self.base_batch_size

            # 3) adjust for remaining samples
            remaining = n - ptr
            if remaining < batch_size:
                if self.drop_last:
                    break
                else:
                    batch_size = remaining

            batch_indices = indices[ptr: ptr + batch_size]
            ptr += batch_size

            # 4) yield (idx, size) pairs
            yield [(idx, size) for idx in batch_indices]
