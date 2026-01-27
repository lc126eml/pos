import math
import random

class DynamicResolutionBatchSampler:
    """
    Yields batches of (idx, size) with dynamic batch size so that
    batch_size * size^2 approx base_batch_size * base_img_size^2.
    """

    def __init__(
        self,
        dataset,
        image_sizes,
        base_batch_size,
        base_img_size,
        shuffle: bool = True,
        drop_last: bool = True,
        size_schedule: str = "batch",
        hold_batches: int = 0,
        seed: int = 0,
    ):
        self.dataset_len = len(dataset)
        self.image_sizes = list(image_sizes)
        self.base_batch_size = base_batch_size
        self.base_img_size = base_img_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.size_schedule = size_schedule
        self.hold_batches = hold_batches
        self.seed = seed
        self.epoch = 0

        self.pixel_budget = base_batch_size * (base_img_size ** 2)

        avg_size_sq = sum(s * s for s in self.image_sizes) / len(self.image_sizes)
        self.avg_batch_size = self.pixel_budget / avg_size_sq

    def __len__(self):
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
        size = None
        batches_since_change = 0
        if self.size_schedule == "epoch":
            size = rng.choice(self.image_sizes)

        while ptr < n:
            if self.size_schedule == "batch":
                size = rng.choice(self.image_sizes)
            elif self.size_schedule == "batches":
                if self.hold_batches <= 0:
                    size = rng.choice(self.image_sizes)
                elif size is None or batches_since_change >= self.hold_batches:
                    size = rng.choice(self.image_sizes)
                    batches_since_change = 0

            pixels_per_sample = size * size
            if pixels_per_sample > 0:
                batch_size = max(1, self.pixel_budget // pixels_per_sample)
            else:
                batch_size = self.base_batch_size

            remaining = n - ptr
            if remaining < batch_size:
                if self.drop_last:
                    break
                batch_size = remaining

            batch_indices = indices[ptr: ptr + batch_size]
            ptr += batch_size

            yield [(idx, size) for idx in batch_indices]
            batches_since_change += 1


class DistributedDynamicResolutionBatchSampler:
    """
    TPU-friendly dynamic batch sampler that keeps batch shapes identical across ranks.
    Each rank sees a disjoint shard of the shuffled indices, while sharing the same
    size schedule (so all ranks use the same resolution per step).
    """

    def __init__(
        self,
        dataset,
        image_sizes,
        base_batch_size,
        base_img_size,
        num_replicas: int,
        rank: int,
        shuffle: bool = True,
        drop_last: bool = True,
        size_schedule: str = "epoch",
        hold_batches: int = 0,
        seed: int = 0,
    ):
        self.dataset_len = len(dataset)
        self.image_sizes = list(image_sizes)
        self.base_batch_size = base_batch_size
        self.base_img_size = base_img_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.size_schedule = size_schedule
        self.hold_batches = hold_batches
        self.seed = seed
        self.epoch = 0

        self.pixel_budget = base_batch_size * (base_img_size ** 2)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def _shard_indices(self, rng):
        indices = list(range(self.dataset_len))
        if self.shuffle:
            rng.shuffle(indices)
        if self.drop_last:
            total = (len(indices) // self.num_replicas) * self.num_replicas
            indices = indices[:total]
        return indices[self.rank::self.num_replicas]

    def _num_batches(self, rng, indices):
        ptr = 0
        n = len(indices)
        count = 0
        size = None
        batches_since_change = 0
        if self.size_schedule == "epoch":
            size = rng.choice(self.image_sizes)
        while ptr < n:
            if self.size_schedule == "batch":
                size = rng.choice(self.image_sizes)
            elif self.size_schedule == "batches":
                if self.hold_batches <= 0:
                    size = rng.choice(self.image_sizes)
                elif size is None or batches_since_change >= self.hold_batches:
                    size = rng.choice(self.image_sizes)
                    batches_since_change = 0
            pixels_per_sample = size * size
            if pixels_per_sample > 0:
                batch_size = max(1, self.pixel_budget // pixels_per_sample)
            else:
                batch_size = self.base_batch_size

            remaining = n - ptr
            if remaining < batch_size:
                if self.drop_last:
                    break
                batch_size = remaining
            ptr += batch_size
            count += 1
            batches_since_change += 1
        return count

    def __len__(self):
        rng = random.Random(self.seed + self.epoch)
        indices = self._shard_indices(rng)
        return self._num_batches(rng, indices)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        indices = self._shard_indices(rng)
        ptr = 0
        n = len(indices)
        size = None
        batches_since_change = 0
        if self.size_schedule == "epoch":
            size = rng.choice(self.image_sizes)

        while ptr < n:
            if self.size_schedule == "batch":
                size = rng.choice(self.image_sizes)
            elif self.size_schedule == "batches":
                if self.hold_batches <= 0:
                    size = rng.choice(self.image_sizes)
                elif size is None or batches_since_change >= self.hold_batches:
                    size = rng.choice(self.image_sizes)
                    batches_since_change = 0
            pixels_per_sample = size * size
            if pixels_per_sample > 0:
                batch_size = max(1, self.pixel_budget // pixels_per_sample)
            else:
                batch_size = self.base_batch_size

            remaining = n - ptr
            if remaining < batch_size:
                if self.drop_last:
                    break
                batch_size = remaining

            batch_indices = indices[ptr: ptr + batch_size]
            ptr += batch_size
            yield [(idx, size) for idx in batch_indices]
            batches_since_change += 1
