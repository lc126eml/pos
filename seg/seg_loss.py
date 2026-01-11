import torch
import torch.nn as nn
import torch.nn.functional as F



class MMSegCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=-1, loss_weight=1.0, avg_non_ignore=True):
        super().__init__()
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight
        self.avg_non_ignore = avg_non_ignore

    def forward(self, logits, target):
        # logits: (N, C, H, W), target: (N, H, W)
        if not self.avg_non_ignore:
            return self.loss_weight * F.cross_entropy(
                logits, target, ignore_index=self.ignore_index, reduction='mean'
            )

        # reduction='none' then average only over valid pixels
        loss = F.cross_entropy(
            logits, target, ignore_index=self.ignore_index, reduction='none'
        )  # (N, H, W)
        valid = (target != self.ignore_index)
        denom = valid.sum().clamp_min(1).to(loss.dtype)
        return self.loss_weight * (loss[valid].sum() / denom)
        
class MMSegDiceLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=-1, smooth=1.0, loss_weight=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.loss_weight = loss_weight

    def forward(self, logits, target):
        # logits: (N, C, H, W), target: (N, H, W)
        N, C, H, W = logits.shape
        prob = F.softmax(logits, dim=1)

        valid = (target != self.ignore_index)
        target_clamped = target.clamp_min(0)  # safe for ignore pixels

        # Flatten
        prob = prob.view(N, C, -1)                 # (N, C, HW)
        target_flat = target_clamped.view(N, -1)   # (N, HW)
        valid_flat = valid.view(N, -1).float()     # (N, HW)

        total = 0.0
        count = 0

        for c in range(C):
            p = prob[:, c, :] * valid_flat
            t = (target_flat == c).float() * valid_flat

            inter = (p * t).sum(dim=1)
            union = p.sum(dim=1) + t.sum(dim=1)

            dice = (2.0 * inter + self.smooth) / (union + self.smooth)
            total += (1.0 - dice).mean()
            count += 1

        return self.loss_weight * (total / max(count, 1))
