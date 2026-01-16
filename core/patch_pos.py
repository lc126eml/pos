import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional, Dict, Tuple

LossType = Literal["mse", "smooth_l1", "l1"]
class PatchRowColRegressionCriterion(nn.Module):
    def __init__(
        self,
        feat_dim,
        grid_h,
        grid_w,
        normalize=True,
        huber_beta=None,
        loss_type: LossType = "smooth_l1",
    ):
        """
        Predict row and column index of each patch via regression (single resolution).

        Args:
            feat_dim (int): Dimension of patch features (D)
            grid_h (int): Number of patch rows (fixed)
            grid_w (int): Number of patch columns (fixed)
            normalize (bool): If True, normalize row/col targets to [0, 1]
            huber_beta (float|None): SmoothL1 beta (only used when loss_type="smooth_l1")
            loss_type (str): "l1", "smooth_l1", or "mse"
        """
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.normalize = normalize

        # Regression heads: scalar row / scalar col
        self.row_mlp = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)   # scalar row index
        )

        self.col_mlp = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)   # scalar col index
        )

        if loss_type == "l1":
            self.loss_fn = nn.L1Loss()
        elif loss_type == "smooth_l1":
            if huber_beta is None:
                self.loss_fn = nn.SmoothL1Loss()
            else:
                self.loss_fn = nn.SmoothL1Loss(beta=huber_beta)
        elif loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

        # Precompute row/col targets once (N = grid_h * grid_w)
        rows_2d = torch.arange(grid_h, dtype=torch.float32).unsqueeze(1).repeat(1, grid_w)
        cols_2d = torch.arange(grid_w, dtype=torch.float32).unsqueeze(0).repeat(grid_h, 1)

        if normalize:
            rows_2d = rows_2d / (grid_h - 1)
            cols_2d = cols_2d / (grid_w - 1)

        # Flatten to 1D (N,)
        row_targets = rows_2d.flatten()
        col_targets = cols_2d.flatten()

        # Non-persistent buffers: device-aware but excluded from state_dict.
        self.register_buffer("row_targets", row_targets, persistent=False)  # (N,)
        self.register_buffer("col_targets", col_targets, persistent=False)  # (N,)

    def forward(self, feats):
        """
        Args:
            feats: (B, N, D) patch features, N = grid_h * grid_w

        Returns:
            avg_loss: scalar, average of row and column regression losses
        """
        B, N, D = feats.shape
        assert N == self.grid_h * self.grid_w, f"Expected N = grid_h * grid_w = {self.grid_h * self.grid_w}, got N = {N}"

        # (B*N, D)
        x = feats.reshape(-1, D)

        # Repeat targets for batch: (N,) -> (B*N,)
        row_targets = self.row_targets.repeat(B)
        col_targets = self.col_targets.repeat(B)

        # Predict rows and columns: (B*N, 1) -> (B*N,)
        row_pred = self.row_mlp(x).squeeze(-1)
        col_pred = self.col_mlp(x).squeeze(-1)

        loss_row = self.loss_fn(row_pred, row_targets)
        loss_col = self.loss_fn(col_pred, col_targets)

        return (loss_row + loss_col) / 2.0

class PatchRowColRegressionCriterionDynamic(nn.Module):
    def __init__(
        self,
        feat_dim,
        grid_h,
        grid_w,
        normalize=True,
        loss_type: LossType = "smooth_l1",
    ):
        """
        Predict row and column index of each patch via regression,
        supporting dynamic training resolutions.

        Args:
            feat_dim (int): Dimension of patch features (D)
            grid_h (int): Max number of patch rows (upper bound)
            grid_w (int): Max number of patch columns (upper bound)
            normalize (bool): If True, normalize row/col targets to [0, 1]
                              based on the *current* hp/wp for each batch.
            loss_type (str): "l1", "smooth_l1", or "mse"
        """
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.normalize = normalize

        # MLP for row regression (scalar output)
        self.row_mlp = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)   # scalar row index
        )

        # MLP for column regression (scalar output)
        self.col_mlp = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)   # scalar col index
        )

        if loss_type == "l1":
            self.loss_fn = nn.L1Loss()
        elif loss_type == "smooth_l1":
            self.loss_fn = nn.SmoothL1Loss()
        elif loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

        # Precompute integer row/col indices (max grid) as floats
        rows = torch.arange(grid_h, dtype=torch.float32).unsqueeze(1).repeat(1, grid_w)  # (grid_h, grid_w)
        cols = torch.arange(grid_w, dtype=torch.float32).unsqueeze(0).repeat(grid_h, 1)  # (grid_h, grid_w)

        # Non-persistent buffers: device-aware but excluded from state_dict.
        self.register_buffer("row_index_full", rows, persistent=False)  # (grid_h, grid_w)
        self.register_buffer("col_index_full", cols, persistent=False)  # (grid_h, grid_w)

    def forward(self, feats, hp=None, wp=None):
        """
        Args:
            feats: (B, N, D) patch features, with N = hp * wp for this batch.
            hp, wp: number of patch rows / columns used for this batch
                    (single scalar each; one resolution per batch).

        Returns:
            avg_loss: scalar, average of row and column regression losses.
        """
        B, N, D = feats.shape

        # Fallback to maximum grid if hp/wp not given
        if hp is None:
            hp = self.grid_h
        if wp is None:
            wp = self.grid_w

        assert N == hp * wp, f"Expected N = hp * wp = {hp * wp}, got N = {N}"

        # Flatten features: (B*N, D)
        x = feats.reshape(-1, D)

        # Slice the index grids to current resolution: (hp, wp)
        row_idx_2d = self.row_index_full[:hp, :wp]  # [0..hp-1]
        col_idx_2d = self.col_index_full[:hp, :wp]  # [0..wp-1]

        if self.normalize:
            # Normalize to [0, 1] based on current hp/wp
            row_idx_2d = row_idx_2d / max(hp - 1, 1)
            col_idx_2d = col_idx_2d / max(wp - 1, 1)

        # Flatten to 1D and repeat for batch: (hp*wp,) -> (B*hp*wp,)
        row_targets = row_idx_2d.flatten().repeat(B)
        col_targets = col_idx_2d.flatten().repeat(B)

        # Predict rows and columns: (B*N, 1) -> (B*N,)
        row_pred = self.row_mlp(x).squeeze(-1)
        col_pred = self.col_mlp(x).squeeze(-1)

        # Regression losses
        loss_row = self.loss_fn(row_pred, row_targets)
        loss_col = self.loss_fn(col_pred, col_targets)

        return (loss_row + loss_col) / 2.0

class PatchRowColCriterionDynamic(nn.Module):
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
        rows = torch.arange(grid_h).unsqueeze(1).repeat(1, grid_w)
        cols = torch.arange(grid_w).unsqueeze(0).repeat(grid_h, 1)
        self.register_buffer("row_labels", rows, persistent=False)
        self.register_buffer("col_labels", cols, persistent=False)

    def forward(self, feats, hp=None, wp=None):
        """
        Args:
            feats: (B, N, D) patch features, N = grid_h * grid_w
            wp, hp: (B,) number of patches in each row/column
        Returns:
            avg_loss: scalar, sum of row and column classification losses
        """
        B, N, D = feats.shape
        # assert N == self.grid_h * self.grid_w, f"Expected {self.grid_h*self.grid_w} patches, got {N}"

        x = feats.reshape(-1, D)  # (B*N, D)

        if hp is None or wp is None:
            hp = self.grid_h
            wp = self.grid_w
        # Repeat labels for batch
        row_labels = self.row_labels[:hp, :wp].flatten().repeat(B)
        col_labels = self.col_labels[:hp, :wp].flatten().repeat(B)

        # Predict rows and columns
        row_logits = self.row_mlp(x)
        col_logits = self.col_mlp(x)

        # Compute cross-entropy loss for rows and columns
        loss_row = self.ce(row_logits, row_labels)
        loss_col = self.ce(col_logits, col_labels)

        return (loss_row + loss_col) / 2  # average

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
        self.register_buffer("row_labels", rows, persistent=False)
        self.register_buffer("col_labels", cols, persistent=False)

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


# if Use_Row_Col_Loss:
#     grid_h, grid_w = model.patch_embed.grid_size
#     rowcol_loss = PatchRowColCriterion(
#         feat_dim=model.embed_dim,
#         grid_h=grid_h,
#         grid_w=grid_w
#     ).to(DEVICE)
#     print("✅ Row-Column loss initialized.")

class PatchPositionCriterion(nn.Module):
    def __init__(self, feat_dim, hidden_dim=256, num_classes=None):
        """
        Args:
            feat_dim (int): Feature dimension of each patch (D)
            hidden_dim (int): Hidden layer size for MLP
            num_classes (int): Number of patches (grid_h * grid_w)
        """
        super().__init__()
        self.num_classes = num_classes
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
        self.ce = nn.CrossEntropyLoss()

        # Precompute patch position labels once
        self.register_buffer("patch_positions", torch.arange(num_classes), persistent=False)  # shape (num_patches,)
        
    def forward(self, feats):
        """
        Args:
            feats: (B, N, D) patch features
        Returns:
            avg_loss: scalar, mean cross-entropy over all patches
        """
        B, N, D = feats.shape
        assert N == self.num_classes, f"Expected {self.num_classes} patches, got {N}"

        # Flatten batch and patches: (B*N, D)
        x = feats.reshape(-1, D)
        # Repeat labels for all images in batch: (B*N,)
        labels = self.patch_positions.repeat(B)
        # Predict positions
        logits = self.mlp(x)
        # Compute CE loss
        loss = self.ce(logits, labels)
        return loss

class PatchPositionRegressionCriterion(nn.Module):
    def __init__(self, feat_dim, num_classes, normalize=True):
        """
        Predict patch position index via regression (single resolution).

        Args:
            feat_dim (int): Feature dimension of each patch (D)
            num_classes (int): Number of patches (grid_h * grid_w)
            normalize (bool): If True, normalize position targets to [0, 1]
        """
        super().__init__()
        self.num_classes = num_classes
        self.normalize = normalize

        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)   # scalar position index
        )

        self.loss_fn = nn.SmoothL1Loss()

        # Precompute patch position targets once
        position_targets = torch.arange(num_classes, dtype=torch.float32)
        if normalize:
            position_targets = position_targets / max(num_classes - 1, 1)
        self.register_buffer("position_targets", position_targets, persistent=False)  # (N,)

    def forward(self, feats):
        """
        Args:
            feats: (B, N, D) patch features
        Returns:
            loss: scalar, SmoothL1 loss over all patches
        """
        B, N, D = feats.shape
        assert N == self.num_classes, f"Expected {self.num_classes} patches, got {N}"

        # Flatten batch and patches: (B*N, D)
        x = feats.reshape(-1, D)
        # Repeat targets for batch: (N,) -> (B*N,)
        targets = self.position_targets.repeat(B)
        # Predict positions: (B*N, 1) -> (B*N,)
        pred = self.mlp(x).squeeze(-1)
        # Compute regression loss
        loss = self.loss_fn(pred, targets)
        return loss

class PatchPositionRegressionCriterionDynamic(nn.Module):
    def __init__(self, feat_dim, max_patch_count, normalize=True):
        """
        Predict patch position index via regression, supporting dynamic resolutions.

        Args:
            feat_dim (int): Feature dimension of each patch (D)
            max_patch_count (int): Max number of patches (upper bound)
            normalize (bool): If True, normalize position targets to [0, 1]
                              based on the *current* patch count for each batch.
        """
        super().__init__()
        self.max_patch_count = max_patch_count
        self.normalize = normalize

        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)   # scalar position index
        )

        self.loss_fn = nn.SmoothL1Loss()

        positions = torch.arange(max_patch_count, dtype=torch.float32)
        self.register_buffer("position_index_full", positions, persistent=False)

    def forward(self, feats):
        """
        Args:
            feats: (B, N, D) patch features.
        Returns:
            loss: scalar, SmoothL1 loss over all patches.
        """
        B, N, D = feats.shape
        if N > self.max_patch_count:
            raise ValueError(f"Expected N <= max_patch_count={self.max_patch_count}, got N={N}")

        # Flatten features: (B*N, D)
        x = feats.reshape(-1, D)

        # Slice position indices to current patch count: (N,)
        pos_idx = self.position_index_full[:N]

        if self.normalize:
            pos_idx = pos_idx / max(N - 1, 1)

        # Repeat for batch: (N,) -> (B*N,)
        targets = pos_idx.repeat(B)

        # Predict positions: (B*N, 1) -> (B*N,)
        pred = self.mlp(x).squeeze(-1)

        loss = self.loss_fn(pred, targets)
        return loss
        
# if Use_Patch_Position_Loss:
#     position_loss = PatchPositionCriterion(
#         feat_dim=model.embed_dim,
#         num_classes=model.patch_embed.num_patches
#     ).to(DEVICE)
