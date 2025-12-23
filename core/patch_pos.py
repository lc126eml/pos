import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional, Dict, Tuple

LossType = Literal["mse", "smooth_l1", "l1"]

class PatchRowColRegressionCriterionSimple(nn.Module):
    def __init__(self, feat_dim, grid_h, grid_w, normalize=True):
        """
        Predict row and column index of each patch via regression (single resolution).

        Args:
            feat_dim (int): Dimension of patch features (D)
            grid_h (int): Number of patch rows (fixed)
            grid_w (int): Number of patch columns (fixed)
            normalize (bool): If True, normalize row/col targets to [0, 1]
        """
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.normalize = normalize

        # Regression heads: scalar row / scalar col
        self.row_mlp = nn.Sequential(
            nn.Linear(feat_dim, 1)   # scalar row index
        )

        self.col_mlp = nn.Sequential(
            nn.Linear(feat_dim, 1)   # scalar col index
        )

        self.loss_fn = nn.SmoothL1Loss()

        # Precompute row/col targets once (N = grid_h * grid_w)
        rows_2d = torch.arange(grid_h, dtype=torch.float32).unsqueeze(1).repeat(1, grid_w)
        cols_2d = torch.arange(grid_w, dtype=torch.float32).unsqueeze(0).repeat(grid_h, 1)

        if normalize:
            rows_2d = rows_2d / (grid_h - 1)
            cols_2d = cols_2d / (grid_w - 1)

        # Flatten to 1D (N,)
        row_targets = rows_2d.flatten()
        col_targets = cols_2d.flatten()

        # Register as buffers so they move with .to(device)
        self.register_buffer("row_targets", row_targets)  # (N,)
        self.register_buffer("col_targets", col_targets)  # (N,)

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
class PatchRowColRegressionCriterionFast(nn.Module):
    # deprecated, mlp shared
    def __init__(
        self,
        feat_dim: int,
        grid_h: int,
        grid_w: int,
        normalize: bool = True,
        loss_type: LossType = "smooth_l1",
        huber_beta: Optional[float] = None,  # only used for smooth_l1; None => PyTorch default
    ):
        """
        Predict row/col index of each patch via regression (single resolution).

        Args:
            feat_dim: feature dim D
            grid_h, grid_w: patch grid size (fixed)
            normalize: if True, targets are normalized to [0,1]
            loss_type: one of {"mse","smooth_l1","l1"}
            huber_beta: SmoothL1 transition point. If None, use PyTorch default.
        """
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.normalize = normalize
        self.loss_type = loss_type
        self.huber_beta = huber_beta

        # One head, two outputs: (row, col)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),
        )
        self._init_mlp()

        # Precompute targets: (1, N, 2)
        rows = torch.arange(self.grid_h, dtype=torch.float32).unsqueeze(1).repeat(1, self.grid_w)
        cols = torch.arange(self.grid_w, dtype=torch.float32).unsqueeze(0).repeat(self.grid_h, 1)

        if self.normalize:
            rows = rows / max(self.grid_h - 1, 1)
            cols = cols / max(self.grid_w - 1, 1)

        targets = torch.stack([rows.flatten(), cols.flatten()], dim=-1).unsqueeze(0)  # (1, N, 2)
        self.register_buffer("targets", targets)

        if loss_type == "mse":
            self._loss = lambda pred, tgt: F.mse_loss(pred, tgt, reduction="mean")
        elif loss_type == "l1":
            self._loss = lambda pred, tgt: F.l1_loss(pred, tgt, reduction="mean")
        elif loss_type == "smooth_l1":
            if huber_beta is None:
                self._loss = lambda pred, tgt: F.smooth_l1_loss(pred, tgt, reduction="mean")
            else:
                self._loss = lambda pred, tgt: F.smooth_l1_loss(pred, tgt, reduction="mean", beta=huber_beta)
        else:
            raise ValueError(f"Unsupported loss_type={self.loss_type}. Use 'mse', 'smooth_l1', or 'l1'.")

    def _init_mlp(self):
        # First linear: standard init
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.zeros_(self.mlp[0].bias)

        # Last linear: near-zero so it doesn't dominate early training
        nn.init.xavier_uniform_(self.mlp[2].weight)
        # nn.init.normal_(self.mlp[2].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.mlp[2].bias)

        # Optional: start at center of [0,1] if normalize=True
        # This is often stable for L1 / SmoothL1
        if self.normalize:
            self.mlp[2].bias.data.fill_(0.5)
    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """
        feats: (B, N, D), N = grid_h * grid_w
        returns: scalar loss
        """
        B, N, D = feats.shape
        if N != self.grid_h * self.grid_w:
            raise ValueError(f"Expected N={self.grid_h*self.grid_w}, got N={N}")

        pred = self.mlp(feats)                # (B, N, 2)
        tgt = self.targets.expand(B, -1, -1)  # view, no allocation

        return self._loss(pred.float(), tgt.float())

class PatchRowColRegressionCriterion(nn.Module):
    def __init__(self, feat_dim, grid_h, grid_w, normalize=True, huber_beta=None):
        """
        Predict row and column index of each patch via regression (single resolution).

        Args:
            feat_dim (int): Dimension of patch features (D)
            grid_h (int): Number of patch rows (fixed)
            grid_w (int): Number of patch columns (fixed)
            normalize (bool): If True, normalize row/col targets to [0, 1]
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

        if huber_beta is None:
            self.loss_fn = nn.SmoothL1Loss()
        else:
            self.loss_fn = nn.SmoothL1Loss(beta=0.5/self.grid_h)

        # Precompute row/col targets once (N = grid_h * grid_w)
        rows_2d = torch.arange(grid_h, dtype=torch.float32).unsqueeze(1).repeat(1, grid_w)
        cols_2d = torch.arange(grid_w, dtype=torch.float32).unsqueeze(0).repeat(grid_h, 1)

        if normalize:
            rows_2d = rows_2d / (grid_h - 1)
            cols_2d = cols_2d / (grid_w - 1)

        # Flatten to 1D (N,)
        row_targets = rows_2d.flatten()
        col_targets = cols_2d.flatten()

        # Register as buffers so they move with .to(device)
        self.register_buffer("row_targets", row_targets)  # (N,)
        self.register_buffer("col_targets", col_targets)  # (N,)

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

class PatchRowColRegressionCriterionDynamicFast(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        max_grid_h: int,
        max_grid_w: int,
        normalize: bool = True,
        loss_type: LossType = "smooth_l1",
        huber_beta: Optional[float] = None,
        cache_targets: bool = True,
    ):
        """
        Fast dynamic-resolution row/col regression.

        Args:
            feat_dim: feature dim D
            max_grid_h/w: upper bound of patch grid (used only for validation)
            normalize: targets in [0,1] based on current hp/wp
            loss_type: {"mse","smooth_l1","l1"}
            huber_beta: SmoothL1 beta; None uses PyTorch default
            cache_targets: cache target tensors per (hp,wp) to avoid recompute
        """
        super().__init__()
        self.max_grid_h = int(max_grid_h)
        self.max_grid_w = int(max_grid_w)
        self.normalize = bool(normalize)
        self.cache_targets = bool(cache_targets)

        # Fused head: predict (row, col)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),
        )

        # Pre-bind loss function to avoid forward-time branching
        if loss_type == "mse":
            self._loss = lambda pred, tgt: F.mse_loss(pred, tgt, reduction="mean")
        elif loss_type == "l1":
            self._loss = lambda pred, tgt: F.l1_loss(pred, tgt, reduction="mean")
        elif loss_type == "smooth_l1":
            if huber_beta is None:
                self._loss = lambda pred, tgt: F.smooth_l1_loss(pred, tgt, reduction="mean")
            else:
                self._loss = lambda pred, tgt: F.smooth_l1_loss(pred, tgt, reduction="mean", beta=huber_beta)
        else:
            raise ValueError(f"Unsupported loss_type={self.loss_type}. Use 'mse', 'smooth_l1', or 'l1'.")

        # Python cache: {(hp,wp,device_str,dtype_str): targets(1,N,2)}
        # Safe because targets are small and resolutions are from a limited set.
        self._tgt_cache: Dict[Tuple[int, int, str, str], torch.Tensor] = {}

    @torch.no_grad()
    def _make_targets(self, hp: int, wp: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Build targets of shape (1, N, 2) for current (hp,wp), on correct device/dtype.
        Uses torch operations on the target device (GPU-friendly).
        """
        # row indices: (hp,1) broadcast to (hp,wp)
        rows = torch.arange(hp, device=device, dtype=dtype).unsqueeze(1).expand(hp, wp)
        cols = torch.arange(wp, device=device, dtype=dtype).unsqueeze(0).expand(hp, wp)

        if self.normalize:
            rows = rows / max(hp - 1, 1)
            cols = cols / max(wp - 1, 1)

        # (hp*wp,2) -> (1,N,2)
        tgt = torch.stack((rows.reshape(-1), cols.reshape(-1)), dim=-1).unsqueeze(0)
        return tgt  # (1, N, 2)

    def _get_targets(self, hp: int, wp: int, feats: torch.Tensor) -> torch.Tensor:
        """
        Retrieve cached targets, or create and cache them.
        """
        device = feats.device
        # Use feats.dtype so loss runs in same dtype under autocast
        dtype = feats.dtype

        key = (hp, wp, str(device), str(dtype))
        if self.cache_targets:
            tgt = self._tgt_cache.get(key, None)
            if tgt is None:
                tgt = self._make_targets(hp, wp, device=device, dtype=dtype)
                self._tgt_cache[key] = tgt
            return tgt
        else:
            return self._make_targets(hp, wp, device=device, dtype=dtype)

    def forward(self, feats: torch.Tensor, hp: int, wp: int) -> torch.Tensor:
        """
        feats: (B, N, D), where N = hp*wp
        hp/wp: ints (one resolution per batch)

        returns: scalar loss
        """
        B, N, D = feats.shape

        # Optional validation (remove for max speed once stable)
        # if hp > self.max_grid_h or wp > self.max_grid_w:
        #     raise ValueError(f"hp/wp=({hp},{wp}) exceed max=({self.max_grid_h},{self.max_grid_w})")
        # if N != hp * wp:
        #     raise ValueError(f"Expected N=hp*wp={hp*wp}, got N={N}")

        pred = self.mlp(feats)                    # (B, N, 2)
        tgt = self._get_targets(hp, wp, feats)    # (1, N, 2)
        tgt = tgt.expand(B, -1, -1)               # (B, N, 2), view

        return self._loss(pred, tgt)

class PatchRowColRegressionCriterionDynamic(nn.Module):
    def __init__(self, feat_dim, grid_h, grid_w, normalize=True):
        """
        Predict row and column index of each patch via regression,
        supporting dynamic training resolutions.

        Args:
            feat_dim (int): Dimension of patch features (D)
            grid_h (int): Max number of patch rows (upper bound)
            grid_w (int): Max number of patch columns (upper bound)
            normalize (bool): If True, normalize row/col targets to [0, 1]
                              based on the *current* hp/wp for each batch.
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

        self.loss_fn = nn.SmoothL1Loss()

        # Precompute integer row/col indices (max grid) as floats
        rows = torch.arange(grid_h, dtype=torch.float32).unsqueeze(1).repeat(1, grid_w)  # (grid_h, grid_w)
        cols = torch.arange(grid_w, dtype=torch.float32).unsqueeze(0).repeat(grid_h, 1)  # (grid_h, grid_w)

        self.register_buffer("row_index_full", rows)  # (grid_h, grid_w)
        self.register_buffer("col_index_full", cols)  # (grid_h, grid_w)

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
        self.register_buffer("row_labels", rows)
        self.register_buffer("col_labels", cols)

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
        self.register_buffer("patch_positions", torch.arange(num_classes))  # shape (num_patches,)
        
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
        self.register_buffer("position_targets", position_targets)  # (N,)

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
        self.register_buffer("position_index_full", positions) 

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
