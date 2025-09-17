import torch
import torch.nn as nn

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
        
# if Use_Patch_Position_Loss:
#     position_loss = PatchPositionCriterion(
#         feat_dim=model.embed_dim,
#         num_classes=model.patch_embed.num_patches
#     ).to(DEVICE)