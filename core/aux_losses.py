if Use_Relative_Direction_Loss==8:
    direction_map = {
                # (0, 0): 0,   # same patch (center)
                (0, 1): 1,   # right
                (0, -1): 2,  # left
                (1, 0): 3,   # down
                (-1, 0): 4,  # up
                (1, 1): 5,   # down-right
                (1, -1): 6,  # down-left
                (-1, 1): 7,  # up-right
                (-1, -1): 8  # up-left
            }
    rc_classes = 9
if Use_Relative_Direction_Loss==4:
    direction_map = {
                # (0, 0): 0,   # same patch (center)
                (0, 1): 1,   # right
                (0, -1): 2,  # left
                (1, 0): 3,   # down
                (-1, 0): 4,  # up
                # (1, 1): 5,   # down-right
                # (1, -1): 6,  # down-left
                # (-1, 1): 7,  # up-right
                # (-1, -1): 8  # up-left
            }
    rc_classes = 5
if Use_Relative_Direction_Loss>0:
    def get_relative_direction(p1, p2):
        """
        Example placeholder function.
        Define mapping from relative (dy, dx) to class id (0–8).
        """
        dy = p2[0] - p1[0]
        dx = p2[1] - p1[1]   
        if Discard_FAR>0:
            if abs(dy) > Discard_FAR or abs(dx) > Discard_FAR:
                return -1
        return direction_map.get((dy,dx), 0)  # default self if outside range
            
    def precompute_patch_pairs(grid_h, grid_w):
        pos = [(i, j) for i in range(grid_h) for j in range(grid_w)]
        pairs, labels = [], []
    
        for idx1, p1 in enumerate(pos):
            for idx2, p2 in enumerate(pos):
                if idx1 == idx2:
                    continue
                label = get_relative_direction(p1, p2)  # 0–8 classes
                if label < 0:
                    continue
                pairs.append((idx1, idx2))
                labels.append(label)
    
        return torch.tensor(pairs), torch.tensor(labels)
    
    # Example: 14x14 patches
    grid_h, grid_w = model.patch_embed.grid_size
    pairs, labels = precompute_patch_pairs(grid_h, grid_w)
    labels = labels.to(DEVICE)
    pairs = pairs.to(DEVICE)
    # pairs.shape = (num_pairs, 2), labels.shape = (num_pairs,)

    pos_mask = labels > 0
    pos_indices = torch.nonzero(pos_mask, as_tuple=False).squeeze()
    
    # Sample positive indices with Bernoulli probability
    num_pos = len(pos_indices)

     # Identify negative pairs (labels == 0)
    neg_mask = labels == 0
    neg_indices = torch.nonzero(neg_mask, as_tuple=False).squeeze()
    num_neg = len(neg_indices)

    def random_sample(pairs, labels, sample_rate=0.5):
        """
        Samples pairs such that the number of negative pairs (label 0) equals the number of 
        sampled positive pairs (labels > 0), where positive pairs are sampled with the given rate.
        
        Args:
            pairs (torch.Tensor): Tensor of shape (num_pairs, 2) containing pair indices.
            labels (torch.Tensor): Tensor of shape (num_pairs,) containing labels.
            sample_rate (float): Probability for sampling each positive pair.
        
        Returns:
            sampled_pairs (torch.Tensor): Sampled pairs.
            sampled_labels (torch.Tensor): Corresponding sampled labels.
        """
                
        sample_mask_pos = torch.rand(num_pos, device=pos_indices.device) < sample_rate
        sampled_pos_indices = pos_indices[sample_mask_pos]
        num_sampled_neg = min(round(len(sampled_pos_indices) * Neg_Ratio), num_neg)
        
        perm = torch.randperm(num_neg, device=neg_indices.device)[:num_sampled_neg]
        sampled_neg_indices = neg_indices[perm]
        
        # Combine and shuffle indices
        all_sampled_indices = torch.cat([sampled_pos_indices, sampled_neg_indices])
        perm = torch.randperm(len(all_sampled_indices), device=all_sampled_indices.device)
        all_sampled_indices = all_sampled_indices[perm]
        
        # Extract sampled pairs and labels
        sampled_pairs = pairs[all_sampled_indices]
        sampled_labels = labels[all_sampled_indices]
        del perm, all_sampled_indices, sample_mask_pos, sampled_pos_indices, sampled_neg_indices        
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()
        
        return sampled_pairs, sampled_labels
    
    class RelativeDirectionCriterion(nn.Module):
        def __init__(self, feat_dim, hidden_dim=256, num_classes=rc_classes, sample_rate=1.0):
            """
            Args:
                feat_dim (int): Dimension of input features (D).
                hidden_dim (int): Hidden layer size for MLP.
                num_classes (int): Number of relative direction classes (default 9 for 8-neighbor + self).
            """
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(2 * feat_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes)
            )
            self.ce = nn.CrossEntropyLoss()
            self.sample_rate = sample_rate
            
        def forward(self, feats, pairs=pairs, labels=labels, chunk_size=196):
            # feats: (B, N, D)
            sampled_pairs, sampled_labels = random_sample(pairs, labels, sample_rate=self.sample_rate)
            idx1, idx2 = sampled_pairs[:, 0], sampled_pairs[:, 1]
            total_loss = 0.0
            count = 0
        
            for start in range(0, len(sampled_pairs), chunk_size):
                end = min(len(sampled_pairs), start + chunk_size)
                p1, p2 = idx1[start:end], idx2[start:end]
                lbl = sampled_labels[start:end]
                actual_chunk_size = len(p1)
        
                f1 = feats[:, p1, :]  # (B, chunk, D)
                f2 = feats[:, p2, :]
                labels_expanded = lbl.repeat(feats.size(0))

                f1 = f1.reshape(-1, feats.size(-1))
                f2 = f2.reshape(-1, feats.size(-1))
                x = torch.cat([f1, f2], dim=-1)
                logits = self.mlp(x)
                loss = self.ce(logits, labels_expanded)
                total_loss += loss * actual_chunk_size
                count += actual_chunk_size
            del sampled_pairs, sampled_labels, idx1, idx2, p1, p2, lbl           
        
            return total_loss / count
            
    # direction_loss = RelativeDirectionCriterion(feat_dim=model.embed_dim, num_classes=rc_classes, sample_rate=SAMPLE_RATE).to(DEVICE)

