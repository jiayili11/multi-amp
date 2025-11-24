import torch
import torch.nn.functional as F

def kl_divergence_loss(p, q):
    """Compute symmetric KL divergence between two probability distributions."""
    # p and q are logits
    log_p = F.log_softmax(p, dim=-1)
    log_q = F.log_softmax(q, dim=-1)
    
    # KL(p || q)
    kl_p_q = F.kl_div(log_p, F.softmax(q.detach(), dim=-1), reduction='batchmean')
    # KL(q || p)
    kl_q_p = F.kl_div(log_q, F.softmax(p.detach(), dim=-1), reduction='batchmean')
    
    # Symmetric KL divergence
    return (kl_p_q + kl_q_p) / 2


class SupConLoss(torch.nn.Module):
    """
    Supervised Contrastive Learning loss function - CORRECTED VERSION
    """
    def __init__(self, temperature=0.1, contrast_mode='all'):  # Lower temperature
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode

    def forward(self, features, labels):
        """
        Args:
            features: hidden vector of shape [bsz, n_views, dim] or [bsz, dim]
            labels: ground truth of shape [bsz]
        """
        device = features.device
        
        # Ensure features are 2D [batch_size, feature_dim]
        if len(features.shape) == 3:
            # If multi-view [batch_size, n_views, feature_dim]
            batch_size, n_views, feature_dim = features.shape
            features = features.contiguous().view(batch_size * n_views, feature_dim)
            labels = labels.repeat(n_views)
        elif len(features.shape) != 2:
            raise ValueError('Features shape should be [bsz, dim] or [bsz, n_views, dim]')
        
        batch_size = features.shape[0]
        
        # Feature normalization - key for contrastive learning!
        features = F.normalize(features, p=2, dim=1)
        
        # Create label mask
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # Compute similarity matrix
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )
        
        # Subtract max for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # Create mask, exclude self-contrast
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # Compute contrastive loss
        exp_logits = torch.exp(logits) * logits_mask
        
        # Count positive samples for each anchor
        positive_pair_counts = mask.sum(1)
        
        # Compute log probability
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        
        # Compute mean log probability
        mean_log_prob_pos = (mask * log_prob).sum(1) / (positive_pair_counts + 1e-8)
        
        # Only compute loss for anchors with positive samples
        valid_indices = positive_pair_counts > 0
        if valid_indices.sum() > 0:
            loss = -mean_log_prob_pos[valid_indices].mean()
        else:
            loss = torch.tensor(0.0, device=device)
        
        return loss
    

# ==============================================================================
# New: Secondary structure prediction loss function
# ==============================================================================

import torch
import torch.nn.functional as F

class FocalLoss(torch.nn.Module):
    """Focal Loss for class imbalance"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None
    
    def forward(self, inputs, targets):
        weight = None
        if self.alpha is not None:
            weight = self.alpha.to(dtype=inputs.dtype, device=inputs.device)
        
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class StructureAwareLoss(torch.nn.Module):
    def __init__(self, num_classes=3, class_weights=None, gamma=2.0, 
                 continuity_weight=0.05, use_crf=True, ignore_index=-1):
        super().__init__()
        self.num_classes = num_classes
        self.continuity_weight = continuity_weight
        self.use_crf = use_crf
        self.ignore_index = ignore_index
        
        self.focal_loss = FocalLoss(alpha=class_weights, gamma=gamma)
        
        if use_crf:
            from torchcrf import CRF
            self.crf = CRF(num_classes, batch_first=True)
    
    def forward(self, logits, targets, mask=None):
        B, L, C = logits.shape
        device = logits.device
        
        if mask is None:
            mask = torch.ones(B, L, device=device)
        
        mask_bool = mask.bool()
        valid_mask = (
            mask_bool & 
            (targets != self.ignore_index) & 
            (targets >= 0) & 
            (targets < self.num_classes)
        )
        
        # Focal Loss
        logits_flat = logits.reshape(-1, C)
        targets_flat = targets.reshape(-1)
        valid_flat = valid_mask.reshape(-1)
        
        if valid_flat.sum() > 0:
            focal = self.focal_loss(logits_flat[valid_flat], targets_flat[valid_flat])
        else:
            focal = torch.tensor(0.0, device=device, dtype=logits.dtype)
        
        # CRF Loss
        crf_loss = torch.tensor(0.0, device=device, dtype=logits.dtype)
        if self.use_crf and mask_bool.sum() > 0:
            targets_for_crf = targets.clone()
            targets_for_crf[targets == self.ignore_index] = 0
            logits_fp32 = logits.float()
            
            try:
                crf_loss_fp32 = -self.crf(
                    logits_fp32.contiguous(),
                    targets_for_crf.contiguous(),
                    mask=mask_bool.contiguous(),
                    reduction='mean'
                )
                crf_loss = crf_loss_fp32.to(dtype=logits.dtype)
            except Exception as e:
                print(f"⚠️ CRF failed: {e}")
        
        # Continuity Loss
        pred_labels = logits.argmax(dim=-1)
        transitions = (pred_labels[:, 1:] != pred_labels[:, :-1]).float()
        valid_transitions = transitions * valid_mask[:, 1:].float()
        
        if valid_mask[:, 1:].sum() > 0:
            continuity_loss = (valid_transitions.sum() / valid_mask[:, 1:].sum().float()).to(dtype=logits.dtype)
        else:
            continuity_loss = torch.tensor(0.0, device=device, dtype=logits.dtype)
        
        total_loss = focal + crf_loss + self.continuity_weight * continuity_loss
        
        return {
            'loss': total_loss,
            'focal_loss': focal.item(),
            'crf_loss': crf_loss.item(),
            'continuity_loss': continuity_loss.item()
        }
    
    def decode(self, logits, mask=None):
        if not self.use_crf:
            return logits.argmax(dim=-1).tolist()
        
        if mask is None:
            mask = torch.ones(logits.shape[:2], device=logits.device)
        
        logits_fp32 = logits.float()
        return self.crf.decode(logits_fp32, mask=mask.bool())
# ==============================================================================
# Helper function: compute class weights
# ==============================================================================

def compute_ss_class_weights(dataset, num_classes=3, ignore_index=-1):
    """
    Automatically compute class weights for secondary structure
    Args:
        dataset: PeptideDataset instance
        num_classes: 3 (H/C/E, excluding padding)
        ignore_index: padding index value (-1)
    Returns:
        torch.Tensor: [num_classes] weight vector
    """
    class_counts = torch.zeros(num_classes)
    
    print("Computing secondary structure class distribution...")
    for i in range(len(dataset)):
        sample = dataset[i]
        if sample is None:  # Skip invalid samples
            continue
        
        ss_labels = sample['ss_target']  # [L]
        
        # Only count valid positions (excluding padding)
        valid_mask = (ss_labels != ignore_index) & (ss_labels >= 0) & (ss_labels < num_classes)
        
        for c in range(num_classes):
            class_counts[c] += ((ss_labels == c) & valid_mask).sum()
    
    # Prevent division by zero
    class_counts = class_counts.clamp(min=1)
    
    # Compute inverse proportion weights
    total = class_counts.sum()
    class_weights = total / (num_classes * class_counts)
    
    # Normalize
    class_weights = class_weights / class_weights.sum() * num_classes
    
    print(f"\n📊 Secondary Structure Class Statistics:")
    # According to SS_MAP order: H(0), C(1), E(2)
    ss_names = ['H', 'C', 'E']
    
    for i, name in enumerate(ss_names):
        print(f"  {name}: {int(class_counts[i]):,} ({class_counts[i]/total*100:.2f}%)")
    
    print(f"\n⚖️ Computed Class Weights:")
    for i, name in enumerate(ss_names):
        print(f"  {name}: {class_weights[i]:.4f}")
    
    return class_weights