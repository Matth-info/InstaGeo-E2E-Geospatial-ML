 
import torch
import torch.nn as nn
import torch.nn.functional as F

class MulticlassDiceLoss(nn.Module):
    """Multiclass Dice Loss with Ignore Index support."""
    
    def __init__(self, num_classes, softmax_dim=None, ignore_index=-1):
        super().__init__()
        self.num_classes = num_classes
        self.softmax_dim = softmax_dim
        self.ignore_index = ignore_index

    def forward(self, logits, targets, reduction='mean', smooth=1e-6):
        """
        Computes the Dice loss for multiclass segmentation while ignoring a specific class index.

        Args:
            logits: Predicted tensor of shape (B, C, H, W).
            targets: Ground truth tensor of shape (B, H, W) with class indices.
            reduction: Reduction method (ignored, as dice loss computes a mean inherently).
            smooth: Smoothing term to avoid division by zero.

        Returns:
            Dice loss.
        """
        # Apply softmax if needed
        probabilities = logits
        if self.softmax_dim is not None:
            probabilities = F.softmax(logits, dim=self.softmax_dim)

        # Ensure targets are long integers
        targets = targets.long()

        # Create a mask for valid pixels (ignore -1 labels)
        valid_mask = targets != self.ignore_index  # Shape: (B, H, W)

        # Set ignored labels to zero temporarily (for one-hot encoding)
        masked_targets = targets.clone()
        masked_targets[~valid_mask] = 0  # Set ignored pixels to class 0 (temporary)

        # One-hot encode targets
        targets_one_hot = F.one_hot(masked_targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        # Apply the mask: Exclude ignored pixels from loss computation
        valid_mask = valid_mask.unsqueeze(1).expand_as(targets_one_hot)  # (B, C, H, W)
        targets_one_hot = targets_one_hot * valid_mask
        probabilities = probabilities * valid_mask

        # Compute Dice loss
        intersection = (targets_one_hot * probabilities).sum(dim=(0, 2, 3))
        union = targets_one_hot.sum(dim=(0, 2, 3)) + probabilities.sum(dim=(0, 2, 3))

        # Compute per-class Dice loss
        dice_coeff = (2. * intersection + smooth) / (union + smooth)
        dice_loss = -dice_coeff.log()

        # Ignore ignored class in final loss computation
        if 0 <= self.ignore_index < self.num_classes:
            dice_loss = torch.cat([dice_loss[:self.ignore_index], dice_loss[self.ignore_index+1:]])

        return dice_loss.mean()

    