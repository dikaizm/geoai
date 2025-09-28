import torch
import torch.nn as nn
import torch.nn.functional as F

class DECBLoss(nn.Module):
    """
    Dynamic Effective Class Balanced Loss for semantic segmentation.
    Reference: Zhou et al. 2023, Remote Sensing 15(7), 1768.
    """
    def __init__(self, num_classes, beta=0.9999, loss_type="ce", gamma=2.0, ignore_index=255, device="cpu"):
        """
        Args:
            num_classes (int): number of classes.
            beta (float): smoothing factor for effective number of samples.
            loss_type (str): "ce" (cross entropy) or "focal".
            gamma (float): focal loss gamma (only used if loss_type="focal").
            ignore_index (int): label value to ignore in loss computation.
            device (str): "cuda", "mps", or "cpu".
        """
        super().__init__()
        self.num_classes = num_classes
        self.beta = beta
        self.loss_type = loss_type
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.device = device

    def effective_num(self, samples_per_class):
        """Compute effective number of samples."""
        return (1.0 - torch.pow(self.beta, samples_per_class)) / (1.0 - self.beta)

    def forward(self, logits, targets):
        """
        Args:
            logits: (N, C, H, W) raw model outputs.
            targets: (N, H, W) ground-truth labels with values in [0..num_classes-1].
        """
        N, C, H, W = logits.shape
        assert C == self.num_classes, f"Expected {self.num_classes} classes, got {C}"

        # Flatten predictions and targets
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)
        targets_flat = targets.view(-1)

        # Remove ignore_index pixels
        mask = targets_flat != self.ignore_index
        logits_flat = logits_flat[mask]
        targets_flat = targets_flat[mask]

        if logits_flat.numel() == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # Count samples per class in this batch
        samples_per_class = torch.bincount(targets_flat, minlength=C).float()

        # Effective numbers
        effective_num = self.effective_num(samples_per_class)
        class_weights = (1.0 / effective_num).to(self.device)
        class_weights = class_weights / class_weights.sum() * C

        if self.loss_type == "ce":
            loss = F.cross_entropy(
                logits_flat, targets_flat,
                weight=class_weights,
                ignore_index=self.ignore_index
            )
        elif self.loss_type == "focal":
            ce_loss = F.cross_entropy(
                logits_flat, targets_flat,
                weight=class_weights,
                reduction="none",
                ignore_index=self.ignore_index
            )
            pt = torch.exp(-ce_loss)  # prob of correct class
            focal_loss = ((1 - pt) ** self.gamma) * ce_loss
            loss = focal_loss.mean()
        else:
            raise ValueError(f"Unknown loss_type {self.loss_type}")

        return loss