import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction  # 'mean' or 'sum'

    def forward(self, logits, targets):
        # Step 1: Compute probabilities using softmax
        probs = F.softmax(logits, dim=1)  # Shape: [batch_size, num_classes, height, width]
        print(f"[DEBUG] Softmax probabilities:\n{probs}")

        print(f"[DEBUG] Min Prob: {probs.min().item()}, Max Prob: {probs.max().item()}")
        print(f"[DEBUG] Sum of Probabilities per Pixel:\n{probs.sum(dim=1)}")


        # Step 2: One-hot encode the targets
        num_classes = logits.size(1)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2)
        print(f"[DEBUG] One-hot encoded targets:\n{targets_one_hot}")

        # Step 3: Extract probabilities for the target classes
        probs_for_targets = (probs * targets_one_hot).sum(dim=1).clamp(min=1e-10)  # Shape: [batch_size, height, width]
        print(f"[DEBUG] Probabilities for target classes:\n{probs_for_targets}")

        print(f"[DEBUG] Min Prob for Targets: {probs_for_targets.min().item()}")
        print(f"[DEBUG] Max Prob for Targets: {probs_for_targets.max().item()}")


        # Step 4: Calculate focal weights
        focal_weight = (1 - probs_for_targets) ** self.gamma  # Shape: [batch_size, height, width]
        print(f"[DEBUG] Focal weights:\n{focal_weight}")

        print(f"[DEBUG] Min Focal Weight: {focal_weight.min().item()}, Max Focal Weight: {focal_weight.max().item()}")

        # Step 5: Compute log probabilities
        log_probs = torch.log(probs_for_targets + 1e-10)  # Avoid log(0)
        print(f"[DEBUG] Log probabilities:\n{log_probs}")

        print(f"[DEBUG] Min Log Prob: {log_probs.min().item()}, Max Log Prob: {log_probs.max().item()}")
        print(f"[DEBUG] Any Log Probs NaN: {torch.isnan(log_probs).any()}")


        # Step 6: Apply alpha (class weights)
        if isinstance(self.alpha, torch.Tensor):
            alpha = self.alpha.view(1, -1, 1, 1)  # Reshape for broadcasting
            alpha = torch.gather(self.alpha, 0, targets.unsqueeze(-1).to(dtype=torch.long)).squeeze(-1)  # Extract alpha for target classes
        else:
            alpha = self.alpha  # Scalar
        print(f"[DEBUG] Alpha weights:\n{alpha}")

        # Step 7: Compute the focal loss
        loss = -alpha * focal_weight * log_probs  # Element-wise loss
        print(f"[DEBUG] Per-pixel focal loss:\n{loss}")

        # Step 8: Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, reduction='mean'):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction

    def forward(self, logits, targets):
        # Apply softmax to logits for multi-class segmentation
        probs = torch.softmax(logits, dim=1)
        targets_one_hot = nn.functional.one_hot(targets, num_classes=logits.shape[1]).permute(0, 3, 1, 2)
        targets_one_hot = targets_one_hot.to(logits.dtype)
        
        true_pos = torch.sum(probs * targets_one_hot, dim=(2, 3))  # Element-wise multiplication
        false_neg = torch.sum(targets_one_hot * (1 - probs), dim=(2, 3))
        false_pos = torch.sum((1 - targets_one_hot) * probs, dim=(2, 3))

        tversky_index = true_pos / (true_pos + self.alpha * false_neg + self.beta * false_pos + 1e-7)
        loss = 1 - tversky_index

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
