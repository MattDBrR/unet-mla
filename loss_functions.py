import torch
import torch.nn as nn
import torch.nn.functional as F 

#generated
class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, pos_weight=None, eps=1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.eps = eps
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, target):
        # logits: [B,1,H,W], target: [B,1,H,W] float {0,1}
        bce = self.bce(logits, target)

        probs = torch.sigmoid(logits)
        probs = probs.flatten(1)
        target_f = target.flatten(1)

        inter = (probs * target_f).sum(dim=1)
        denom = probs.sum(dim=1) + target_f.sum(dim=1)
        dice = (2 * inter + self.eps) / (denom + self.eps)
        dice_loss = 1 - dice.mean()

        return self.bce_weight * bce + self.dice_weight * dice_loss


class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    def forward(self, logits, target, w_map):
        """
        logits: [B,1,H,W]
        target: [B,1,H,W] float {0,1}
        w_map:  [B,1,H,W] float
        """
        loss_map = self.bce(logits, target)      # [B,1,H,W]
        return (loss_map * w_map).mean()

class UNetLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target, weight_map):
        if target.dtype != torch.long:
            target = target.long()

        if target.ndim == 4:
            target = target.squeeze(1)
        if weight_map.ndim == 4:
            weight_map = weight_map.squeeze(1)
            
        loss = F.cross_entropy(pred, target, reduction='none')  # (B, H, W)
        
        weighted_loss = loss * weight_map
        
        return weighted_loss.mean()