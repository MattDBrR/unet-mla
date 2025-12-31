import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import BCEWithLogitsLoss

class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.8, bce_weight=0.2):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()
    
    def dice_loss(self, pred, target, smooth=1e-6):
        pred = torch.sigmoid(pred)

        pred = pred.contiguous()
        target = target.contiguous()
        
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice
    
    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        bce = self.bce(pred, target)
        return self.dice_weight * dice + self.bce_weight * bce
    



#generated
import torch
import torch.nn as nn

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
        """
        Loss original de U-Net con weight map
        
        Args:
            pred: (B, C, H, W) - logits (C=num_classes, típicamente 2)
            target: (B, H, W) - clases ground truth (long tensor)
            weight_map: (B, H, W) - mapa de pesos
        """
        if target.dtype != torch.long:
            target = target.long()

        if target.ndim == 4:
            target = target.squeeze(1)
        if weight_map.ndim == 4:
            weight_map = weight_map.squeeze(1)
            
        # Cross entropy sin reducción
        loss = F.cross_entropy(pred, target, reduction='none')  # (B, H, W)
        
        # Aplicar weight map
        weighted_loss = loss * weight_map
        
        # Promediar
        return weighted_loss.mean()