import torch
import numpy as np
from typing import Union

#generated
class SegmentationMetrics:
    def __init__(self, threshold: float = 0.5, eps: float = 1e-7):
        self.threshold = threshold
        self.eps = eps
    
    def compute_iou(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        pred = self._to_binary_pred(pred)
        target = self._to_binary_target(target)
        
        pred = pred.float().flatten()
        target = target.float().flatten()
        
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        
        iou = (intersection + self.eps) / (union + self.eps)
        
        return iou.item()
    
    def compute_dice(self, pred: torch.Tensor, target: torch.Tensor) -> float:        
        pred = self._to_binary_pred(pred)
        target = self._to_binary_target(target)
        
        pred = pred.float().flatten()
        target = target.float().flatten()
        
        intersection = (pred * target).sum()
        
        dice = (2.0 * intersection + self.eps) / (pred.sum() + target.sum() + self.eps)
        
        return dice.item()
    
    def compute_pixel_accuracy(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        pred = self._to_binary_pred(pred)
        target = self._to_binary_target(target)
        
        pred = pred.float().flatten()
        target = target.float().flatten()
        
        correct = (pred == target).sum()
        total = target.numel()
        
        accuracy = correct / total
        
        return accuracy.item()
    
    def compute_precision_recall(self, pred: torch.Tensor, target: torch.Tensor) -> tuple:
        pred = self._to_binary_pred(pred)
        target = self._to_binary_target(target)
        
        pred = pred.float().flatten()
        target = target.float().flatten()
        
        tp = (pred * target).sum()
        fp = (pred * (1 - target)).sum()
        fn = ((1 - pred) * target).sum()
        
        precision = (tp + self.eps) / (tp + fp + self.eps)
        recall = (tp + self.eps) / (tp + fn + self.eps)
        
        return precision.item(), recall.item()
    
    def compute_all_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        iou = self.compute_iou(pred, target)
        dice = self.compute_dice(pred, target)
        accuracy = self.compute_pixel_accuracy(pred, target)
        precision, recall = self.compute_precision_recall(pred, target)
   
        f1 = 2 * (precision * recall) / (precision + recall + self.eps)
        
        return {
            'iou': iou,
            'dice': dice,
            'pixel_accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def _binarize(self, pred: torch.Tensor) -> torch.Tensor:
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)
        
        return (pred > self.threshold).float()
    
    def _to_binary_pred(self, pred: torch.Tensor) -> torch.Tensor:
        # If it's multichannel
        if pred.ndim == 4 and pred.shape[1] > 1:
            pred = torch.softmax(pred, dim=1)
            pred = torch.argmax(pred, dim=1)  # [B, H, W]
            return pred.float()
        
        #1 channel
        elif pred.ndim == 4 and pred.shape[1] == 1:
            pred = pred.squeeze(1)  # [B, H, W]
            return self._binarize(pred)
        
        elif pred.ndim == 3:
            return self._binarize(pred)
        
        else:
            raise ValueError(f"Formato de pred no soportado: {pred.shape}")

    def _to_binary_target(self, target: torch.Tensor) -> torch.Tensor:
        #remove channel
        if target.ndim == 4:
            target = target.squeeze(1)  # [B, H, W]
        
        #binary
        return (target > 0).float()

class MetricsTracker:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.iou_sum = 0.0
        self.dice_sum = 0.0
        self.accuracy_sum = 0.0
        self.precision_sum = 0.0
        self.recall_sum = 0.0
        self.count = 0
    
    def update(self, metrics: dict):
        self.iou_sum += metrics['iou']
        self.dice_sum += metrics['dice']
        self.accuracy_sum += metrics['pixel_accuracy']
        self.precision_sum += metrics['precision']
        self.recall_sum += metrics['recall']
        self.count += 1
    
    def get_averages(self) -> dict:
        if self.count == 0:
            return {
                'iou': 0.0,
                'dice': 0.0,
                'pixel_accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0
            }
        
        return {
            'iou': self.iou_sum / self.count,
            'dice': self.dice_sum / self.count,
            'pixel_accuracy': self.accuracy_sum / self.count,
            'precision': self.precision_sum / self.count,
            'recall': self.recall_sum / self.count
        }
    
    def __str__(self) -> str:
        avg = self.get_averages()
        return (
            f"IoU: {avg['iou']:.4f} | "
            f"Dice: {avg['dice']:.4f} | "
            f"Acc: {avg['pixel_accuracy']:.4f} | "
            f"Prec: {avg['precision']:.4f} | "
            f"Rec: {avg['recall']:.4f}"
        )

