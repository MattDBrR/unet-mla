import torch 
import torch.nn as nn
from torch.optim import AdamW
from u_net import UNet_mla
from unet_dataset import UNetDataset, DatasetName
from torch.utils.data import DataLoader
from loss_functions import *
from torch.optim.lr_scheduler import ReduceLROnPlateau

from enum import Enum

from metrics import * 

import numpy as np
from tqdm import tqdm
import os

from torch.utils.data import random_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(2025)
if torch.cuda.is_available():
    torch.cuda.manual_seed(2025)



class Interface:
    def __init__(
        self,
        out_channels: int = 2,
        epochs: int = 50,
        batch_size: int = 8,
        learning_rate: float = 1e-5,
        weight_decay: float = 1e-4,
        pos_weight_fg: float | None = None,
        optimizer=AdamW,
        save_model_path: str = "weights",
        val_split: float = 0.2,
        num_workers: int = 4,
        data_path: str = "data",
        dataset_name: DatasetName = DatasetName.DIC_HELA,
        load_model_path: str | None = None,
        augment: bool = True,
        device: str = "cuda",
        metrics_thr: int = 0.5,
        original_mode: bool = True
    ):
        
        self.original_mode = original_mode
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        self.out_channels = out_channels
        self.model_instance = UNet_mla(out_channels=self.out_channels).to(self.device)
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_class = optimizer

        self.save_model_path = os.path.join(save_model_path, dataset_name)
        self.val_split = val_split
        self.num_workers = num_workers

        data_path = data_path

        self.data_path = data_path
        self.augment = augment
        self.dataset_name = dataset_name

        os.makedirs(self.save_model_path, exist_ok=True)
        
        pos_weight = None
        if pos_weight_fg is not None:
            pos_weight = torch.tensor([pos_weight_fg], device=self.device)

        if self.out_channels == 1:
            self.criterion = WeightedBCEWithLogitsLoss(pos_weight=pos_weight).to(self.device)
        else:
            self.criterion = UNetLoss()
    
        # Dataset and dataloader
        g = torch.Generator().manual_seed(2025)
        self.full_dataset = UNetDataset(self.data_path, dataset_name=dataset_name,augment=self.augment, mode='train')
        self.dataset_test = UNetDataset(self.data_path, dataset_name=dataset_name, mode='test')
        full_dataset_val = UNetDataset(self.data_path, dataset_name=dataset_name,augment=False, mode='train')


        val_size = int(len(self.full_dataset) * val_split)
        train_size = len(self.full_dataset) - val_size

        subset_train, subset_val = random_split(
                                                self.full_dataset, 
                                                [train_size, val_size],
                                                generator=g
                                             )
        self.dataset_train = subset_train
        self.dataset_val = torch.utils.data.Subset(full_dataset_val, subset_val.indices)

            
        print(f"Total training samples: {len(self.dataset_train)}")
        print(f"Total val samples: {len(self.dataset_val)}")
        print(f"Total test samples: {len(self.dataset_test)}")

        self.train_dataloader = DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

        self.val_dataloader = DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

        self.test_dataloader = DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

        self.optimizer = self.optimizer_class(
            self.model_instance.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )
        self.best_loss = np.inf

        if load_model_path is not None:
            self._load_model(load_model_path)

        self.metrics = SegmentationMetrics(threshold=metrics_thr)
        self.train_metrics = MetricsTracker()
        self.val_metrics = MetricsTracker()

    

    def _prep_mask_and_wmap(self, mask, w_map, H, W):
        # Center-crop
        mask = self._center_crop_to(mask, H, W)
        w_map = self._center_crop_to(w_map, H, W)
        
        if self.out_channels == 1:
            # Para BCE: necesitamos [B, 1, H, W]
            mask = mask.float()
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            if w_map.ndim == 3:
                w_map = w_map.unsqueeze(1)
            
            # Normalizar mask si es necesario
            if mask.max() > 1.0:
                mask = mask / 255.0
        
        else:  # out_channels == 2
            # Para CrossEntropy: necesitamos [B, H, W] long
            mask = mask.long()
            if mask.ndim == 4:
                mask = mask.squeeze(1)  # Quitar canal
            if w_map.ndim == 4:
                w_map = w_map.squeeze(1)
        
        return mask, w_map
        
    def _save_best(self, epoch, val_loss,name='best'):
        os.makedirs(self.save_model_path, exist_ok=True)
        path = os.path.join(self.save_model_path, name+".pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state": self.model_instance.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "val_loss": float(val_loss),
                'scheduler_state': self.scheduler.state_dict()
            },
            path,
        )

    def _load_model(self,path):
        checkpoint = torch.load(
            path,
            map_location=self.device
        )
        self.model_instance.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])

        print(
            f"Loaded best model from epoch {checkpoint['epoch']} "
            f"(val loss = {checkpoint['val_loss']:.4f})"
        )


    def _center_crop_to(self, x, H_out, W_out):
        # Convert to tensor if needed
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)

        if x.ndim == 4:  # (B,C,H,W)
            _, _, H, W = x.shape
            top = (H - H_out) // 2
            left = (W - W_out) // 2
            return x[:, :, top:top+H_out, left:left+W_out]

        if x.ndim == 3:  # (B,H,W) or (C,H,W)
            H, W = x.shape[-2], x.shape[-1]
            top = (H - H_out) // 2
            left = (W - W_out) // 2
            return x[..., top:top+H_out, left:left+W_out]

        if x.ndim == 2:  # (H,W)
            H, W = x.shape
            top = (H - H_out) // 2
            left = (W - W_out) // 2
            return x[top:top+H_out, left:left+W_out]

        raise ValueError(f"x must be 2D/3D/4D, got ndim={x.ndim}")
    
    import torch.nn.functional as F

    def _pad_or_crop_center(self, x, H, W, mode="reflect"):
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)

        # asegurar (B,H,W)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        elif x.ndim == 4:
            x = x[:, 0]

        _, h, w = x.shape

        # --- PAD ---
        pad_h = max(0, H - h)
        pad_w = max(0, W - w)

        if pad_h > 0 or pad_w > 0:
            pad_top    = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left   = pad_w // 2
            pad_right  = pad_w - pad_left

            x = F.pad(
                x,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode=mode
            )

        # --- CROP (por si quedó más grande) ---
        _, h, w = x.shape
        top  = (h - H) // 2
        left = (w - W) // 2

        return x[:, top:top+H, left:left+W]



    
    
    def train(self):
        train_losses = []
        val_losses = []

        train_ious = []
        val_ious = []


        model = self.model_instance

        for epoch in range(self.epochs):
            model.train()
            train_running_loss = 0
            self.train_metrics.reset()

            for idx,img_mask_w in enumerate(tqdm(self.train_dataloader,position=0,leave=True)):
                img = img_mask_w[0].float().to(self.device,non_blocking=True)
                mask = img_mask_w[1].float().to(self.device,non_blocking=True)
                w_map = img_mask_w[2].float().to(self.device,non_blocking=True)


                logits = model(img)
                H, W = logits.shape[-2:]

                #mask_cut, w_map_cut = self._prep_mask_and_wmap(mask, w_map, H, W)
                mask_cut = self._pad_or_crop_center(mask, H, W, mode="reflect")
                w_map_cut = self._pad_or_crop_center(w_map, H, W, mode="reflect")

                mask_cut = (mask_cut > 0).long()

                loss = self.criterion(logits, mask_cut, w_map_cut)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                train_running_loss += loss.item()

                with torch.no_grad():
                    batch_metrics = self.metrics.compute_all_metrics(logits,mask_cut)
                    self.train_metrics.update(batch_metrics)
            
            train_loss = train_running_loss / max(1, len(self.train_dataloader))
            train_losses.append(train_loss)
            train_metrics_avg = self.train_metrics.get_averages()
            train_ious.append(train_metrics_avg['iou'])

            val_loss,val_metrics_avg = self.validate()
            val_losses.append(val_loss)
            val_ious.append(val_metrics_avg['iou'])

            if self.val_split > 0:
                self.scheduler.step(val_loss)
            else: 
                self.scheduler.step(train_loss)

            current_lr = self.optimizer.param_groups[0]["lr"]
            print(f"LR now: {current_lr:.2e}")
            
            
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self._save_best(epoch=epoch, val_loss=val_loss)

                print(f"Best model saved (val loss = {val_loss:.4f})")

            

            print("-" * 30)
            print(f"Epoch {epoch+1}/{self.epochs}")
            print(f"Train loss: {train_loss:.6f}")
            print(f"Val   loss: {val_loss:.6f}")
            print("-" * 30)

        #save last
        self._save_best(epoch,val_loss,'last')
        print(f"Last model saved (val loss = {val_loss:.4f})")


        return train_losses,val_losses,train_ious,val_ious

    @torch.no_grad()
    def validate(self):
        model = self.model_instance
        model.eval()

        val_running_loss = 0
        self.val_metrics.reset()

        for img,mask,w_map in tqdm(self.val_dataloader, desc='Val',leave=False):
            img = img.float().to(self.device,non_blocking = True)
            mask = mask.float().to(self.device,non_blocking = True)
            w_map = w_map.float().to(self.device,non_blocking = True)

            logits = model(img)
            H,W = logits.shape[-2:]

            #mask_cut, w_map_cut = self._prep_mask_and_wmap(mask, w_map, H, W)
            mask_cut = self._pad_or_crop_center(mask, H, W, mode="reflect")
            w_map_cut = self._pad_or_crop_center(w_map, H, W, mode="reflect")

            mask_cut = (mask_cut > 0).long()
            
            loss = self.criterion(logits, mask_cut, w_map_cut)

            val_running_loss += loss.item()

            batch_metrics = self.metrics.compute_all_metrics(logits,mask_cut)
            self.val_metrics.update(batch_metrics)

        val_loss = val_running_loss / max(1, len(self.val_dataloader))
        val_metrics_avg = self.val_metrics.get_averages()

        return val_loss, val_metrics_avg
    
    @torch.no_grad()
    def predict_proba(self, img):
        """
        img: [B,1,H,W] float tensor
        returns: probs [B,1,H,W] aligned with model output spatial size
        """
        self.model_instance.eval()
        img = img.float().to(self.device)
        logits = self.model_instance(img)
        return torch.sigmoid(logits)

    @torch.no_grad()
    def predict_mask(self, img, thr=0.5):
        probs = self.predict_proba(img)
        return (probs > thr).float()


