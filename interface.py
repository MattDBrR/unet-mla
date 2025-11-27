import torch 
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from u_net import UNet_mla
from unet_dataset import UNetDataset
from torch.utils.data import DataLoader
from loss_functions import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import KFold

import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(2025)
torch.cuda.manual_seed(2025)


class Interface:
    def __init__(
                self,
                epochs: int = 50,
                batch_size: int = 8,
                learning_rate: float = 0.001,
                weight_decay: float = 1e-4,
                loss_function: torch.optim = CombinedLoss,
                optimizer: torch.optim = AdamW,
                scheduler: torch.optim.lr_scheduler = None,
                save_model_path: str = "weights",
                val_split: float = 0.2,
                num_workers: int = 4,
                data_path: str = "data",
                load_model_path: str = None,
                augment: bool = True,
                use_kfold: bool = True,
                k_folds: int = 5
                ):
        
        self.model_instance = UNet_mla().to(device)

        if load_model_path:
            try:
                checkpoint = torch.load(load_model_path,map_location=device)
                if isinstance(checkpoint,dict) and "model_state_dict" in checkpoint:
                    self.start_epoch = self.load_checkpoint(load_model_path)
                else:
                    self.model_instance.load_state_dict(checkpoint)
            except Exception as e:
                print(f"Failed to load model from {load_model_path}: {e}")

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer(self.model_instance.parameters(),lr=learning_rate,weight_decay=weight_decay)
        self.save_model_path = save_model_path
        self.val_split = val_split
        self.use_kfold = use_kfold
        self.best_loss = float('inf')
        self.best_f1_score = 0.0

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        self.num_workers = num_workers
        self.augment = augment


        #Data configuration
        self.data_path = data_path
    
        #Dataset and dataloader

        self.full_dataset = UNetDataset(self.data_path, augment=True, mode='train')
        self.dataset_test = UNetDataset(self.data_path, mode='test')

        print(f"Total training samples: {len(self.full_dataset)}")
        print(f"Total test samples: {len(self.dataset_test)}")

        if self.use_kfold:
            self.kfold = KFold(n_splits=k_folds, shuffle=True, random_state=2025)
            print(f"Using {k_folds}-Fold Cross-Validation")
        else:
            print(f"Using single train/val split ({int((1-val_split)*100)}/{int(val_split*100)})")
        

        self.test_dataloader = DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        
        self.loss_function = loss_function()

    def train(self):
        self.model_instance.train()
        start = getattr(self, 'start_epoch', 0)

        for epoch in range(start, self.epochs):
            running_loss = 0.0

            for i, (img, mask) in enumerate(tqdm.tqdm(
                self.train_dataloader, 
                desc=f'Training Epoch {epoch+1}/{self.epochs}'
            )):
                img = img.to(device)
                mask = mask.to(device)

                output = self.model_instance(img)
                loss = self.loss_function(output, mask)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
            
            avg_train_loss = running_loss / len(self.train_dataloader)
            
            avg_val_loss = self.validate()
            
            print(f'Epoch {epoch+1}/{self.epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}')

            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(avg_val_loss)  
                else:
                    self.scheduler.step()
                
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f'Learning rate: {current_lr:.6f}')
            
            if avg_val_loss < self.best_loss:
                self.best_loss = avg_val_loss
                self.save_checkpoint(epoch)

    def validate(self):
        self.model_instance.eval()
        running_loss = 0.0
        
        with torch.no_grad():
            for img, mask in self.val_dataloader:
                img, mask = img.to(device), mask.to(device)
                output = self.model_instance(img)
                loss = self.loss_function(output, mask)
                running_loss += loss.item()
        
        return running_loss / len(self.val_dataloader)
    
    def predict(self):
        self.model_instance.eval()
        predictions = []
        
        with torch.no_grad():
            for img in tqdm.tqdm(self.test_dataloader, desc='Predicting'):
                if isinstance(img, tuple): 
                    img = img[0]
                
                img = img.to(device)
                output = self.model_instance(img)
                
                probs = torch.sigmoid(output)
                
                pred_masks = (probs > 0.5).float()
                
                predictions.append(pred_masks.cpu())
        
        return torch.cat(predictions, dim=0)
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=device)
        self.model_instance.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  # ✅
        
        start_epoch = checkpoint.get('epoch', 0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        print(f"Checkpoint loaded from epoch {start_epoch}")
        return start_epoch
    
    def save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model_instance.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None  # ✅
        }
        
        os.makedirs(self.save_model_path, exist_ok=True)
        path = f"{self.save_model_path}/best_model.pth"
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")
    


    



