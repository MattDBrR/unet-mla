import os
import torch 
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import tifffile as tiff
import numpy as np
from scipy import ndimage
from scipy.ndimage import zoom, map_coordinates
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import distance_transform_edt
from skimage.measure import label
import cv2
from enum import Enum


class DatasetName(str, Enum):
    DIC_HELA= 'dic-hela'
    EM_SEG = 'em-seg'
    PHC_U373 = 'phc-u373'
    MIXED = 'mixed'

class UNetDataset(Dataset): 
    def __init__(
                self, 
                data_path: str = 'data',
                dataset_name: DatasetName = DatasetName.DIC_HELA,
                augment: bool = True,
                mode: str = 'train',
                p_flip_h: float = 0.5, 
                p_flip_v: float = 0.5,
                max_degree: float = 15.0, 
                max_shift_px: int = 28,
                elastic_sigma: int = 10,
                elastic_grid: int = 3,
                original_mode: bool = True):
        
        data_path = os.path.join(data_path,dataset_name)
        #data_path = data_path + dataset_name
        print('Dataset in ',data_path)

        self.data_path = data_path + f'/{mode}'
        self.dataset_name = dataset_name

        if dataset_name == DatasetName.MIXED:
            img_roots = [self.data_path+'/01',self.data_path+'/02',
                         self.data_path+'/03',self.data_path+'/04',self.data_path+'/05']
        elif not dataset_name == DatasetName.EM_SEG:
            img_roots = [self.data_path+'/01',self.data_path+'/02']
        else:
            img_roots = [self.data_path + '/imgs']

        

        test = mode == 'test'

        self.original_mode = original_mode
        self.context = 96 #lost pixels


        if not test:
            if dataset_name == DatasetName.MIXED:
                mask_roots = [self.data_path+'/01_GT',self.data_path+'/02_GT',
                              self.data_path+'/03_GT',self.data_path+'/04_GT',
                              self.data_path+'/05_GT']
            elif  dataset_name != DatasetName.EM_SEG:
                mask_roots = [self.data_path+'/01_GT/SEG',self.data_path+'/02_GT/SEG']
            else:
                mask_roots = [self.data_path + '/labels']
        else:
            mask_roots = None

        self.samples = []

        nb_folders = len(img_roots)

        for i in range(nb_folders):
            img_root = img_roots[i]
            if not test:
                mask_root = mask_roots[i]

                if dataset_name == DatasetName.EM_SEG:
                    masks = sorted([f for f in os.listdir(mask_root) if f.endswith(".jpg")])
                elif dataset_name == DatasetName.MIXED:
                    masks = sorted([f for f in os.listdir(mask_root) if f.endswith(".tif") or f.endswith(".jpg")])

                else:
                    masks = sorted([f for f in os.listdir(mask_root) if f.endswith(".tif")])

                for mask_name in masks:
                    if dataset_name == DatasetName.EM_SEG:
                        id = mask_name.replace("train-labels","").replace(".jpg","")
                        img_name = 'train-volume'+id+'.jpg'
                    elif dataset_name == DatasetName.MIXED:
                        if i < 4:
                            id = mask_name.replace("man_seg","").replace(".tif","")
                            img_name = 't'+id+'.tif'
                        else:
                            id = mask_name.replace("train-labels","").replace(".jpg","")
                            img_name = 'train-volume'+id+'.jpg'
                    else:
                        id = mask_name.replace("man_seg","").replace(".tif","")
                        img_name = 't'+id+'.tif'


                    img_path = os.path.join(img_root,img_name)
                    mask_path = os.path.join(mask_root,mask_name)

                    if os.path.exists(img_path):
                        self.samples.append((img_path,mask_path))
            else:
                img_root = img_roots[i]
                if dataset_name == DatasetName.EM_SEG:
                    imgs = sorted([f for f in os.listdir(img_root) if f.endswith(".jpg")])
                    
                else:
                    imgs = sorted([f for f in os.listdir(img_root) if f.endswith(".tif")])
                
                for img_name in imgs:
                    img_path = os.path.join(img_root,img_name)
                    
                    if os.path.exists(img_path):
                        self.samples.append(img_path)
        if not test:
            self.augment = augment
        else:
            self.augment = False 
            
        self.p_flip_h = p_flip_h
        self.p_flip_v = p_flip_v
        self.max_degree = max_degree
        self.max_shift_px = max_shift_px
        self.elastic_sigma = elastic_sigma
        self.elastic_grid = elastic_grid

        self.test = test
                 
    def __len__(self):
        return len(self.samples)

    
    #got from git
    def compute_unet_weight_map(self, mask, wc=None, w0=10, sigma=5):

        if wc is None:
            wc = {
                0: 1,  # background
                1: 1   # object
            }

        weight_map = np.zeros_like(mask, dtype=np.float32)
        
        for k, v in wc.items():
            weight_map[mask == k] = v 

        labeled_mask, num_cells = label(mask, return_num=True, connectivity=2)

        if num_cells < 2:
            return weight_map

        h, w = mask.shape
        distance_maps = np.ones((h, w, num_cells), dtype=np.float32) * 1e6 

        for i in range(num_cells):
            current_cell = (labeled_mask == (i + 1))
            dmap = distance_transform_edt(1 - current_cell.astype(int))
            distance_maps[:, :, i] = dmap

        distance_maps.sort(axis=2)
        d1 = distance_maps[:, :, 0]
        d2 = distance_maps[:, :, 1]
        
        border_loss = w0 * np.exp(-((d1 + d2)**2) / (2 * sigma**2))
        
        weight_map = weight_map + (border_loss * (mask == 0))
        
        return weight_map.astype(np.float32)
    

    def elastic_transform_triplet(self,image, label, weight_map, sigma=10, random_state=None):
        """
        Augmentation élastique synchronisée sur le triplet (Img, Label, Weights).
        Paramètres fixés selon le papier : Grille 3x3, Sigma 10[cite: 126, 127].
        """
        if random_state is None:
            random_state = np.random.RandomState(None)

        h, w = image.shape[:2]
        grid_size = self.elastic_grid
        
        # Génération vecteurs déplacement (Gaussian distribution, std=10) [cite: 127]
        grid_x = random_state.normal(0, sigma, size=(grid_size, grid_size))
        grid_y = random_state.normal(0, sigma, size=(grid_size, grid_size))

        # Interpolation Bicubique de la grille [cite: 128]
        yi = np.linspace(0, grid_size - 1, h)
        xi = np.linspace(0, grid_size - 1, w)
        xy_grid = np.meshgrid(yi, xi, indexing='ij')

        dx = map_coordinates(grid_x, xy_grid, order=3, mode='reflect')
        dy = map_coordinates(grid_y, xy_grid, order=3, mode='reflect')

        Y, X = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        indices = [Y + dy, X + dx]

        # Application
        image_warped = map_coordinates(image, indices, order=1, mode='reflect')
        label_warped = map_coordinates(label, indices, order=0, mode='reflect') # Nearest Neighbor
        weight_warped = map_coordinates(weight_map, indices, order=1, mode='reflect')

        return image_warped, label_warped, weight_warped

    
    def __getitem__(self, idx: int):
        
        if not self.test:
            img_path, mask_path = self.samples[idx]
        else:
            img_path = self.samples[idx]
            mask_path = None

        image = cv2.imread(img_path,cv2.IMREAD_UNCHANGED) 
        #image = (image > 0).astype(int)
        #image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        if mask_path is not None:
            mask  = cv2.imread(mask_path,cv2.IMREAD_UNCHANGED)
            if 'em-seg' in self.data_path or ('mixed' in self.data_path and idx > 51):
                mask = (mask > 127).astype(np.uint8)

            #else:
            #    mask = (mask > 0).astype(np.uint8)


        else:
            mask = None

        if mask is not None:
            w_map = self.compute_unet_weight_map(mask)
            assert not np.isnan(w_map).any(), f"NaN in w_map at idx {idx}"
            assert not np.isinf(w_map).any(), f"Inf in w_map at idx {idx}"
            assert w_map.min() >= 0, f"Negative weights at idx {idx}"

            if self.augment and np.random.rand() > 0.5:
                image, mask, w_map = self.elastic_transform_triplet(
                    image, mask, w_map, sigma=self.elastic_sigma
                )
        else:
            w_map = None

        image = torch.from_numpy(image).float()

        if image.max() > 1: 
            image = image / 255.0

        if image.ndim == 2:
            image = image.unsqueeze(0)   # (1,H,W)
        elif image.ndim == 3:
            image = image.permute(2,0,1)  # (C,H,W)

        context = self.context  # 92
        image = F.pad(
            image,
            pad=(context, context, context, context),
            mode="reflect"
        )

        if self.test:
            return image


        mask = (torch.from_numpy(mask).long() > 0).long()


        w_map = torch.from_numpy(w_map).float()

        return image, mask, w_map
            






