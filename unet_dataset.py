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


class UNetDataset(Dataset): 
    def __init__(
                self, 
                data_path: str,
                augment: bool = True,
                mode: str = 'train',
                p_flip_h: float = 0.5, 
                p_flip_v: float = 0.5,
                max_degree: float = 15.0, 
                max_shift_px: int = 28,
                elastic_sigma: int = 10,
                elastic_grid: int = 3):
        
        self.data_path = data_path + f'/{mode}'
        img_roots = [self.data_path+'/01',self.data_path+'/02']

        test = mode == 'test'

        if not test:
            mask_roots = [self.data_path+'/01_GT/SEG',self.data_path+'/02_GT/SEG']
        else:
            mask_roots = None

        self.samples = []

        for i in range(2):
            img_root = img_roots[i]
            if not test:
                mask_root = mask_roots[i]
                masks = sorted([f for f in os.listdir(mask_root) if f.endswith(".tif")])

                for mask_name in masks:
                    id = mask_name.replace("man_seg","").replace(".tif","")
                    img_name = 't'+id+'.tif'

                    img_path = os.path.join(img_root,img_name)
                    mask_path = os.path.join(mask_root,mask_name)

                    if os.path.exists(img_path):
                        self.samples.append((img_path,mask_path))
            else:
                img_root = img_roots[i]
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

    def _resize_pair(self, image, mask=None, size=(572, 572)):
        if image.ndim == 2:
            image = image.unsqueeze(0)
        
        image = TF.resize(
            image, size,
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True
        )

        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            mask = TF.resize(
                mask, size,
                interpolation=transforms.InterpolationMode.NEAREST
            )
            mask = mask.squeeze(0)

        return image, mask

    
    #got from git
    def compute_unet_weight_map(self, mask, wc=None, w0=10, sigma=5):
        """
        Genera el Weight Map.
        Corrección: Ahora incluye wc (pesos de clase) para balancear objeto vs fondo.
        """
        # Si no se pasan pesos, definimos por defecto que el objeto (1) pesa más
        if wc is None:
            wc = {
                0: 1,  # Fondo
                1: 5   # Objeto (Interior de la célula)
            }

        # 1. GENERAR MAPA DE PESOS BASE (wc)
        # En lugar de np.ones_like, usamos los valores de wc
        weight_map = np.zeros_like(mask, dtype=np.float32)
        
        # Asignamos el peso correspondiente a cada píxel según si es fondo (0) u objeto (1)
        for k, v in wc.items():
            # mask == k selecciona los pixeles de esa clase
            weight_map[mask == k] = v 

        # ---------------------------------------------------------
        # 2. CALCULAR PESOS DE BORDE (w0 * exp(...))
        # (Esta parte de tu código estaba bien, la mantenemos igual)
        
        labeled_mask, num_cells = label(mask, return_num=True, connectivity=2)

        if num_cells < 2:
            # Si hay menos de 2 células, no hay "fronteras" entre células que separar.
            # Devolvemos solo el mapa de pesos de clase.
            return weight_map

        h, w = mask.shape
        distance_maps = np.ones((h, w, num_cells), dtype=np.float32) * 1e6 

        for i in range(num_cells):
            current_cell = (labeled_mask == (i + 1))
            # Distancia euclidiana inversa
            dmap = distance_transform_edt(1 - current_cell.astype(int))
            distance_maps[:, :, i] = dmap

        # Ordenar para obtener d1 y d2
        distance_maps.sort(axis=2)
        d1 = distance_maps[:, :, 0]
        d2 = distance_maps[:, :, 1]
        
        # Calcular la pérdida de borde
        border_loss = w0 * np.exp(-((d1 + d2)**2) / (2 * sigma**2))
        
        # Aplicar borde SOLO al fondo (mask == 0), sumándolo al peso base
        weight_map = weight_map + (border_loss * (mask == 0))
        
        return weight_map.astype(np.float32)
    
    def _elastic_transform_paired(image, label, sigma=10, random_state=None):
        if random_state is None:
            random_state = np.random.RandomState(None)

        h, w = image.shape[:2]
        
        grid_size = 3
        grid_x = random_state.normal(0, sigma, size=(grid_size, grid_size))
        grid_y = random_state.normal(0, sigma, size=(grid_size, grid_size))

        yi = np.linspace(0, grid_size - 1, h)
        xi = np.linspace(0, grid_size - 1, w)
        xy_grid = np.meshgrid(yi, xi, indexing='ij')

        dx = ndimage.map_coordinates(grid_x, xy_grid, order=3, mode='reflect')
        dy = ndimage.map_coordinates(grid_y, xy_grid, order=3, mode='reflect')

        Y, X = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        indices = [Y + dy, X + dx]

        image_warped = ndimage.map_coordinates(
            image, indices, order=1, mode='reflect'
        )

        label_warped = ndimage.map_coordinates(
            label, indices, order=0, mode='reflect'
        )

        return image_warped, label_warped

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

        image = tiff.imread(img_path) 

        if mask_path is not None:
            mask  = tiff.imread(mask_path)
        else:
            mask = None

        if self.augment and mask is not None:
            w_map = self.compute_unet_weight_map(mask)
            if np.random.rand() > 0.5:
                print('Applying transformation')
                image,mask,w_map = self.elastic_transform_triplet(image,mask,w_map,
                                                              sigma=self.elastic_sigma)


        else:
            if mask is not None:
                w_map = self.compute_unet_weight_map(mask)
            else:
                w_map = None

        image = torch.from_numpy(image).float()

        if image.max() > 1: 
            image = image / 255.0

        if image.ndim == 2:
            image = image.unsqueeze(0)   # (1,H,W)
        elif image.ndim == 3:
            image = image.permute(2,0,1)  # (C,H,W)

        if mask is not None:
            mask  = torch.from_numpy(mask).float()
            w_map = torch.from_numpy(w_map).float()

        image, mask = self._resize_pair(image, mask,size=(572,572))

        if w_map is not None:
            w_map = w_map.unsqueeze(0)  # (1, H, W) para resize
            w_map = TF.resize(
                w_map, 
                size=(572, 572),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True
            )
            w_map = w_map.squeeze(0)


        if self.test:
            return image

        mask = (mask > 0).long()

        return image,mask,w_map
    






