import os
import torch 
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import tifffile as tiff


class UNetDataset(Dataset): 
    def __init__(self, data_path, augment=True,
                 p_flip_h=0.5, p_flip_v=0.5,
                 max_degree=15, max_shift_px=28,
                 elastic_sigma=10, elastic_grid=3):
        
        self.data_path = data_path 
        img_roots = [data_path+'/01',data_path+'/02']
        mask_roots = [data_path+'/01_GT/SEG',data_path+'/02_GT/SEG']
        self.samples = []

        for i in range(2):
            img_root = img_roots[i]
            mask_root = mask_roots[i]
            masks = sorted([f for f in os.listdir(mask_root) if f.endswith(".tif")])

            for mask_name in masks:
                id = mask_name.replace("man_seg","").replace(".tif","")
                img_name = 't'+id+'.tif'

                img_path = os.path.join(img_root,img_name)
                mask_path = os.path.join(mask_root,mask_name)

                if os.path.exists(img_path):
                    self.samples.append((img_path,mask_path))

        self.augment = augment
        self.p_flip_h = p_flip_h
        self.p_flip_v = p_flip_v
        self.max_degree = max_degree
        self.max_shift_px = max_shift_px
        self.elastic_sigma = elastic_sigma
        self.elastic_grid = elastic_grid

        self.jitter = transforms.ColorJitter(
                brightness = 0.1,
                contrast = 0.1
            )
                 
    def __len__(self):
        return len(self.samples)

    def _resize_pair(self, image, mask, size=(572,572)):
        image = TF.resize(image, size)
        mask  = TF.resize(mask, size, interpolation=transforms.InterpolationMode.NEAREST)

        return image, mask
    
    def _random_affine_pair(self, image, mask):
        degree = torch.empty(1).uniform_(-self.max_degree, self.max_degree).item()
        dx = torch.randint(-self.max_shift_px, self.max_shift_px+1, (1,)).item()
        dy = torch.randint(-self.max_shift_px, self.max_shift_px+1, (1,)).item()
        image = TF.affine(
            image, degree, (dx, dy), scale=1.0, shear=0,
            interpolation=transforms.InterpolationMode.BILINEAR
        )
        mask = TF.affine(
            mask, degree, (dx, dy), scale=1.0, shear=0,
            interpolation=transforms.InterpolationMode.NEAREST
        )
        return image, mask
    
    def _random_flip_pair(self, image, mask):
        if torch.rand(1) < self.p_flip_h:
            image = TF.hflip(image)
            mask  = TF.hflip(mask)
        if torch.rand(1) < self.p_flip_v:
            image = TF.vflip(image)
            mask  = TF.vflip(mask)
        return image, mask
    
    def _random_elastic_pair(self, image, mask):
        C,H,W = image.shape

        disp_x = torch.normal(mean=0.0,std=self.elastic_sigma,size=(self.elastic_grid, self.elastic_grid)).to(image.device)
        disp_y = torch.normal(mean=0.0,std=self.elastic_sigma,size=(self.elastic_grid, self.elastic_grid)).to(image.device)
        
        disp_x = disp_x.unsqueeze(0).unsqueeze(0)
        disp_y = disp_y.unsqueeze(0).unsqueeze(0)

        disp_x_full = F.interpolate(disp_x,size=(H,W),mode='bicubic')
        disp_y_full = F.interpolate(disp_y,size=(H,W),mode='bicubic')

        disp_x_norm = disp_x_full / (W/2)
        disp_y_norm = disp_y_full / (H/2)
        disp_x_norm = disp_x_norm.squeeze(1)
        disp_y_norm = disp_y_norm.squeeze(1)
        disp_grid = torch.stack([disp_x_norm, disp_y_norm], dim=-1)

        xs = torch.linspace(-1,1,W).to(image.device)
        ys = torch.linspace(-1,1,H).to(image.device)
        grid_y, grid_x = torch.meshgrid(ys,xs,indexing='ij')

        base_grid = torch.stack([grid_x,grid_y],dim=-1)

        base_grid = base_grid.unsqueeze(0).to(image.device)

        deformed_grid = base_grid + disp_grid

        image_in = image.unsqueeze(0)
        mask_in = mask.unsqueeze(0).float()

        image_out = F.grid_sample(image_in,deformed_grid,mode='bilinear',
                                    padding_mode='border',align_corners=False)
        mask_out = F.grid_sample(mask_in,deformed_grid,mode='nearest',
                                    padding_mode='border',align_corners=False).squeeze(0).squeeze(0)

        image = image_out.squeeze(0)
        mask = mask_out.round().long()
        return image, mask


    def _random_photometric(self, image):
        image = self.jitter(image)
        return image.clamp(0,1)
    
    def __getitem__(self, idx: int):
        
        img_path, mask_path = self.samples[idx]

        image = tiff.imread(img_path) 
        mask  = tiff.imread(mask_path)

        image = torch.from_numpy(image).float()
        mask  = torch.from_numpy(mask)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        
        image = image / 255.0
        mask = mask.long()

        if image.ndim == 2:
            image = image.unsqueeze(0)   # (1,H,W)
        elif image.ndim == 3:
            image = image.permute(2,0,1)  # (C,H,W)


        image, mask = self._resize_pair(image, mask)

        if self.augment:
            image, mask = self._random_affine_pair(image, mask)
            image, mask = self._random_flip_pair(image, mask)
            image, mask = self._random_elastic_pair(image, mask)
            image = self._random_photometric(image)

        mask = mask.squeeze(0).long()
        return image,mask
    






