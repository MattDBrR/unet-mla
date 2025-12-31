import torch.nn as nn
import torch
import torch.nn.functional as F

class UNet_mla(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        
        self.lvl1D = self._get_seq_unet(in_channels=in_channels,
                                        out_channels=64)
        self.lvl2D = self._get_seq_unet(64,128)
        self.lvl3D = self._get_seq_unet(128,256)
        self.lvl4D = self._get_seq_unet(256,512)
        self.lvl5D = self._get_seq_unet(512,1024)
        
        self.lvl1U = self._get_seq_unet(1024,512)
        self.lvl2U = self._get_seq_unet(512,256)
        self.lvl3U = self._get_seq_unet(256,128)
        self.lvl4U = self._get_seq_unet(128,64)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(128, 64,  kernel_size=2, stride=2)
    
            
        self.last = nn.Conv2d(in_channels = 64,
                              out_channels = out_channels,
                              kernel_size = 1)
        
        self.pool = nn.MaxPool2d(2)

        self.apply(self._init_weights)



    def forward(self, x):

        x1 = self.lvl1D(x)
        p1 = self.pool(x1)
        x2 = self.lvl2D(p1)
        p2 = self.pool(x2)
        x3 = self.lvl3D(p2)
        p3 = self.pool(x3)
        x4 = self.lvl4D(p3)
        p4 = self.pool(x4)
        x5 = self.lvl5D(p4)

        u4 = self.up4(x5)
        x4c = self._crop(x4,u4)
        cat4 = torch.cat([x4c,u4],dim=1)
        y4 = self.lvl1U(cat4)

        u3 = self.up3(y4)
        x3c = self._crop(x3,u3)
        cat3 = torch.cat([x3c,u3],dim=1)
        y3 = self.lvl2U(cat3)

        u2 = self.up2(y3)
        x2c = self._crop(x2,u2)
        cat2 = torch.cat([x2c,u2],dim=1)
        y2 = self.lvl3U(cat2)

        u1 = self.up1(y2)
        x1c = self._crop(x1,u1)
        cat1 = torch.cat([x1c,u1],dim=1)
        y1 = self.lvl4U(cat1)

        y = self.last(y1)

        return y


    def _crop(self, skip,target):
        B, C, Hs, Ws = list(skip.size())
        B, C2, Ht, Wt = list(target.size())

        dh = Hs - Ht
        dw = Ws - Wt

        top = dh//2
        left = dw//2
        bottom = top + Ht 
        right = left + Wt 

        return skip[:,:,top:bottom,left:right]

    def _get_seq_unet(self,in_channels,out_channels,batch_norm=True):
        if not batch_norm:
            return nn.Sequential(
                nn.Conv2d(in_channels = in_channels, out_channels= out_channels,
                        kernel_size=3),
                nn.ReLU(inplace = True),
                nn.Conv2d(in_channels = out_channels, out_channels= out_channels,
                        kernel_size=3),
                nn.ReLU(inplace=True)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

