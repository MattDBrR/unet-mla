import torch.nn as nn
import torch
import torch.nn.functional as F

class UNet_mla(nn.Module):
    def __init__(self, 
                 in_channels: int =1,
                 out_channels: int =2, 
                 original: bool = True,
                 dropout: bool = True,
                 skip_connections: bool = True):
        super().__init__()

        self.original = original
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropoutB = dropout
        self.skip_connections = skip_connections
        
        self.lvl1D = self._get_seq_unet(in_channels=in_channels,
                                        out_channels=64)
        self.lvl2D = self._get_seq_unet(64,128)
        self.lvl3D = self._get_seq_unet(128,256)
        self.lvl4D = self._get_seq_unet(256,512)
        self.lvl5D = self._get_seq_unet(512,1024)

        if dropout:
            self.dropout = nn.Dropout2d(p = 0.5)
        
        self.lvl1U = self._get_seq_unet(1024 if skip_connections else 512, 512)
        self.lvl2U = self._get_seq_unet(512 if skip_connections else 256, 256)
        self.lvl3U = self._get_seq_unet(256 if skip_connections else 128, 128)
        self.lvl4U = self._get_seq_unet(128 if skip_connections else 64, 64)
        
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(128, 64,  kernel_size=2, stride=2)
    
            
        self.last = nn.Conv2d(in_channels = 64,
                              out_channels = out_channels,
                              kernel_size = 1)
        
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)

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
        if self.dropoutB:
            x5 = self.dropout(x5)

        u4 = self.up4(x5)
        if self.skip_connections:
            cat4 = self._crop_and_concat(x4, u4)
        else:
            cat4 = u4
        y4 = self.lvl1U(cat4)

        u3 = self.up3(y4)
        if self.skip_connections:
            cat3 = self._crop_and_concat(x3, u3)
        else:
            cat3 = u3
        y3 = self.lvl2U(cat3)

        u2 = self.up2(y3)
        if self.skip_connections:
            cat2 = self._crop_and_concat(x2, u2)
        else:
            cat2 = u2
        y2 = self.lvl3U(cat2)

        u1 = self.up1(y2)
        if self.skip_connections:
            cat1 = self._crop_and_concat(x1, u1)
        else:
            cat1 = u1
        y1 = self.lvl4U(cat1)

        y = self.last(y1)

        return y


    def _crop_and_concat(self, encoder_feature, decoder_feature):
        if self.original:
            # Center crop
            _, _, H_enc, W_enc = encoder_feature.shape
            _, _, H_dec, W_dec = decoder_feature.shape
            
            crop_h = (H_enc - H_dec) // 2
            crop_w = (W_enc - W_dec) // 2
            
            encoder_cropped = encoder_feature[
                :, :, 
                crop_h:crop_h + H_dec, 
                crop_w:crop_w + W_dec
            ]
        else:
            encoder_cropped = encoder_feature

        return torch.cat([encoder_cropped, decoder_feature], dim=1)

    def _get_seq_unet(self,in_channels,out_channels):
        padding = 0 if self.original else 1

        # we add batchnorm even if the paper doesn't mention it
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

