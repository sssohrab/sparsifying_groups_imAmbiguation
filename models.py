import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tools import KBest
#######################################################
class downConvBlock_I(nn.Module):
    def __init__(self, in_size, out_size, scale_factor):
        super(downConvBlock_I, self).__init__()
        #
        self.in_size = in_size
        self.out_size = out_size
        self.scale_factor = scale_factor
        #
        self.resConv = nn.Conv2d(self.in_size, self.out_size, 1, padding=0, stride=self.scale_factor)
        self.conv1 =   nn.Conv2d(self.in_size, self.out_size, 3, padding=1, stride=1)
        self.conv2 =   nn.Conv2d(self.out_size, self.out_size, 5, padding=2, stride=self.scale_factor, groups=5)
        self.bn    =   nn.BatchNorm2d(num_features=self.out_size)
        #
    def forward(self, x):
        res = self.resConv(x)
        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = x + res
        x = F.leaky_relu(x)
        return x

#######################################################
class downConvBlock_II(nn.Module):
    def __init__(self, in_size, out_size, scale_factor):
        super(downConvBlock_II, self).__init__()
        #
        self.in_size = in_size
        self.out_size = out_size
        self.scale_factor = scale_factor
        #
        self.resConv = nn.Conv2d(self.in_size, self.out_size, 1, padding=0, stride=self.scale_factor)
        self.conv1 =   nn.Conv2d(self.in_size, self.in_size, 3, padding=1, stride=1)
        self.conv2 =   nn.Conv2d(self.in_size, self.in_size, 5, padding=2, stride=1)
        self.conv3 =   nn.Conv2d(self.in_size, self.out_size, 1, padding=0, stride=self.scale_factor)
        self.bn    =   nn.BatchNorm2d(num_features=self.out_size)
    #
    def forward(self, x):
        res = self.resConv(x)
        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.bn(x)
        x = x + res
        x = F.leaky_relu(x)
        return x



##################################    
class upConvBlock_I(nn.Module):
    def __init__(self, in_size, out_size, scale_factor):
        super(upConvBlock_I, self).__init__()
        #
        self.in_size = in_size
        self.out_size = out_size
        self.scale_factor = scale_factor
        #
        self.up  = nn.Upsample(scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        self.resConv = nn.Conv2d(self.in_size, self.out_size, 1)
        self.conv1 = nn.Conv2d(self.in_size, self.out_size, 5, padding=2, stride=1)
        self.gn = nn.GroupNorm(num_groups=1, num_channels=self.out_size)  # Does it help?
        self.conv2 = nn.Conv2d(self.out_size, self.out_size, 3, padding=1, stride=1)
        # self.bn    = nn.BatchNorm2d(num_features=self.out_size)  # BN is detrimental for up-sampling layers!
       # 
    def forward(self, x):
        x = self.up(x)
        res = self.resConv(x)
        x = self.conv1(x)
        x = self.gn(x)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = x + res
        #x = self.bn(x)
        return x
###################################    
class upConvBlock_II(nn.Module):
    def __init__(self, in_size, out_size, scale_factor):
        super(upConvBlock_II, self).__init__()
        #
        self.in_size = in_size
        self.out_size = out_size
        self.scale_factor = scale_factor
        #
        self.up  = nn.Upsample(scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        self.resConv = nn.Conv2d(self.in_size, self.out_size, 1)
        self.conv1 = nn.Conv2d(self.in_size, self.in_size, 3, padding=1, stride=1)
        self.gn = nn.GroupNorm(num_groups=self.in_size, num_channels=self.in_size)  # Does it help?
        self.conv2 = nn.Conv2d(self.in_size, self.out_size, 1, padding=0, stride=1)
        #self.bn    = nn.BatchNorm2d(num_features=self.out_size)
       # 
    def forward(self, x):
        x = self.up(x)
        res = self.resConv(x)
        x = self.conv1(x)
        x = self.gn(x)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = x + res
        #x = self.bn(x)
        return x
##################################
class groupedLinearEncoder(nn.Module):
    def __init__(self, size, neck_dim, k):
        super(groupedLinearEncoder, self).__init__()
        #
        self.size = size   # (num_channels, H, W)
        self.in_dim = size[1] * size[2]
        self.num_filts = size[0]
        self.neck_dim = neck_dim
        self.k = k
        #
        self.LinearLayers = nn.ModuleList()
        for l in range(self.num_filts):
            lin = nn.Linear(self.in_dim,self.neck_dim,bias=False)
            self.LinearLayers.append(lin)
    #
    def forward(self,x):
        batch_size = x.shape[0]
        device = x.device
        code = torch.zeros(batch_size,self.num_filts,self.neck_dim).to(device)
        for l in range(self.num_filts):
            c = self.LinearLayers[l](x[:,l,:,:].view(batch_size,-1))
            c = KBest(c,self.k,symmetric=True)
            code[:,l,:] = c
            #
        return code
##################################
class groupedLinearDecoder(nn.Module):
    def __init__(self, size, weights):
        super(groupedLinearDecoder, self).__init__()
        #
        self.size = size   # (num_channels, H, W)
        self.num_filts = size[0]
        self.weights = weights
        #
    def forward(self,code):
        batch_size = code.shape[0]
        device = code.device
        x_hat = torch.zeros(batch_size,self.num_filts, self.size[1],self.size[2]).to(device)
        for l in range(self.num_filts):
            x_hat[:,l,:,:] = F.linear(
                code[:,l,:],weight=self.weights[l].weight.t()).view(
                batch_size,1,self.size[1],self.size[2]
            ).squeeze(1)
        return x_hat
##################################
class Encoder(nn.Module):
    def __init__(self, image_size, num_blocks, num_filts, scale_factor, num_codes, neck_dim, k):
        super(Encoder, self).__init__()
        #
        self.image_size = image_size  # Here I mean (C,H,W)
        self.num_blocks = num_blocks
        self.num_filts = num_filts
        self.scale_factor = scale_factor
        self.num_codes =  num_codes
        self.neck_dim = neck_dim
        self.k = k
        #
        assert(self.num_blocks == len(self.num_filts))
        assert(self.num_blocks == len(self.scale_factor))
        #
        self.downBlocks_I = nn.ModuleList()
        self.downBlocks_II = nn.ModuleList()
        in_size =  self.image_size[0]
        s = int(np.prod(self.scale_factor))
        for l in range(num_blocks):
            self.downBlocks_I.append(downConvBlock_I(in_size, self.num_filts[l], self.scale_factor[l]))
            self.downBlocks_II.append(downConvBlock_II(in_size, self.num_filts[-1], int(s)))
            in_size = num_filts[l]
            s /= scale_factor[l]
        #
        self.convPoint = nn.Conv2d(self.num_filts[-1] * (num_blocks + 1), self.num_codes, 1)
        self.LinearLayers = groupedLinearEncoder(
            (self.num_codes,
                int(image_size[1]/(np.prod(self.scale_factor))),
                int(image_size[2]/(np.prod(self.scale_factor)))),
            self.neck_dim, self.k
            )
    #
    def forward(self,x):
        for l in range(self.num_blocks):
            if l == 0:
                res = self.downBlocks_II[l](x)
            else:
                res = torch.cat((res, self.downBlocks_II[l](x)),1)
            x = self.downBlocks_I[l](x)
        res = torch.cat((res, x), 1)
        res = self.convPoint(res)
        code = self.LinearLayers(res)
        return code
 ##################################   
class Decoder(nn.Module):
    def __init__(self, image_size, num_blocks, num_filts, scale_factor, num_codes, weights):
        super(Decoder, self).__init__()
        #
        self.image_size = image_size
        self.num_blocks = num_blocks
        self.num_filts = num_filts[::-1]  #  reversed for symmetry 
        self.num_filts.append(self.image_size[0]) # image channels appended as last filter size
        self.scale_factor = scale_factor[::-1]
        self.num_codes = num_codes
        self.weights = weights
       # 
        assert(self.num_blocks == len(self.num_filts) - 1)
        assert(self.num_blocks == len(self.scale_factor))
        #
        self.LinearLayers = groupedLinearDecoder(
            (self.num_codes,
             int(image_size[2]/(np.prod(self.scale_factor))), int(image_size[1]/(np.prod(self.scale_factor)))),
            self.weights)
        self.convPoint = nn.Conv2d(self.num_codes, self.num_filts[0] * (num_blocks + 1),  1)
        #
        in_size = self.num_filts[0]
        s = 1
        self.upBlocks_I = nn.ModuleList()
        self.upBlocks_II = nn.ModuleList()
        for l in range(self.num_blocks):
            self.upBlocks_I.append(upConvBlock_I(in_size, self.num_filts[l+1], self.scale_factor[l]))
            s *= self.scale_factor[l]
            self.upBlocks_II.append(upConvBlock_II(self.num_filts[0], self.num_filts[l+1], int(s)))
            in_size = self.num_filts[l+1]
    # 
    def forward(self, code):
        x_hat_tmp = self.LinearLayers(code)
        x_hat_tmp = self.convPoint(x_hat_tmp)
        x_hat = x_hat_tmp[:, 0:self.num_filts[0],:,:]
        for l in range(self.num_blocks):
            x_hat = self.upBlocks_I[l](x_hat)
            res = x_hat_tmp[:, self.num_filts[0] * (l+1): self.num_filts[0] * (l+2),:,:]
            res = self.upBlocks_II[l](res)
            x_hat = x_hat + res
            if l < self.num_blocks -1:  # This is to avoid relu on the last block, in case sigmoid is used.
                x_hat = F.leaky_relu(x_hat)
        return x_hat
        #return x_hat.sigmoid_()
 ##################################   
class Autoencoder(nn.Module):
    def __init__(self, image_size, num_blocks, num_filts, scale_factor, num_codes, neck_dim, k):
        super(Autoencoder, self).__init__()
        #
        self.image_size = image_size
        self.num_blocks = num_blocks
        self.num_filts = num_filts
        self.scale_factor = scale_factor
        self.num_codes = num_codes
        self.neck_dim = neck_dim
        self.k = k
        #
        self.encoder = Encoder(self.image_size,
                                      self.num_blocks, self.num_filts, self.scale_factor,
                                      self.num_codes, self.neck_dim,self.k
                                     )
        self.decoder = Decoder(self.image_size, self.num_blocks,
                                       self.num_filts, self.scale_factor, self.num_codes,
                                       #None
                                       self.encoder.LinearLayers.LinearLayers
                                      )
        #
    def forward(self,x):
        code = self.encoder(x)
        x_hat = self.decoder(code)
        return x_hat, code
 ################################## 
