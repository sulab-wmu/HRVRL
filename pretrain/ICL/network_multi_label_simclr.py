# -*- coding: utf-8 -*-
import torchvision.models
from torch import nn
import torch
from functools import partial
import torch.nn.functional as F

from torchvision.models.convnext import convnext_tiny



class MultiAV_RIP_SimCLR(nn.Module):
    def __init__(self,
                 num_classes=128,pretrain=False
                 ):

        super(MultiAV_RIP_SimCLR, self).__init__()
        self.num_classes = num_classes

        cut = 6
        base_model = convnext_tiny
        norm_layer = partial(LayerNorm2d, eps=1e-6)
        if pretrain:
            #layers = list(base_model(weights=ConvNeXt_Tiny_Weights,stochastic_depth_prob=0.8).features)[:cut]
            layers = list(base_model(stochastic_depth_prob=0.8).features)[:cut]
        else:
            layers = list(base_model(stochastic_depth_prob=0.1).features)[:cut]

        base_layers = nn.Sequential(*layers)
        self.sn_unet = base_layers

        #self.avg = GeM(2)
        self.avg = nn.AdaptiveAvgPool2d(1)
        # self.final = nn.Sequential(nn.Conv2d(384*2, self.num_classes, kernel_size=(1, 1), stride=(1, 1), bias=False),nn.Flatten(1))
        if pretrain:
            
            self.head = nn.Sequential(norm_layer(384),nn.Flatten(1), nn.Linear(384, 384),nn.GELU(), nn.Linear(384, self.num_classes))
            self.head2 = nn.Sequential(norm_layer(384),nn.Flatten(1), nn.Linear(384,2))
            #self.head2 = nn.Linear(128, 2)
        else:
            self.head = nn.Sequential(norm_layer(384),nn.Flatten(1), nn.Linear(384, 384),nn.GELU(), nn.Linear(384, self.num_classes))
            self.head2 = nn.Sequential(norm_layer(384),nn.Flatten(1), nn.Linear(384,2))
            #self.head2 = nn.Linear(128, 2)


        if not pretrain:
            print("===============train from scratch================")
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x,x2):

        out = {}
        out['positive']=[]
        out['negative'] = []
        
        for key, value in x.items():
            # value[0], value[1]
            for val in value:
                val = self.sn_unet(val)
                val = self.avg(val)
                val = self.head(val)

                out[key].append(val)

        
        x2 = self.sn_unet(x2)
        x2 = self.avg(x2)
        x2  = self.head2(x2)

        return out,x2




def load_checkpoint(model,pt):
    # self.netG.load_state_dict(torch.load(model_path), strict=False)
    #pt = torch.load(model_path)
    model_static = model.state_dict()
    pt_={}
    for k, v in pt.items():
        if k in model_static and 'head' not in k:
            pt_[k]=v
        else:
            print(k)
            
    #pt_ = {k: v for k, v in pt.items() if k in model_static}
    model_static.update(pt_)
    model.load_state_dict(model_static)
    print("Loading pretrained model")
    return model
class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x
if __name__ == '__main__':
    


    

    m = MultiAV(num_classes=2)
    


    x = torch.randn((2,3,256,256))



    # y = torch.tensor([[0,1,2],[1,1,2]]).float()
    out = m(x)

    print(m(x).shape)
