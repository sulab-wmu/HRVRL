import torch
from torch import nn
from functools import partial
from torchvision.models import convnext_tiny,ConvNeXt_Tiny_Weights
import torch.nn.functional as F
class MultiAV_RIP(nn.Module):
    def __init__(self,
                 num_classes=2,pretrain=True,drop_path=0.1
                 ):

        super(MultiAV_RIP, self).__init__()
        self.num_classes = num_classes

        cut = 6
        base_model = convnext_tiny
        norm_layer = partial(LayerNorm2d, eps=1e-6)
        if pretrain:
            #layers = list(base_model(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1,stochastic_depth_prob=drop_path).features)[:cut]
            #layers = list(base_model(stochastic_depth_prob=0.8).features)[:cut]
            layers = list(base_model(stochastic_depth_prob=drop_path).features)[:cut]
        else:
            layers = list(base_model(stochastic_depth_prob=drop_path).features)[:cut]

        base_layers = nn.Sequential(*layers)
        self.sn_unet = base_layers

        #self.avg = GeM(2)
        self.avg = nn.AdaptiveAvgPool2d(1)
        # self.final = nn.Sequential(nn.Conv2d(384*2, self.num_classes, kernel_size=(1, 1), stride=(1, 1), bias=False),nn.Flatten(1))
        if pretrain:
            #self.head = nn.Sequential(norm_layer(384),nn.Flatten(1),nn.Linear(384, self.num_classes))
            self.head = nn.Sequential(norm_layer(384),nn.Flatten(1), nn.Linear(384, 384),nn.BatchNorm1d(384,affine=True),nn.GELU(), nn.Linear(384, self.num_classes))
            #self.head = nn.Sequential(nn.LayerNorm(384), nn.GELU(),nn.Linear(384, self.num_classes))
        else:
            #self.head = nn.Sequential(norm_layer(384),nn.Flatten(1),nn.Linear(384, self.num_classes))
            #self.head = nn.Sequential(nn.LayerNorm(384), nn.GELU(),nn.Linear(384, self.num_classes))
            self.head = nn.Sequential(norm_layer(384),nn.Flatten(1), nn.Linear(384, 384),nn.BatchNorm1d(384,affine=True),nn.GELU(), nn.Linear(384, self.num_classes))
        if not pretrain:
            print("===============train from scratch================")
            self.apply(self._init_weights)

    def _init_weights(self, m):
       if isinstance(m, (nn.Conv2d, nn.Linear)):
           nn.init.trunc_normal_(m.weight, std=.02)
           #nn.init.trunc_normal_(m.weight, std=2e-5)
           nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = self.sn_unet(x)
        if len(x.shape) == 4 and x.shape[2] != x.shape[3]:
            B, H, W, C = x.shape
            x = x.permute(0, 3, 1, 2).contiguous()
        elif len(x.shape) == 3:
            B, L, C = x.shape
            h = int(L ** 0.5)
            x = x.view(B, h, h, C)
            x = x.permute(0, 3, 1, 2).contiguous()
        else:
            B, C,H, W = x.shape
            x = x

        x = self.avg(x)
        #x = x.view(B,-1)

        high_out = self.head(x)

        return high_out



class MultiAV2(nn.Module):
    def __init__(self, num_classes=2,pretrain=True):
        super(MultiAV2, self).__init__()
        base_model = convnext_tiny
        cut = 6
        if not pretrain:
            print("===============train from scratch================")
            layers = list(base_model().features)[:cut]
        else:
            print("=================train from imagenet================")
            layers = list(base_model(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1).features)[:cut]

        base_layers = nn.Sequential(*layers)
        self.high_level_classifier = nn.Sequential(
                            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                                        nn.Conv2d(384, 384*2, kernel_size=(3, 3), stride=(1, 1),padding=1, bias=False),
                                                    nn.BatchNorm2d(768),
                                                                #nn.ReLU(inplace=True),
                                                                        )
        self.adapt_global = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes
        self.mid  = nn.Sequential(nn.Flatten(1),nn.Linear(384*2, 384,bias=False),nn.BatchNorm1d(384),nn.GELU(),nn.Linear(384, 96,bias=False),nn.BatchNorm1d(96),nn.GELU())
        self.sn_unet = base_layers
        self.head = nn.Linear(96, self.num_classes)
    def forward(self, x):
        x = self.sn_unet(x)
        if len(x.shape) == 4 and x.shape[2] != x.shape[3]:
            B, H, W, C = x.shape
            x = x.permute(0, 3, 1, 2).contiguous()
        elif len(x.shape) == 3:
            B, L, C = x.shape
            h = int(L ** 0.5)
            x = x.view(B, h, h, C)
            x = x.permute(0, 3, 1, 2).contiguous()
        else:
            x=x
        high_out = x.clone()
        high_out = self.high_level_classifier(high_out)
        high_out_global = self.adapt_global(high_out)
        high_out = self.mid(high_out_global)
        high_out = self.head(high_out)
        return high_out





class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return torch.nn.functional.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)

    def __repr__(self):
        return self.__class__.__name__ + \
            '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
            ', ' + 'eps=' + str(self.eps) + ')'


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

def load_checkpoint(model,model_path):
    # self.netG.load_state_dict(torch.load(model_path), strict=False)
    checkpoint = torch.load(model_path, map_location='cpu')
    pt = None
    for model_key in ['model','model_ema']:
        if model_key in checkpoint:
            pt = checkpoint[model_key]
            print("Load state_dict by model_key = %s" % model_key)
            break
    if pt==None:
        pt = checkpoint
    model_static = model.state_dict()
    pt_ = {k: v for k, v in pt.items() if k in model_static and 'head' not in k}
    model_static.update(pt_)
    model.load_state_dict(model_static)
    print("Loading pretrained model for Generator from " + model_path)
    return model


if __name__ == '__main__':
#     from  collections import OrderedDict
    x = torch.randn((2,3,256,256))
    m = MultiAV_RIP(pretrain=False)
    print(m(x).shape)

#
#     y1 = Trans(image_size=256,patch_size=16,num_layers=12,num_heads=8,hidden_dim=768,mlp_dim=768*4,dropout=0.0,representation_size=768*2)
#     #weight_path = r'E:\eye_paper\AUV-GAN\checkpoint.pth.tar'
#     weight_path_2 = r'E:\eye_paper\AUV-GAN\trans_best_epoch90_imagenet1k.pth'
#     #pt = torch.load(weight_path,map_location='cpu')['state_dict']
#     pt_2 = torch.load(weight_path_2, map_location='cpu')
#     y1.load_state_dict(pt_2,strict=True)
#     # pt_single = OrderedDict()
#     # py_tmp = y1.state_dict()
#     # for k,v in pt.items():
#     #     k = k.split('module.')[-1]
#     #     pt_single[k] = v
#     #torch.save(pt_single,'./trans_best_epoch90_imagenet1k.pth')
#
#     #
#     # y1.load_state_dict(py_tmp,strict=True)
#     # y1.heads = nn.Linear(768,1)
#     # y = TransformerBottleNeck(x.shape[2],x.shape[1],x.shape[1],8,4)
#     print(y1(x).shape)

if __name__ == '__main__':



    x = torch.randn((2, 3,224,224))

    x = x
    y1 = MultiAV_RIP()
    out = y1(x)

    out = torch.squeeze(out,1)
    print(out.ge(0.5).float())

    # score_max_index = out.argmax()
    # print(score_max_index)
    # score_max = out[0, score_max_index]
    # score_max.backward()
    # print(y4.grad.data)
    # y1 = Trans(image_size=256, patch_size=16, num_layers=3, num_heads=8, hidden_dim=384, mlp_dim=384 * 4, dropout=0.0,
    #            representation_size=384 * 2)
    # #weight_path = r'./trans_best_epoch90_imagenet1k.pth'
    # #pt = torch.load(weight_path, map_location="cpu")
    # # py_tmp = y1.state_dict()
    # # for k,v in pt.items():
    # #     if k in y1.state_dict().keys() and 'conv_proj' not in k:
    # #         py_tmp.update({k:v})
    # #
    # #y1.state_dict().keys() == pt.keys()
    # # y1.load_state_dict(py_tmp,strict=True)
    # #y1.load_state_dict(pt, strict=True)
    #
    # # y1.heads = nn.Linear(768,1)
    # y1.heads = nn.Linear(768, 2)

    # pe = PositionalEmbedding(d_model=384)

    # # y = TransformerBottleNeck(x.shape[2],x.shape[1],x.shape[1],8,4)
    # print(y1(y4).shape)
    # from torchvision.models.vision_transformer import Encoder
    #
    # seq_length=64
    # num_layers=3
    # num_heads=8
    # hidden_dim=384
    # mlp_dim=384 * 4
    # dropout=0.0
    # attention_dropout=0.0
    # norm_layer=partial(nn.LayerNorm, eps=1e-6)
    #
    # s = Encoder(64,3,8,384,384*4,0.0,0.0,norm_layer)
    # print(s(x1).shape)

