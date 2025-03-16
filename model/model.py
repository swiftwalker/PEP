import torch
import torch.nn as nn
from .darknet_backbone import CSPDarknet
from .swin_t_backbone import SwinTransformer
from .fpn import FPN
from .head import Edge_hm_head

class Edge_Detect_Net(nn.Module):
    def __init__(self,point_nums, backbone, fpn_channel):
        super(Edge_Detect_Net, self).__init__()
        if backbone['name'] == 'darknet':
            self.backbone = CSPDarknet(dep_mul= 1.0, wid_mul=backbone['wid_mul'])
        elif backbone['name'] == 'swin':
            self.backbone = SwinTransformer(img_size=512,window_size=8, in_chans=1 ,embed_dim=backbone['embed_dim'])
        else:
            raise NotImplementedError
        self.fpn = FPN(depth=4, in_channels=fpn_channel['in_channels'], out_channels=fpn_channel['out_channels'])
        self.head = Edge_hm_head(in_channel=fpn_channel['out_channels'][0],
                                 out_channel = point_nums + 1)
        self.apply(weights_init)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.fpn(x)
        x = self.head(x)
        return x
    
def weights_init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)