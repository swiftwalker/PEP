import torch
import torch.nn as nn
from .nn_blocks import *

class FPN(nn.Module):
    def __init__(self, 
                 depth, 
                 in_channels, 
                 out_channels, ):
        '''
        Args:
        'depth' | int : number of layers in FPN
        'in_channels' | list[int] : number of input channels for each tensor from backbone (from high to low resolution)
        'out_indexes' | list[int] : index of layers to output
        'out_channels' | list[int] : number of output channels for each tensor from FPN (from high to low resolution)
        '''
        
        super(FPN, self).__init__()
        assert depth == len(in_channels) , "depth and in_channels dismatch"
        
        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        lateral_base_channel = 64
        lateral_out_channels = [lateral_base_channel * 2 ** i for i in range(depth)]
        self.lateral_convs = self.get_lateral_convs(in_channels, lateral_out_channels)
        self.fpn_stream = self.get_fpn_stream(lateral_out_channels)
        self.up_sample = nn.Upsample(scale_factor=2, mode='nearest')
    
    def get_lateral_convs(self, in_channels, out_channels, ksize=1, stride=1, groups=1, bias=False, act="silu"):
        # out_channels here is the number of channels for the output of lateral convolutions (not the final output)
        return nn.ModuleList([BaseConv(in_channels[i], out_channels[i], ksize, stride, groups, bias, act) for i in range(self.depth)])
    
    def get_fpn_stream(self, lateral_out_channels, act="silu"):
        out_tmp = [self.out_channels[i] for i in range(1, self.depth)] + [0]
        in_channels = [lateral_out_channels[i] + out_tmp[i] for i in range(self.depth)]  
        out_channels = self.out_channels
        return nn.ModuleList([self.get_fpn_conv_block(in_channels[i], out_channels[i], act) for i in range(self.depth)])
        
    def get_fpn_conv_block(self, in_channel, out_channel, act="silu"):
        return CSPLayer(in_channel, out_channel, n=1, act=act)

    def forward(self, x):
        # x is a list of tensors from backbone (from high to low resolution)
        # lateral convolutions
        lateral_out = [lateral_conv(x[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        # FPN stream
        x = lateral_out[-1]
        for i in range(self.depth-1, 0, -1):
            x = self.fpn_stream[i](x)
            x = self.up_sample(x)
            x = torch.cat((x, lateral_out[i-1]), dim=1)
        x = self.fpn_stream[0](x)
        return x
