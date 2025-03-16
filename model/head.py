import torch
import torch.nn as nn
from .nn_blocks import *

class base_block(nn.Module):
    def __init__(self,in_channel,out_channel,act='sigmoid'):
        super(base_block,self).__init__()
        self.conv1 = BaseConv(in_channel, in_channel, 3, stride=1, act="silu")
        self.conv2 = nn.Conv2d(in_channel, out_channel, 1, stride=1)
        if act == 'sigmoid':
            self.act1 = nn.Sigmoid()
        elif act == 'tanh':
            self.act1 = nn.Tanh()
    
    def forward(self,x):
        y = x + self.conv1(x)
        return self.act1(self.conv2(y))

class Edge_hm_head(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Edge_hm_head,self).__init__()
        self.hm_head = base_block(in_channel, out_channel, act='sigmoid')
        self.offset_head = base_block(in_channel, 2, act='tanh')
        
    def forward(self,x):
        return self.hm_head(x) ,self.offset_head(x)
    

# import torch
# import torch.nn as nn
# from .nn_blocks import *

# class Edge_hm_head(nn.Module):
#     def __init__(self,in_channel,out_channel):
#         super(Edge_hm_head,self).__init__()
#         self.conv1 = BaseConv(in_channel, in_channel, 3, stride=1, act="silu")
#         self.conv2 = nn.Conv2d(in_channel, out_channel, 1, stride=1, padding=0)
#         self.act = nn.Sigmoid()
    
#     def forward(self,x):
#         y = x + self.conv1(x)
#         return self.act(self.conv2(y))