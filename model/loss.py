import torch
import torch.nn as nn
import torch.nn.functional as F


def _neg_loss(pred, gt, alpha, beta):
    ''' 
    Modified focal loss. Exactly the same as CornerNet.
    Runs faster and costs a little bit more memory
    Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    neg_weights = torch.pow(1 - gt, beta)
    loss = 0
    pos_loss = torch.log(pred+(1e-8)) * torch.pow(1 - pred, alpha) * pos_inds
    neg_loss = torch.log(1 - pred +(1e-8)) * torch.pow(pred, alpha) * neg_weights * neg_inds
    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss)/num_pos
    return loss

class Con_Loss(nn.Module):
    def __init__(self,alpha,beta):
        super(Con_Loss,self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.loss = nn.BCEWithLogitsLoss()
    def forward(self, pred, target):
        #pred = pred.mean(dim=1)
        pred = pred.max(dim=1)[0]
        return self.loss(pred, target)

class Grad_Loss(nn.Module):
    def __init__(self,device):
        super(Grad_Loss,self).__init__()
        self.sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], requires_grad=False,dtype=torch.float32).view(1, 1, 3, 3).to(device)
        self.sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], requires_grad=False,dtype=torch.float32).view(1, 1, 3, 3).to(device)
    
    def loss(self, pred, target):
        log_pred = torch.log(pred + 1e-6)
        log_one_minus_pred = torch.log(1 - pred + 1e-6)
        return (target * log_pred * log_one_minus_pred + (1 - target) * log_one_minus_pred * log_one_minus_pred).mean()
    
    def forward(self, img, hm):
        grad_x = F.conv2d(img, self.sobel_x, padding=1)
        grad_y = F.conv2d(img, self.sobel_y, padding=1)
        grad = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        grad_scaled = grad / (grad.flatten(2).max(dim=2, keepdim=True)[0].view(grad.shape[0],1,1,1) + 1e-8)
        h,w = hm.shape[2:]
        # downsample grad to hm size
        grad_scaled = F.interpolate(grad_scaled, size=(h,w), mode='bilinear', align_corners=False)
        #mean_hm = hm.mean(dim=1, keepdim=True) # (b,1,h,w)
        max_hm = hm.max(dim=1, keepdim=True)[0] # (b,1,h,w)
        return self.loss(max_hm, grad_scaled)
        

class Offset_loss(nn.Module):
    def __init__(self):
        super(Offset_loss,self).__init__()
        self.loss = nn.L1Loss()
    
    def forward(self, offset_map_pred, hm_gt, offset_gt):
        '''
        shape:
        offset_map_pred: (b,2,h,w)
        hm_gt: (b,n,h,w)
        offset_gt: (b,n,2)
        '''
        max_indices = torch.argmax(hm_gt.flatten(2), dim=2) # (b,n,h,w) -> (b,n)
        max_indices = max_indices.unsqueeze(1).expand(-1,2,-1) # (b,n) -> (b,1,n) -> (b,2,n)
        #根据索引取出对应的offset
        offset_pred = offset_map_pred.flatten(2).gather(2,max_indices).permute(0, 2, 1) # (b,2,h,w) -> (b,2,h*w) -> (b,2,n) -> (b,n,2)
        
        return self.loss(offset_pred, offset_gt) / offset_gt.shape[1]

class Loss(nn.Module):
    def __init__(self,device):
        super(Loss,self).__init__()
        self.heatmap_loss = _neg_loss
        self.edge_loss = Con_Loss(alpha=2, beta=4)
        self.offset_loss = Offset_loss()
        self.grad_loss = Grad_Loss(device)

    def forward(self, pred, target,a=1,b=1):
        heatmap_pred,offset_map_pred = pred
        heatmap_gt,offset_gt,src_img = target
        hm_c_ed = heatmap_gt[:,:-1,:,:]
        hm_con = heatmap_gt[:,-1,:,:]
        heatmap_loss = self.heatmap_loss(heatmap_pred, hm_c_ed, alpha=2, beta=4)
        con_loss = self.edge_loss(heatmap_pred, hm_con)
        offset_loss = self.offset_loss(offset_map_pred, hm_c_ed, offset_gt)
        grad_loss = self.grad_loss(src_img, hm_c_ed)
        
        return heatmap_loss + a*offset_loss + b*(con_loss + grad_loss)
        # return heatmap_loss + con_loss + offset_loss + grad_loss
        
        # heatmap*1 + a * offset + b (con + grad)
        # a=0.1,1,10 b=0.1,1,10
        
        
        ## no norm loss
        # return heatmap_loss + offset_loss
        
        ## no con loss
        # return heatmap_loss + a*offset_loss + b*grad_loss
    
        # no grad loss
        # return heatmap_loss + a*offset_loss + b*con_loss
