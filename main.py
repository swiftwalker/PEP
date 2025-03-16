import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import argparse
import os
import time
from tqdm import tqdm
import sys

from tool.dataset import TrainDataSet
from model.model import Edge_Detect_Net
from model.loss import Loss
from tool.utils import load_configs

def gpu(i=0):
    return torch.device(f'cuda:{i}')

def create_exp() -> str:
    # get current time
    t = time.localtime()
    #get machine name
    machine_name = os.uname().nodename
    # create experiment name ,hour minute second has leading zero
    year = str(t.tm_year)[-2:]
    month = str(t.tm_mon).zfill(2)
    day = str(t.tm_mday).zfill(2)
    hour = str(t.tm_hour).zfill(2)
    minute = str(t.tm_min).zfill(2)
    second = str(t.tm_sec).zfill(2)
    exp = f'{machine_name}_{year}.{month}.{day}_{hour}.{minute}.{second}'
    return exp

def train_one_epoch(model, optimizer, data_loader, device, epoch ,loss_function, show_pbar = True,a=1,b=1):
    model.train()
    accu_loss = torch.zeros(1).to(device[0])  # 累计损失
    optimizer.zero_grad()
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout) if show_pbar else data_loader
    for step, data in enumerate(data_loader):
        input,(param,hm,offset) = data
        sample_num += input.shape[0]  #B*C*H*W
        pred = model((input.to(device[0])))
        param = param.to(device[0])
        hm = hm.to(device[0])
        offset = offset.to(device[0])
        target = (hm,offset,(input.to(device[0])))
        loss = loss_function(pred, target,a,b)
        loss.backward()
        accu_loss += loss.detach()  

        if show_pbar:
            data_loader.desc = "[train epoch {}] loss: {:.3f}".format(epoch,accu_loss.item() / (step + 1))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1)


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, loss_function, show_pbar = True, a=1,b=1):
    model.eval()
    accu_loss = torch.zeros(1).to(device[0])  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout) if show_pbar else data_loader
    for step, data in enumerate(data_loader):
        input,(param,hm,offset) = data
        sample_num += input.shape[0]  #B*C*H*W
        pred = model((input.to(device[0])))
        param = param.to(device[0])
        hm = hm.to(device[0])
        offset = offset.to(device[0])
        target = (hm,offset,(input.to(device[0])))
        loss = loss_function(pred, target,a,b)
        accu_loss += loss
        if show_pbar:
            data_loader.desc = "[valid epoch {}] loss: {:.3f}".format(epoch,accu_loss.item() / (step + 1))

    return accu_loss.item() / (step + 1)

def create_dataloader(data_path, batch_size, val_rate):
    # 单数据集
    if isinstance(data_path, str):
        total_dataset = TrainDataSet(data_path)
        train_len = int(len(total_dataset) * (1 - val_rate))
        val_len = len(total_dataset) - train_len
        train_dataset, val_dataset = random_split(total_dataset, [train_len, val_len])
        train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=8)
        val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle=False, num_workers=8)
        return train_loader, val_loader
    # 多数据集
    elif isinstance(data_path, list):
        datasets = [TrainDataSet(path) for path in data_path]
        total_dataset = torch.utils.data.ConcatDataset(datasets)
        train_len = int(len(total_dataset) * (1 - val_rate))
        val_len = len(total_dataset) - train_len
        train_dataset, val_dataset = random_split(total_dataset, [train_len, val_len])
        train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=16)
        val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle=False, num_workers=16)
        return train_loader, val_loader
    # 异常
    else:
        raise ValueError('data_path type error')

def main(configs):  
    
    device=[gpu(i) for i in configs['device']]
    result_path = configs['result_path']
    exp_name = configs['exp_name']
    log_path = os.path.join(result_path, exp_name, create_exp())
    pth_save_path = os.path.join(log_path, 'weights')
    log_file = os.path.join(log_path, 'log.txt')
    if not os.path.exists(pth_save_path):
        os.makedirs(pth_save_path)
    SW = SummaryWriter(log_path)

    try:
        with open(log_file, 'w') as f:
            f.write('training configs:\n')
            for k, v in configs.items():
                f.write(f'  {k}: {v}\n')
        print('Start preparing data...')
        train_loader, val_loader = create_dataloader(configs['data_path'], configs['batch_size'], configs['val_rate'])
        # 实例化模型
        model = Edge_Detect_Net(point_nums=32, backbone=configs['backbone'],fpn_channel=configs['fpn_channels']).to(device[0])
        
        #随机初始化model的参数
        def weights_init(m):
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        model.apply(weights_init)
        lr = configs['lr']
        optimizer = optim.Adam(model.parameters(), lr = lr)
        loss_function = Loss(device=device[0])
        
        best_loss = 1e10
        best_epoch = 0
        
        SW.add_text(tag = 'params', text_string = str(args))
        
        if not os.path.exists(pth_save_path):
            os.makedirs(pth_save_path)
        os.system('clear')
        print('Start training')    
        epochs = configs['epochs']
        hyper_a = configs['hyper_a']
        hyper_b = configs['hyper_b']
        for epoch in range(1, epochs+1): 
            # train
            # if epoch == 100:
            #      optimizer.param_groups[0]['lr'] /= 10
            # if epoch == 30:
            #     optimizer.param_groups[0]['lr'] = args.lr/100
            
            train_loss = train_one_epoch(model=model,
                                        optimizer=optimizer,
                                        data_loader=train_loader,
                                        device=device,epoch=epoch,
                                        loss_function=loss_function,
                                        show_pbar=configs['show_pbar'],
                                        a=hyper_a,b=hyper_b)

            val_loss = evaluate(model=model,
                                data_loader=val_loader,
                                device=device,
                                epoch=epoch,
                                loss_function=loss_function,
                                show_pbar=configs['show_pbar'],
                                a=hyper_a,b=hyper_b)
            with open(log_file, 'a') as f:
                f.write(f'Epoch {epoch}/{epochs} train_loss: {train_loss:.4f} val_loss: {val_loss:.4f}\n')
            SW.add_scalar(tag = 'loss_epoch/train', scalar_value = train_loss, global_step = epoch)
            SW.add_scalar(tag = 'loss_epoch/val', scalar_value = val_loss, global_step = epoch)

            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                save_dict = {'model':model.state_dict(),'configs':configs}
                torch.save(save_dict, os.path.join(pth_save_path, f'model_best.pth'))
            
            save_epochs = [int(epochs * p) for p in [0.2, 0.4, 0.6, 0.8, 1.0]]
            if epoch in save_epochs:
                torch.save(save_dict, os.path.join(pth_save_path, f'model_epoch_{epoch}.pth'))
            print(f'Epoch {epoch}/{epochs} train_loss: {train_loss:.4f} val_loss: {val_loss:.4f}')
            SW.add_scalar(tag='best_epoch', scalar_value=best_epoch, global_step=epoch)
            
        with open(log_file, 'a') as f:
            f.write(f'Best epoch: {best_epoch} with val_loss: {best_loss:.4f}\n')
            f.write(f'Training finished!\n')
        print('Training finished!')
    
    except Exception as e:
        print(f"Training interrupted due to: {str(e)}")
        # 关闭 SummaryWriter
        SW.close()
        # 修改日志目录名称
        failed_log_path = log_path + '_failed'
        with open(log_file, 'a') as f:
            f.write(f'Training interrupted due to: {str(e)}\n')
        os.rename(log_path, failed_log_path)
        print(f"Log directory renamed to: {failed_log_path}")

        raise e    
    except KeyboardInterrupt as e:
        print(f"Training interrupted due to keyboard interrupt")
        # 关闭 SummaryWriter
        SW.close()
        # 修改日志目录名称
        failed_log_path = log_path + '_failed'
        with open(log_file, 'a') as f:
            f.write(f'Training interrupted due to keyboard interrupt\n')
        os.rename(log_path, failed_log_path)
        print(f"Log directory renamed to: {failed_log_path}")
        raise e   
    
    finally:
        SW.close()        
       
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Load a module from a file path.")
    parser.add_argument('module_path', type=str, help='Path to the Python module file (e.g. a.py).')
    args = parser.parse_args()
    configs = load_configs(args.module_path)
    main(configs)