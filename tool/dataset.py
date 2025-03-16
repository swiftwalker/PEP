import os
import json
import torch
import numpy as np
import torchvision.transforms as transforms
import pickle
import h5py

from PIL import Image
from torch.utils.data import Dataset
from .utils import ellipse_param_size_adjust

class NormalizeTransform:
    def __call__(self,img):
        mean = torch.mean(img)
        std = torch.std(img)
        return transforms.functional.normalize(img,[mean],[std])

class TrainDataSet(Dataset):
    def __init__(self, data_path ):
        self.bin_path = os.path.join(data_path,'train.bin')
        self.index_path = os.path.join(data_path,'train.txt')
        self.read_index()
        self.create_transform()
        
    def read_index(self):
        self.offsets = []
        with open(self.index_path, 'r') as index_file:
            for line in index_file:
                idx, offset = line.strip().split()
                self.offsets.append(int(offset))
    
    def sparse2dense(self, csr_data):
        dense_data = []
        for map in csr_data:
            dense_data.append(map.toarray())
        return np.array(dense_data)
    
    def create_transform(self): # 创建图片转换器
        self.transform = {
                'img': transforms.Compose([transforms.ToTensor(),
                                           NormalizeTransform()
                                           ]),
            }
    def __len__(self):
        return len(self.offsets)
    
    def __getitem__(self, item):
        with open(self.bin_path, 'rb') as bin_file:
            bin_file.seek(self.offsets[item])
            frame_data = pickle.load(bin_file)
        img = Image.fromarray(frame_data['image']/255.0)
        ellipse_param = frame_data['ellipse_param']
        hm = self.sparse2dense(frame_data['heatmap'])
        offset = frame_data['offset']
        
        return (self.transform['img'](img),
                (torch.tensor(ellipse_param),
                 torch.tensor(hm),
                 torch.tensor(offset)
                 )
                )
        
def get_test_dataset(data_path, re_size = (512,512), tag='h5'):
    if tag == 'h5':
        return TestDataSet_h5(data_path, re_size)
    else:
        pass
    
class TestDataSet_h5(Dataset):
    def __init__(self, data_path, re_size):
        self.hdf5_file = h5py.File(data_path, 'r')
        image_data = self.hdf5_file['images']
        image_info = self.hdf5_file['info']
        self.length = len(image_data)
        src_sz = image_data[0].shape[0:]
        self.params = []
        self.namelist = []
        for msg in image_info:
            ellipse = json.loads(msg)['ellipse']
            ellipse_param = (ellipse[0][0], ellipse[0][1], ellipse[1][0], ellipse[1][1], ellipse[2])
            self.params.append(ellipse_param_size_adjust(ellipse_param, src_sz, re_size))
            self.namelist.append(json.loads(msg)['file_name'])
    
        self.transform = transforms.Compose([transforms.Resize(re_size),
                                             transforms.ToTensor(),
                                             NormalizeTransform()])
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, item):
        label = torch.tensor(self.params[item])
        image_data = self.hdf5_file['images']
        image_info = self.hdf5_file['info']
        img = self.transform(Image.fromarray(image_data[item]))
        name = json.loads(image_info[item])['file_name']
        return img, label, name

    def _get(self, item):
        label = self.params[item]
        image_data = self.hdf5_file['images']
        image_info = self.hdf5_file['info']
        img = image_data[item]
        name = json.loads(image_info[item])['file_name']
        return img, label, name
    
    def name2index(self, name):
        return self.namelist.index(name)
    
    def __del__(self):
        self.hdf5_file.close()
        