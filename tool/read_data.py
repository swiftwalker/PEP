import h5py
import numpy as np
from PIL import Image
import time
import json

# 打开 HDF5 文件
hdf5_file_path = 'data/oursdata_train_raw.h5'

with h5py.File(hdf5_file_path, 'r') as hdf5_file:
    #查看文件中的所有键
    print('Keys:', list(hdf5_file.keys()))
    image_data = hdf5_file['images']
    image_info = hdf5_file['info']
    
    start = time.time()
    for i in range(len(image_data)):
        img = image_data[i]
        msg = json.loads(image_info[i])
        print('Image shape:', img.shape)
        print('Image info:', msg)
        break
        
    end = time.time()
    print('Time:', end - start)
    
    # print('Image shape:', img.shape)
    # print('Image info:', msg)

    



    