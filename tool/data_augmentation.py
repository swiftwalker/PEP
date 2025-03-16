import torch
import torch.nn as nn
import numpy as np
import cv2
from tqdm import tqdm
import random
import os
import h5py
import json
import pickle
import scipy.sparse as sp
import concurrent.futures
import time
from scipy import integrate
from scipy.optimize import fsolve
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from .utils import *

random.seed(time.time())

def gauss2d_con_heatmap(size, x, y, a, b, angle, sigma, ):
    '''
    input:
        size: the size of the heatmap
        sigma: the sigma of the gaussian kernel
        x,y: the center of the heatmap
        a,b: the long and short axes of the ellipse
        angle: the angle of the ellipse
    output:
        2d contour heatmap
    '''

    def block_fcn(x,y,f1,f2,la,sigma): 
        dist_2 = (np.sqrt((x-f1[0])**2+(y-f1[1])**2) + np.sqrt((x-f2[0])**2+(y-f2[1])**2)-la)**2
        return np.exp(-dist_2/(2*sigma**2))
    la = 2*max(a,b)
    angle = angle*np.pi/180
    focus = np.array(calculate_foci_np(a,b,angle)) + np.array([x,y])
    x_index ,y_index = np.meshgrid(np.arange(size[1]),np.arange(size[0]))
    npmap = block_fcn(x_index,y_index,focus[0],focus[1],la,sigma)
    return np.expand_dims(npmap,axis=0)

def gauss2d_heatmap(size,x, y, a, b, angle, sigma,):
    '''
    input:
        params: list, [x, y, a, b, angle]
        size: the size of the heatmap
        sigma: the sigma of the gaussian kernel
    output:
        2d heatmap
    '''

    def block_fcn(x, y, a, b, angle, x0, y0): 
        return np.exp(-((x-x0)*np.cos(angle)+(y-y0)*np.sin(angle))**2/(2*(a*sigma)**2) - ((x-x0)*np.sin(angle)-(y-y0)*np.cos(angle))**2/(2*(b*sigma)**2))
    angle = angle * np.pi / 180
    x_index, y_index = np.meshgrid(np.arange(size[1]), np.arange(size[0]))
    x = round(x)
    y = round(y)
    x = min(max(x, 0), size[1] - 1)
    y = min(max(y, 0), size[0] - 1)
    npmap = block_fcn(x_index, y_index, a, b, angle, x, y)
    return np.expand_dims(npmap, axis=0)

def calculate_foci_np(a, b, angle):
    '''
    a : scalar
    b : scalar
    angle : scalar
    '''
    semi_major = max(a, b)
    semi_minor = min(a, b)
    c = np.sqrt(semi_major**2 - semi_minor**2)
    if a > b:
        focus1 = np.array([c, 0])
        focus2 = np.array([-c, 0])
    else:
        focus1 = np.array([0, c])
        focus2 = np.array([0, -c])
    cos, sin = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([[cos, -sin], [sin, cos]])
    focus = np.array([focus1, focus2])
    focus = np.dot(rotation_matrix, focus.T).T
    return focus

def random_select(edgenums):
    const = [10,12,14]
    random_add = const[random.randint(0,2)]
    total_num = edgenums + random_add
    #从0到total_num-1的循环数组中挑选连续的random_add个数
    start = random.randint(0,total_num-1)
    end = (start + random_add) % total_num
    #找到未被选中的数的索引
    if end > start:
        selected = list(range(start,end))
        unselected = list(range(start)) + list(range(end,total_num))
    else:
        unselected = list(range(end,start))    
        selected = list(range(end)) + list(range(start,total_num))
    return selected,unselected,random_add

def generate_ellipse_points_standard(a, b, n, xtol=np.deg2rad(0.1), epsabs=0.1, epsrel=0.1):
    def ellipse_arclength_integrand(theta):
        return np.sqrt((a * np.sin(theta))**2 + (b * np.cos(theta))**2)

    def ellipse_arclength():
        return integrate.quad(ellipse_arclength_integrand, 0, 2 * np.pi, epsabs=epsabs, epsrel=epsrel)[0]

    def cumulative_arclength(theta):
        return integrate.quad(ellipse_arclength_integrand, 0, theta, epsabs=epsabs, epsrel=epsrel)[0]

    def arclength_to_theta(s, theta0):
        def equation(theta):
            return cumulative_arclength(theta) - s
        return fsolve(equation, theta0, xtol=xtol)[0]

    L = ellipse_arclength()
    s_values = np.linspace(0, L, n, endpoint=False)
    points = []
    theta0 = 0
    for s in s_values:
        theta = arclength_to_theta(s, theta0)
        points.append((a * np.cos(theta), b * np.sin(theta)))
        theta0 = theta  # 更新初始值为当前theta
    return np.array(points)

def generate_ellipse_points_equidist(param, n):
    xc, yc, a, b, angle = param
    angle = np.deg2rad(angle)
    # 生成标准位置上的椭圆点
    points = generate_ellipse_points_standard(a, b, n)
    # 创建旋转矩阵
    rotation_matrix = np.array([[np.cos(angle), np.sin(angle)],
                                [-np.sin(angle), np.cos(angle)]])
    # 旋转和平移椭圆点
    transformed_points = np.dot(points, rotation_matrix) + np.array([xc, yc])
    return transformed_points

def get_mask_pos(mask_points):
    # 找到mask_points的外接矩形
    left = int(min(mask_points[:,0]))
    right = int(max(mask_points[:,0]))
    top = int(min(mask_points[:,1]))
    bottom = int(max(mask_points[:,1]))
    return left,right,top,bottom

def custom_cut(img, edgenums, ellipse_param):
    # selected_points: 被遮挡的点索引
    # unselected_points: 未被遮挡的点索引
    selected_points,unselected_points,add = random_select(edgenums)
    ellipse_points = generate_ellipse_points_equidist(ellipse_param,edgenums + add)
    mask_points = ellipse_points[selected_points]
    mask_rect = get_mask_pos(mask_points)
    #生成0-255之间，形状为img大小的随机矩阵
    mask = np.random.rand(*img.shape) * 255
    img[mask_rect[2]:mask_rect[3],mask_rect[0]:mask_rect[1]] = mask[mask_rect[2]:mask_rect[3],mask_rect[0]:mask_rect[1]]
    unse_points = ellipse_points[unselected_points,:]
    
    return img, unse_points, mask_rect

def gauss_blur(img, ksize=(5,5), sigma= 5):
    img = cv2.GaussianBlur(img, ksize, sigma)
    return img

def random_gauss_blur(img):
    ksize = random.choice([(3,3),(5,5),(7,7)])
    sigma = random.randint(3,11)
    return gauss_blur(img, ksize, sigma)


def process_each_frame(frame_data, config):
    '''
    完成对每一帧数据的处理：
    图像的resize 参数的调整
    遮挡数据生成 遮挡图像的生成 遮挡热力图的生成
    '''
    
    edgenum = 32 if config['edgenum'] == None else config['edgenum'] # 构成椭圆边的离散点数
    augmentation_ratio = 5 if config['augmentation_ratio'] == None else config['augmentation_ratio'] # 数据增强倍数
    cut_percent = 0.8 if config['cut_percent'] == None else config['cut_percent'] # 遮挡数据比例
    dst_size = (512,512) if config['dst_size'] == None else config['dst_size'] # 目标图像大小
    hm_size = (128,128) if config['hm_size'] == None else config['hm_size'] # 热力图大小
    
    out_data = []
    for i in range(augmentation_ratio):
        is_cut = random.random() < cut_percent
        img = frame_data['image']
        ellipse_param = frame_data['ellipse_param']
        # img = random_gauss_blur(img) # for training
        
        src_size = img.shape[:2]
        img = cv2.resize(img, dst_size)
        ellipse_param = ellipse_param_size_adjust(ellipse_param, src_size, dst_size)
        if is_cut:
            img, unse_points, mask_rect = custom_cut(img, edgenum, ellipse_param)

        else:
            mask_rect = None
            unse_points = generate_ellipse_points_equidist(ellipse_param, edgenum)
        #记录中心点和边缘点降采样导致的偏移
        offset = np.zeros([edgenum+1,2])
        
        down_ellipse_param = ellipse_param_size_adjust(ellipse_param, dst_size, hm_size)
        down_radio = np.array(hm_size) / np.array(dst_size)
        
        down_unse_points = unse_points * down_radio
        offset[1:,:] = down_unse_points - np.round(down_unse_points)
        
        down_mask_rect = [int(i*hm_size[0]/dst_size[0]) for i in mask_rect] if mask_rect is not None else None
        mask = np.zeros(hm_size)
        if mask_rect is not None:
            mask[down_mask_rect[2]:down_mask_rect[3],down_mask_rect[0]:down_mask_rect[1]] = 1
            
        # expand mask 8 pixels
        mask = cv2.dilate(mask, np.ones((8,8),np.uint8))
        mask = mask.astype(np.float32)
            
        # heatmap_con = gauss2d_con_heatmap(hm_size, *down_ellipse_param, 1) # for training
        heatmap_con = gauss2d_con_heatmap(hm_size, *down_ellipse_param, 5) # for visual display
        heatmap_con = heatmap_con * (1 - mask)
        heatmap_center = gauss2d_heatmap(hm_size, *down_ellipse_param, 0.15)
        heatmap_edge = np.zeros([edgenum, hm_size[0], hm_size[1]])

        
        center_tmp = ellipse_param[:2] * down_radio
        offset[0] = center_tmp - np.round(center_tmp)

        for i,point in enumerate(down_unse_points):
            heatmap_edge[i,:,:] = gauss2d_heatmap(hm_size, *point, 2.5, 2.5, 0, 1) # for visual display
            # heatmap_edge[i,:,:] = gauss2d_heatmap(hm_size, *point, 2.5, 2.5, 0, 0.1) # for training
        heatmap = np.concatenate([heatmap_center,heatmap_edge,heatmap_con],axis=0)
        heatmap = heatmap.astype(np.float32)
        csr_hm = [sp.csr_matrix(map) for map in heatmap]
        
        out_data.append({'image': img,
                         'ellipse_param': ellipse_param,
                         'heatmap': csr_hm,
                         'offset': offset,
                        })
    return out_data


def read_process(data_path, save_path, num_workers=8):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    bin_filename = os.path.join(save_path,'train.bin')
    index_filename = os.path.join(save_path,'train.txt')
    # 如果已经存在这两个文件，先删除
    if os.path.exists(bin_filename):
        os.remove(bin_filename)
    if os.path.exists(index_filename):
        os.remove(index_filename)
    
    with h5py.File(data_path, 'r') as data_file , open(bin_filename, 'wb') as bin_file, open(index_filename, 'w') as index_file:
    #查看文件中的所有键
        print('Keys:', list(data_file.keys()))
        image_data = data_file['images']
        image_info = data_file['info']
        aug_config = {'edgenum': 32, 
                      'augmentation_ratio': 5, 
                      'cut_percent': 0.8, 
                      'dst_size': (512,512),
                      'hm_size': (128,128)}
        # 使用 ThreadPoolExecutor 进行并行处理
        with ProcessPoolExecutor(max_workers = num_workers) as executor:
            futures = []
            for i in range(len(image_data)):
                img = image_data[i]
                msg = json.loads(image_info[i])
                ellipse = msg['ellipse']
                ellipse_param = (ellipse[0][0], ellipse[0][1], ellipse[1][0]/2, ellipse[1][1]/2, ellipse[2])
                frame_data = {'image': img, 'ellipse_param': ellipse_param}

                # 提交多线程任务
                futures.append(executor.submit(process_each_frame, frame_data, aug_config))
            
            # 获取每个线程的结果
            for i, future in tqdm(enumerate(as_completed(futures)), total=len(futures)):
                processed_data = future.result()
                for j, iter_data in enumerate(processed_data):
                    offset = bin_file.tell()
                    pickle.dump(iter_data, bin_file)
                    index_file.write(f'{i * aug_config["augmentation_ratio"] + j} {offset}\n')
    

