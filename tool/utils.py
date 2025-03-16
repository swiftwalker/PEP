import numpy as np
import importlib.util
import sys
import os

def abt2mat(a,b,t):
    mat1 =np.array([[np.cos(t),np.sin(t)],[-np.sin(t),np.cos(t)]])
    mat2 = np.array([[1/a**2,0],[0,1/b**2]])
    mat = mat1.T@mat2@mat1
    return mat
def mat2abt(mat):
    w,v = np.linalg.eig(mat)
    w = np.sqrt(1/w)
    a,b = w[0],w[1]
    t = np.arctan2(v[1,0],v[0,0])
    return a,b,t  

def ellipse_param_size_adjust(ellipse_param, src_size, dst_size, radian_input = False):
    '''
    input:
        ellipse_param: (x,y,a,b,t) or ((x,y),(a,b),t)
        src_size: (w,h)
        dst_size: (w,h)
    output:
        ellipse_param: (x,y,a,b,t) or ((x,y),(a,b),t)
    '''
    w_ratio = dst_size[1]/src_size[1]
    h_ratio = dst_size[0]/src_size[0]
    if len(ellipse_param) == 5:
        x,y,a,b,t = ellipse_param
    elif len(ellipse_param) == 3:
        (x,y),(a,b),t = ellipse_param
    else:
        raise ValueError('ellipse_param length error')
    x = x*w_ratio
    y = y*h_ratio
    if not radian_input:
        t = t*np.pi/180
    mat1 = np.array([[1/w_ratio,0],[0,1/h_ratio]])
    mat2 = abt2mat(a,b,t)
    mat = mat1.T@mat2@mat1
    a,b,t = mat2abt(mat)
    if not radian_input:
        t = t*180/np.pi
    if len(ellipse_param) == 5:
        return x,y,a,b,t
    else:
        return (x,y),(a,b),t
    
def load_configs(path):
    module_name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.configs
    