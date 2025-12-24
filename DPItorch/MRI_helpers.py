import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as functional

torch.set_default_dtype(torch.float32)
import torch.optim as optim
import pickle
import math

###############################################################################
# Define the loss functions for MRI imaging
###############################################################################
def Loss_kspace_diff(sigma):
	def func(y_true, y_pred):
		return torch.mean(torch.abs(y_pred - y_true), (1, 2, 3)) / sigma
	return func

def Loss_kspace_diff2(sigma):
	def func(y_true, y_pred):
		return torch.mean((y_pred - y_true)**2, (1, 2, 3)) / (sigma)**2
	return func

def Loss_l1(y_pred):
	# image prior - sparsity loss
	return torch.mean(torch.abs(y_pred), (-1, -2))

def Loss_TSV(y_pred):
	# image prior - total squared variation loss
	return torch.mean((y_pred[:, 1::, :] - y_pred[:, 0:-1, :])**2, (-1, -2)) + torch.mean((y_pred[:, :, 1::] - y_pred[:, :, 0:-1])**2, (-1, -2))

def Loss_TV(y_pred):
	# image prior - total variation loss
	return torch.mean(torch.abs(y_pred[:, 1::, :] - y_pred[:, 0:-1, :]), (-1, -2)) + torch.mean(torch.abs(y_pred[:, :, 1::] - y_pred[:, :, 0:-1]), (-1, -2))

# def Loss_TV(y_pred):
# 	# image prior - total variation loss
# 	eps = 1e-24
# 	return torch.mean(torch.sqrt((y_pred[:, 1::, :]-y_pred[:, 0:-1, :])**2+eps), (-1, -2)) + torch.mean(torch.sqrt((y_pred[:, :, 1::]-y_pred[:, :, 0:-1])**2+eps), (-1, -2))

def fft2c_torch(img):
    """
    使用PyTorch实现的快速傅里叶变换函数（使用新版本torch.fft API）
    
    参数:
        img: 实数域图像，形状为(batch_size, npix, npix)
        
    返回:
        kspace: 复数域K空间数据，形状为(batch_size, npix, npix, 2)
               其中最后一个维度表示实部和虚部
    
    说明:
        此函数对应于MATLAB中的fft2函数，并进行了归一化处理
        使用新版本PyTorch API (torch.fft模块) 实现，替代旧版的torch.fft函数
    """
    x = img.unsqueeze(-1)  # (batch_size, npix, npix, 1)
    x = torch.cat([x, torch.zeros_like(x)], -1)  # (batch_size, npix, npix, 2)
    x_complex = torch.view_as_complex(x)  # 转换为复数tensor
    kspace_complex = torch.fft.fft2(x_complex, norm="ortho")  # 执行2D FFT
    kspace_real = kspace_complex.real.unsqueeze(-1)  # (batch, npix, npix, 1)
    kspace_imag = kspace_complex.imag.unsqueeze(-1)  # (batch, npix, npix, 1)
    kspace = torch.cat([kspace_real, kspace_imag], dim=-1)  # (batch, npix, npix, 2)
    return kspace
