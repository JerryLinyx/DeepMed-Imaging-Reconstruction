#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DPI_PET.py  – Deep Probabilistic Imaging on PET (single-scan training)

代码总体逻辑：
-----------
1. 数据加载与预处理
   - 读取PET活动图(activity map)和投影数据(sinogram)
   - 选择特定时间帧和ROI切片进行重建
   - 支持64或128尺寸的图像处理

2. 系统矩阵(G_sparse)处理
   - 创建稀疏系统矩阵的PyTorch表示
   - 支持CUDA加速的大规模稀疏矩阵计算
   - 自动适应64或128图像尺寸

3. 生成模型构建
   - 使用RealNVP或Glow作为生成模型
   - 设置可学习的缩放因子以适应PET图像值域
   - 支持预训练模型加载

4. 损失函数设计
   - 泊松负对数似然作为数据保真项
   - 使用TV和L1正则化约束图像特性
   - logdet项促进生成模型多样性

5. 训练与优化
   - 自动适应GPU内存的批处理策略
   - 支持学习率调度和梯度裁剪
   - 实时监控重建质量(MSE, PSNR, SSIM)

6. 结果保存与评估
   - 生成包含详细参数的模型文件名
   - 保存重建图像和评估指标

运行示例:
Linux:
python DPI_PET.py \\
  --activity_path   /gpfsdata/home/Zhaobo_yuxuan/work/FlowPET/dataset/brain64_tumor_activity_map/brain64_tumor_FDG_K1_40min.mat \\
  --sinogram_path   /gpfsdata/home/Zhaobo_yuxuan/work/FlowPET/dataset/brain64_tumor_sinogram/brain64_tumor_FDG_K1_40min.mat \\
  --gmat_path       /gpfsdata/home/Zhaobo_yuxuan/work/FlowPET/dataset/G_system_matrix_64.mat \\
  --ri_path         /gpfsdata/home/Zhaobo_yuxuan/work/FlowPET/dataset/brain64_tumor_sinogram/brain64_tumor_FDG_K1_40min_ri.mat \\
  --ci_path         /gpfsdata/home/Zhaobo_yuxuan/work/FlowPET/dataset/brain64_tumor_sinogram/brain64_tumor_FDG_K1_40min_ci.mat \\
  --ytrue_path      /gpfsdata/home/Zhaobo_yuxuan/work/FlowPET/dataset/brain64_tumor_sinogram/brain64_tumor_FDG_K1_40min_ytrue.mat \\
  --save_dir       /path/to/output/checkpoint/pet \\
  --model_form     realnvp \\
  --image_size     64

Windows:
python DPI_PET.py ^
  --activity_path  ../dataset/brain64_tumor_activity_map/brain64_tumor_FDG_K1_40min.mat ^
  --sinogram_path  ../dataset/brain64_tumor_sinogram/brain64_tumor_FDG_K1_40min.mat ^
  --gmat_path      ../dataset/G_system_matrix_64.mat ^
  --ri_path        ../dataset/brain64_tumor_sinogram/brain64_tumor_FDG_K1_40min_ri.mat ^
  --ci_path        ../dataset/brain64_tumor_sinogram/brain64_tumor_FDG_K1_40min_ci.mat ^
  --ytrue_path     ../dataset/brain64_tumor_sinogram/brain64_tumor_FDG_K1_40min_ytrue.mat ^
  --save_dir       ./checkpoint/pet64 ^
  --model_form     realnvp ^
  --image_size     64

使用128x128尺寸:
python DPI_PET.py ^
  --activity_path ../dataset/brain128_tumor_activity_map/brain128_tumor_FDG_K1_40min.mat ^
  --sinogram_path ../dataset/brain128_tumor_sinogram/brain128_tumor_FDG_K1_40min.mat ^
  --gmat_path     ../dataset/G_system_matrix.mat ^
  --ri_path       ../dataset/brain128_tumor_sinogram/brain128_tumor_FDG_K1_40min_ri.mat ^
  --ci_path       ../dataset/brain128_tumor_sinogram/brain128_tumor_FDG_K1_40min_ci.mat ^
  --ytrue_path    ../dataset/brain128_tumor_sinogram/brain128_tumor_FDG_K1_40min_ytrue.mat ^
  --save_dir      ./checkpoint/pet128 ^
  --image_size    128 ^
  --n_flow        16 ^
  --n_epoch       10000 ^
  --lr            5e-5 ^
  --lr_decay ^
  --lr_step       2000 ^
  --lr_gamma      0.5
"""

# ---------- imports ----------
import os, argparse, numpy as np, matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 黑体，适用于 Windows 中文系统
matplotlib.rcParams['axes.unicode_minus'] = False    # 正常显示负号


import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as functional

torch.set_default_dtype(torch.float32)
import torch.optim as optim

import scipy.io, h5py
from scipy.sparse import csc_matrix
from pathlib import Path
from tqdm import trange
import time
import gc

# --- Distributed Training Imports ---
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
# os import is already above
# --- End Distributed Training Imports ---

from generative_model import realnvpfc_model, glow_model

# --- Distributed Training Setup ---
local_rank = -1
world_size = 1
if 'LOCAL_RANK' in os.environ:
    dist.init_process_group(backend='nccl') # or 'gloo' for CPU
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    if local_rank == 0: # Print init message only from rank 0
        print(f"[Rank {local_rank}/{world_size}] Distributed training initialized. Device: {device}")
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# --- End Distributed Training Setup ---


print("Torch version:", torch.__version__)
if local_rank == -1 or local_rank == 0: # Print only for single GPU or rank 0
    print("Device selected:", device)
if device.type == "cuda":
    if local_rank == -1 or local_rank == 0:
        print("CUDA device name :", torch.cuda.get_device_name(0 if local_rank == -1 else local_rank))
        # 打印初始GPU内存使用情况
        print(f"初始GPU内存: 已用 {torch.cuda.memory_allocated(device)/1024**3:.2f}GB, "
              f"保留 {torch.cuda.memory_reserved(device)/1024**3:.2f}GB")
else:
    if local_rank == -1 or local_rank == 0:
        print("CUDA not available, using CPU -- slower training")

# 定义内存监控函数
def print_gpu_memory(prefix=""):
    """打印当前GPU内存使用情况"""
    if torch.cuda.is_available() and (local_rank == -1 or local_rank ==0): # Print only for rank 0 or single GPU
        allocated = torch.cuda.memory_allocated(device)/1024**3
        reserved = torch.cuda.memory_reserved(device)/1024**3
        print(f"{prefix} [Rank {local_rank if local_rank != -1 else 0}] GPU内存: 已用 {allocated:.2f}GB, 保留 {reserved:.2f}GB")

# ---------- utils ----------
def load_mat(fp: str, key: str):
    """读取 v7 / v7.3 .mat"""
    try:
        return scipy.io.loadmat(fp)[key]
    except NotImplementedError:
        with h5py.File(fp, 'r') as f:
            return np.array(f[key]).T

def save_img(arr, path, cmap='gray'):
    arr_n = (arr - arr.min()) / (np.ptp(arr) + 1e-8)
    plt.imshow(arr_n, cmap=cmap); plt.axis('off'); plt.tight_layout(pad=0)
    plt.savefig(path, bbox_inches='tight', pad_inches=0); plt.close()

def poisson_nll(lam, y, eps=1e-8):
    """泊松负对数似然
    参数:
        lam: 模型预测的泊松分布参数 λ (活度投影)
        y: 观测数据 (探测器计数)
        eps: 数值稳定性常数
    返回:
        负对数似然 (y * log(λ) - λ) 的负值
        
    注意:
        - 这里使用mean()对批次和像素进行平均，使损失值稳定，不受批大小影响
        - 图片中使用sum()会导致损失值随批次大小和图像尺寸线性增长
        - 训练过程中使用mean()，最终评估时可考虑使用sum()以获得总似然
    """
    lam = torch.clamp(lam, min=eps)
    # 使用mean版本 (默认)
    return (lam - y * torch.log(lam)).mean()
    
    # # 使用sum版本 (如图片所示，需要对应调整权重)
    # return (lam - y * torch.log(lam)).sum()

def loss_tv(x):
    """各向同性总变差正则化
    
    注意:
    - MRI_helpers.py也使用mean实现，使损失对图像大小不敏感
    - 对于边缘差异取平均，保持与总体损失的尺度一致
    """
    dy = torch.abs(x[:, :-1, :] - x[:, 1:, :]).mean()
    dx = torch.abs(x[:, :, :-1] - x[:, :, 1:]).mean()
    return dx + dy

def loss_l1(x):
    """L1稀疏性约束
    
    注意:
    - 使用mean与MRI_helpers.py保持一致
    - 避免损失随图像大小线性增长
    """
    return torch.mean(torch.abs(x))

class ImgLogScale(nn.Module):
    """可学习的全局缩放因子
    
    在DPI框架中的作用:
    1. 对生成模型输出进行缩放，使其适应PET活度图像的值域范围
    2. 优化过程中，自动调整缩放因子，减少手动设置的需要
    3. 使得Flow模型能够专注于学习图像的结构特征，而不必同时学习整体强度
    
    在PET成像中的特殊意义:
    - PET图像值域可能跨越几个数量级，缩放因子帮助处理这种宽值域
    - 活度值始终为正，softplus+缩放的组合能很好地满足这一约束
    - 每个器官/组织的摄取值可能差异很大，全局缩放有助于整体调整
    
    注意:
    - 这种设计模仿了DPI_MRI.py中的做法，但在PET中可能更为重要
    - 原始Flow模型通常生成[-1,1]或标准正态值域的输出，而PET活度通常是任意正实数
    - 如果图像值域相对稳定且已知，也可以使用固定的缩放因子代替
    """
    def __init__(self, init_scale=1.0):
        super().__init__()
        self.log_scale = nn.Parameter(torch.tensor([np.log(init_scale)], dtype=torch.float32))

    def forward(self):
        return self.log_scale

# ---------- CLI ----------
parser = argparse.ArgumentParser("Deep Probabilistic Imaging Trainer for PET")

"""
RealNVP模型参数设置说明：
-------------------------
MRI与PET参数对比：

1. 流层数(n_flow)
   - MRI使用16层，适合复杂k空间变换
   - PET也使用16层(已调整)，以便捕捉活度分布的复杂结构
   - 若训练慢，可先用8层快速测试，实际重建推荐16层

2. 学习率(lr)
   - MRI使用1e-5，较保守
   - PET使用5e-5(已调整)，在稳定性和收敛速度间取得平衡
   - 过大的学习率(>1e-4)可能导致不稳定

3. 训练轮数(n_epoch)
   - MRI使用3000轮
   - PET默认设为10000轮(已调整)，适合64×64和128×128的图像
   - 128×128图像可能需要更多轮数收敛

4. 正则化参数
   - MRI和PET均使用总变差(TV)和L1正则化
   - TV权重均为1e3，保持结构平滑
   - 调整logdet权重可影响生成多样性
"""

# 主要路径参数 - 基于参数化的图像尺寸
# 通过args.image_size参数化所有路径，避免硬编码尺寸
parser.add_argument("--activity_path", default="../dataset/brain64_tumor_activity_map/brain64_tumor_FDG_K1_40min.mat", 
                    help="活度图路径，与args.image_size指定的尺寸相匹配")
parser.add_argument("--sinogram_path", default="../dataset/brain64_tumor_sinogram/brain64_tumor_FDG_K1_40min.mat",
                    help="投影数据路径，与args.image_size指定的尺寸相匹配")
parser.add_argument("--gmat_path", default="../dataset/G_system_matrix_64.mat", 
                    help="系统矩阵路径，需要与args.image_path匹配(如G_system_matrix_64.mat或G_system_matrix.mat)")
# parser.add_argument("--rmat_path", default="../dataset/r_noise_map.mat", help="MAT file with ri (old background noise)") # 旧的参数，将被替换
parser.add_argument("--save_dir", default="./checkpoint/pet64-flow16-0")

# 新增的路径参数
parser.add_argument("--image_size", type=int, default=64, help="图像的边长 (例如 64 表示 64x64)")
parser.add_argument("--ri_path", default="../dataset/brain64_tumor_sinogram/brain64_tumor_FDG_K1_40min_ri.mat", 
                    help="背景噪声/随机事件路径 (Poisson 噪声的一部分)")
# parser.add_argument("--ci_path", default="../dataset/brain64_tumor_sinogram/brain64_tumor_FDG_K1_40min_ci.mat", 
#                     help="探测器效率扰动路径")
parser.add_argument("--ytrue_path", default="../dataset/brain64_tumor_sinogram/brain64_tumor_FDG_K1_40min_ytrue.mat", 
                    help="无噪声真实投影路径 (用于验证)")

parser.add_argument("--frame_idx", type=int, default=10, help="time frame index")
parser.add_argument("--roi_idx",   type=int, default=30, help="slice/ROI index")

# Flow model
parser.add_argument("--model_form", choices=["realnvp", "glow"], default="realnvp")
parser.add_argument("--n_flow",    type=int, default=16, help="流层数，MRI中使用16层，PET中至少需要16层才能捕捉复杂结构")
parser.add_argument("--n_block",   type=int, default=4)  # for Glow

# Training hyper-params
parser.add_argument("--n_epoch", type=int, default=1000, help="训练轮数，对于复杂PET图像建议至少10k轮")
parser.add_argument("--n_batch", type=int, default=64, help="批量大小，内存不足时可适当减小")
parser.add_argument("--lr",      type=float, default=5e-5, help="学习率，MRI使用1e-5，PET建议使用5e-5")
parser.add_argument("--logdet",  type=float, default=1.0)
parser.add_argument("--tv",      type=float, default=1.0)
parser.add_argument("--l1",      type=float, default=0.0)
parser.add_argument("--clip",    type=float, default=1e-2)
parser.add_argument("--pretrained", default="", help="(optional) pre-trained generator .pth")
parser.add_argument("--lr_decay", action="store_true", help="是否使用学习率衰减")
parser.add_argument("--lr_step", type=int, default=10000, help="学习率衰减步长")
parser.add_argument("--lr_gamma", type=float, default=0.5, help="学习率衰减因子")

# ---------- 删除不需要的参数 ----------
# parser.add_argument("--resize", type=int, default=0, help="如果>0，将图像resize到该尺寸×该尺寸")
# parser.add_argument("--prefix", type=str, default="", help="文件命名前缀，方便区分不同实验")
# parser.add_argument("--fast_test", action="store_true", help="快速测试模式，训练更少的轮次")
# parser.add_argument("--verbose", action="store_true", help="详细输出模式，显示更多信息")
parser.add_argument("--cache_matrix_on_gpu", action="store_true", help="将系统矩阵存储在GPU上（需要更多显存）")

# --- Add seed argument for better control in distributed setting ---
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
# --- End seed argument ---

args = parser.parse_args()

# --- Adjust save_dir if running distributed to avoid conflicts if needed, or ensure only rank 0 writes ---
# For simplicity, we'll let rank 0 handle all saving to args.save_dir
if local_rank == -1 or local_rank == 0:
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

if local_rank != -1:
    dist.barrier() # Ensure rank 0 creates dir before other ranks proceed

if local_rank == -1 or local_rank == 0:
    print("---------- 1. 读取 PET 数据 (activity & sinogram) ----------")
act4d  = load_mat(args.activity_path,  'tracer')      # (H,W,S,T)
sino4d = load_mat(args.sinogram_path, 'sinogram')     # (nb,na,S,T)

if local_rank == -1 or local_rank == 0:
    print("✔ activity shape:", act4d.shape)   # 应为 (args.image_size,args.image_size,S,T)
    print("✔ sinogram shape:", sino4d.shape)  # 应为 (nb,na,S,T)

# 从 activity map 的形状推断S和T，H和W由args.image_size决定
_, _, S, T = act4d.shape 
H, W = args.image_size, args.image_size
N_pix = H * W
orig_H, orig_W = H, W # 固定为args.image_size，不再有resize逻辑

frame = np.clip(args.frame_idx, 0, T-1)
roi   = np.clip(args.roi_idx,   0, S-1)

if local_rank == -1 or local_rank == 0:
    print(f"====== Using image_size: {H}x{W}, frame: {frame}, roi: {roi} ======")

# ground-truth activity (仅用于可视化对比)
act_gt_full_res = act4d[:, :, roi, frame] # 这是从原始文件读取的，可能与args.image_size不同
# 确保act_gt与目标image_size一致
if act_gt_full_res.shape[0] != H or act_gt_full_res.shape[1] != W:
    if local_rank == -1 or local_rank == 0:
        print(f"[Resize] Ground truth activity map from {act_gt_full_res.shape} to {H}x{W}")
    import cv2
    act_gt = cv2.resize(act_gt_full_res, (H, W), interpolation=cv2.INTER_AREA)
else:
    act_gt = act_gt_full_res

# 观测 sinogram → 展平向量 y (这是带噪声的观测)
y_vec = torch.tensor(
    sino4d[:, :, roi, frame].T.flatten(),
    dtype=torch.float32, device=device)               # (M,)
if local_rank == -1 or local_rank == 0:
    print("✔ 观测数据 y_vec (noisy sinogram) shape:", y_vec.shape)

if local_rank == -1 or local_rank == 0:
    print("---------- 2. 系统矩阵 G_sparse & 新增的 ci, ri, ytrue ----------")
# 根据 image_size 动态确定系统矩阵路径
if args.image_size == 64:
    gmat_file = "G_system_matrix_64.mat"
    # 自动调整相关路径的默认值以匹配64x64的示例数据
    if "../dataset/brain128_tumor" in args.activity_path: # This logic might be problematic if paths change
        if local_rank == -1 or local_rank == 0:
            print("[INFO] Adapting paths from 128 to 64 based on activity_path.")
        args.activity_path = args.activity_path.replace("brain128_tumor", "brain64_tumor")
        args.sinogram_path = args.sinogram_path.replace("brain128_tumor", "brain64_tumor")
        args.ri_path = args.ri_path.replace("brain128_tumor", "brain64_tumor")
        # args.ci_path = args.ci_path.replace("brain128_tumor", "brain64_tumor")
        args.ytrue_path = args.ytrue_path.replace("brain128_tumor", "brain64_tumor")
    if local_rank == -1 or local_rank == 0:
        print(f"[INFO] 使用64x64数据集路径和{gmat_file}")
elif args.image_size == 128:
    gmat_file = "G_system_matrix.mat"
    # 确保路径是128版本
    if "brain64_tumor" in args.activity_path: # This logic might be problematic
        if local_rank == -1 or local_rank == 0:
            print("[INFO] Adapting paths from 64 to 128 based on activity_path.")
        args.activity_path = args.activity_path.replace("brain64_tumor", "brain128_tumor")
        args.sinogram_path = args.sinogram_path.replace("brain64_tumor", "brain128_tumor")
        args.ri_path = args.ri_path.replace("brain64_tumor", "brain128_tumor")
        # args.ci_path = args.ci_path.replace("brain64_tumor", "brain128_tumor")
        args.ytrue_path = args.ytrue_path.replace("brain64_tumor", "brain128_tumor")
    if local_rank == -1 or local_rank == 0:
        print(f"[INFO] 使用128x128数据集路径和{gmat_file}")
else:
    # Ensure this error is raised by all or handled by rank 0 exiting
    if local_rank == -1 or local_rank == 0:
        raise ValueError(f"不支持的image_size: {args.image_size}。请使用64或128。")
    # If distributed, other ranks might hang if only rank 0 raises.
    # Consider sys.exit() or a more robust distributed error handling.
    # For now, let's assume if rank 0 errors out, the job stops.

# 动态构建gmat_path的上层目录
gmat_dir = Path(args.gmat_path).parent
actual_gmat_path = gmat_dir / gmat_file

if local_rank == -1 or local_rank == 0:
    print(f"[系统矩阵] 尝试从以下路径加载: {actual_gmat_path}")
Gsp = csc_matrix(load_mat(str(actual_gmat_path), 'G_sparse'))

# 是否缓存系统矩阵在CUDA中（可以加速计算但需要更多GPU内存）
cache_system_matrix = args.cache_matrix_on_gpu

# 检查系统矩阵尺寸与图像尺寸(orig_H, orig_W)是否匹配
# 系统矩阵是针对 orig_H * orig_W 的，即args.image_size
if Gsp.shape[1] != orig_H * orig_W:
    if local_rank == -1 or local_rank == 0:
        print(f"[警告] 系统矩阵列数({Gsp.shape[1]})与期望的活动图尺寸({orig_H}×{orig_W}={orig_H*orig_W})不匹配")
        print(f"       请确保系统矩阵 ({actual_gmat_path}) 与图像尺寸 (args.image_size={args.image_size}) 匹配，否则可能导致重建结果不正确")

if local_rank == -1 or local_rank == 0:
    print("✔ G shape:", Gsp.shape, "| non-zeros:", Gsp.nnz,
          f"({Gsp.nnz / (Gsp.shape[0] * Gsp.shape[1]):.2%} non zeros)")

# 创建稀疏张量
coo = Gsp.tocoo()
# 系统矩阵的 device 处理: `device` is now process-specific if distributed
A_torch_device = device if cache_system_matrix else 'cpu'
A_torch = torch.sparse_coo_tensor(
    np.vstack((coo.row, coo.col)), coo.data, Gsp.shape,
    dtype=torch.float32, device=A_torch_device).coalesce()

if local_rank == -1 or local_rank == 0:
    print(f"✔ 系统矩阵存储在{'GPU' if A_torch_device != 'cpu' else 'CPU'}上 (Target device for A_torch: {A_torch_device})")
    print("✔ torch sparse A:", A_torch.shape, "| dtype:", A_torch.dtype, "| device:", A_torch.device)

# 加载 ci (探测器效率扰动)
# ci_slice = load_mat(args.ci_path, 'ci')[:, :, roi, frame] 
# ci_vec = torch.tensor(ci_slice.T.flatten(), dtype=torch.float32, device=device)
# if local_rank == -1 or local_rank == 0:
# print("✔ 探测器效率 ci_vec shape:", ci_vec.shape)

# 加载 ri (背景噪声/随机事件)
ri_slice = load_mat(args.ri_path, 'ri')[:, :, roi, frame]  
ri_vec = torch.tensor(ri_slice.T.flatten(), dtype=torch.float32, device=device)
if local_rank == -1 or local_rank == 0:
    print("✔ 背景噪声 ri_vec shape:", ri_vec.shape)

# 加载 ytrue (无噪声真实投影)
ytrue_slice = load_mat(args.ytrue_path, 'ytrue')[:, :, roi, frame]
ytrue_vec = torch.tensor(ytrue_slice.T.flatten(), dtype=torch.float32, device=device)
if local_rank == -1 or local_rank == 0:
    print("✔ 无噪声真实投影 ytrue_vec shape:", ytrue_vec.shape)

# 验证 ytrue (可选但推荐)
# 将真实活度图展平以匹配系统矩阵的输入
# Ensure act_gt_vec is on the same device as A_torch for matmul, then move result to current process's device
act_gt_vec_device = A_torch.device if A_torch.is_cuda else device # Use A_torch's device if it's on CUDA, else current process's device
act_gt_vec = torch.tensor(act_gt.flatten(order='F'), dtype=torch.float32, device=act_gt_vec_device) 

# 计算 y_calc_from_gt = ci * (A @ act_gt) + ri
# 注意: A_torch @ act_gt_vec 需要 act_gt_vec 是列向量 [N,1]
proj_gt_no_ci_ri = torch.sparse.mm(A_torch.to(act_gt_vec.device), act_gt_vec.unsqueeze(1)).squeeze().to(device)
# 确保所有张量在同一设备上
# proj_gt_no_ci_ri = proj_gt_no_ci_ri.to(device) # Already done in the line above

# 分两步验证 - 第一步：验证 ytrue ≈ ci * (A @ act_gt)，不考虑ri
# y_calc_no_ri = ci_vec * proj_gt_no_ci_ri 
y_calc_no_ri = proj_gt_no_ci_ri 
mse_ytrue_no_ri = torch.mean((ytrue_vec - y_calc_no_ri)**2)
if local_rank == -1 or local_rank == 0:
    print(f"[验证第一步] ytrue与(A@act_gt)的MSE: {mse_ytrue_no_ri.item():.4e}")

# 第二步：验证完整模型 y ≈ ci * (A @ act_gt) + ri
# y_calc_from_gt = ci_vec * proj_gt_no_ci_ri + ri_vec
y_calc_from_gt = proj_gt_no_ci_ri + ri_vec
mse_ytrue_validation = torch.mean((ytrue_vec - y_calc_from_gt)**2)
if local_rank == -1 or local_rank == 0:
    print(f"[验证第二步] ytrue与完整模型((A@act_gt)+ri)的MSE: {mse_ytrue_validation.item():.4e}")

# 第三步：验证y与ytrue+噪声的一致性
mse_y_vs_ytrue_ri = torch.mean((y_vec - (ytrue_vec + ri_vec))**2)
if local_rank == -1 or local_rank == 0:
    print(f"[验证第三步] y与(ytrue+ri)的MSE: {mse_y_vs_ytrue_ri.item():.4e}")

# 确定是否存在问题
if mse_ytrue_no_ri.item() > 1e-5 and (local_rank == -1 or local_rank == 0) :  # 第一步验证阈值
    print(f"[警告] ytrue与ci*(A@act_gt)差异较大，系统矩阵A或探测器效率ci可能有问题！")
    
    # 添加可视化代码来帮助诊断问题
    print("\n===== 开始诊断 ytrue 验证差异问题 =====")
    
    # 1. 检查并打印活度图和投影数据的基本信息
    print(f"活度图形状: {act_gt.shape}, 最小值: {act_gt.min():.4f}, 最大值: {act_gt.max():.4f}, 均值: {act_gt.mean():.4f}")
    print(f"观测投影形状: {y_vec.shape}, 最小值: {y_vec.min().item():.4f}, 最大值: {y_vec.max().item():.4f}, 均值: {y_vec.mean().item():.4f}")
    print(f"ytrue投影形状: {ytrue_vec.shape}, 最小值: {ytrue_vec.min().item():.4f}, 最大值: {ytrue_vec.max().item():.4f}, 均值: {ytrue_vec.mean().item():.4f}")
    print(f"计算投影(不含ri)形状: {y_calc_no_ri.shape}, 最小值: {y_calc_no_ri.min().item():.4f}, 最大值: {y_calc_no_ri.max().item():.4f}, 均值: {y_calc_no_ri.mean().item():.4f}")
    print(f"计算投影(含ri)形状: {y_calc_from_gt.shape}, 最小值: {y_calc_from_gt.min().item():.4f}, 最大值: {y_calc_from_gt.max().item():.4f}, 均值: {y_calc_from_gt.mean().item():.4f}")
    
    # 2. 检查ci和ri的统计信息
    # print(f"ci形状: {ci_vec.shape}, 最小值: {ci_vec.min().item():.4f}, 最大值: {ci_vec.max().item():.4f}, 均值: {ci_vec.mean().item():.4f}")
    print(f"ri形状: {ri_vec.shape}, 最小值: {ri_vec.min().item():.4f}, 最大值: {ri_vec.max().item():.4f}, 均值: {ri_vec.mean().item():.4f}")
    
    # 3. 计算其他差异指标
    mse_y_ytrue = torch.mean((y_vec - ytrue_vec)**2).item()
    rel_error_no_ri = torch.mean(torch.abs(ytrue_vec - y_calc_no_ri) / (ytrue_vec + 1e-8)).item()
    print(f"观测y与ytrue之间的MSE: {mse_y_ytrue:.4e}")
    print(f"ytrue与ci*(A@act_gt)之间的相对误差: {rel_error_no_ri:.4%}")
    
    # 4. 保存可视化图像进行比较
    # Ensure all prints and plt saving in this block are conditional
    debug_dir = Path(args.save_dir) / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存活度图
    save_img(act_gt, debug_dir / "act_gt.png")
    print(f"✔ 已保存活度图到 {debug_dir}/act_gt.png")
    
    # 重塑和保存投影数据 (sinogram)
    nb, na = sino4d.shape[0], sino4d.shape[1]
    try:
        y_reshaped = y_vec.cpu().numpy().reshape(na, nb).T
        ytrue_reshaped = ytrue_vec.cpu().numpy().reshape(na, nb).T
        y_calc_no_ri_reshaped = y_calc_no_ri.cpu().numpy().reshape(na, nb).T
        y_calc_with_ri_reshaped = y_calc_from_gt.cpu().numpy().reshape(na, nb).T
        
        # 保存投影图像
        plt.figure(figsize=(20, 5))
        
        plt.subplot(141)
        plt.imshow(y_reshaped, cmap='hot')
        plt.title('观测投影 (y)')
        plt.colorbar()
        
        plt.subplot(142)
        plt.imshow(ytrue_reshaped, cmap='hot')
        plt.title('真实投影 (ytrue)')
        plt.colorbar()
        
        plt.subplot(143)
        plt.imshow(y_calc_no_ri_reshaped, cmap='hot')
        plt.title('计算投影 (ci*(A@act_gt))')
        plt.colorbar()
        
        plt.subplot(144)
        plt.imshow(y_calc_with_ri_reshaped, cmap='hot')
        plt.title('计算投影 (ci*(A@act_gt)+ri)')
        plt.colorbar()
        
        plt.tight_layout()
        plt.savefig(debug_dir / "projections_comparison.png")
        plt.close()
        print(f"✔ 已保存投影对比图到 {debug_dir}/projections_comparison.png")
        
        # 保存差异图
        plt.figure(figsize=(15, 5))
        
        plt.subplot(131)
        diff_no_ri = ytrue_reshaped - y_calc_no_ri_reshaped
        plt.imshow(diff_no_ri, cmap='bwr', vmin=-np.abs(diff_no_ri).max(), vmax=np.abs(diff_no_ri).max())
        plt.title('ytrue与ci*(A@act_gt)的差异')
        plt.colorbar()
        
        plt.subplot(132)
        rel_diff_no_ri = diff_no_ri / (ytrue_reshaped + 1e-8)
        plt.imshow(rel_diff_no_ri, cmap='bwr', vmin=-0.5, vmax=0.5)
        plt.title('相对差异 (不含ri)')
        plt.colorbar()
        
        plt.subplot(133)
        plt.imshow(ri_vec.cpu().numpy().reshape(na, nb).T, cmap='hot')
        plt.title('背景噪声 (ri)')
        plt.colorbar()
        
        plt.tight_layout()
        plt.savefig(debug_dir / "projections_difference.png")
        plt.close()
        print(f"✔ 已保存差异图到 {debug_dir}/projections_difference.png")
        
    except Exception as e:
        print(f"投影数据可视化失败: {str(e)}")
    
    print("\n⚠️ 可能的问题原因：")
    print("1. 系统矩阵与当前图像尺寸不匹配")
    print("2. ci或ytrue文件与活度图不对应")
    print("3. ytrue可能是另一种方式生成的，而非简单的ci*(A@act_gt)")
    print("4. 数据集未按预期对齐（如roi_idx或frame_idx不正确）")
    print("请查看保存的可视化图像以进一步诊断问题")
    print("===== 诊断信息结束 =====\n")

if local_rank == -1 or local_rank == 0:
    print("---------- 3. 构造 / 加载 Flow 生成器 ----------")
start_time = time.time()

# N_pix 传递给生成器的是args.image_size*args.image_size
if args.model_form == 'realnvp':
    Gnet = realnvpfc_model.RealNVP(N_pix, args.n_flow, affine=True).to(device)
else:
    # Glow 模型的输入参数需要传递H（图像高度）
    n_channel = 1 # Default to 1, can be adjusted if needed based on image_size
    z_shapes = glow_model.calc_z_shapes(n_channel, H, args.n_flow, args.n_block)
    Gnet = glow_model.Glow(n_channel, args.n_flow, args.n_block, affine=True, conv_lu=True).to(device)

# 确保模型处于训练模式
Gnet.train()

if local_rank != -1: # If distributed
    Gnet = DDP(Gnet, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True) # find_unused_parameters for safety

elapsed = time.time() - start_time
if local_rank == -1 or local_rank == 0:
    print(f"✔ Flow model form: {args.model_form} | Built in {elapsed:.3f} seconds")

if args.pretrained:
    model_to_load = Gnet.module if local_rank != -1 else Gnet
    # Load state dict to CPU first to avoid device mismatches, then to model's device
    state_dict = torch.load(args.pretrained, map_location='cpu')
    model_to_load.load_state_dict(state_dict)
    if local_rank == -1 or local_rank == 0:
        print("Pre-trained weights loaded:", args.pretrained)

if local_rank == -1 or local_rank == 0:
    print("---------- 4. 可学习全局缩放因子 ----------")
# 初始化缩放因子为活度图均值，适应不同尺寸图像的绝对强度
init_scale = np.mean(act_gt)  # act_gt is numpy array, same for all processes

# 是否使用可学习的缩放因子
use_learnable_scale = True  # 将此设置为False可以尝试不使用缩放因子

if use_learnable_scale:
    logscale = ImgLogScale(init_scale).to(device)
    if local_rank != -1: # If distributed
        logscale = DDP(logscale, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    if local_rank == -1 or local_rank == 0:
        print("[Logscale] 使用可学习缩放因子，初始值 =", init_scale)
else:
    if local_rank == -1 or local_rank == 0:
        print("[Logscale] 不使用可学习缩放因子，将使用固定缩放或无缩放")


if local_rank == -1 or local_rank == 0:
    print("---------- 5. 优化器 ----------")
# Optimizer setup: parameters from DDP-wrapped models are fine.
if use_learnable_scale:
    optimizer = optim.Adam(list(Gnet.parameters()) + list(logscale.parameters()),
                           lr=args.lr)
else:
    optimizer = optim.Adam(Gnet.parameters(), lr=args.lr)

# 参考MRI方法，计算权重 (These calculations are the same for all processes)
nz_ratio = Gsp.nnz / (Gsp.shape[0] * Gsp.shape[1])  # 系统矩阵非零元素比例
mean_act_gt = np.mean(act_gt)
if mean_act_gt < 1e-6: # 防止除以非常小的值导致权重过大
    if local_rank == -1 or local_rank == 0: # Print warning only once
        print(f"[警告] 真实活度图均值 ({mean_act_gt:.2e}) 非常小，可能导致正则化权重计算不稳定。请检查数据。")
    mean_act_gt = 1e-6 # 使用一个小的正值代替

logdet_w = args.logdet * 1e-4 / (nz_ratio * N_pix) 
tv_w = args.tv / (0.5 * mean_act_gt * W)

if args.l1 > 0:
    l1_w = args.l1 / (mean_act_gt * N_pix) 
else:
    l1_w = 0.0

if local_rank == -1 or local_rank == 0:
    print(f"[权重] logdet={logdet_w:.3e} (来自args.logdet={args.logdet:.1e}), "
          f"TV={tv_w:.3e} (来自args.tv={args.tv:.1e}), "
          f"L1={l1_w:.3e} (来自args.l1={args.l1:.1e})")
    print(f"[INFO] 损失函数均使用 'mean' 版本进行聚合。数据保真项(data loss)将在损失中占主导地位。")

loss_rec = {'total': [], 'data': [], 'logdet': [], 'tv': [], 'l1': []} # Rank 0 collects loss history

if local_rank == -1 or local_rank == 0:
    print("---------- 6. 训练循环 ----------")
tick = time.time()  # 起始时间

# 固定打印间隔 - 不再依赖verbose参数
print_interval = 100  # 每100轮打印一次详细信息
# checkpoint_interval = 1000 # Not used explicitly for saving best model, best model saved at end or on improvement

# 初始化最佳loss跟踪 (only rank 0 needs to track this for saving best model)
best_loss = float('inf')
best_epoch = -1
no_improve_count = 0
early_stopping_patience = 30  # 连续30次检查（30*print_interval epochs）没有改善则停止 -> updated comment

# 设置随机种子确保可复现性
current_seed = args.seed
if local_rank != -1:
    current_seed += local_rank # Different seed for each process to vary z sampling

torch.manual_seed(current_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(current_seed) # Use manual_seed_all for distributed
    # cudnn.deterministic and benchmark might be better set based on global rank 0 decision or consistently
    if local_rank == -1 or local_rank == 0: # Or set for all
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False # Set to False if deterministic is True
np.random.seed(current_seed)
if local_rank == -1 or local_rank == 0:
    print(f"[Rank {local_rank if local_rank != -1 else 0}] Seed set to {current_seed}")


# --- tqdm for rank 0 only ---
epoch_iterator = range(args.n_epoch)
if local_rank == -1 or local_rank == 0:
    epoch_iterator = trange(args.n_epoch, desc="Training Epoch")

for epoch in epoch_iterator:
    # 1) 采 z - 维度由N_pix决定，固定为args.image_size*args.image_size
    if args.model_form == 'realnvp':
        z = torch.randn(args.n_batch, N_pix, device=device)
    else: # glow
        z = [torch.randn(args.n_batch, *s, device=device) for s in z_shapes]

    # 2) 生成正值图像 & logdet
    if local_rank != -1: # Distributed training
        img_raw, logdet_val = Gnet.module.reverse(z) # Use Gnet.module for DDP
    else: # Single GPU training
        img_raw, logdet_val = Gnet.reverse(z)
    img_raw = img_raw.view(-1, H, W)  # H,W固定为args.image_size
    
    if use_learnable_scale:
        if local_rank != -1: # Distributed
            current_log_scale_val = logscale.module.forward() # Call forward on the original module
        else: # Single GPU
            current_log_scale_val = logscale.forward() # Or simply logscale()
        img_pos = torch.nn.functional.softplus(img_raw) * torch.exp(current_log_scale_val)
        
        det_softplus = torch.sum(img_raw - torch.nn.functional.softplus(img_raw), (1, 2))
        # current_log_scale_val is Parameter(torch.tensor([value])), so access .item() or ensure it's broadcastable
        # For DDP, logscale() returns the tensor. It should be [1] or scalar.
        det_scale = current_log_scale_val.squeeze() * H * W # Squeeze if it's [1]
        logdet_val = logdet_val + det_softplus + det_scale
    else:
        img_pos = torch.nn.functional.softplus(img_raw)
        det_softplus = torch.sum(img_raw - torch.nn.functional.softplus(img_raw), (1, 2))
        logdet_val = logdet_val + det_softplus
        # fixed_scale logic if needed would go here, affecting logdet_val

    img_vec = img_pos.permute(0, 2, 1).contiguous().view(args.n_batch, -1)
    
    # 3) 前向投影 λ = ci * (A·x) + ri
    # Ensure A_torch is on the correct device for matmul with img_vec
    # img_vec is on `device` (current process's GPU)
    # A_torch is on `A_torch_device` (either 'cpu' or current process's GPU if cache_matrix_on_gpu)
    
    A_op_device = img_vec.device # Operations should happen on the device of img_vec
    A_for_op = A_torch.to(A_op_device)

    if A_for_op.device.type != 'cuda' or not cache_system_matrix: # If A is on CPU, or not cached (redundant check but for clarity)
        lam = torch.zeros(args.n_batch, A_for_op.shape[0], device=A_op_device)
        for b in range(args.n_batch):
            # This loop is slow. Vectorize if possible.
            # However, sparse.mm is usually efficient.
            proj_no_ci_ri = torch.sparse.mm(A_for_op, img_vec[b:b+1].t()).squeeze() 
            lam[b] = proj_no_ci_ri + ri_vec # ri_vec is on `device` (A_op_device)
    else: # A_torch is on GPU and cached
        img_vec_t = img_vec.t() 
        proj_no_ci_ri_batch = torch.sparse.mm(A_for_op, img_vec_t).t() 
        lam = proj_no_ci_ri_batch + ri_vec.unsqueeze(0)

    # 4) 损失函数计算
    y_batch = y_vec.unsqueeze(0).expand_as(lam) # y_vec is on `device` (A_op_device)
    data_term = poisson_nll(lam, y_batch)
    
    logdet_term = -logdet_w * torch.mean(logdet_val) 
    tv_term = tv_w * loss_tv(img_pos) if tv_w > 0 else torch.tensor(0.0, device=device) # ensure tensor for addition
    l1_term = l1_w * loss_l1(img_pos) if l1_w > 0 else torch.tensor(0.0, device=device) # ensure tensor
    loss = data_term + logdet_term + tv_term + l1_term

    # 5) 反向传播
    optimizer.zero_grad()
    loss.backward() # DDP handles gradient synchronization
    # Clipping gradients for both models if use_learnable_scale
    # Parameters are obtained from the optimizer, which already has all relevant params
    nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], args.clip)
    
    optimizer.step()
    
    # 记录损失 (only rank 0)
    if local_rank == -1 or local_rank == 0:
        loss_rec['total'].append(loss.item())
        loss_rec['data'].append(data_term.item())
        loss_rec['logdet'].append(logdet_term.item())
        loss_rec['tv'].append(tv_term.item() if tv_w > 0 else 0) # Use .item()
        loss_rec['l1'].append(l1_term.item() if l1_w > 0 else 0) # Use .item()
    
    if epoch % 10 == 0 and torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    if epoch % print_interval == 0:
        if local_rank == -1 or local_rank == 0:
            elapsed = time.time() - tick
            
            with torch.no_grad():
                sample_img_np = img_pos[0].cpu().numpy() # From rank 0's batch
                cur_mse = np.mean((sample_img_np - act_gt)**2) # act_gt is numpy
                cur_psnr = 10 * np.log10(np.max(act_gt)**2 / (cur_mse + 1e-9)) if cur_mse > -1e9 else float('inf') # avoid div by zero
                
                has_ssim = False
                cur_ssim = 0
                try:
                    from skimage.metrics import structural_similarity as ssim
                    cur_ssim = ssim(sample_img_np, act_gt, data_range=np.max(act_gt))
                    has_ssim = True
                except ImportError:
                    pass # ssim not available
                
                # Loss components for print (from rank 0's calculation)
                current_data_item = data_term.item()
                current_logdet_item = logdet_term.item()
                current_tv_item = tv_term.item() if tv_w > 0 else 0
                current_l1_item = l1_term.item() if l1_w > 0 else 0
                
                total_abs_loss = abs(current_data_item) + abs(current_logdet_item) + \
                                 abs(current_tv_item) + abs(current_l1_item)
                                 
                data_ratio = abs(current_data_item) / (total_abs_loss + 1e-10) * 100
                logdet_ratio = abs(current_logdet_item) / (total_abs_loss + 1e-10) * 100
                tv_ratio = abs(current_tv_item) / (total_abs_loss + 1e-10) * 100
                l1_ratio = abs(current_l1_item) / (total_abs_loss + 1e-10) * 100
            
                print(f"Epoch {epoch:04d}/{args.n_epoch} | total {loss.item():.4e} | "
                      f"data {current_data_item:.4e} | logdet {current_logdet_item:.4e} | "
                      f"TV {current_tv_item:.4e} | "
                      f"L1 {current_l1_item:.4e}")
                print(f"[损失占比] data: {data_ratio:.1f}%, logdet: {logdet_ratio:.1f}%, "
                      f"TV: {tv_ratio:.1f}%, L1: {l1_ratio:.1f}%")
                print(f"[质量评估] MSE: {cur_mse:.6f}, PSNR: {cur_psnr:.2f} dB"
                      f"{f', SSIM: {cur_ssim:.4f}' if has_ssim else ''} | "
                      f"[Time] {print_interval} iters in {elapsed:.2f}s "
                      f"({elapsed/print_interval:.2f}s/iter, ETA: {elapsed/print_interval*(args.n_epoch-epoch)/3600:.1f}h)")
                
                print_gpu_memory(f"Epoch {epoch}")

                # Update tqdm description
                if hasattr(epoch_iterator, 'set_description'):
                    epoch_iterator.set_description(
                        f"E{epoch} L{loss.item():.2e} PSNR{cur_psnr:.1f}"
                    )

                current_total_loss_item = loss.item()
                if current_total_loss_item < best_loss:
                    best_loss = current_total_loss_item
                    best_epoch = epoch
                    no_improve_count = 0
                    print(f"[最佳模型] 更新! Loss: {best_loss:.6e}, Epoch: {best_epoch}")
                else:
                    no_improve_count += 1
                
                if data_ratio < 40 and epoch > 100:
                    print(f"[警告] 数据项占比过低({data_ratio:.1f}%)，可能导致过度正则化。"
                          f"考虑进一步降低tv_w值(当前:{tv_w:.2e})或调高学习率。")
        
        tick = time.time() # Reset tick for rank 0 timing
        
        # Early stopping logic (coordinated)
        should_stop_tensor = torch.tensor(0, dtype=torch.int, device=device)
        if local_rank == -1 or local_rank == 0:
            if no_improve_count >= early_stopping_patience:
                print(f"\n[早停] 连续 {no_improve_count} 次检查 ({no_improve_count * print_interval} epochs) loss 未改善，停止训练。")
                should_stop_tensor.fill_(1)
        
        if local_rank != -1: # If distributed, synchronize stop signal from rank 0
            dist.broadcast(should_stop_tensor, src=0)

        if should_stop_tensor.item() == 1:
            break # All processes break the loop
    
    # 每100轮保存一次当前生成的图像 (only rank 0)
    if (epoch % 100 == 0 or epoch == args.n_epoch - 1) and (local_rank == -1 or local_rank == 0):
        with torch.no_grad():
            sample_img_np = img_pos[0].cpu().numpy() # From rank 0's batch
            # Add rank to filename if debugging distributed setup, otherwise not needed for final single image
            # recon_img_path = str(Path(f"{args.save_dir}/recon_epoch{epoch:04d}_rank{local_rank if local_rank != -1 else 0}.png"))
            recon_img_path = str(Path(f"{args.save_dir}/recon_epoch{epoch:04d}.png"))
            save_img(sample_img_np, recon_img_path)

if local_rank == -1 or local_rank == 0:
    print("---------- 7. 保存最终模型 ----------")
# 保存最终模型 (only rank 0)
if local_rank == -1 or local_rank == 0:
    # Access .module when DDP wrapped
    model_to_save_gnet_state = Gnet.module.state_dict() if local_rank != -1 else Gnet.state_dict()
    
    final_model_dict = {
        'model_state_dict': model_to_save_gnet_state,
        'image_size': args.image_size,
        'n_flow': args.n_flow, # Save key hyperparams
        'n_block': args.n_block,
        'model_form': args.model_form
    }

    if use_learnable_scale:
        logscale_state_to_save = logscale.module.state_dict() if local_rank != -1 else logscale.state_dict()
        final_model_dict['logscale_state_dict'] = logscale_state_to_save
        # Access original parameter for the value, not the DDP wrapper
        logscale_param_val = (logscale.module.log_scale if local_rank != -1 else logscale.log_scale).item()
        final_model_dict['scale_factor'] = np.exp(logscale_param_val)
    else:
        final_model_dict['use_learnable_scale'] = False
        # final_model_dict['fixed_scale'] = init_scale # This is just the initial scale

    # 构建简化的文件名
    model_params = f"{args.model_form}_flow{args.n_flow}"
    reg_params = f"ld{args.logdet:.0e}_tv{args.tv:.0e}_l1{args.l1:.0e}"
    train_params = f"lr{args.lr}_batch{args.n_batch}" # Batch size here is per-GPU
    size_params = f"size{args.image_size}"
    scale_type = "adaptScale" if use_learnable_scale else "fixedScale"

    checkpoint_path_final = f"{args.save_dir}/{model_params}_{reg_params}_{train_params}_{size_params}_{scale_type}_final_epoch{epoch+1}.pth"

    try:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True) 
        torch.save(final_model_dict, checkpoint_path_final)
        print(f"✔ Final model saved to {checkpoint_path_final}")
    except Exception as e:
        print(f"⚠️ 保存最终模型失败: {e}")

    loss_curve_path = str(Path(f"{args.save_dir}/loss_{model_params}_{reg_params}_{train_params}_{size_params}_{scale_type}_epoch{epoch+1}.npy"))
    try:
        np.save(loss_curve_path, loss_rec) # loss_rec collected by rank 0
        print(f"✔ Loss curve saved to {loss_curve_path}")
    except Exception as e:
        print(f"⚠️ 保存损失曲线失败: {e}")

    print(f"✔ 训练完成，总轮数: {epoch+1}, 最佳轮数: {best_epoch}, 最佳loss: {best_loss:.6e}")
    print(f"✔ 最终模型已保存，可使用单独的评估脚本进行结果可视化和分析")

# --- Distributed Training Cleanup ---
if local_rank != -1:
    dist.destroy_process_group()
    if local_rank == 0: # Print cleanup message only from rank 0
        print(f"[Rank {local_rank}] Distributed training finished and cleaned up.")
# --- End Distributed Training Cleanup ---
