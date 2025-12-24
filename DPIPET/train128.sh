#!/bin/bash
#BSUB -J train1281622
#BSUB -n 4
#BSUB -R "span[ptile=2] select[ngpus>0] rusage[ngpus_shared=4]"
#BSUB -q gpuq
#BSUB -o output1281622
#BSUB -e error1281622

echo "[INFO] Checking environment..."
hostname
whoami
which python
which conda

echo "[INFO] Loading conda environment..."
source ~/.bashrc
conda activate pytorch || { echo "[ERROR] Conda activate failed"; exit 1; }

echo "[INFO] Starting PyTorch job..."
python --version || { echo "[ERROR] Python not found"; exit 1; }

python -u /gpfsdata/home/Zhaobo_yuxuan/work/FlowPET/DPItorch/DPI_PET.py \
  --activity_path   /gpfsdata/home/Zhaobo_yuxuan/work/FlowPET/dataset/brain128_tumor_activity_map/brain128_tumor_FDG_K1_40min.mat \
  --sinogram_path   /gpfsdata/home/Zhaobo_yuxuan/work/FlowPET/dataset/brain128_tumor_sinogram/brain128_tumor_FDG_K1_40min.mat \
  --gmat_path       /gpfsdata/home/Zhaobo_yuxuan/work/FlowPET/dataset/G_system_matrix_128.mat \
  --ri_path         /gpfsdata/home/Zhaobo_yuxuan/work/FlowPET/dataset/brain128_tumor_sinogram/brain128_tumor_FDG_K1_40min_ri.mat \
  --ytrue_path      /gpfsdata/home/Zhaobo_yuxuan/work/FlowPET/dataset/brain128_tumor_sinogram/brain128_tumor_FDG_K1_40min_ytrue.mat \
  --save_dir      /gpfsdata/home/Zhaobo_yuxuan/work/FlowPET/checkpoint/train1281622 \
  --model_form     glow \
  --image_size     128  \
  --n_epoch        50000 \
  --n_flow         16 \
  --n_batch        32 \
  --logdet 100 \
  --tv 25 || { echo "[ERROR] Python script execution failed"; exit 1; }
  
echo "[INFO] Job finished successfully."

