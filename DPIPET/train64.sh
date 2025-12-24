#!/bin/bash
#BSUB -J train641611
#BSUB -n 4
#BSUB -R "span[ptile=2] select[ngpus>0] rusage[ngpus_shared=4]"
#BSUB -q gpuq
#BSUB -o output641611
#BSUB -e error641611

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
  --activity_path   /gpfsdata/home/Zhaobo_yuxuan/work/FlowPET/dataset/brain64_tumor_activity_map/brain64_tumor_FDG_K1_40min.mat \
  --sinogram_path   /gpfsdata/home/Zhaobo_yuxuan/work/FlowPET/dataset/brain64_tumor_sinogram/brain64_tumor_FDG_K1_40min.mat \
  --gmat_path       /gpfsdata/home/Zhaobo_yuxuan/work/FlowPET/dataset/G_system_matrix_64.mat \
  --ri_path         /gpfsdata/home/Zhaobo_yuxuan/work/FlowPET/dataset/brain64_tumor_sinogram/brain64_tumor_FDG_K1_40min_ri.mat \
  --ytrue_path      /gpfsdata/home/Zhaobo_yuxuan/work/FlowPET/dataset/brain64_tumor_sinogram/brain64_tumor_FDG_K1_40min_ytrue.mat \
  --save_dir      /gpfsdata/home/Zhaobo_yuxuan/work/FlowPET/checkpoint/train641611 \
  --model_form     glow \
  --image_size     64  \
  --n_epoch        100000 \
  --n_flow         16 \
  --n_batch        64 \
  --tv 5 || { echo "[ERROR] Python script execution failed"; exit 1; }
  
echo "[INFO] Job finished successfully."

