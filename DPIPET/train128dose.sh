#!/bin/bash
#BSUB -J train1281617
#BSUB -n 4
#BSUB -R "span[ptile=2] select[ngpus>0] rusage[ngpus_shared=4]"
#BSUB -q gpuq
#BSUB -o output1281617
#BSUB -e error1281617

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

python -u /gpfsdata/home/Zhaobo_yuxuan/work/FlowPET/DPItorch/DPI_PET_Scale.py \
  --activity_path   /gpfsdata/home/Zhaobo_yuxuan/work/FlowPET/dataset/brain128_tumor_activity_map/brain128_tumor_FDG_K1_40min.mat \
  --sinogram_path   /gpfsdata/home/Zhaobo_yuxuan/work/FlowPET/dataset/brain128_tumor_dose_output/dose_7e+07brain128_tumor_FDG_K1_40min.mat \
  --gmat_path       /gpfsdata/home/Zhaobo_yuxuan/work/FlowPET/dataset/G_system_matrix_128_dose.mat \
  --ri_path         /gpfsdata/home/Zhaobo_yuxuan/work/FlowPET/dataset/brain128_tumor_dose_output/dose_7e+07/brain128_tumor_FDG_K1_40min_ri.mat \
  --ytrue_path      /gpfsdata/home/Zhaobo_yuxuan/work/FlowPET/dataset/brain128_tumor_dose_output/dose_7e+07/brain128_tumor_FDG_K1_40min_ytrue.mat \
  --scale_path      /gpfsdata/home/Zhaobo_yuxuan/work/FlowPET/dataset/brain128_tumor_dose_output/dose_7e+07/brain128_tumor_FDG_K1_40min_scale.mat \
  --save_dir      /gpfsdata/home/Zhaobo_yuxuan/work/FlowPET/checkpoint/train1281617 \
  --model_form     glow \
  --image_size     128  \
  --n_epoch        50000 \
  --n_flow         16 \
  --n_batch        32 \
  --tv 1 || { echo "[ERROR] Python script execution failed"; exit 1; }
  
echo "[INFO] Job finished successfully."

