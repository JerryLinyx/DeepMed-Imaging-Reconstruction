#!/bin/bash
#BSUB -J train1281609
#BSUB -n 4
#BSUB -R "span[ptile=2] select[ngpus>0] rusage[ngpus_shared=4]"
#BSUB -q gpuq
#BSUB -o output1281609
#BSUB -e error1281609

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

# --- torchrun Distributed Training Setup ---
# LSB_HOSTS is a space-separated list of hostnames allocated to the job.
# We'll use the first host as the master for rendezvous.
# Make sure LSB_HOSTS is available and correctly formatted in your LSF environment.
if [ -z "$LSB_HOSTS" ]; then
  echo "[ERROR] LSB_HOSTS environment variable is not set. Cannot determine master address."
  exit 1
fi
MASTER_HOST=$(echo $LSB_HOSTS | awk '{print $1}')
MASTER_PORT=29500 # You can change this port if needed

echo "[INFO] Master Host for torchrun: $MASTER_HOST"
echo "[INFO] Master Port for torchrun: $MASTER_PORT"
echo "[INFO] LSB_JOBID for rdzv_id: $LSB_JOBID"
# --- End torchrun Distributed Training Setup ---

torchrun --nnodes= \
         --nproc_per_node=2 \
         --rdzv_id=$LSB_JOBID \
         --rdzv_backend=c10d \
         --rdzv_endpoint="$MASTER_HOST:$MASTER_PORT" \
  /gpfsdata/home/Zhaobo_yuxuan/work/FlowPET/DPItorch/DPI_PET_T.py \
  --activity_path   /gpfsdata/home/Zhaobo_yuxuan/work/FlowPET/dataset/brain128_tumor_activity_map/brain128_tumor_FDG_K1_40min.mat \
  --sinogram_path   /gpfsdata/home/Zhaobo_yuxuan/work/FlowPET/dataset/brain128_tumor_sinogram/brain128_tumor_FDG_K1_40min.mat \
  --gmat_path       /gpfsdata/home/Zhaobo_yuxuan/work/FlowPET/dataset/G_system_matrix_128.mat \
  --ri_path         /gpfsdata/home/Zhaobo_yuxuan/work/FlowPET/dataset/brain128_tumor_sinogram/brain128_tumor_FDG_K1_40min_ri.mat \
  --ytrue_path      /gpfsdata/home/Zhaobo_yuxuan/work/FlowPET/dataset/brain128_tumor_sinogram/brain128_tumor_FDG_K1_40min_ytrue.mat \
  --save_dir      /gpfsdata/home/Zhaobo_yuxuan/work/FlowPET/checkpoint/train1281609 \
  --model_form     glow \
  --image_size     128  \
  --n_epoch        50000 \
  --n_flow         16 \
  --n_batch        16 \
  --tv             20 \
  --cache_matrix_on_gpu \
  || { echo "[ERROR] torchrun Python script execution failed"; exit 1; }
  
echo "[INFO] Job finished successfully."