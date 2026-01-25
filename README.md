# DeepMed-Imaging-Reconstruction
Normalizing flow research sandbox for medical imaging (PET, MRI)

## References
- Deep Probabilistic Imaging (AAAI 2021): https://github.com/HeSunPU/DPI  
- Normalizing Flows are Capable Generative Models (TARFlow): https://github.com/apple/ml-tarflow


## PET Unconditional Training using TARFlow

Prepare FID stats
```bash
torchrun --standalone --nproc_per_node=2 prepare_fid_stats.py --dataset=pet --img_size=128
```

2xH100, 6mins/epoch 
```bash
torchrun --standalone --nproc_per_node=2 train.py --dataset=pet --img_size=128 --channel_size=1\
  --patch_size=4 --channels=1024 --blocks=8 --layers_per_block=8\
  --noise_std=0.15 --batch_size=32 --epochs=150 --lr=1e-5 --nvp --cfg=0 --drop_label=0.1\
  --sample_freq=15 --logdir=runs/pet128 --num_samples=256 
```

Set noise_std=0.01
```bash
torchrun --standalone --nproc_per_node=2 train.py --dataset=pet --img_size=128 --channel_size=1\
  --patch_size=4 --channels=1024 --blocks=8 --layers_per_block=8\
  --noise_std=0.01 --batch_size=32 --epochs=300 --lr=1e-5 --nvp --cfg=0 --drop_label=0.1\
  --sample_freq=20 --logdir=runs/pet128 --num_samples=256
```
![](assets/pet_tarflow_sample_001.png)

Resize dataset to 12000
```bash
python resize_dataset.py --size 12000
torchrun --standalone --nproc_per_node=2 prepare_fid_stats.py --dataset=pet_12000 --img_size=128
```
```bash
torchrun --standalone --nproc_per_node=2 train.py --dataset=pet_12000 --img_size=128 --channel_size=1\
  --patch_size=4 --channels=1024 --blocks=8 --layers_per_block=8\
  --noise_std=0.01 --batch_size=32 --epochs=300 --lr=1e-5 --nvp --cfg=0 --drop_label=0.1\
  --sample_freq=20 --logdir=runs/pet128_12000 --num_samples=256
```

### AFHQ Training using TARFlow
AFHQ dataset: https://www.kaggle.com/datasets/dimensi0n/afhq-512

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("dimensi0n/afhq-512")

print("Path to dataset files:", path)
```
#### Prepare FID stats
```bash
torchrun --standalone --nproc_per_node=2 prepare_fid_stats.py --dataset=afhq --img_size=256
```
```bash
torchrun --standalone --nproc_per_node=2 prepare_fid_stats.py --dataset=afhq --img_size=128
```
```bash
torchrun --standalone --nproc_per_node=2 prepare_fid_stats.py --dataset=afhq --img_size=64
```
#### Training
Original config: need to run on 4 nodes, 32 GPUs total
```bash  
  torchrun --standalone --nproc_per_node=8 train.py --dataset=afhq --img_size=128 --channel_size=3\
  --patch_size=4 --channels=1024 --blocks=8 --layers_per_block=8\
  --noise_std=0.15 --batch_size=768 --epochs=320 --lr=1e-4 --nvp --cfg=0 --drop_label=0.1\
  --sample_freq=20 --logdir=runs/afhq128
```
Other configs remain the same, batch_size reduced to 1/12, 5 minutes/epoch, correspondingly need 2 hours, occupying 93GB x 2
```bash  
  torchrun --standalone --nproc_per_node=2 train.py --dataset=afhq --img_size=128 --channel_size=3\
  --patch_size=4 --channels=1024 --blocks=8 --layers_per_block=8\
  --noise_std=0.15 --batch_size=64 --epochs=27 --lr=1e-5 --nvp --cfg=0 --drop_label=0.1\
  --sample_freq=3 --logdir=runs/afhq128
```
Other configs remain the same, batch_size reduced to 1/24, 9 minutes/epoch, correspondingly need 2 hours, occupying 93GB
```bash  
CUDA_VISIBLE_DEVICES=1 \
torchrun --standalone --nproc_per_node=1 train.py --dataset=afhq --img_size=128 --channel_size=3\
  --patch_size=4 --channels=1024 --blocks=8 --layers_per_block=8\
  --noise_std=0.15 --batch_size=32 --epochs=150 --lr=1e-5 --nvp --cfg=0 --drop_label=0.1\
  --sample_freq=10 --logdir=runs/afhq128
```

Channels reduced to half, and batch_size reduced to 1/12, need 21 days, occupying 44GB x 2
```bash  
  torchrun --standalone --nproc_per_node=2 train.py --dataset=afhq --img_size=128 --channel_size=3\
  --patch_size=4 --channels=512 --blocks=8 --layers_per_block=8\
  --noise_std=0.15 --batch_size=64 --epochs=15360 --lr=1e-4 --nvp --cfg=0 --drop_label=0.1\
  --sample_freq=20 --logdir=runs/afhq128
```

Channels reduced to half, and batch_size reduced to 1/6, need 10.6 days, occupying 82GB x 2
```bash  
  torchrun --standalone --nproc_per_node=2 train.py --dataset=afhq --img_size=128 --channel_size=3\
  --patch_size=4 --channels=512 --blocks=8 --layers_per_block=8\
  --noise_std=0.15 --batch_size=128 --epochs=7680 --lr=1e-4 --nvp --cfg=0 --drop_label=0.1\
  --sample_freq=20 --logdir=runs/afhq128
```

## Unified Model Interface

We provide a unified interface to interact with RealNVP, Glow, and Tarflow models using `unified_model.py`.

### Quick Start

```python
import torch
from unified_model import ModelFactory

# 1. RealNVP
realnvp_config = {'ndim': 32, 'n_flow': 4, 'affine': True, 'seqfrac': 4}
model = ModelFactory.create_model('realnvp', realnvp_config)
x = torch.randn(10, 32)
z, logdet, _ = model.forward(x)
x_recon = model.inverse(z)

# 2. Glow
glow_config = {'in_channel': 3, 'n_flow': 4, 'n_block': 3, 'affine': True, 'conv_lu': True}
model = ModelFactory.create_model('glow', glow_config)
x = torch.randn(2, 3, 32, 32)
z_outs, logdet, _ = model.forward(x)
# z_outs is a list of multi-scale latents
x_recon = model.inverse(z_outs)

# 3. Tarflow
tarflow_config = {
    'in_channels': 3, 'img_size': 32, 'patch_size': 4, 'channels': 64,
    'num_blocks': 4, 'layers_per_block': 4, 'nvp': True
}
model = ModelFactory.create_model('tarflow', tarflow_config)
x = torch.randn(2, 3, 32, 32)
z, logdets, _ = model.forward(x)
x_recon = model.inverse(z)
```

### 4. PET Model Sampling with Refinement (Langevin Dynamics)

This example demonstrates how to generate a high-quality PET image by refining an initial sample using Langevin dynamics (gradient descent on the energy/likelihood).

```python
import torch
from unified_model import ModelFactory

# Configuration specific to PET 12000 checkpoint
pet_noise_std = 0.01  # Noise level used during training
pet_config = {
    'in_channels': 1,       # PET is 1 channel
    'img_size': 128,
    'patch_size': 4,
    'channels': 1024,
    'num_blocks': 8,
    'layers_per_block': 8,
    'nvp': True
}

# 1. Initialize Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_adapter = ModelFactory.create_model('tarflow', pet_config)
model_adapter.to(device)
# model_adapter.model.load_state_dict(torch.load('path/to/checkpoint.pth'))

# 2. Sample from Gaussian Prior
batch_size = 1
num_patches = (pet_config['img_size'] // pet_config['patch_size']) ** 2
latent_dim = pet_config['in_channels'] * (pet_config['patch_size'] ** 2)
z = torch.randn(batch_size, num_patches, latent_dim, device=device)

# 3. Initial Inverse Mapping (Latent -> Image)
x_sample = model_adapter.inverse(z)

# 4. Refinement Loop (Denoising / Langevin Dynamics)
# We optimize the image x to maximize its likelihood (minimize energy)
for p in model_adapter.parameters():
    p.requires_grad = False

x_sample = x_sample.detach().clone()
x_sample.requires_grad = True

# Heuristic learning rate
lr = batch_size * (pet_config['img_size'] ** 2) * pet_config['in_channels'] * (pet_noise_std ** 2)

# Forward pass to get latent z and log-determinant for the current image
z_new, logdet_new, _ = model_adapter.forward(x_sample)

# Calculate Loss (Negative Log Likelihood)
# get_loss computes: 0.5 * z^2 - logdet
loss = model_adapter.get_loss(z_new, logdet_new)

# Compute gradients w.r.t input image
grad = torch.autograd.grad(loss, [x_sample])[0]

# Update image (Gradient Descent Step)
x_refined = x_sample.data - lr * grad
```
