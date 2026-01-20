# DeepMed-Imaging-Reconstruction
Normalizing flow research sandbox for medical imaging (PET, MRI)



## References
- Deep Probabilistic Imaging (AAAI 2021): https://github.com/HeSunPU/DPI  
- Normalizing Flows are Capable Generative Models (TARFlow): https://github.com/apple/ml-tarflow

```bash
python prepare_fid_stats.py --dataset=pet --img_size=128

torchrun --standalone --nproc_per_node=2 train.py --dataset=pet --img_size=128 --channel_size=1\
  --patch_size=4 --channels=1024 --blocks=8 --layers_per_block=8\
  --noise_std=0.15 --batch_size=32 --epochs=150 --lr=1e-5 --nvp --cfg=0 --drop_label=0.1\
  --sample_freq=15 --logdir=runs/pet128 --num_samples=256 
```

noise_std=0.01
```bash
torchrun --standalone --nproc_per_node=2 train.py --dataset=pet --img_size=128 --channel_size=1\
  --patch_size=4 --channels=1024 --blocks=8 --layers_per_block=8\
  --noise_std=0.01 --batch_size=32 --epochs=300 --lr=1e-5 --nvp --cfg=0 --drop_label=0.1\
  --sample_freq=20 --logdir=runs/pet128 --num_samples=256
```


```bash
python resize_dataset.py --size 12000
python prepare_fid_stats.py --dataset=pet_12000 --img_size=128
```

```bash
torchrun --standalone --nproc_per_node=2 train.py --dataset=pet_12000 --img_size=128 --channel_size=1\
  --patch_size=4 --channels=1024 --blocks=8 --layers_per_block=8\
  --noise_std=0.01 --batch_size=32 --epochs=300 --lr=1e-5 --nvp --cfg=0 --drop_label=0.1\
  --sample_freq=20 --logdir=runs/pet128_12000 --num_samples=256
```


