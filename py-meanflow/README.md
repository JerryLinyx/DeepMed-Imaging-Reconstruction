# Mean Flows: PyTorch + GPU Implementation

<div align="center">
<img width="800" alt="Image" src="https://github.com/user-attachments/assets/2adc06a5-c3bf-41c8-acfa-54c822e7c07b" />
</div>


This is a PyTorch+GPU re-implementation for the CIFAR-10 experiments in [Mean Flows for One-step Generative Modeling](https://arxiv.org/abs/2505.13447). The original experiments were done in JAX+TPU.

## Installation

This repo was tested in PyTorch 2.7.1 and uses `torch.compile`. Compilation may depend on PyTorch versions.

```
conda env create -f environment.yml
conda activate meanflow
```

## Demo

Run `demo.ipynb` for a demo of 1-step generation and FID evaluation. This demo should produce <2.9 FID.

<div align="center">
<img width="480" alt="Image" src="https://github.com/user-attachments/assets/11966c45-25c6-44e5-ae24-75b29e697b9b" />
</div>


## Training

Run the script `cifar10_v1.sh` to train from scratch with 8 GPUs.
It is an improved configuration that can approach ~2.9 FID at 16000 epochs (800k iterations with batch 128x8). It takes 0.21s/iter in 8x H200 GPUs. The checkpoint in `demo.ipynb` (~2.80 FID) is from this script.

The original configuration used in the paper is in `cifar10_v0.sh`.


## Note on JVP

Users may be unfamiliar with the JVP (Jacobian-vector product) operation, which MeanFlow is based on. While JVP is straightforward to implement in JAX, its correct implementation in PyTorch is worth a closer look.

#### DDP

The op `torch.func.jvp` does not support a DDP (`DistributedDataParallel`) object. In your code, you may need to replace `model` with `model.module` to allow `torch.func.jvp` to run. However, doing so may bypass the gradient synchronization normally handled by DDP, **with no error reported**.

In our code, we handle this by `synchronize_gradients(model)`, with a sanity check `gradient_sanity_check`.

#### Compilation

The memory and speed of JVP can greatly benefit from compilation, in both JAX and PyTorch. In our code, this is done by:
```
compiled_train_step = torch.compile(
    train_step,
    disable=not args.compile,
)
```
where `train_step` is:
```
def train_step(model_without_ddp, *args, **kwargs):
    loss = model_without_ddp.forward_with_loss(*args, **kwargs)
    loss.backward(create_graph=False)
    return loss
```
Optionally, we also put `update_ema()` into `train_step` for compilation.

#### Alternative to Compilation

If you don't want to compile (for example, some of your ops are not supported), we recommend to compute `dudt` by `torch.func.jvp` under `torch.no_grad()`:
```
u_pred = u_func(z, t, r)
with torch.no_grad():
    _, dudt = torch.func.jvp(u_func, (z, t, r), (v, dtdt, drdt))
```
The function prediction `u_pred` is computed separately. In this way, computing `dudt` does not introduce substantial additional memory usage, and its time cost is roughly equivalent to a forward and backward pass. If you want `u_func` to share the dropout masks, consider backing up rng states by `cpu_rng_state = torch.get_rng_state(); cuda_rng_state = torch.cuda.get_rng_state()` and restoring by `torch.set_rng_state(cpu_rng_state); torch.cuda.set_rng_state(cuda_rng_state)` before and after the call of `u_func`.

## References

This repo is based on the following repos:

* [Flow Matching repo](https://github.com/facebookresearch/flow_matching)
* [EDM repo](https://github.com/NVlabs/edm)

See also:

* [Our MeanFlow JAX repo](https://github.com/Gsunshine/meanflow) with ImageNet experiments.
* [A third-party MeanFlow PyTorch repo](https://github.com/zhuyu-cs/MeanFlow) with reproduced ImageNet results.
