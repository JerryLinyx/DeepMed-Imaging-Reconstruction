# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
# Copyright (c) Meta Platforms, Inc. and affiliates.

import datetime
import logging
import os
import sys
import time
from pathlib import Path

from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
from models.model_configs import instantiate_model
from train_arg_parser import get_args_parser

from training import distributed_mode
from training.data_transform import get_transform_cifar, get_transform_mnist
from training.eval_loop import eval_model
from training.load_and_save import load_model, save_model
from training.train_loop import train_one_epoch, train_step
from torchmetrics.aggregation import MeanMetric
import models.rng as rng

from torch.utils.tensorboard import SummaryWriter

torch.set_float32_matmul_precision('high')

logger = logging.getLogger(__name__)


def print_model(model):
    logger.info("=" * 91)
    num_params = 0
    for name, param in model.named_parameters():
        param_std = param.std().item()
        if param.requires_grad:
            num_params += param.numel()
            logger.info(f"{name:48} | {str(list(param.shape)):24} | std: {param_std:.6f}")
    logger.info("=" * 91)
    logger.info(f"Total params: {num_params}")


def get_data_loader(args, is_for_fid):
    if args.dataset == "cifar10":
        transforms = get_transform_cifar(is_for_fid)
        dataset = datasets.CIFAR10(
            root=args.data_path,
            train=True,
            download=True,
            transform=transforms,
        )
    elif args.dataset == "mnist":  # 3x32x32 MNIST for fast development
        transforms = get_transform_mnist()
        dataset = datasets.MNIST(
            root=args.data_path,
            train=True,
            download=True,
            transform=transforms,
        )
    else:
        raise NotImplementedError(f"Unsupported dataset {args.dataset}")

    logger.info(dataset)

    logger.info("Intializing DataLoader")
    num_tasks = distributed_mode.get_world_size()
    global_rank = distributed_mode.get_rank()
    sampler = torch.utils.data.DistributedSampler(
        dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        worker_init_fn=partial(rng.worker_init_fn, rank=global_rank),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=not is_for_fid,  # for FID evaluation, we want to keep all samples
    )
    logger.info(str(sampler))
    return data_loader


def main(args):
    distributed_mode.init_distributed_mode(args)

    print(f"Rank: {distributed_mode.get_rank()}")
    print(f"World Size: {distributed_mode.get_world_size()}")

    if distributed_mode.get_rank() == 0:
        logging.basicConfig(
            level=logging.INFO,
            stream=sys.stdout,
            format="%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    else:
        logger.addHandler(logging.NullHandler())

    logger.info("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    logger.info("{}".format(args).replace(", ", ",\n"))
    if distributed_mode.is_main_process():
        # create tensorboard
        os.makedirs(args.output_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.output_dir)
        logger.info(f"Tensorboard writer created at {args.output_dir}")
    else:
        log_writer = None
        logger.info('Writer not created.')

    device = torch.device(args.device)

    # set the seeds
    seed = args.seed + distributed_mode.get_rank()  # legacy. TODO: rng.fold_in 
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    logger.info(f"Initializing Dataset: {args.dataset}")
    data_loader_train = get_data_loader(args, is_for_fid=False)
    data_loader_fid = get_data_loader(args, is_for_fid=True)

    # define the model
    logger.info("Initializing Model")
    model = instantiate_model(args)

    model.to(device)

    model_without_ddp = model
    print_model(model)

    eff_batch_size = args.batch_size * distributed_mode.get_world_size()

    logger.info(f"Learning rate: {args.lr:.2e}")

    logger.info(f"Effective batch size: {eff_batch_size}")

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu],
            find_unused_parameters=False,
            broadcast_buffers=False,
            static_graph=True,
            gradient_as_bucket_view=True
        )
        model_without_ddp = model.module

    optimizer = torch.optim.Adam(  # Note: Adam, not AdamW
        model_without_ddp.net.parameters(),  # only the "net" parameters
        lr=args.lr,
        betas=args.optimizer_betas,
        weight_decay=0.0
    )

    warmup_iters = args.warmup_epochs * len(data_loader_train)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-8 / args.lr, end_factor=1.0, total_iters=warmup_iters,)
    main_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, total_iters=args.epochs * len(data_loader_train), factor=1.0)
    lr_schedule = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_iters])

    logger.info(f"Optimizer: {optimizer}")
    logger.info(f"Learning-Rate Schedule: {lr_schedule}")

    load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        lr_schedule=lr_schedule,
    )

    compiled_train_step = torch.compile(
        train_step,
        disable=not args.compile,
    )

    batch_loss = MeanMetric().to(device, non_blocking=True)
    batch_time = MeanMetric().to(device, non_blocking=True)
    batch_loss.reset()
    batch_time.reset()

    meters = {'batch_loss': batch_loss, 'batch_time': batch_time,}

    logger.info(f"Start from {args.start_epoch} to {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if not args.eval_only:
            train_one_epoch(
                model=model,
                compiled_train_step=compiled_train_step,
                data_loader=data_loader_train,
                optimizer=optimizer,
                lr_schedule=lr_schedule,
                device=device,
                epoch=epoch,
                log_writer=log_writer,
                args=args,
                meters=meters
            )

        if args.output_dir and (
            (args.eval_frequency > 0 and (epoch + 1) % args.eval_frequency == 0)
            or args.eval_only
            or args.test_run
        ):
            if not args.eval_only:
                save_model(
                    args=args,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    lr_schedule=lr_schedule,
                    epoch=epoch,
                )
                logging.info(f"Saved checkpoint to {args.output_dir}")

            # Eval ema model:
            net_eval = model_without_ddp.net_ema
            ema_decay = net_eval.ema_decay
            eval_stats = eval_model(model, net_eval, data_loader_fid, device, epoch=epoch, args=args, suffix=f'_ema{ema_decay}')
            if log_writer is not None and "fid" in eval_stats:
                logging.info(f"Eval {epoch + 1} epochs finished: FID_ema{ema_decay}: {eval_stats['fid']}")
                log_writer.add_scalar(f"FID_ema{ema_decay}", eval_stats["fid"], epoch + 1)

            # Eval extra ema model:
            for i in range(len(model_without_ddp.ema_decays)):
                net_eval = model_without_ddp._modules[f"net_ema{i + 1}"]
                ema_decay = net_eval.ema_decay
                eval_stats = eval_model(model, net_eval, data_loader_fid, device, epoch=epoch, args=args, suffix=f'_ema{ema_decay}')
                if log_writer is not None and "fid" in eval_stats:
                    logging.info(f"Eval {epoch + 1} epochs finished: FID_ema{ema_decay}: {eval_stats['fid']}")
                    log_writer.add_scalar(f"FID_ema{ema_decay}", eval_stats["fid"], epoch + 1)

            # Eval no-ema model:
            net_eval = model_without_ddp.net
            eval_stats = eval_model(model, net_eval, data_loader_fid, device, epoch=epoch, args=args, suffix='_noema')
            if log_writer is not None and "fid" in eval_stats:
                logging.info(f"Eval {epoch + 1} epochs finished: FID w/o ema: {eval_stats['fid']}")
                log_writer.add_scalar("FID", eval_stats["fid"], epoch + 1)

        if args.test_run or args.eval_only:
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Training time {total_time_str}")


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
