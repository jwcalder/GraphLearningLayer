import sys
import time
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

import matplotlib.pyplot as plt

from utils import adjust_learning_rate, warmup_learning_rate, AverageMeter
from utils import set_optimizer, save_model, print_model_param_stats
from utils import FileLogger, set_loader, set_model
from losses import SupConLoss
from config.cli import parse_option


def is_master_process() -> bool:
    """Return True if current process is the master (rank 0) or if DDP is not initialized."""
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def train(train_loader, model, criterion_supcon, criterion_simclr, optimizer, epoch, opt, gamma: float):
    """
    Run one epoch of training by jointly optimizing SupCon and SimCLR losses.

    Supported batch structures:
      - (indices, images, labels)  -> 3-tuple (e.g., DatasetWithScore)
      - (images, labels)           -> 2-tuple (e.g., vanilla torchvision dataset)

    In both cases, `images` must be a list of two tensors (TwoCropTransform applied).
    Final loss: gamma * loss_supcon + (1 - gamma) * loss_simclr
    """
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_total_meter = AverageMeter()
    loss_supcon_meter = AverageMeter()
    loss_simclr_meter = AverageMeter()

    master = is_master_process()
    end = time.time()

    for idx, batch in enumerate(train_loader):
        # -------------------------
        # Normalize batch structure
        # -------------------------
        # Expected:
        #   images: [tensor(B, C, H, W), tensor(B, C, H, W)]
        #   labels: tensor(B)
        if isinstance(batch, (list, tuple)):
            if len(batch) == 3:
                indices, images, labels = batch
            elif len(batch) == 2:
                images, labels = batch
                indices = None  # indices are not used here
            else:
                raise ValueError(f"Unexpected batch structure with length {len(batch)}")
        else:
            raise ValueError(f"Unexpected batch type: {type(batch)}")

        # measure data loading time
        data_time.update(time.time() - end)

        # images: list of two augmented views -> concat along batch dim
        if not (isinstance(images, (list, tuple)) and len(images) == 2):
            raise ValueError("Expected 'images' to be a list/tuple of two views from TwoCropTransform.")

        images = torch.cat([images[0], images[1]], dim=0)

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        bsz = labels.shape[0]

        # warm-up learning rate (if configured in opts)
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # forward to get features
        _, features = model(images)

        # split two views and reshape to [B, 2, C]
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)  # [B, 2, C]

        # all-gather across GPUs so negatives come from all devices
        features = concat_all_gather(features, requires_grad=True)  # [B*world, 2, C]
        labels_all = concat_all_gather(labels)  # [B*world]

        # compute both losses with their own temperatures (encoded in the criteria)
        loss_supcon = criterion_supcon(features, labels_all)
        loss_simclr = criterion_simclr(features)
        loss = gamma * loss_supcon + (1.0 - gamma) * loss_simclr

        # update metrics
        loss_total_meter.update(loss.item(), bsz)
        loss_supcon_meter.update(loss_supcon.item(), bsz)
        loss_simclr_meter.update(loss_simclr.item(), bsz)

        # SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # logging (only on master to avoid clutter)
        if master and ((idx + 1) % opt.print_freq_ss == 0):
            print(
                'Train (Joint): [{0}][{1}/{2}]\t'
                'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                'supcon {lsc.val:.3f} ({lsc.avg:.3f})\t'
                'simclr {lsm.val:.3f} ({lsm.avg:.3f})\t'.format(
                    epoch, idx + 1, len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=loss_total_meter,
                    lsc=loss_supcon_meter,
                    lsm=loss_simclr_meter
                )
            )
            sys.stdout.flush()

    # return average total loss
    return loss_total_meter.avg


def concat_all_gather(tensor, requires_grad=True):
    """
    All-gather and concatenate tensors from all processes.
    If requires_grad=False, uses no_grad gather (faster but remote parts are constants).
    """
    if not dist.is_available() or not dist.is_initialized():
        return tensor

    world_size = dist.get_world_size()
    if world_size == 1:
        return tensor

    # Prefer autograd-friendly all_gather if available
    if requires_grad and hasattr(torch.distributed.nn.functional, "all_gather"):
        from torch.distributed.nn.functional import all_gather as all_gather_fn
        gathered = all_gather_fn(tensor)
        return torch.cat(list(gathered), dim=0)

    # Fallback: manual all_gather without grad (remote chunks are constants)
    tensors_gather = [torch.zeros_like(tensor) for _ in range(world_size)]
    with torch.no_grad():
        dist.all_gather(tensors_gather, tensor)
    return torch.cat(tensors_gather, dim=0)


def main_worker(local_rank, opt):
    """Main worker for single- or multi-GPU training with joint SupCon+SimCLR loss."""
    # attach ranks to opts for downstream utils that may expect them
    opt.local_rank = local_rank
    opt.rank = local_rank  # for single-node spawn, rank == local_rank

    # initialize DDP if requested
    if getattr(opt, 'distributed', False):
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")

        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
            world_size=opt.world_size,
            rank=opt.rank
        )
        torch.cuda.set_device(local_rank)

    # build data loaders (per-rank sampler should be set inside set_loader)
    # Expected train_tuple:
    #   (label_train_dataset_score, label_train_loader_score, unlabel_train_loader[, full_train_loader])
    train_loaders, _ = set_loader(opt, augment_type='weak', twoviews=True, return_full=True)
    label_train_loader_score = train_loaders[1]
    full_train_loader = train_loaders[3] if len(train_loaders) >= 4 else None

    # choose a unified loader that provides two views and labels
    # prefer full_train_loader if available; otherwise fall back to labeled-score loader
    current_loader = full_train_loader if full_train_loader is not None else label_train_loader_score

    # build model / optimizer
    model = set_model(opt)
    optimizer = set_optimizer(opt, model)

    # Temperatures are read from `opt` with defaults; no global constants
    tau_supcon: float = getattr(opt, 'tau_supcon', 0.07)
    tau_simclr: float = getattr(opt, 'tau_simclr', 0.15)

    # build two separate criteria with different temperatures
    criterion_supcon = SupConLoss(temperature=tau_supcon)  # used with labels: SupCon
    criterion_simclr = SupConLoss(temperature=tau_simclr)  # used without labels: SimCLR-style

    # gamma for loss blending (formerly alpha)
    gamma: float = getattr(opt, 'gamma', 0.5)

    master = is_master_process()

    # ensure save folder exists (especially important before children try to write)
    if master:
        os.makedirs(opt.save_folder, exist_ok=True)
        print_model_param_stats(model, encoder_attr_name="encoder")
        print(f"[Config] gamma={gamma:.3f}, tau_supcon={tau_supcon:.3f}, tau_simclr={tau_simclr:.3f}")

    # training loop (joint optimization every epoch)
    for epoch in range(1, opt.epochs + 1):
        # for DDP samplers (e.g., DistributedSampler), set epoch for shuffling
        if getattr(opt, 'distributed', False) and hasattr(current_loader.sampler, 'set_epoch'):
            current_loader.sampler.set_epoch(epoch)

        loss = train(current_loader, model, criterion_supcon, criterion_simclr, optimizer,
                     epoch, opt, gamma, tau_supcon, tau_simclr)

        if master:
            print(f'[Joint SupCon+SimCLR] Epoch {epoch}, TotalLoss {loss:.4f}, '
                  f'gamma={gamma:.3f}, tau_supcon={tau_supcon:.3f}, tau_simclr={tau_simclr:.3f}')

            # periodic checkpoint saving (master only)
            save_freq = getattr(opt, 'save_freq', 0) or 0
            if save_freq > 0 and (epoch % save_freq == 0):
                ckpt_path = os.path.join(
                    opt.save_folder, f'pretrain_joint_ckpt_epoch_{epoch}.pth'
                )
                mdl = model.module if hasattr(model, 'module') else model
                save_model(mdl, optimizer, opt, epoch, ckpt_path)

        # Optional: prevent workers from racing too far ahead of rank 0
        if getattr(opt, 'distributed', False):
            dist.barrier()

    # always save a final checkpoint at the end (master only)
    if master and opt.epochs >= 1:
        last_ckpt = os.path.join(opt.save_folder, f'pretrain_joint_ckpt_last.pth')
        mdl = model.module if hasattr(model, 'module') else model
        save_model(mdl, optimizer, opt, epoch, last_ckpt)

    # clean up DDP
    if getattr(opt, 'distributed', False):
        dist.destroy_process_group()


def main(opt):
    """Entry point which either spawns DDP workers or runs a single process."""
    if getattr(opt, 'distributed', False):
        mp.spawn(main_worker, nprocs=opt.world_size, args=(opt,))
    else:
        main_worker(0, opt)


if __name__ == '__main__':
    opt = parse_option()

    # Make sure save folder exists before logging
    os.makedirs(opt.save_folder, exist_ok=True)

    # Prepare timestamped log file path
    txt_path_template = os.path.join(opt.save_folder, 'output_record_{}.txt')
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    txt_path = txt_path_template.format(timestamp)

    # Redirect stdout to both console and file using FileLogger
    with open(txt_path, "w") as f:
        logger = FileLogger(f, sys.stdout)
        sys.stdout = logger
        try:
            # Optionally print all parsed options for reproducibility
            if getattr(opt, 'print_all_parameters', False):
                for key, value in vars(opt).items():
                    print(f"{key}: {value}")
            main(opt)
        finally:
            # restore original stdout
            sys.stdout = sys.__stdout__
