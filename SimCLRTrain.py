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
from utils import set_loader, set_model
from losses import SupConLoss
from config.cli import parse_option

def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss      
        _, features = model(images)
        # Split two views (per-GPU batch size)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)  # [B, 2, C]
        # All-gather across GPUs so that negatives come from all devices
        features = concat_all_gather(features, requires_grad=True)  # shape [B*world, 2, C]
        labels_all = concat_all_gather(labels)  # [B*world]

        # Compute loss using global batch
        if opt.pretrain_method == 'SupCon':
            loss = criterion(features, labels_all)
        elif opt.pretrain_method == 'SimCLR':
            loss = criterion(features)
        else:
            raise ValueError(f'contrastive method not supported: {opt.pretrain_method}')

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq_ss == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg


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

    if requires_grad and hasattr(torch.distributed.nn.functional, "all_gather"):
        # Preserve grad if PyTorch provides distributed all_gather with autograd support
        from torch.distributed.nn.functional import all_gather as all_gather_fn
        gathered = all_gather_fn(tensor)
        return torch.cat(list(gathered), dim=0)

    # Fallback: no grad all_gather (remote chunks act as constants)
    tensors_gather = [torch.zeros_like(tensor) for _ in range(world_size)]
    with torch.no_grad():
        dist.all_gather(tensors_gather, tensor)
    return torch.cat(tensors_gather, dim=0)

def main_worker(local_rank, opt):
    # Setup distributed if needed
    opt.local_rank = local_rank
    opt.rank = local_rank  # single-node spawn: rank == local_rank

    if getattr(opt, 'distributed', False):
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(backend="nccl",
                                init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
                                world_size=opt.world_size,
                                rank=opt.rank)
        torch.cuda.set_device(local_rank)

    # Build data loader (per-rank sampler is set in utils.set_loader)
    _, train_loader = set_loader(opt, loader_suffix='Encoder Pretrain',
                                 augment_type=opt.augment_type_ss,
                                 twoviews=True, p_label=False, train=True, score_dataset=False)

    # Build model & criterion & optimizer
    model = set_model(opt)
    optimizer = set_optimizer(opt, model)
    criterion = SupConLoss(temperature=opt.temp)

    # Print model/encoder/head parameter stats only on rank 0
    is_master = (not getattr(opt, "distributed", False)) or (getattr(opt, "rank", 0) == 0)

    if is_master:
        print_model_param_stats(model, encoder_attr_name="encoder")

    # Training
    train_loss_record = []
    for epoch in range(1, opt.epochs + 1):
        if getattr(opt, 'distributed', False) and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        loss = train(train_loader, model, criterion, optimizer, epoch, opt)

        if (not getattr(opt, 'distributed', False)) or (opt.rank == 0):
            print(f'Epoch {epoch}, Loss {loss:.4f}')
            train_loss_record.append(loss)

    if (not getattr(opt, 'distributed', False)) or (opt.rank == 0):
        print(f'Epoch {epoch}, Loss {loss:.4f}')
        train_loss_record.append(loss)

        # Save checkpoint periodically
        if getattr(opt, 'save_freq', None) is not None and opt.save_freq > 0 and (epoch % opt.save_freq == 0):
            ckpt_path = os.path.join(opt.save_folder, f'ckpt_epoch_{epoch}.pth')
            save_model(model, optimizer, opt, epoch, ckpt_path)

    if getattr(opt, 'distributed', False):
        dist.destroy_process_group()

def main(opt):
    # Auto-spawn per GPU when distributed
    if getattr(opt, 'distributed', False):
        mp.spawn(main_worker, nprocs=opt.world_size, args=(opt,))
    else:
        main_worker(0, opt)

# === in __main__ keep your parse_option() and folder creation ===
if __name__ == '__main__':
    opt = parse_option()
    if not os.path.exists(opt.save_folder):
        os.makedirs(opt.save_folder)
    main(opt)