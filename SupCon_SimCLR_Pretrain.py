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
from utils import FileLogger, set_loader, set_model, print_loader_info, test_GL_NP
from losses import SupConLoss
from config.cli import parse_option


def is_master_process() -> bool:
    """Return True if current process is the master (rank 0) or if DDP is not initialized."""
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def concat_all_gather(tensor, requires_grad: bool = True):
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


def _normalize_batch(batch):
    """Normalize a batch into (images, labels). Labels can be None for unlabeled data.
    Expected:
      - images: [v1, v2]
      - labels: tensor of shape [B] for labeled; can be None for unlabeled
    Supported raw batch forms:
      - (indices, images, labels)
      - (images, labels)
    """
    if not isinstance(batch, (list, tuple)):
        raise ValueError(f"Unexpected batch type: {type(batch)}")

    if len(batch) == 3:
        _, images, labels = batch
    elif len(batch) == 2:
        images, labels = batch
    else:
        raise ValueError(f"Unexpected batch structure with length {len(batch)}")

    if not (isinstance(images, (list, tuple)) and len(images) == 2):
        raise ValueError("Expected 'images' to be a list/tuple of two views from TwoCropTransform.")

    return images, labels


def train_two_loaders(
    labeled_loader,
    unlabeled_loader,
    model,
    criterion_supcon,
    criterion_simclr,
    optimizer,
    epoch,
    opt,
    gamma: float,
):
    """
    Train one epoch using two different dataloaders:
      - SupCon loss is computed only on `labeled_loader` batches.
      - SimCLR loss is computed only on `unlabeled_loader` batches.
    Final loss per step: gamma * supcon + (1 - gamma) * simclr (single backward/step).
    """
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_total_meter = AverageMeter()
    loss_supcon_meter = AverageMeter()
    loss_simclr_meter = AverageMeter()

    master = is_master_process()
    end = time.time()

    # Steps per epoch: align by cycling the shorter loader
    steps = max(len(labeled_loader), len(unlabeled_loader))
    it_labeled = iter(labeled_loader)
    it_unlabeled = iter(unlabeled_loader)

    for step in range(steps):
        # -------------------------
        # Fetch a labeled batch (for SupCon)
        # -------------------------
        try:
            batch_l = next(it_labeled)
        except StopIteration:
            it_labeled = iter(labeled_loader)
            batch_l = next(it_labeled)

        # -------------------------
        # Fetch an unlabeled batch (for SimCLR)
        # -------------------------
        try:
            batch_u = next(it_unlabeled)
        except StopIteration:
            it_unlabeled = iter(unlabeled_loader)
            batch_u = next(it_unlabeled)

        # Measure data loading time
        data_time.update(time.time() - end)

        # --- Labeled path: SupCon ---
        images_l, labels_l = _normalize_batch(batch_l)
        imgs_l = torch.cat([images_l[0], images_l[1]], dim=0)

        # --- Unlabeled path: SimCLR ---
        images_u, _ = _normalize_batch(batch_u)
        imgs_u = torch.cat([images_u[0], images_u[1]], dim=0)

        if torch.cuda.is_available():
            imgs_l = imgs_l.cuda(non_blocking=True)
            imgs_u = imgs_u.cuda(non_blocking=True)
            labels_l = labels_l.cuda(non_blocking=True)

        # Warm-up learning rate once per global step
        warmup_learning_rate(opt, epoch, step, steps, optimizer)

        # -------------------------
        # Forward for labeled -> SupCon
        # -------------------------
        _, feats_l = model(imgs_l)
        bsz_l = labels_l.shape[0]
        f1_l, f2_l = torch.split(feats_l, [bsz_l, bsz_l], dim=0)
        feats_l = torch.cat([f1_l.unsqueeze(1), f2_l.unsqueeze(1)], dim=1)  # [B, 2, C]

        feats_l = concat_all_gather(feats_l, requires_grad=True)
        labels_all = concat_all_gather(labels_l)
        loss_supcon = criterion_supcon(feats_l, labels_all)

        # -------------------------
        # Forward for unlabeled -> SimCLR
        # -------------------------
        _, feats_u = model(imgs_u)
        bsz_u = imgs_u.shape[0] // 2
        f1_u, f2_u = torch.split(feats_u, [bsz_u, bsz_u], dim=0)
        feats_u = torch.cat([f1_u.unsqueeze(1), f2_u.unsqueeze(1)], dim=1)  # [B, 2, C]

        feats_u = concat_all_gather(feats_u, requires_grad=True)
        loss_simclr = criterion_simclr(feats_u)

        # -------------------------
        # Combine and optimize (single backward/step)
        # -------------------------
        loss = gamma * loss_supcon + (1.0 - gamma) * loss_simclr

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update meters (you may weight differently if desired)
        loss_total_meter.update(loss.item(), bsz_l)
        loss_supcon_meter.update(loss_supcon.item(), bsz_l)
        loss_simclr_meter.update(loss_simclr.item(), bsz_u)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Logging (master only)
        if master and ((step + 1) % getattr(opt, "print_freq_ss", 10) == 0):
            print(
                'Train (TwoLoaders): [{0}][{1}/{2}]\t'
                'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                'supcon {lsc.val:.3f} ({lsc.avg:.3f})\t'
                'simclr {lsm.val:.3f} ({lsm.avg:.3f})\t'.format(
                    epoch, step + 1, steps,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=loss_total_meter,
                    lsc=loss_supcon_meter,
                    lsm=loss_simclr_meter
                )
            )
            sys.stdout.flush()

    # Return average total loss for the epoch
    return loss_total_meter.avg


def main_worker(local_rank, opt):
    """Main worker for single- or multi-GPU training with separate loaders for SupCon and SimCLR."""
    # Attach ranks to opts for downstream utils that may expect them
    opt.local_rank = local_rank
    opt.rank = local_rank  # for single-node spawn, rank == local_rank

    # Initialize DDP if requested
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

    # Build data loaders (per-rank sampler should be set inside set_loader)
    # Expected train_loaders layout:
    #   (label_train_dataset_score, label_train_loader_score, unlabel_train_loader[, full_train_loader])
    train_loaders, eval_loaders = set_loader(opt, augment_type='weak', twoviews=True, return_full=True)
    label_train_loader_score = train_loaders[1]
    # Prefer the wider unlabeled/full loader for SimCLR
    full_train_loader = train_loaders[3] if len(train_loaders) >= 4 else None

    # Dataset info (master prints)
    print_loader_info("label_train_loader_score", label_train_loader_score)
    print_loader_info("full_train_loader", full_train_loader)

    eval_labeled_train_loader, eval_unlabeled_train_loader, test_loader_eval, _ = eval_loaders
    # Eval loaders
    print_loader_info("eval_labeled_train_loader", eval_labeled_train_loader)
    print_loader_info("eval_unlabeled_train_loader", eval_unlabeled_train_loader)
    print_loader_info("test_loader_eval", test_loader_eval)
    test_acc_record = []
    
    # Ensure we have both loaders
    if full_train_loader is None:
        raise ValueError("full_train_loader is required for SimCLR loss but is None.")

    # Build model / optimizer
    model = set_model(opt)
    optimizer = set_optimizer(opt, model)
    
    # test model before training
    test_acc = test_GL_NP(model, eval_labeled_train_loader, test_loader_eval, opt, unlabel_train_loader=eval_unlabeled_train_loader)
    test_acc_record.append(test_acc)

    # Temperatures from options (with defaults)
    tau_supcon: float = getattr(opt, 'tau_supcon', 0.07)
    tau_simclr: float = getattr(opt, 'tau_simclr', 0.15)

    # Build two criteria with different temperatures
    criterion_supcon = SupConLoss(temperature=tau_supcon)   # used with labels: SupCon
    criterion_simclr = SupConLoss(temperature=tau_simclr)   # used without labels: SimCLR-style

    # Gamma for loss blending
    gamma: float = getattr(opt, 'gamma', 0.5)

    master = is_master_process()

    # Ensure save folder exists and print config
    if master:
        os.makedirs(opt.save_folder, exist_ok=True)
        print_model_param_stats(model, encoder_attr_name="encoder")
        print(f"[Config] gamma={gamma:.3f}, tau_supcon={tau_supcon:.3f}, tau_simclr={tau_simclr:.3f}")

    # Training loop
    for epoch in range(1, opt.epochs + 1):
        # For DDP samplers (e.g., DistributedSampler), set epoch for shuffling
        if getattr(opt, 'distributed', False):
            if hasattr(label_train_loader_score.sampler, 'set_epoch'):
                label_train_loader_score.sampler.set_epoch(epoch)
            if hasattr(full_train_loader.sampler, 'set_epoch'):
                full_train_loader.sampler.set_epoch(epoch)

        # Train one epoch using two loaders
        loss = train_two_loaders(
            labeled_loader=label_train_loader_score,
            unlabeled_loader=full_train_loader,
            model=model,
            criterion_supcon=criterion_supcon,
            criterion_simclr=criterion_simclr,
            optimizer=optimizer,
            epoch=epoch,
            opt=opt,
            gamma=gamma
        )

        if master:
            print(f'[SupCon (labeled) + SimCLR (unlabeled)] Epoch {epoch}, '
                  f'TotalLoss {loss:.4f}, gamma={gamma:.3f}, '
                  f'tau_supcon={tau_supcon:.3f}, tau_simclr={tau_simclr:.3f}')

            # Periodic checkpoint saving (master only)
            save_freq = getattr(opt, 'save_freq', 0) or 0
            if save_freq > 0 and (epoch % save_freq == 0):
                ckpt_path = os.path.join(
                    opt.save_folder, f'pretrain_joint_ckpt_epoch_{epoch}.pth'
                )
                mdl = model.module if hasattr(model, 'module') else model
                save_model(mdl, optimizer, opt, epoch, ckpt_path)
                
                # test model at this checkpoint
                test_acc = test_GL_NP(model, eval_labeled_train_loader, test_loader_eval, opt, unlabel_train_loader=eval_unlabeled_train_loader)
                test_acc_record.append(test_acc)

        # Optional: prevent workers from racing too far ahead of rank 0
        if getattr(opt, 'distributed', False):
            dist.barrier()

    # Always save a final checkpoint at the end (master only)
    if master and opt.epochs >= 1:
        last_ckpt = os.path.join(opt.save_folder, f'pretrain_joint_ckpt_last.pth')
        mdl = model.module if hasattr(model, 'module') else model
        save_model(mdl, optimizer, opt, epoch, last_ckpt)

    # Clean up DDP
    if getattr(opt, 'distributed', False):
        dist.destroy_process_group()
        
    # save test acc record
    if master:
        acc_record_path = os.path.join(opt.save_folder, 'test_acc_record.npy')
        np.save(acc_record_path, np.array(test_acc_record))
        print(f"Test accuracy record saved to {acc_record_path}")


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
            # Restore original stdout
            sys.stdout = sys.__stdout__
