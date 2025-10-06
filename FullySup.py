import sys
import time
import os

import numpy as np
import matplotlib.pyplot as plt

# Added explicit torch imports (used below)
import torch
import torch.nn as nn
import torch.nn.functional as F

# from itertools import cycle, islice  # no longer needed

####
from utils import adjust_learning_rate, warmup_learning_rate, AverageMeter
from utils import set_optimizer, save_model
from utils import FileLogger, test_GL_NP, test_network
from utils import set_loader, set_model, print_loader_info, print_dataset_info
from losses import *

from GLL import LaplaceLearningSparseHard
from visualize import visualize
from config.cli import parse_option


## --cp_load_path ./simclr_ckpt_epoch_1000.pth
def train(train_loader, base_loader, unlabel_train_loader,
          train_dataset, model, optimizer, epoch, opt):
    """One epoch training (memory-safe).

    Key fixes for memory stability:
    1) Avoid itertools.cycle cache by manually restarting iterators on exhaustion.
    2) Do NOT create a new iterator for base_loader every step. Preload the base batch once per epoch.
    3) Keep DataLoader workers stable by not spawning/tearing down repeatedly.
    """

    # Enable grads
    for param in model.parameters():
        param.requires_grad = True

    model.train()

    lap = LaplaceLearningSparseHard.apply
    # criterion = nn.CrossEntropyLoss()
    criterion = custom_ce_loss

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    correct_num = 0
    data_count = 0

    # ---- Preload the base batch ONCE per epoch (base_loader has a single big batch) ----
    # This prevents building a fresh iterator (and new workers) on every training step.
    base_iter = iter(base_loader)
    base_images, base_labels = next(base_iter)

    # Move base tensors to device once per epoch
    if torch.cuda.is_available() and (opt.dev != 'cpu'):
        base_images = base_images.cuda(non_blocking=True)
        base_labels = base_labels.cuda(non_blocking=True)

    # Precompute label matrix once per epoch (used in 'gl' mode)
    if opt.sup_train_type == 'gl':
        # NOTE: use opt.num_classes if available; falls back to 10 otherwise
        num_classes = getattr(opt, "num_classes", 10)
        label_matrix_epoch = F.one_hot(base_labels, num_classes=num_classes).float()

    # ---- Build manual iterators for loaders to avoid cycle() caching all batches in memory ----
    if unlabel_train_loader is None:
        target_steps = len(train_loader)
        l_iter = iter(train_loader)
        u_iter = None
    else:
        # Run for the longer length while restarting the shorter iterator when it exhausts.
        target_steps = max(len(train_loader), len(unlabel_train_loader))
        l_iter = iter(train_loader)
        u_iter = iter(unlabel_train_loader)

    for idx in range(target_steps):
        # Measure data time start
        data_start = time.time()

        # Fetch labeled batch (restart iterator if exhausted)
        if unlabel_train_loader is None:
            batch_l = next(l_iter, None)
            if batch_l is None:
                l_iter = iter(train_loader)
                batch_l = next(l_iter)
            indices, images, labels = batch_l
        else:
            batch_l = next(l_iter, None)
            if batch_l is None:
                l_iter = iter(train_loader)
                batch_l = next(l_iter)
            indices, images_l, labels = batch_l

            # Fetch unlabeled batch (restart iterator if exhausted)
            batch_u = next(u_iter, None)
            if batch_u is None:
                u_iter = iter(unlabel_train_loader)
                batch_u = next(u_iter)
            images_u, _ = batch_u

            # Concatenate labeled + unlabeled for forward; loss computed only on labeled
            images = torch.cat([images_l, images_u], dim=0)

        # Move current batch to device
        if torch.cuda.is_available() and (opt.dev != 'cpu'):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        bsz = labels.shape[0]

        # Update data_time meter
        data_time.update(time.time() - data_start)

        # Warm-up LR (per step) if enabled
        if opt.warm:
            # For steps_per_epoch use len(train_loader) to keep schedule stable
            warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # Step-wise LR adjust (if your schedule intends epoch-level, consider moving outside loop)
        if opt.adjust_lr:
            adjust_learning_rate(opt, optimizer, epoch)

        # ---- Forward & loss ----
        if opt.sup_train_type == 'gl':
            # Concatenate preloaded base_images with current images
            images_cat = torch.cat((base_images, images), dim=0)
            _, features = model(images_cat)
            pred = lap(features, label_matrix_epoch, opt.temp, opt.epsilon)
            pred = pred[:len(labels)]  # keep only labeled part for loss
            loss = criterion(pred, labels)
        else:
            pred, _ = model(images)
            pred = pred[:len(labels)]
            loss = criterion(pred, labels)

        # Compute training accuracy stats on the labeled portion
        pred_labels = torch.argmax(pred, dim=1)
        correct_num += torch.sum(torch.eq(pred_labels, labels)).item()
        data_count += len(pred)

        # Optionally update sample scores (for 'score' mode)
        if (opt.sup_train_type == 'gl'
            and epoch % opt.gl_update_base_epochs == 0
            and opt.gl_update_base_mode == 'score'):
            if opt.gl_score_type == 'entropy':
                batch_size, num_classes_pred = pred.shape
                one_hot_targets = F.one_hot(labels, num_classes=num_classes_pred).to(pred.dtype)
                scores = -torch.sum(one_hot_targets * torch.log(pred + 1e-8), 1)
            elif opt.gl_score_type == 'l2':
                scores = 1 - torch.sum(pred ** 2, 1)
            else:
                raise ValueError(opt.gl_score_type)
            for data_ind, score in zip(indices, scores):
                train_dataset.update_score(data_ind, score)

        # ---- Backprop & step ----
        losses.update(loss.item(), bsz)
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), opt.temp)
        optimizer.step()

        # Timing
        batch_time.update(time.time() - end)
        end = time.time()

        # NaN guard
        is_nan = torch.stack([torch.isnan(p).any() for p in model.parameters()]).any()
        if is_nan:
            print('nan value')

        # Logging
        if (idx + 1) % opt.print_freq_ss == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, target_steps,
                   batch_time=batch_time, data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg, correct_num / max(data_count, 1)


def main(opt):
    # Build data loaders
    train_loaders, eval_loaders = set_loader(opt, augment_type=opt.augment_type)
    train_dataset_ss, train_loader_ss, unlabel_train_loader = train_loaders
    eval_labeled_train_loader, eval_unlabeled_train_loader, test_loader_eval = eval_loaders

    print("✓ Data loaders generated successfully.")

    # Dataset info
    print_dataset_info("train_dataset_ss", train_dataset_ss)

    # Train loaders
    print_loader_info("train_loader_ss", train_loader_ss)
    print_loader_info("unlabel_train_loader", unlabel_train_loader)

    # Eval loaders
    print_loader_info("eval_labeled_train_loader", eval_labeled_train_loader)
    print_loader_info("eval_unlabeled_train_loader", eval_unlabeled_train_loader)
    print_loader_info("test_loader_eval", test_loader_eval)

    # Build model and optimizer
    model = set_model(opt)
    optimizer = set_optimizer(opt, model)

    # Records
    train_loss_record = []
    test_acc_record = []
    plot_epochs = []

    # Initial eval
    epoch = 0
    plot_epochs.append(epoch)
    if opt.sup_train_type == 'gl':
        test_acc = test_GL_NP(model, eval_labeled_train_loader, test_loader_eval, opt, unlabel_train_loader=eval_unlabeled_train_loader)
    elif opt.sup_train_type == 'mlp':
        _ = test_GL_NP(model, eval_labeled_train_loader, test_loader_eval, opt, unlabel_train_loader=eval_unlabeled_train_loader)
        test_acc = test_network(model, eval_labeled_train_loader, test_loader_eval, opt, predictor='MLP')
    else:
        raise ValueError(opt.sup_train_type)
    test_acc_record.append(test_acc)

    # Select base subset once before training
    base_dataset_ss = train_dataset_ss.select_base_data(
        opt.num_base_data,
        class_uniform_sample=opt.class_uni_sample,
        seed=opt.seed,
        mode='random'
    )

    base_loader_ss = torch.utils.data.DataLoader(
        base_dataset_ss,
        batch_size=len(base_dataset_ss),
        shuffle=True,
        num_workers=0,          
        pin_memory=False,       
        sampler=None
    )

    # Train epochs
    for epoch in range(1 + opt.start_epochs, opt.epochs + 1):
        time1 = time.time()

        loss, train_acc = train(
            train_loader_ss, base_loader_ss, unlabel_train_loader,
            train_dataset_ss, model, optimizer, epoch, opt
        )

        time2 = time.time()
        print('epoch {}, total time {:.2f}, loss {:.2f}, train acc {:.2f}'.format(
            epoch, time2 - time1, loss, train_acc * 100))

        # Optionally update base set every N epochs
        if opt.sup_train_type == 'gl' and epoch % opt.gl_update_base_epochs == 0:
            base_dataset_ss = train_dataset_ss.select_base_data(
                opt.num_base_data,
                class_uniform_sample=opt.class_uni_sample,
                seed=None,
                mode=opt.gl_update_base_mode
            )
            base_loader_ss = torch.utils.data.DataLoader(
                base_dataset_ss,
                batch_size=len(base_dataset_ss),
                shuffle=True,
                num_workers=0,     # keep single-process
                pin_memory=False,  # keep unpinned
                sampler=None
            )
            print(f'Base dataset has been updated with {len(base_dataset_ss)} samples.')

        # Record train loss
        train_loss_record.append(loss)

        # Periodic test & checkpoint
        if epoch % opt.plot_freq_ss == 0:
            plot_epochs.append(epoch)
            if opt.sup_train_type == 'gl':
                test_acc = test_GL_NP(model, eval_labeled_train_loader, test_loader_eval, opt, unlabel_train_loader=eval_unlabeled_train_loader)
            elif opt.sup_train_type == 'mlp':
                _ = test_GL_NP(model, eval_labeled_train_loader, test_loader_eval, opt, unlabel_train_loader=eval_unlabeled_train_loader)
                test_acc = test_network(model, eval_labeled_train_loader, test_loader_eval, opt, predictor='MLP')
            else:
                raise ValueError(opt.sup_train_type)
            test_acc_record.append(test_acc)

            save_file = os.path.join(opt.save_folder, f'ckpt_epoch_{epoch}.pth')
            save_model(model, optimizer, opt, epoch, save_file)

            record_path = os.path.join(opt.save_folder, 'loss_acc_records.npy')
            record_dic = {'epoch': epoch,
                          'train_loss_record': train_loss_record,
                          'test_acc_record': test_acc_record}
            np.save(record_path, record_dic)

            # Training loss plot
            plt.figure(figsize=(10, 5))
            plt.plot(train_loss_record, label='Train Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Training Loss Over Epochs')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(opt.save_folder, 'train_loss_plot.png'))
            plt.close()

            # Test accuracy plot
            plt.figure(figsize=(10, 5))
            plt.plot(plot_epochs, test_acc_record, label='Test Accuracy', color='green')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.title('Test Accuracy Over Epochs')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(opt.save_folder, 'test_acc_plot.png'))
            plt.close()

    # Save the last model
    save_file = os.path.join(opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)

    path = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}'.format(epoch=opt.epochs))
    visualize(save_file, opt.model, base=base_loader_eval, TSNE=opt.TSNE,
              head=True, save_dir=path, head_type=opt.head_type)

    record_path = os.path.join(opt.save_folder, 'loss_acc_records.npy')
    record_dic = {'epoch': epoch,
                  'train_loss_record': train_loss_record,
                  'test_acc_record': test_acc_record}
    np.save(record_path, record_dic)


if __name__ == '__main__':
    opt = parse_option()
    if not os.path.exists(opt.save_folder):
        os.makedirs(opt.save_folder)

    txt_path_template = os.path.join(opt.save_folder, 'output_record_{}.txt')
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    txt_path = txt_path_template.format(timestamp)

    with open(txt_path, "w") as f:
        logger = FileLogger(f, sys.stdout)
        sys.stdout = logger
        try:
            if getattr(opt, "print_all_parameters", False):
                for key, value in vars(opt).items():
                    print(f"{key}: {value}")
            main(opt)
        finally:
            sys.stdout = sys.__stdout__
