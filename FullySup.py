import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import os

from itertools import cycle, islice

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
    """one epoch training"""
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

    if unlabel_train_loader is None:
        data_iter = train_loader
    else:
        target_steps = max(len(train_loader), len(unlabel_train_loader))
        data_iter = islice(zip(cycle(train_loader), cycle(unlabel_train_loader)), target_steps)
        
    for idx, data in enumerate(data_iter):
        base_images, base_labels = next(iter(base_loader))
        data_time.update(time.time() - end)
        
        if unlabel_train_loader is None:
            indices, images, labels = data
        else:
            (indices, images_l, labels), (images_u, _) = data
            images = torch.cat([images_l, images_u], dim=0)
            labels = labels

        if torch.cuda.is_available() & (opt.dev != 'cpu'):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            base_images = base_images.cuda(non_blocking=True)
            base_labels = base_labels.cuda(non_blocking=True)
        bsz = labels.shape[0]
        # print('bsz is ', bsz)

        # warm-up learning rate
        if opt.warm:
            warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)
        if opt.adjust_lr:
            adjust_learning_rate(opt, optimizer, epoch)

        if opt.sup_train_type == 'gl':
            label_matrix = nn.functional.one_hot(base_labels, num_classes=10).float()
            images = torch.cat((base_images, images), dim=0)  # Put the base images on top of unlabel images
            _, features = model(images)
            pred = lap(features, label_matrix, opt.temp, opt.epsilon)
            pred = pred[:len(labels)]
            loss = criterion(pred, labels)
        else:
            pred, _ = model(images)
            pred = pred[:len(labels)]
            loss = criterion(pred, labels)
        pred_labels = torch.argmax(pred, dim=1)
        correct_num += torch.sum(torch.eq(pred_labels, labels)).item()
        data_count += len(pred)

        if opt.sup_train_type == 'gl' and epoch % opt.gl_update_base_epochs == 0 and opt.gl_update_base_mode == 'score':
            if opt.gl_score_type == 'entropy':
                batch_size, num_classes = pred.shape
                one_hot_targets = F.one_hot(labels, num_classes=num_classes).to(pred.dtype)
                scores = -torch.sum(one_hot_targets * torch.log(pred + 1e-8), 1)
            elif opt.gl_score_type == 'l2':
                scores = 1 - torch.sum(pred ** 2, 1)
            else:
                raise ValueError(opt.gl_score_type)
            for data_ind, score in zip(indices, scores):
                train_dataset.update_score(data_ind, score)

        # update metric
        losses.update(loss.item(), bsz)
        # SGD
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), opt.temp)
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        is_nan = torch.stack([torch.isnan(p).any() for p in model.parameters()]).any()
        if is_nan:
            print('nan value')

        # print info
        if (idx + 1) % opt.print_freq_ss == 0:
            print('Train: [{0}][{1}/{2}]\t'
                'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg, correct_num / data_count


def main(opt):
    # build data loader
    # _, train_loader_ss, train_dataset_ss = set_loader(opt, loader_suffix='Supervised Training',
    #                                                 augment_type=opt.augment_type,
    #                                                 twoviews=False, p_label=False, train=True,
    #                                                 score_dataset=True)
    # base_loader_eval, test_loader_eval = set_loader(opt, loader_suffix='Test', augment_type='no',
    #                                                 twoviews=False, p_label=False, train=False)
    # _, train_loader_eval = set_loader(opt, loader_suffix='Test', augment_type='no',
    #                                 twoviews=False, p_label=False, train=True)
    train_loaders, eval_loaders = set_loader(opt, augment_type=opt.augment_type)
    train_dataset_ss, train_loader_ss, unlabel_train_loader = train_loaders
    base_loader_eval, train_loader_eval, test_loader_eval = eval_loaders

    print("✓ Data loaders generated successfully.")

    # Dataset info
    print_dataset_info("train_dataset_ss", train_dataset_ss)

    # Train loaders
    print_loader_info("train_loader_ss", train_loader_ss)
    print_loader_info("unlabel_train_loader", unlabel_train_loader)

    # Eval loaders
    print_loader_info("base_loader_eval", base_loader_eval)
    print_loader_info("train_loader_eval", train_loader_eval)
    print_loader_info("test_loader_eval", test_loader_eval)
    
    # build model and criterion
    model = set_model(opt)
    optimizer = set_optimizer(opt, model)

    # training routine
    train_loss_record = []
    test_acc_record = []
    plot_epochs = []

    epoch = 0
    plot_epochs.append(epoch)
    if opt.sup_train_type == 'gl':
        test_acc = test_GL_NP(model, base_loader_eval, test_loader_eval, opt, train_loader=train_loader_eval)
    elif opt.sup_train_type == 'mlp':
        _ = test_GL_NP(model, base_loader_eval, test_loader_eval, opt, train_loader=train_loader_eval)
        test_acc = test_network(model, base_loader_eval, test_loader_eval, opt, predictor='MLP')
    else:
        raise ValueError(opt.sup_train_type)
    test_acc_record.append(test_acc)
    base_dataset_ss = train_dataset_ss.select_base_data(opt.num_base_data, class_uniform_sample=opt.class_uni_sample,
                                                    seed=opt.seed, mode='random')
    base_loader_ss = torch.utils.data.DataLoader(
        base_dataset_ss, batch_size=len(base_dataset_ss), shuffle=True,
        num_workers=opt.num_workers, pin_memory=True, sampler=None)

    for epoch in range(1 + opt.start_epochs, opt.epochs + 1):
        # train for one epoch
        time1 = time.time()
        loss, train_acc = train(train_loader_ss, base_loader_ss, unlabel_train_loader, 
                                train_dataset_ss, model, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}, loss {:.2f}, train acc {:.2f}'.format(epoch, time2 - time1,
                                                                                  loss, train_acc * 100))

        if opt.sup_train_type == 'gl' and epoch % opt.gl_update_base_epochs == 0:
            base_dataset_ss = train_dataset_ss.select_base_data(opt.num_base_data, class_uniform_sample=opt.class_uni_sample,
                                                               seed=None, mode=opt.gl_update_base_mode)
            base_loader_ss = torch.utils.data.DataLoader(
                base_dataset_ss, batch_size=len(base_dataset_ss), shuffle=True,
                num_workers=opt.num_workers, pin_memory=True, sampler=None)
            print(f'Base dataset has been updated with {len(base_dataset_ss)} samples.')
        ## record loss and acc
        train_loss_record.append(loss)

        if epoch % opt.plot_freq_ss == 0:
            plot_epochs.append(epoch)
            if opt.sup_train_type == 'gl':
                test_acc = test_GL_NP(model, base_loader_eval, test_loader_eval, opt, train_loader=train_loader_eval)
            elif opt.sup_train_type == 'mlp':
                _ = test_GL_NP(model, base_loader_eval, test_loader_eval, opt, train_loader=train_loader_eval)
                test_acc = test_network(model, base_loader_eval, test_loader_eval, opt, predictor='MLP')
            else:
                raise ValueError(opt.sup_train_type)
            test_acc_record.append(test_acc)

            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)
            path = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch_num}'.format(epoch_num=epoch))
            visualize(save_file, opt.model, base=base_loader_eval, TSNE=opt.TSNE,
                      head=True, save_dir=path, head_type=opt.head_type)
            os.remove(save_file)

            record_path = os.path.join(opt.save_folder, 'loss_acc_records.npy')
            record_dic = {'epoch': epoch, 'train_loss_record': train_loss_record, 'test_acc_record': test_acc_record}
            np.save(record_path, record_dic)

            plt.figure(figsize=(10, 5))
            plt.plot(train_loss_record, label='Train Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Training Loss Over Epochs')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(opt.save_folder, 'train_loss_plot.png'))
            plt.close()

            plt.figure(figsize=(10, 5))
            plt.plot(plot_epochs, test_acc_record, label='Test Accuracy', color='green')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.title('Test Accuracy Over Epochs')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(opt.save_folder, 'test_acc_plot.png'))
            plt.close()

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)

    path = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}'.format(epoch=opt.epochs))
    # visualize(save_file,opt.model,TSNE=opt.TSNE,head=False,save_dir=path)
    visualize(save_file, opt.model, base=base_loader_eval, TSNE=opt.TSNE,
            head=True, save_dir=path, head_type=opt.head_type)

    record_path = os.path.join(opt.save_folder, 'loss_acc_records.npy')
    record_dic = {'epoch': epoch, 'train_loss_record': train_loss_record, 'test_acc_record': test_acc_record}
    np.save(record_path, record_dic)


if __name__ == '__main__':
    opt = parse_option()
    if not os.path.exists(opt.save_folder):
        os.makedirs(opt.save_folder)

    txt_path_template = os.path.join(opt.save_folder, 'output_record_{}.txt')
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    txt_path = txt_path_template.format(timestamp)
    # if not os.path.isfile(txt_path):
    #     open(txt_path, 'w').close()

    with open(txt_path, "w") as f:
        logger = FileLogger(f, sys.stdout)
        sys.stdout = logger
        try:
            if opt.print_all_parameters:
                for key, value in vars(opt).items():
                    print(f"{key}: {value}")
            main(opt)
        finally:
            sys.stdout = sys.__stdout__
