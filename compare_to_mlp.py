import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import os

####
from utils import adjust_learning_rate, warmup_learning_rate, AverageMeter
from utils import set_optimizer, save_model
from losses import *

from GLL import LaplaceLearningSparseHard
# from VELoss import VarianceExpectationSparseHard
from config.cli import parse_option
from utils import FileLogger, test_GL_NP, test_network
from utils import set_loader, set_model


## --cp_load_path ./simclr_ckpt_epoch_1000.pth

def train(train_loader, base_loader, train_dataset, model, optimizer, epoch, opt):
    """one epoch training"""
    for param in model.parameters():
        param.requires_grad = True

    model.train()

    lap = LaplaceLearningSparseHard.apply
    # criterion = nn.CrossEntropyLoss()
    criterion = custom_ce_loss

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_gl = AverageMeter()
    losses_mlp = AverageMeter()

    end = time.time()
    correct_num = 0
    data_count = 0

    for idx, (indices, images, labels) in enumerate(train_loader):
        base_images, base_labels = next(iter(base_loader))
        data_time.update(time.time() - end)

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
            loss = criterion(pred, labels)
        else:
            label_matrix = nn.functional.one_hot(base_labels, num_classes=10).float()
            images = torch.cat((base_images, images), dim=0)  # Put the base images on top of unlabel images
            pred, features = model(images)
            gl_pred = lap(features, label_matrix, opt.temp, opt.epsilon)
            loss_gl = criterion(gl_pred, labels).item()
            pred = pred[len(base_images):]
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
        if opt.sup_train_type == 'gl':
            losses_gl.update(loss.item(), bsz)
        elif opt.sup_train_type == 'mlp':
            losses_gl.update(loss_gl, bsz)
            losses_mlp.update(loss.item(), bsz)
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
                  'GL loss {loss_gl.val:.3f} ({loss_gl.avg:.3f})\t'
                  'MLP loss {loss_mlp.val:.3f} ({loss_mlp.avg:.3f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss_gl=losses_gl, loss_mlp=losses_mlp))
            sys.stdout.flush()

    return losses_gl.avg, losses_mlp.avg, correct_num / data_count


def main(opt):
    # build data loader
    train_loader_sup, _ = set_loader(opt, loader_suffix='MLP Classifier Pretrain',
                                     augment_type=opt.augment_type_sup,
                                     twoviews=False, p_label=False, train=True, score_dataset=False)
    _, train_loader_ss, train_dataset_ss = set_loader(opt, loader_suffix='Fully Supervised Training',
                                                      augment_type=opt.augment_type_ss,
                                                      twoviews=False, p_label=False, train=True,
                                                      score_dataset=True)
    base_loader_eval, test_loader_eval = set_loader(opt, loader_suffix='Test', augment_type='no',
                                                    twoviews=False, p_label=False, train=False)
    _, train_loader_eval = set_loader(opt, loader_suffix='Test', augment_type='no',
                                      twoviews=False, p_label=False, train=True)

    # build model and criterion
    model = set_model(opt)
    optimizer = set_optimizer(opt, model)

    # training routine
    train_loss_gl_record = []
    train_loss_mlp_record = []
    train_acc_record = []
    test_acc_mlp_record = []
    test_acc_gl_record = []
    plot_epochs = []

    epoch = 0
    # plot_epochs.append(epoch)
    if opt.sup_train_type == 'gl':
        test_acc_gl = test_GL_NP(model, base_loader_eval, test_loader_eval, opt, train_loader=train_loader_eval)
    elif opt.sup_train_type == 'mlp':
        test_acc_gl = test_GL_NP(model, base_loader_eval, test_loader_eval, opt, train_loader=train_loader_eval)
        test_acc_mlp = test_network(model, base_loader_eval, test_loader_eval, opt, predictor='MLP')
    else:
        raise ValueError(opt.sup_train_type)
    # test_acc_record.append(test_acc)
    base_dataset_ss = train_dataset_ss.select_base_data(opt.num_train, class_uniform_sample=opt.class_uni_sample,
                                                        seed=opt.seed, mode='random')
    base_loader_ss = torch.utils.data.DataLoader(
        base_dataset_ss, batch_size=len(base_dataset_ss), shuffle=True,
        num_workers=opt.num_workers, pin_memory=True, sampler=None)

    for epoch in range(1 + opt.start_epochs, opt.epochs + 1):

        # train for one epoch
        time1 = time.time()
        loss_gl, loss_mlp, train_acc = train(train_loader_ss, base_loader_ss, train_dataset_ss, model, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}, loss gl {:.2f}, loss mlp {:.2f}, train acc {:.2f}'
              .format(epoch, time2 - time1, loss_gl, loss_mlp, train_acc * 100))

        if opt.sup_train_type == 'gl' and epoch % opt.gl_update_base_epochs == 0:
            base_dataset_ss = train_dataset_ss.select_base_data(opt.num_train,
                                                                class_uniform_sample=opt.class_uni_sample,
                                                                seed=None, mode=opt.gl_update_base_mode)
            base_loader_ss = torch.utils.data.DataLoader(
                base_dataset_ss, batch_size=len(base_dataset_ss), shuffle=True,
                num_workers=opt.num_workers, pin_memory=True, sampler=None)
            print(f'Base dataset has been updated with {len(base_dataset_ss)} samples.')
        ## record loss and acc
        train_loss_gl_record.append(loss_gl)
        train_loss_mlp_record.append(loss_mlp)
        train_acc_record.append(train_acc)

        if epoch % opt.plot_freq_ss == 0:
            plot_epochs.append(epoch)
            if opt.sup_train_type == 'gl':
                test_acc_gl = test_GL_NP(model, base_loader_eval, test_loader_eval, opt, train_loader=train_loader_eval)
            elif opt.sup_train_type == 'mlp':
                test_acc_gl = test_GL_NP(model, base_loader_eval, test_loader_eval, opt, train_loader=train_loader_eval)
                test_acc_mlp = test_network(model, base_loader_eval, test_loader_eval, opt, predictor='MLP')
                test_acc_mlp_record.append(test_acc_mlp)
            else:
                raise ValueError(opt.sup_train_type)
            test_acc_gl_record.append(test_acc_gl)

            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)
            # os.remove(save_file)

            record_path = os.path.join(opt.save_folder, 'loss_acc_records.npy')
            record_dic = {'epoch': epoch,
                          'train_loss_gl_record': train_loss_gl_record,
                          'train_loss_mlp_record': train_loss_mlp_record,
                          'test_acc_record': test_acc_mlp_record,
                          'test_acc_gl_record': test_acc_gl_record}
            np.save(record_path, record_dic)

            plt.figure(figsize=(10, 5))
            plt.plot(train_loss_gl_record, label='Train Loss GL')
            if opt.sup_train_type == 'mlp':
                plt.plot(train_loss_mlp_record, label='Train Loss MLP')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Training Loss Over Epochs')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(opt.save_folder, 'train_loss_plot.png'))
            plt.close()

            plt.figure(figsize=(10, 5))
            plt.plot(plot_epochs, train_acc_record, label='Train Accuracy', color='green')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.title('Train Accuracy Over Epochs')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(opt.save_folder, 'train_acc_plot.png'))
            plt.close()

            plt.figure(figsize=(10, 5))
            plt.plot(plot_epochs, test_acc_gl_record, label='Test GL Accuracy', color='green')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.title('Test GL Accuracy Over Epochs')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(opt.save_folder, 'test_acc_gl_plot.png'))
            plt.close()

            if opt.sup_train_type == 'mlp':
                plt.figure(figsize=(10, 5))
                plt.plot(plot_epochs, test_acc_mlp_record, label='Test MLP Accuracy', color='green')
                plt.xlabel('Epochs')
                plt.ylabel('Accuracy')
                plt.title('Test MLP Accuracy Over Epochs')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(opt.save_folder, 'test_acc_mlp_plot.png'))
                plt.close()

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)

    path = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}'.format(epoch=opt.epochs))

    record_path = os.path.join(opt.save_folder, 'loss_acc_records.npy')
    record_dic = {'epoch': epoch,
                  'train_loss_gl_record': train_loss_gl_record,
                  'train_loss_mlp_record': train_loss_mlp_record,
                  'test_acc_record': test_acc_mlp_record,
                  'test_acc_gl_record': test_acc_gl_record}
    np.save(record_path, record_dic)


if __name__ == '__main__':
    opt = parse_option()
    if not os.path.exists(opt.save_folder):
        os.makedirs(opt.save_folder)

    for key, value in vars(opt).items():
        print(f"{key}: {value}")

    txt_path_template = os.path.join(opt.save_folder, 'output_record_{}.txt')
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    txt_path = txt_path_template.format(timestamp)
    # if not os.path.isfile(txt_path):
    #     open(txt_path, 'w').close()

    with open(txt_path, "w") as f:
        logger = FileLogger(f, sys.stdout)
        sys.stdout = logger
        try:
            main(opt)
        finally:
            sys.stdout = sys.__stdout__
