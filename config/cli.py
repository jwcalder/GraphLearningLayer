import argparse
import os

import torch
import math
def str_or_float(value):
    try:
        return float(value)
    except ValueError:
        return value

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--dev', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='cpu or cuda')

    parser.add_argument('--print_freq_sup', type=int, default=1,
                        help='print frequency')
    parser.add_argument('--print_freq_ss', type=int, default=10,
                        help='print frequency')
    # parser.add_argument('--save_freq', type=int, default=10,
    #                     help='save frequency')
    parser.add_argument('--plot_freq_sup', type=int, default=15,
                        help='plot frequency for supervised training-- deprecated')
    parser.add_argument('--plot_freq_ss', type=int, default=15,
                        help='plot frequency for semi-supervised training-- deprecated')
    parser.add_argument('--batch_size', type=int, default=1250,
                        help='batch_size')
    parser.add_argument('--test_batch_size', type=int, default=1250,
                        help='batch_size for test set')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of training epochs')
    parser.add_argument('--cp_load_path', type=str, default='no',
                        help='path to the checkpoint. no means to train from scratch.')
    parser.add_argument('--train_mode', type=str, default='Sup_and_SS',
                        choices=['Sup_and_SS', 'Sup_only', 'SS_only'],
                        help='training mode')
    parser.add_argument('--train_prefix', type=str, default='',
                        help='training prefix of the folder')
    parser.add_argument('--no_softmax', action='store_true',
                        help='not use a softmax layer at the end of the mlp predictor')
    parser.add_argument('--start_epochs', type=int, default=0,
                        help='the start of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='400,500,600,700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='using warm-up learning rate')
    parser.add_argument('--adjust_lr', action='store_true',
                        help='adjusting learning rate')
    parser.add_argument('--Adam', action='store_true',
                        help='use Adam optimizer or not')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'mnist', 'fashion_mnist'], help='dataset')
    parser.add_argument('--ds_stepsize', type=int, default=1, help='the step size to downsample the dataset')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')
    parser.add_argument('--augment_type_sup', type=str, default='strong',
                        choices=['no', 'weak', 'strong'], help='augmentation intensity for supervised part')
    parser.add_argument('--augment_type_ss', type=str, default='strong',
                        choices=['no', 'weak', 'strong'], help='augmentation intensity for semi-supervised part')
    parser.add_argument('--num_train', type=int, default=250, help='total num of training samples')

    # method
    parser.add_argument('--sup_method', type=str, default='SupCE',
                        choices=['SupCE', 'SupCon'], help='choose method')
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='dimension of latent features')
    parser.add_argument('--head_type', type=str, default='mlp',
                        choices=['mlp', 'linear', 'no'], help='dataset')
    parser.add_argument('--TSNE', action='store_true',
                        help='use TSNE embedding')

    # general parameters
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')
    parser.add_argument('--tau', type=float, default=1e-8,
                        help='tau for CG regularization')
    parser.add_argument('--epsilon', type=str_or_float, default=1,
                        help='epsilon for knn weight matrix, can be auto or any float.')
    parser.add_argument('--alpha', type=float, default=1,
                        help='alpha for the sym CE loss.')

    # uncertainty parameters
    parser.add_argument('--beta', type=float, default=0,
                        help='beta for the VE loss.')
    parser.add_argument('--n_samples_ve', type=int, default=50,
                        help='number of samples for the VE loss.')

    # supervised only
    parser.add_argument('--sup_train_type', type=str, default='no',
                        choices=['no', 'gl', 'mlp'], help='type of supervised training')
    parser.add_argument('--pretrain_lr_multiply', type=int, default=1,
                        help='the multiplier of the pretrain learning rate.')
    parser.add_argument('--sup_epochs', type=int, default=0,
                        help='Number of supervised pretraining epochs.')
    parser.add_argument('--gl_update_base_epochs', type=int, default=1,
                        help='Number of epochs between each update of GL base data.')
    parser.add_argument('--gl_update_base_mode', type=str, default='score',
                        choices=['random', 'score'], help='Mode to update GL base data.')
    parser.add_argument('--gl_score_type', type=str, default='entropy',
                        choices=['entropy', 'l2'], help='Type of GL scores.')

    # alternative pseudo label training
    parser.add_argument('--mlp_train_steps', type=int, default=1,
                        help='number of training steps for mlp training in each epoch')
    parser.add_argument('--gl_train_steps', type=int, default=1,
                        help='number of training steps for gl training in each epoch')
    parser.add_argument('--thresh', type=float, default=0.9,
                        help='threshold for trustable gl pseudo labels.')
    parser.add_argument('--thresh_mlp', type=float, default=0.95,
                        help='threshold for trustable mlp pseudo labels.')
    parser.add_argument('--DV_plabels', action='store_true',
                        help='use double verified pseudo labels')
    parser.add_argument('--plabel_update_epochs', type=int, default=5,
                        help='Number of epochs to update the plabel after.')

    ## CPL
    parser.add_argument('--cpl', action='store_true',
                        help='use curriculum pseudo labels')
    parser.add_argument('--cpl_nonlinear', action='store_true',
                        help='use nonlinear mapping function curriculum pseudo labels')
    parser.add_argument('--cpl_warmup', action='store_true',
                        help='use warm-up threshold for curriculum pseudo labels')


    # other
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--class_rand_sample', action='store_true',
                        help='sample data randomly among classes for base dataset')
    parser.add_argument('--sup_train_time', type=int, default=10, help='number of sup steps with each simclr step')
    parser.add_argument('--print_all_parameters', action='store_true', help='print all parameters')

    opt = parser.parse_args()

    opt.epochs = opt.epochs + + opt.start_epochs

    if opt.class_rand_sample:
        opt.class_uni_sample = False
    else:
        opt.class_uni_sample = True

    # configure cuda/cpu
    if opt.dev == 'cuda':
        k = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(k)]
        print(f"Using {k} GPUs with names {gpu_names}.")
        o = ''
        for i in range(k):
            o = o + str(i) + ','
        os.environ["CUDA_VISIBLE_DEVICES"] = o[0:-1]
    elif opt.dev == 'cpu':
        torch.set_num_threads(24)  # jeff's machine

    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    opt.model_path = f'./save/{opt.train_prefix}_{opt.train_mode}'
    if opt.sup_train_type != 'no':
        opt.model_path = f'{opt.model_path}_{opt.sup_train_type}'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_bsz_{}_method_{}_{}_supaug_{}_ssaug_{}'. \
        format(opt.sup_method, opt.model, opt.batch_size, opt.sup_method,
               opt.train_mode, opt.augment_type_sup, opt.augment_type_ss)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,

    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 1e-2
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    print("save_folder: {}".format(opt.save_folder))
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt
