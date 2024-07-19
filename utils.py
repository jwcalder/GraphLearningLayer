# from __future__ import print_function

import math
import numpy as np
import random

from collections import defaultdict

from PIL import Image

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from torch.utils.data import Dataset
import torch.nn as nn

####
from networks.BuildNet import buildnet
from networks.customCNN import customCNN
from config import datasets_setting
import scipy.sparse as sparse

from GLL import LaplaceLearningSparseHard, knn_sym_dist, stable_conjgrad

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

class NCropTransform:
    """Create N crops/views of the same image"""
    def __init__(self, transform, num_crops=2):
        self.transform = transform
        self.num_crops = num_crops

    def __call__(self, x):
        l=[]
        for i in range(self.num_crops):
            l.append(self.transform(x))
        return l
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(args, optimizer, epoch, lr_multiply=1):
    lr = args.learning_rate
    for param_group in optimizer.param_groups:
        # lr = param_group['lr']
        if args.cosine:
            eta_min = lr * (args.lr_decay_rate ** 3)
            lr = eta_min + (lr - eta_min) * (
                    1 + math.cos(math.pi * epoch / args.epochs)) / 2
        else:
            steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
            if steps > 0:
                lr = lr * (args.lr_decay_rate ** steps)
        param_group['lr'] = lr * lr_multiply

def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer, lr_multiply=1):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * lr_multiply

def set_optimizer(opt, model):
    if isinstance(model, list):
        parameters = []
        for m in model:
            parameters.extend(list(m.parameters()))
    else:
        parameters = model.parameters()

    if opt.Adam:
        optimizer = optim.Adam(parameters,
                               lr=opt.learning_rate,
                               weight_decay=opt.weight_decay)
    else:
        optimizer = optim.SGD(parameters,
                              lr=opt.learning_rate,
                              momentum=opt.momentum,
                              weight_decay=opt.weight_decay)
    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


def get_base_samples_new(dataset, rate=10, num_class=10, seed=None):
    # Set the random seed for reproducibility if provided
    if seed is not None:
        torch.manual_seed(seed)

    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    features, labels = next(iter(loader))

    # Ensure labels are a tensor
    labels = labels if isinstance(labels, torch.Tensor) else torch.tensor(labels)

    # Calculate the number of samples per class
    if isinstance(rate, int):
        num_samples = rate
    elif isinstance(rate, float):
        # Calculate number of samples as a fraction of total
        num_samples = int(rate * len(features) / num_class)

    # Generate the indices for the base samples
    base_samples_indices = []

    for i in range(num_class):
        class_mask = labels == i
        class_indices = torch.where(class_mask)[0]
        class_samples = class_indices[torch.randperm(len(class_indices))[:num_samples]]
        base_samples_indices.append(class_samples)

    # Concatenate indices from all classes
    base_samples_indices = torch.cat(base_samples_indices).long()
    print(base_samples_indices.shape)
    return features[base_samples_indices], labels[base_samples_indices]

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]

        if self.transform:
            x = Image.fromarray(x) if isinstance(x, np.ndarray) else transforms.ToPILImage()(x)
            x = self.transform(x)

        return x, y

class DSCustomDataset(Dataset):
    def __init__(self, dataset, stepsize=1):
        self.dataset = dataset
        self.stepsize = stepsize

    def __len__(self):
        return len(self.dataset) // self.stepsize

    def __getitem__(self, idx):
        new_idx = idx * self.stepsize

        return self.dataset[new_idx][0], self.dataset[new_idx][1]


def prepare_class_indices(dataset):
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)
    return class_indices


def sample_dataset(dataset, num_samples, class_uniform_sample=False, num_classes=None, seed=None):
    if seed is not None:
        np.random.seed(seed)

    if class_uniform_sample:
        if num_classes is None:
            raise ValueError("num_classes must be provided when class_uniform_sample is True")

        if not hasattr(dataset, 'class_indices'):
            dataset.class_indices = prepare_class_indices(dataset)

        samples_per_class = num_samples // num_classes
        selected_indices = [np.random.choice(indices, samples_per_class, replace=False)
                            for indices in dataset.class_indices.values()]
        selected_indices = np.concatenate(selected_indices)
    else:
        selected_indices = np.random.choice(len(dataset), num_samples, replace=False)

    to_tensor_transform = transforms.ToTensor()  # 创建 ToTensor 转换
    tensors, labels = [], []

    for idx in selected_indices:
        image, label = dataset[idx]
        if not torch.is_tensor(image):
            image = to_tensor_transform(image)
        tensors.append(image)
        labels.append(label)

    return torch.stack(tensors), torch.tensor(labels)


def loader_to_numpy(loader, opt, model=None):
    """
    将 PyTorch DataLoader 中的数据和标签转换为 NumPy 数组。
    """
    data_list = []
    label_list = []

    with torch.no_grad():
        for item in loader:
            if len(item) == 2:
                data, labels = item
            else:
                _, data, _, labels = item
            if model is not None:
                if torch.cuda.is_available() & (opt.dev != 'cpu'):
                    data = data.cuda(non_blocking=True)
                _, feature = model(data)
            feature = feature.cpu().numpy()
            labels = labels.cpu().numpy()

            data_list.append(feature[labels >= 0])
            label_list.append(labels[labels >= 0])

    data_array = np.concatenate(data_list, axis=0)
    label_array = np.concatenate(label_list, axis=0)

    return data_array, label_array

class FileLogger(object):
    def __init__(self, fileobj, stdout):
        self.terminal = stdout
        self.log = fileobj

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def set_loader(opt, loader_suffix='Sup', augment_type='weak', twoviews=False, p_label=False, train=True,
               score_dataset=False):
    if opt.dataset == 'cifar10' or opt.dataset == 'cifar100':
        dataset_config = datasets_setting.__dict__[opt.dataset]()
    elif opt.dataset == 'mnist' or opt.dataset == 'fashion_mnist':
        dataset_config = datasets_setting.__dict__[opt.dataset]()
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    weak_transformation = dataset_config['weak_transformation']
    strong_transformation = dataset_config['strong_transformation']
    eval_transformation = dataset_config['eval_transformation']
    num_classes = dataset_config['num_classes']

    print(f'Loader_mode: {loader_suffix}; Augment_type: {augment_type}.')

    if augment_type == 'no':
        transform = eval_transformation
    elif augment_type == 'weak':
        transform = weak_transformation
    else:
        transform = strong_transformation

    # for simclr
    if twoviews:
        transform = TwoCropTransform(transform)

    if opt.dataset == 'cifar10':
        dataset = datasets.CIFAR10(root=opt.data_folder,
                                   transform=None,
                                   train=True,
                                   download=True)
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=transform,
                                         train=train,
                                         download=True)
    elif opt.dataset == 'mnist':
        dataset = datasets.MNIST(root=opt.data_folder,
                                 transform=None,
                                 train=True,
                                 download=True)
        train_dataset = datasets.MNIST(root=opt.data_folder,
                                       transform=transform,
                                       train=train,
                                       download=True)
    elif opt.dataset == 'fashion_mnist':
        dataset = datasets.FashionMNIST(root=opt.data_folder,
                                        transform=None,
                                        train=True,
                                        download=True)

        train_dataset = datasets.FashionMNIST(root=opt.data_folder,
                                              transform=transform,
                                              train=train,
                                              download=True)
    else:
        raise ValueError(opt.dataset)

    if int(opt.ds_stepsize) > 1:
        dataset = DSCustomDataset(dataset, int(opt.ds_stepsize))

    if train:
        batch_size = opt.batch_size
    else:
        batch_size = opt.test_batch_size

    base_data, base_labels = sample_dataset(dataset, opt.num_train, class_uniform_sample=opt.class_uni_sample,
                                            num_classes=num_classes, seed=opt.seed)
    base_dataset = CustomDataset(base_data,
                                 base_labels,
                                 transform=transform)

    base_loader = torch.utils.data.DataLoader(
        base_dataset, batch_size=len(base_dataset), shuffle=True,
        num_workers=opt.num_workers, pin_memory=True, sampler=None)

    if score_dataset:
        train_dataset_new = DatasetWithScore(train_dataset, scores=None)

        train_loader = torch.utils.data.DataLoader(
            train_dataset_new, batch_size=batch_size, shuffle=True,
            num_workers=opt.num_workers, pin_memory=True, sampler=None, drop_last=True)
        return base_loader, train_loader, train_dataset_new
    elif p_label:
        train_dataset_new = DatasetWithPseudoLabel(train_dataset, pred_outputs=None,
                                                   pred_labels=None, num_classes=num_classes)

        train_loader = torch.utils.data.DataLoader(
            train_dataset_new, batch_size=batch_size, shuffle=True,
            num_workers=opt.num_workers, pin_memory=True, sampler=None, drop_last=True)
        return base_loader, train_loader, train_dataset_new
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=opt.num_workers, pin_memory=True, sampler=None, drop_last=True)
        return base_loader, train_loader

def set_loader_sup(opt, loader_mode='Sup', p_label=False):
    '''
    loader_mode should be chosen from ['Sup','SimCLR','SS','Eval']
    '''
    if opt.dataset == 'cifar10' or opt.dataset == 'cifar100':
        dataset_config = datasets_setting.__dict__[opt.dataset]()
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    weak_transformation = dataset_config['weak_transformation']
    strong_transformation = dataset_config['strong_transformation']
    eval_transformation = dataset_config['eval_transformation']
    num_classes = dataset_config['num_classes']

    if loader_mode == 'Sup':
        print(f'Supervised augmentation: {opt.augment_type_sup}.')
        if opt.dataset == 'cifar10':
            dataset = datasets.CIFAR10(root=opt.data_folder,
                                       transform=None,
                                       train=True,
                                       download=True)
        else:
            raise ValueError(opt.dataset)

        base_data, base_labels = sample_dataset(dataset, opt.num_train, class_uniform_sample=opt.class_uni_sample,
                                                num_classes=num_classes, seed=opt.seed)
        if opt.augment_type_sup == 'no':
            transform = eval_transformation
        elif opt.augment_type_sup == 'weak':
            transform = weak_transformation
        else:
            transform = strong_transformation
        if opt.sup_method == 'SupCon':
            transform = TwoCropTransform(transform)
        base_dataset = CustomDataset(base_data,
                                     base_labels,
                                     transform=transform)

        if p_label:
            base_dataset_new = DatasetWithPseudoLabel(base_dataset, pred_outputs=None,
                                                      pred_labels=None, num_classes=num_classes)
            base_loader = torch.utils.data.DataLoader(
                base_dataset_new, batch_size=len(base_dataset), shuffle=True,
                num_workers=opt.num_workers, pin_memory=True, sampler=None)
            return base_loader, base_dataset_new
        else:
            base_loader = torch.utils.data.DataLoader(
                base_dataset, batch_size=len(base_dataset), shuffle=True,
                num_workers=opt.num_workers, pin_memory=True, sampler=None)
            return base_loader
    elif loader_mode == 'SimCLR' or loader_mode == 'SS':
        print(f'Semi-supervised augmentation: {opt.augment_type_ss}.')
        if opt.augment_type_ss == 'no':
            transform = eval_transformation
        elif opt.augment_type_ss == 'weak':
            transform = weak_transformation
        else:
            transform = strong_transformation

        # for simclr
        if loader_mode == 'SimCLR':
            transform = TwoCropTransform(transform)

        if opt.dataset == 'cifar10':
            base_dataset = datasets.CIFAR10(root=opt.data_folder,
                                            transform=None,
                                            train=True,
                                            download=True)
            train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                             transform=transform,
                                             train=True,
                                             download=True)
        else:
            raise ValueError(opt.dataset)

        base_data, base_labels = sample_dataset(base_dataset, opt.num_train, class_uniform_sample=opt.class_uni_sample,
                                                num_classes=num_classes, seed=opt.seed)
        base_dataset = CustomDataset(base_data,
                                     base_labels,
                                     transform=transform)
        train_sampler = None
        base_loader = torch.utils.data.DataLoader(
            base_dataset, batch_size=len(base_dataset), shuffle=(train_sampler is None),
            num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
        if p_label:
            train_dataset_new = DatasetWithPseudoLabel(base_dataset, pred_outputs=None,
                                                      pred_labels=None, num_classes=num_classes)
            train_loader = torch.utils.data.DataLoader(
                train_dataset_new, batch_size=opt.batch_size, shuffle=(train_sampler is None),
                num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)
            return base_loader, train_loader, train_dataset_new
        else:
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
                num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)
            return base_loader, train_loader
    elif loader_mode == 'Eval':
        print(f'Evaluation with no augmentation.')
        if opt.dataset == 'cifar10':
            train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                             transform=eval_transformation,
                                             train=True,
                                             download=True)
            test_dataset = datasets.CIFAR10(root=opt.data_folder,
                                            transform=eval_transformation,
                                            train=False,
                                            download=True)
        else:
            raise ValueError(opt.dataset)
        base_data, base_labels = sample_dataset(train_dataset, opt.num_train, class_uniform_sample=opt.class_uni_sample,
                                                num_classes=num_classes, seed=opt.seed)
        base_dataset = CustomDataset(base_data,
                                     base_labels,
                                     transform=None)
        train_sampler = None
        base_loader = torch.utils.data.DataLoader(
            base_dataset, batch_size=len(base_dataset), shuffle=(train_sampler is None),
            num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler, drop_last=False)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
            num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler, drop_last=False)
        return base_loader, test_loader
    else:
        raise ValueError(loader_mode)


def set_model(opt):
    if opt.dataset == 'cifar10' or opt.dataset == 'cifar100':
        dataset_config = datasets_setting.__dict__[opt.dataset]()
    elif opt.dataset == 'mnist' or opt.dataset == 'fashion_mnist':
        dataset_config = datasets_setting.__dict__[opt.dataset]()
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    num_classes = dataset_config.pop('num_classes')

    if opt.model == 'customCNN' and (opt.dataset == 'mnist' or opt.dataset == 'fashion_mnist'):
        model = customCNN()
    else:
        model = buildnet(name=opt.model, head=opt.head_type,
                         feat_dim=opt.embedding_dim, num_classes=num_classes, softmax=not opt.no_softmax)

    # model = SupConResNet(name=opt.model)
    # model = create_model(num_classes, opt)

    if opt.cp_load_path != 'no':
        model_path = opt.cp_load_path
        model_dict = torch.load(model_path)
        try:
            result = model.load_state_dict(model_dict["model"])
            print(f"Successfully load model from {model_path}. Every key matches exactly.")
        except:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in model_dict["model"].items():
                name = k.replace(".module", "")  # remove `module.`
                new_state_dict[name] = v
            result = model.load_state_dict(new_state_dict, strict=False)
            missing_keys = result.missing_keys
            print("Missing keys:", missing_keys)
            unexpected_keys = result.unexpected_keys
            print("Unexpected keys:", unexpected_keys)
            print(f"Successfully load model from {model_path}")
    else:
        print(f"Initize the model with random parameters")

    # criterion = SupConLoss(temperature=opt.temp)

    if torch.cuda.is_available() & (opt.dev != 'cpu'):
        if torch.cuda.device_count() > 1:  # check for multiple GPU
            model.encoder = torch.nn.DataParallel(model.encoder)
        # model = model.cuda()
        # criterion = criterion.cuda()
        cudnn.benchmark = True

    return model.to(torch.device(opt.dev))  # , criterion

##### laplace learning
def one_hot_encode(labels, n_classes='auto'):
    # Number of labels and number of unique classes
    n_labels = len(labels)
    if n_classes == 'auto':
        n_classes = len(np.unique(labels))

    # Initialize the one-hot encoded matrix
    one_hot_matrix = np.zeros((n_labels, n_classes))

    # Set the appropriate elements to 1
    one_hot_matrix[np.arange(n_labels), labels] = 1

    return one_hot_matrix

def laplace(X, train_labels, knn_num=50, epsilon='auto', n_classes='auto', tau=1e-8):
    '''
    labeled indices are 0,1,2,...,k-1
    '''
    W, _, _, _, _ = knn_sym_dist(X, k=knn_num, epsilon=epsilon)
    L = sparse.csgraph.laplacian(W).tocsr()
    label_matrix = one_hot_encode(train_labels, n_classes)
    k = label_matrix.shape[0]

    Luu = L[k:, k:]  # Lower Right Corner - unlabelled with unlabelled
    Lul = L[k:, :k]  # Lower Left Rectangle - Labelled and Unlabelled

    m = Luu.shape[0]

    Luu = Luu + sparse.spdiags(tau * np.ones(m), 0, m, m).tocsr()

    M = Luu.diagonal()
    M = sparse.spdiags(1 / np.sqrt(M + 1e-10), 0, m, m).tocsr()

    Pred = stable_conjgrad(M * Luu * M,
                           -M * Lul @ label_matrix)  #
    Pred = M * Pred

    return Pred
######

def test_network(model, base_loader, test_loader, opt, predictor='GL'):
    '''
    Directly use the network with Laplace learning layer to do the test
    '''

    lap = LaplaceLearningSparseHard.apply

    model.eval()

    base_images, base_labels = next(iter(base_loader))
    data_count = 0
    correct_num = 0
    for idx, (images, labels) in enumerate(test_loader):
        if torch.cuda.is_available() & (opt.dev != 'cpu'):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            base_images = base_images.cuda(non_blocking=True)
            base_labels = base_labels.cuda(non_blocking=True)

        if predictor == "GL":
            label_matrix = nn.functional.one_hot(base_labels, num_classes=10).float()

            images = torch.cat((base_images, images), dim=0)  # Put the base images on top of unlabel images
            _, features = model(images)

            pred = lap(features, label_matrix, opt.temp, opt.epsilon)
        elif predictor == 'MLP':
            pred, _ = model(images)
        else:
            raise ValueError(predictor)

        pred_labels = torch.argmax(pred, dim=1)
        correct_num += torch.sum(torch.eq(pred_labels, labels)).item()
        data_count += len(pred)

    print('Test set: Accuracy for {} predictor: {}/{} ({:.2f}%)\n'.format(
        predictor, correct_num, data_count,
        100. * correct_num / data_count))
    return 100. * correct_num / data_count


def test_GL_NP(model, base_loader, test_loader, opt, train_loader=None):
    '''
    Transform to numpy and do standard Laplace learning test
    '''
    model.eval()

    test_data, test_label = loader_to_numpy(test_loader, opt, model)
    train_data, train_label = loader_to_numpy(base_loader, opt, model)
    if train_loader is not None:
        unlabeled_train_data, unlabeled_train_label_new = loader_to_numpy(train_loader, opt, model)
        all_data = np.concatenate((train_data, unlabeled_train_data, test_data), axis=0)
    else:
        all_data = np.concatenate((train_data, test_data), axis=0)

    U = laplace(all_data, train_label, knn_num=50, epsilon=opt.epsilon, n_classes='auto', tau=opt.tau)
    pred = np.argmax(U, axis=1)
    correct_num = np.sum(pred[-len(test_data):] == test_label)
    total_test_num = len(test_data)

    print('Test set: Accuracy for GL predictor (Num of train data: {})\t'
          ': {}/{} ({:.2f}%)\n'.format(
        len(train_data), correct_num, total_test_num,
        100. * correct_num / total_test_num))
    return 100. * correct_num / total_test_num


#### for pseudo label training
class DatasetWithPseudoLabel(Dataset):
    def __init__(self, original_dataset, pred_outputs=None,
                 pred_labels=None, num_classes=10):
        self.original_dataset = original_dataset
        self.pred_outputs = pred_outputs
        self.num_classes = num_classes

        if pred_labels is None:
            self.pred_labels = -1 * torch.ones(len(self.original_dataset), dtype=torch.long)
        else:
            self.pred_labels = pred_labels

        self.thresh = 2 * torch.ones(num_classes, dtype=torch.long)

    def __len__(self):
        return len(self.original_dataset)

    def update_pred_labels(self, index, new_pred_label):
        self.pred_labels[index] = new_pred_label

    def update_pred_outputs(self, index, new_pred_outputs):
        if self.pred_outputs is None:
            self.pred_outputs = torch.zeros((len(self.original_dataset), self.num_classes), dtype=torch.float)

        new_pred_outputs = new_pred_outputs.to(dtype=self.pred_outputs.dtype, device=self.pred_outputs.device)

        self.pred_outputs[index] = new_pred_outputs

    def update_thresh(self, new_thresh):
        self.thresh = new_thresh

    def update_all_plabels(self):
        self.pred_labels = convert_outputs_to_pseudo_labels(self.pred_outputs, self.thresh)

    def __getitem__(self, index):
        data, label = self.original_dataset[index]
        if self.pred_outputs is None:
            pred_output = torch.tensor([0])
        else:
            pred_output = self.pred_outputs[index]
        return index, data, label, pred_output, self.pred_labels[index]


def convert_outputs_to_pseudo_labels(outputs, thresh):
    """
    Convert model outputs to pseudo labels based on a threshold or a tensor of thresholds.
    Removes gradient information from the outputs.

    Args:
    outputs (torch.Tensor): The output from the model for a batch of data.
                             This should be a 2D tensor where each row corresponds to a data point
                             and each column to a class probability or logit.
    thresh (float or torch.Tensor): The threshold(s) to decide whether to assign a pseudo label or -1.
                                    If a tensor, its length should match the number of classes.

    Returns:
    torch.Tensor: A 1D tensor containing the pseudo labels for each data point in the batch.
    """
    # Detach the tensor from the computation graph and remove gradient info
    device = outputs.device
    outputs = outputs.detach().cpu()

    # Convert thresh to a tensor if it is a float
    if isinstance(thresh, float):
        thresh = torch.full((outputs.shape[1],), thresh, dtype=outputs.dtype, device=device)
    else:
        thresh = thresh.to(device)

    # Check if the length of thresh matches the number of classes
    if outputs.shape[1] != thresh.shape[0]:
        raise ValueError("Length of thresh does not match the number of classes in outputs")

    max_values, max_indices = torch.max(outputs, dim=1)
    pseudo_labels = torch.full(max_indices.shape, -1, dtype=max_indices.dtype, device=device)
    max_indices, max_values = max_indices.to(device), max_values.to(device)

    # Apply the threshold to each class
    for i in range(outputs.shape[1]):
        # print("max_indices", max_indices.device)
        # print("max_values", max_values.device)
        # print('thresh', thresh.device)
        mask = (max_indices == i) & (max_values > thresh[i])
        pseudo_labels[mask] = i

    return pseudo_labels

#### fully supervised training
class DatasetWithScore(Dataset):
    def __init__(self, original_dataset, scores=None):
        self.original_dataset = original_dataset

        if scores is None:
            self.scores = torch.zeros(len(self.original_dataset), dtype=torch.float)
        else:
            self.scores = scores

        self.class_indices = prepare_class_indices(original_dataset)

    def __len__(self):
        return len(self.original_dataset)

    def update_score(self, index, new_score):
        self.scores[index] = new_score

    def select_base_data(self, num_samples, class_uniform_sample=False, seed=None, mode='random'):
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        if mode == 'random':
            if class_uniform_sample:
                samples_per_class = num_samples // len(self.class_indices)
                selected_indices = []
                for indices in self.class_indices.values():
                    selected_indices.extend(random.sample(indices, min(samples_per_class, len(indices))))
            else:
                selected_indices = random.sample(range(len(self)), num_samples)

        elif mode == 'score':
            if class_uniform_sample:
                samples_per_class = num_samples // len(self.class_indices)
                selected_indices = []
                for class_label, indices in self.class_indices.items():
                    # Sort indices within each class based on scores
                    sorted_class_indices = sorted(indices, key=lambda idx: self.scores[idx], reverse=True)
                    selected_indices.extend(sorted_class_indices[:min(samples_per_class, len(indices))])
            else:
                sorted_indices = sorted(range(len(self)), key=lambda idx: self.scores[idx], reverse=True)
                selected_indices = sorted_indices[:num_samples]
        else:
            raise ValueError(mode)

        to_tensor_transform = transforms.ToTensor()  # 创建 ToTensor 转换
        tensors, labels = [], []

        for idx in selected_indices:
            image, label = self.original_dataset[idx]
            if not torch.is_tensor(image):
                image = to_tensor_transform(image)
            tensors.append(image)
            labels.append(label)

        base_dataset = CustomDataset(torch.stack(tensors),
                                     torch.tensor(labels),
                                     transform=None)
        # base_loader = torch.utils.data.DataLoader(
        #     base_dataset, batch_size=len(base_dataset), shuffle=True,
        #     num_workers=opt.num_workers, pin_memory=True, sampler=None)
        return base_dataset

    def __getitem__(self, index):
        data, label = self.original_dataset[index]
        return index, data, label