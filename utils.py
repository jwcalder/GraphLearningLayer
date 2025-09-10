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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

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
    # state = {
    #     'opt': opt,
    #     'model': model.state_dict(),
    #     'optimizer': optimizer.state_dict(),
    #     'epoch': epoch,
    # }
    # torch.save(state, save_file)
    # print('Checkpoint saved to {}'.format(save_file))
    # del state
    state = {
        "epoch": epoch,
        "model": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "opt": vars(opt) if opt is not None else None,  # keep it serializable
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
    }
    torch.save(state, save_file)
    print(f"Saved checkpoint to {save_file}")


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
    """
    Build a dict: class_id (int) -> list of sample indices for that class.
    Supports labels that are Python int or torch.Tensor.
    Also uses dataset.targets fast-path if available (e.g., torchvision CIFAR-10).
    """
    class_indices = defaultdict(list)

    # Fast path for torchvision datasets
    if hasattr(dataset, "targets"):
        targets = dataset.targets
        # torch.Tensor -> list
        if isinstance(targets, torch.Tensor):
            targets = targets.tolist()

        for idx, y in enumerate(targets):
            # Normalize to Python int
            if isinstance(y, torch.Tensor):
                t = y.detach().cpu()
                if t.ndim == 0 or (t.ndim == 1 and t.numel() == 1):
                    y = int(t.item())
                elif t.ndim == 1:
                    y = int(torch.argmax(t).item())
                else:
                    raise TypeError(f"Unsupported tensor label shape at index {idx}: {tuple(t.shape)}")
            else:
                y = int(y)
            class_indices[y].append(idx)
        return class_indices

    # Generic slow path: iterate dataset
    for idx in range(len(dataset)):
        item = dataset[idx]
        label = item[1] if isinstance(item, (tuple, list)) and len(item) >= 2 else None
        if label is None:
            raise ValueError("Dataset item must be a (data, label) tuple/list.")

        # Normalize to Python int
        if isinstance(label, torch.Tensor):
            t = label.detach().cpu()
            if t.ndim == 0 or (t.ndim == 1 and t.numel() == 1):
                cls_id = int(t.item())
            elif t.ndim == 1:
                cls_id = int(torch.argmax(t).item())
            else:
                raise TypeError(f"Unsupported tensor label shape at index {idx}: {tuple(t.shape)}")
        else:
            cls_id = int(label)

        class_indices[cls_id].append(idx)

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

    to_tensor_transform = transforms.ToTensor()  
    tensors, labels = [], []

    for idx in selected_indices:
        image, label = dataset[idx]
        if not torch.is_tensor(image):
            image = to_tensor_transform(image)
        tensors.append(image)
        labels.append(label)

    return torch.stack(tensors), torch.tensor(labels)

def sample_and_split_dataset(dataset, num_samples, class_uniform_sample=False, num_classes=None, seed=None):
    """
    Split a dataset into two parts:
      1) a sampled subset (using the same sampling logic as `sample_dataset`)
      2) the remaining subset (all items not selected in the sample)

    Args:
        dataset: A dataset implementing __len__ and __getitem__ -> (image, label).
        num_samples (int): Number of samples to draw (without replacement).
        class_uniform_sample (bool): If True, sample uniformly across classes.
        num_classes (int or None): Total number of classes; required if class_uniform_sample is True.
        seed (int or None): Random seed for reproducibility.

    Returns:
        ((sample_tensors, sample_labels), (rest_tensors, rest_labels)):
            - sample_tensors: torch.Tensor of shape [N, C, H, W]
            - sample_labels: torch.LongTensor of shape [N]
            - rest_tensors: torch.Tensor of shape [M, C, H, W]
            - rest_labels: torch.LongTensor of shape [M]
          where N = number of sampled items, M = len(dataset) - N.

    Notes:
        - Sampling is performed without replacement.
        - In class-uniform mode, this function uses `num_samples // num_classes` per class
          (same behavior as the reference function). If `num_samples` is not divisible by
          `num_classes`, the remainder is ignored.
        - All images are converted to tensors using `transforms.ToTensor()` if they aren't already
          torch tensors. Ensure all images have the same spatial size and channels so that
          `torch.stack` succeeds.
    """
    if seed is not None:
        np.random.seed(seed)

    dataset_len = len(dataset)
    if num_samples > dataset_len:
        raise ValueError("num_samples cannot exceed dataset length when sampling without replacement")

    # --- Determine sampled indices with/without class-uniform sampling ---
    if class_uniform_sample:
        if num_classes is None:
            raise ValueError("num_classes must be provided when class_uniform_sample is True")

        # Try to use a cached or helper-prepared index map; otherwise build it here.
        if not hasattr(dataset, 'class_indices'):
            # If a helper exists in the global scope, use it for parity with the reference function.
            if 'prepare_class_indices' in globals() and callable(globals()['prepare_class_indices']):
                dataset.class_indices = prepare_class_indices(dataset)
            else:
                # Build {label: np.ndarray of indices} by scanning the dataset once.
                tmp = {}
                for i in range(dataset_len):
                    _, lbl = dataset[i]
                    tmp.setdefault(lbl, []).append(i)
                dataset.class_indices = {k: np.asarray(v, dtype=np.int64) for k, v in tmp.items()}

        # Basic sanity checks
        if len(dataset.class_indices) < num_classes:
            raise ValueError(
                f"Found {len(dataset.class_indices)} classes, but num_classes={num_classes} was provided."
            )

        samples_per_class = num_samples // num_classes
        if samples_per_class == 0:
            raise ValueError(
                "num_samples is smaller than num_classes; cannot draw at least one per class with uniform sampling."
            )

        selected_chunks = []
        for lbl, indices in dataset.class_indices.items():
            if len(indices) < samples_per_class:
                raise ValueError(
                    f"Not enough items in class {lbl} to draw {samples_per_class} without replacement."
                )
            chosen = np.random.choice(indices, samples_per_class, replace=False)
            selected_chunks.append(chosen)

        selected_indices = np.concatenate(selected_chunks)
    else:
        selected_indices = np.random.choice(dataset_len, num_samples, replace=False)

    # --- Compute remainder indices ---
    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    # Keep remainder in ascending order to preserve dataset order
    rest_indices = np.setdiff1d(np.arange(dataset_len, dtype=np.int64), selected_indices, assume_unique=False)

    # --- Materialize tensors and labels for both splits ---
    to_tensor_transform = transforms.ToTensor()
    sample_imgs, sample_lbls = [], []
    rest_imgs, rest_lbls = [], []

    # Gather sampled subset
    for idx in selected_indices:
        img, lbl = dataset[int(idx)]
        if not torch.is_tensor(img):
            img = to_tensor_transform(img)
        sample_imgs.append(img)
        sample_lbls.append(lbl)

    # Gather remaining subset
    for idx in rest_indices:
        img, lbl = dataset[int(idx)]
        if not torch.is_tensor(img):
            img = to_tensor_transform(img)
        rest_imgs.append(img)
        rest_lbls.append(lbl)

    # Stack into tensors (will fail if shapes are inconsistent across items)
    sample_tensors = torch.stack(sample_imgs) if sample_imgs else torch.empty(0)
    sample_labels = torch.tensor(sample_lbls, dtype=torch.long) if sample_lbls else torch.empty(0, dtype=torch.long)

    rest_tensors = torch.stack(rest_imgs) if rest_imgs else torch.empty(0)
    rest_labels = torch.tensor(rest_lbls, dtype=torch.long) if rest_lbls else torch.empty(0, dtype=torch.long)

    return (sample_tensors, sample_labels), (rest_tensors, rest_labels)

def loader_to_numpy(loader, opt, model=None):
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

def set_loader(opt, augment_type='weak', twoviews=False):
    """
    Build dataloaders for (base, train) with optional distributed support.

    Key points:
        - Keep default collate_fn so that when `twoviews=True`, each batch yields:
            images: [ tensor(B, C, H, W), tensor(B, C, H, W) ]
            labels: tensor(B)
            This matches `images = torch.cat([images[0], images[1]], dim=0)` in your train() loop.
        - Use DistributedSampler when `opt.distributed` is True.
        - Use per-GPU batch sizes: `opt.batch_size_per_gpu` / `opt.test_batch_size_per_gpu`
            (fallback to `opt.batch_size` / `opt.test_batch_size` if fields are not set).
        - Return signatures are unchanged.
    """
    import torch
    from torch.utils.data.distributed import DistributedSampler
    from torchvision import datasets

    # ----- Dataset config (unchanged) -----
    if opt.dataset == 'cifar10' or opt.dataset == 'cifar100':
        dataset_config = datasets_setting.__dict__[opt.dataset]()
    elif opt.dataset == 'mnist' or opt.dataset == 'fashion_mnist':
        dataset_config = datasets_setting.__dict__[opt.dataset]()
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    weak_transformation = dataset_config['weak_transformation']
    strong_transformation = dataset_config['strong_transformation']
    eval_transformation  = dataset_config['eval_transformation']
    num_classes          = dataset_config['num_classes']

    # ----- Choose transform by augment_type -----
    if augment_type == 'no':
        base_transform = eval_transformation
    elif augment_type == 'weak':
        base_transform = weak_transformation
    else:
        base_transform = strong_transformation

    # For SimCLR: wrap transform to produce two views if requested
    # TwoCropTransform is already defined in utils.py and returns [view1, view2]
    transform_for_train = TwoCropTransform(base_transform) if twoviews else base_transform

    # ----- Build torchvision datasets -----
    if opt.dataset == 'cifar10':
        train_dataset   = datasets.CIFAR10(root=opt.data_folder, transform=None, train=True, download=True)
        test_dataset   = datasets.CIFAR10(root=opt.data_folder, transform=None, train=False, download=True)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder, transform=None, train=True, download=True)
        test_dataset   = datasets.CIFAR100(root=opt.data_folder, transform=None, train=False, download=True)
    elif opt.dataset == 'mnist':
        train_dataset = datasets.MNIST(root=opt.data_folder, transform=None, train=True, download=True)
        test_dataset   = datasets.MNIST(root=opt.data_folder, transform=None, train=False, download=True)
    elif opt.dataset == 'fashion_mnist':
        train_dataset = datasets.FashionMNIST(root=opt.data_folder, transform=None, train=True, download=True)
        test_dataset   = datasets.FashionMNIST(root=opt.data_folder, transform=None, train=False, download=True)
    else:
        raise ValueError(opt.dataset)

    # ----- Optional downsample for the base dataset (unchanged) -----
    if int(opt.ds_stepsize) > 1:
        train_dataset = DSCustomDataset(train_dataset, int(opt.ds_stepsize))

    if opt.num_train is None or opt.num_train > len(train_dataset):
        num_train = len(train_dataset) 
    else:
        num_train = opt.num_train
        
    # ----- Per-GPU batch size -----
    train_batch_size = opt.batch_size
    test_batch_size = opt.test_batch_size
    label_train_batch_size = int(num_train / len(train_dataset) * train_batch_size)
    unlabel_train_batch_size = train_batch_size - label_train_batch_size
    
    # ----- Distributed sampler for training dataset -----
    use_dist   = bool(getattr(opt, 'distributed', False))
    world_size = int(getattr(opt, 'world_size', 1))
    rank       = int(getattr(opt, 'rank', 0))
    
    # loader for training (need to split labeled train / unlabeled train)
    (labeled_train_data, labeled_train_labels), (unlabeled_train_data, unlabeled_train_labels) = \
        sample_and_split_dataset(train_dataset, num_train,
        class_uniform_sample=opt.class_uni_sample,
        num_classes=num_classes,
        seed=opt.seed
    )
    # label
    label_train_dataset = CustomDataset(labeled_train_data, 
                                        labeled_train_labels, 
                                        transform=transform_for_train)
    
    if opt.num_train is None:
        unlabel_train_loader = None
    else:
        unlabel_train_dataset = CustomDataset(unlabeled_train_data, 
                                            unlabeled_train_labels, 
                                            transform=transform_for_train)
        unlabel_train_sampler = DistributedSampler(unlabel_train_dataset,
                                        num_replicas=world_size,
                                        rank=rank,
                                        shuffle=True) if use_dist else None
        unlabel_train_loader = torch.utils.data.DataLoader(
            unlabel_train_dataset,
            batch_size=unlabel_train_batch_size,
            shuffle=(unlabel_train_sampler is None),
            num_workers=opt.num_workers,
            pin_memory=True,
            sampler=unlabel_train_sampler,
            drop_last=True,
            collate_fn=None
        )
    
    # ----- score training dataset loader -----
    label_train_dataset_score = DatasetWithScore(label_train_dataset, scores=None)
    train_sampler_score = DistributedSampler(label_train_dataset_score,
                                    num_replicas=world_size,
                                    rank=rank,
                                    shuffle=True) if use_dist else None
    label_train_loader_score = torch.utils.data.DataLoader(
        label_train_dataset_score,
        batch_size=label_train_batch_size,
        shuffle=(train_sampler_score is None),
        num_workers=opt.num_workers,
        pin_memory=True,
        sampler=train_sampler_score,
        drop_last=True,
        collate_fn=None
    )

    # ----- dataloader for evaluation -----
    if opt.num_base_data >= num_train:
        raise ValueError("num_base_data must be smaller than num_train")
    
    base_data, base_labels = sample_dataset(
        label_train_dataset, opt.num_base_data,
        class_uniform_sample=opt.class_uni_sample,
        num_classes=num_classes,
        seed=opt.seed
    )
    base_dataset = CustomDataset(base_data, base_labels, transform=eval_transformation)
    eval_base_loader = torch.utils.data.DataLoader(
        base_dataset,
        batch_size=len(base_dataset),
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        sampler=None
    )
    eval_test_dataset = CustomDataset(test_dataset.data, test_dataset.targets, transform=eval_transformation)
    eval_test_loader = torch.utils.data.DataLoader(
        eval_test_dataset,
        batch_size=test_batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        sampler=None
    )
    eval_train_dataset = CustomDataset(train_dataset.data, train_dataset.targets, transform=eval_transformation)
    eval_train_loader = torch.utils.data.DataLoader(
        eval_train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        sampler=None
    )
    
    return (label_train_dataset_score, label_train_loader_score, unlabel_train_loader), \
            (eval_base_loader, eval_train_loader, eval_test_loader)

# def set_loader(opt, loader_suffix='Sup', augment_type='weak', twoviews=False, p_label=False, train=True,
#                score_dataset=False):
#     if opt.dataset == 'cifar10' or opt.dataset == 'cifar100':
#         dataset_config = datasets_setting.__dict__[opt.dataset]()
#     elif opt.dataset == 'mnist' or opt.dataset == 'fashion_mnist':
#         dataset_config = datasets_setting.__dict__[opt.dataset]()
#     else:
#         raise ValueError('dataset not supported: {}'.format(opt.dataset))

#     weak_transformation = dataset_config['weak_transformation']
#     strong_transformation = dataset_config['strong_transformation']
#     eval_transformation = dataset_config['eval_transformation']
#     num_classes = dataset_config['num_classes']

#     print(f'Loader_mode: {loader_suffix}; Augment_type: {augment_type}.')

#     if augment_type == 'no':
#         transform = eval_transformation
#     elif augment_type == 'weak':
#         transform = weak_transformation
#     else:
#         transform = strong_transformation

#     # for simclr
#     if twoviews:
#         transform = TwoCropTransform(transform)

#     if opt.dataset == 'cifar10':
#         dataset = datasets.CIFAR10(root=opt.data_folder,
#                                    transform=None,
#                                    train=True,
#                                    download=True)
#         train_dataset = datasets.CIFAR10(root=opt.data_folder,
#                                          transform=transform,
#                                          train=train,
#                                          download=True)
#     elif opt.dataset == 'mnist':
#         dataset = datasets.MNIST(root=opt.data_folder,
#                                  transform=None,
#                                  train=True,
#                                  download=True)
#         train_dataset = datasets.MNIST(root=opt.data_folder,
#                                        transform=transform,
#                                        train=train,
#                                        download=True)
#     elif opt.dataset == 'fashion_mnist':
#         dataset = datasets.FashionMNIST(root=opt.data_folder,
#                                         transform=None,
#                                         train=True,
#                                         download=True)

#         train_dataset = datasets.FashionMNIST(root=opt.data_folder,
#                                               transform=transform,
#                                               train=train,
#                                               download=True)
#     else:
#         raise ValueError(opt.dataset)

#     if int(opt.ds_stepsize) > 1:
#         dataset = DSCustomDataset(dataset, int(opt.ds_stepsize))

#     if train:
#         batch_size = opt.batch_size
#     else:
#         batch_size = opt.test_batch_size

#     base_data, base_labels = sample_dataset(dataset, opt.num_base_data, class_uniform_sample=opt.class_uni_sample,
#                                             num_classes=num_classes, seed=opt.seed)
#     base_dataset = CustomDataset(base_data,
#                                  base_labels,
#                                  transform=transform)

#     base_loader = torch.utils.data.DataLoader(
#         base_dataset, batch_size=len(base_dataset), shuffle=True,
#         num_workers=opt.num_workers, pin_memory=True, sampler=None)

#     if score_dataset:
#         train_dataset_new = DatasetWithScore(train_dataset, scores=None)

#         train_loader = torch.utils.data.DataLoader(
#             train_dataset_new, batch_size=batch_size, shuffle=True,
#             num_workers=opt.num_workers, pin_memory=True, sampler=None, drop_last=True)
#         return base_loader, train_loader, train_dataset_new
#     elif p_label:
#         train_dataset_new = DatasetWithPseudoLabel(train_dataset, pred_outputs=None,
#                                                    pred_labels=None, num_classes=num_classes)

#         train_loader = torch.utils.data.DataLoader(
#             train_dataset_new, batch_size=batch_size, shuffle=True,
#             num_workers=opt.num_workers, pin_memory=True, sampler=None, drop_last=True)
#         return base_loader, train_loader, train_dataset_new
#     else:
#         train_loader = torch.utils.data.DataLoader(
#             train_dataset, batch_size=batch_size, shuffle=True,
#             num_workers=opt.num_workers, pin_memory=True, sampler=None, drop_last=True)
#         return base_loader, train_loader



# def set_loader_sup(opt, loader_mode='Sup', p_label=False):
#     '''
#     loader_mode should be chosen from ['Sup','SimCLR','SS','Eval']
#     '''
#     if opt.dataset == 'cifar10' or opt.dataset == 'cifar100':
#         dataset_config = datasets_setting.__dict__[opt.dataset]()
#     else:
#         raise ValueError('dataset not supported: {}'.format(opt.dataset))

#     weak_transformation = dataset_config['weak_transformation']
#     strong_transformation = dataset_config['strong_transformation']
#     eval_transformation = dataset_config['eval_transformation']
#     num_classes = dataset_config['num_classes']

#     if loader_mode == 'Sup':
#         print(f'Supervised augmentation: {opt.augment_type_sup}.')
#         if opt.dataset == 'cifar10':
#             dataset = datasets.CIFAR10(root=opt.data_folder,
#                                        transform=None,
#                                        train=True,
#                                        download=True)
#         else:
#             raise ValueError(opt.dataset)

#         base_data, base_labels = sample_dataset(dataset, opt.num_base_data, class_uniform_sample=opt.class_uni_sample,
#                                                 num_classes=num_classes, seed=opt.seed)
#         if opt.augment_type_sup == 'no':
#             transform = eval_transformation
#         elif opt.augment_type_sup == 'weak':
#             transform = weak_transformation
#         else:
#             transform = strong_transformation
#         if opt.sup_method == 'SupCon':
#             transform = TwoCropTransform(transform)
#         base_dataset = CustomDataset(base_data,
#                                      base_labels,
#                                      transform=transform)

#         if p_label:
#             base_dataset_new = DatasetWithPseudoLabel(base_dataset, pred_outputs=None,
#                                                       pred_labels=None, num_classes=num_classes)
#             base_loader = torch.utils.data.DataLoader(
#                 base_dataset_new, batch_size=len(base_dataset), shuffle=True,
#                 num_workers=opt.num_workers, pin_memory=True, sampler=None)
#             return base_loader, base_dataset_new
#         else:
#             base_loader = torch.utils.data.DataLoader(
#                 base_dataset, batch_size=len(base_dataset), shuffle=True,
#                 num_workers=opt.num_workers, pin_memory=True, sampler=None)
#             return base_loader
#     elif loader_mode == 'SimCLR' or loader_mode == 'SS':
#         print(f'Semi-supervised augmentation: {opt.augment_type_ss}.')
#         if opt.augment_type_ss == 'no':
#             transform = eval_transformation
#         elif opt.augment_type_ss == 'weak':
#             transform = weak_transformation
#         else:
#             transform = strong_transformation

#         # for simclr
#         if loader_mode == 'SimCLR':
#             transform = TwoCropTransform(transform)

#         if opt.dataset == 'cifar10':
#             base_dataset = datasets.CIFAR10(root=opt.data_folder,
#                                             transform=None,
#                                             train=True,
#                                             download=True)
#             train_dataset = datasets.CIFAR10(root=opt.data_folder,
#                                              transform=transform,
#                                              train=True,
#                                              download=True)
#         else:
#             raise ValueError(opt.dataset)

#         base_data, base_labels = sample_dataset(base_dataset, opt.num_base_data, class_uniform_sample=opt.class_uni_sample,
#                                                 num_classes=num_classes, seed=opt.seed)
#         base_dataset = CustomDataset(base_data,
#                                      base_labels,
#                                      transform=transform)
#         train_sampler = None
#         base_loader = torch.utils.data.DataLoader(
#             base_dataset, batch_size=len(base_dataset), shuffle=(train_sampler is None),
#             num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
#         if p_label:
#             train_dataset_new = DatasetWithPseudoLabel(base_dataset, pred_outputs=None,
#                                                       pred_labels=None, num_classes=num_classes)
#             train_loader = torch.utils.data.DataLoader(
#                 train_dataset_new, batch_size=opt.batch_size, shuffle=(train_sampler is None),
#                 num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)
#             return base_loader, train_loader, train_dataset_new
#         else:
#             train_loader = torch.utils.data.DataLoader(
#                 train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
#                 num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)
#             return base_loader, train_loader
#     elif loader_mode == 'Eval':
#         print(f'Evaluation with no augmentation.')
#         if opt.dataset == 'cifar10':
#             train_dataset = datasets.CIFAR10(root=opt.data_folder,
#                                              transform=eval_transformation,
#                                              train=True,
#                                              download=True)
#             test_dataset = datasets.CIFAR10(root=opt.data_folder,
#                                             transform=eval_transformation,
#                                             train=False,
#                                             download=True)
#         else:
#             raise ValueError(opt.dataset)
#         base_data, base_labels = sample_dataset(train_dataset, opt.num_base_data, class_uniform_sample=opt.class_uni_sample,
#                                                 num_classes=num_classes, seed=opt.seed)
#         base_dataset = CustomDataset(base_data,
#                                      base_labels,
#                                      transform=None)
#         train_sampler = None
#         base_loader = torch.utils.data.DataLoader(
#             base_dataset, batch_size=len(base_dataset), shuffle=(train_sampler is None),
#             num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler, drop_last=False)
#         test_loader = torch.utils.data.DataLoader(
#             test_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
#             num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler, drop_last=False)
#         return base_loader, test_loader
#     else:
#         raise ValueError(loader_mode)


def set_model(opt):
    import argparse
    torch.serialization.add_safe_globals([argparse.Namespace])
    
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
        is_contrastive_pretrain = getattr(opt, "pretrain_method", "") in ["SimCLR", "SupCon"] \
                                    and getattr(opt, 'distributed', False)

        model = buildnet(
            name=opt.model,
            head=opt.head_type,
            feat_dim=opt.embedding_dim,
            num_classes=num_classes,
            softmax=not opt.no_softmax,
            include_classifier=not is_contrastive_pretrain,  # <-- disable classifier for SimCLR pretrain
        )

    # model = SupConResNet(name=opt.model)
    # model = create_model(num_classes, opt)

    if opt.cp_load_path != 'no':
        model_path = opt.cp_load_path
        model_dict = torch.load(model_path, weights_only=False)
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

    if torch.cuda.is_available() and (opt.dev != 'cpu'):
        cudnn.benchmark = True

    device = torch.device('cuda', opt.local_rank) if (opt.dev != 'cpu' and torch.cuda.is_available()) else torch.device(opt.dev)
    model = model.to(device)

    if getattr(opt, 'distributed', False):
        # (Optional but recommended) convert BN to SyncBN for contrastive learning
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # Wrap whole model (not only encoder) for correct gradient sync
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank, find_unused_parameters=False)

    return model

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

        to_tensor_transform = transforms.ToTensor()  
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
    
    
### print model parameter statistics
def print_model_param_stats(model, encoder_attr_name: str = "encoder"):
    """
    Print parameter statistics for a model, its encoder (if identifiable), and the head (total - encoder).
    This function unwraps DDP if needed and mirrors the original print format.

    Args:
        model: A PyTorch model, possibly wrapped by DistributedDataParallel (DDP).
        encoder_attr_name: Preferred attribute name to locate the encoder module. Fallback is name prefix "encoder.".
    Returns:
        A dict with summarized stats for optional programmatic use.
    """
    def dedup_params(params_iter):
        """Deduplicate parameters by object identity to avoid double counting."""
        seen = set()
        unique = []
        for p in params_iter:
            pid = id(p)
            if pid not in seen:
                seen.add(pid)
                unique.append(p)
        return unique

    def param_stats(params_iter):
        """Return (total_params, trainable_params, approx_memory_MB, params_list)."""
        params = dedup_params(params_iter)
        total_params = sum(p.numel() for p in params)
        trainable_params = sum(p.numel() for p in params if p.requires_grad)
        mem_bytes = sum(p.numel() * p.element_size() for p in params) if params else 0
        mem_mb = mem_bytes / (1024 ** 2)
        return total_params, trainable_params, mem_mb, params

    # Unwrap DDP if necessary
    model_unwrapped = getattr(model, "module", model)

    # --- Overall model stats ---
    total_params, trainable_params, total_mem_mb, all_params = param_stats(model_unwrapped.parameters())
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Approx param memory: {total_mem_mb:.2f} MB")

    # --- Encoder stats (prefer attribute; fallback to name prefix) ---
    encoder_params_list = []
    enc_total = enc_trainable = enc_mem_mb = 0.0
    encoder_found = False

    encoder = getattr(model_unwrapped, encoder_attr_name, None)
    if encoder is not None:
        enc_total, enc_trainable, enc_mem_mb, encoder_params_list = param_stats(encoder.parameters())
        if enc_total > 0:
            encoder_found = True
            print(f"[encoder] Total params: {enc_total:,}")
            print(f"[encoder] Trainable params: {enc_trainable:,}")
            print(f"[encoder] Approx param memory: {enc_mem_mb:.2f} MB")
        else:
            # encoder exists but has no parameters
            print("[encoder] Found but has no parameters.")
    else:
        # Fallback by name prefix "encoder."
        named_params = list(model_unwrapped.named_parameters(recurse=True))
        encoder_params_list = [p for n, p in named_params if n.startswith(f"{encoder_attr_name}.")]
        if encoder_params_list:
            enc_total, enc_trainable, enc_mem_mb, encoder_params_list = param_stats(encoder_params_list)
            encoder_found = True
            print(f"[encoder] Total params: {enc_total:,}")
            print(f"[encoder] Trainable params: {enc_trainable:,}")
            print(f"[encoder] Approx param memory: {enc_mem_mb:.2f} MB")
        else:
            print("[encoder] Not found or has no parameters.")

    # --- Head (= total - encoder) stats ---
    head_total = head_trainable = head_mem_mb = 0.0
    if encoder_found and encoder_params_list:
        enc_ids = {id(p) for p in encoder_params_list}
        head_params = [p for p in all_params if id(p) not in enc_ids]
        head_total, head_trainable, head_mem_mb, _ = param_stats(head_params)
        print(f"[head = total - encoder] Total params: {head_total:,}")
        print(f"[head = total - encoder] Trainable params: {head_trainable:,}")
        print(f"[head = total - encoder] Approx param memory: {head_mem_mb:.2f} MB")
    else:
        print("[head = total - encoder] Cannot compute because encoder parameters were not identified.")

    # Optional structured return for logging or tests
    return {
        "total": {
            "params": int(total_params),
            "trainable": int(trainable_params),
            "mem_mb": float(total_mem_mb),
        },
        "encoder": {
            "found": bool(encoder_found),
            "params": int(enc_total),
            "trainable": int(enc_trainable),
            "mem_mb": float(enc_mem_mb),
        },
        "head": {
            "computable": bool(encoder_found and bool(encoder_params_list)),
            "params": int(head_total),
            "trainable": int(head_trainable),
            "mem_mb": float(head_mem_mb),
        },
    }

# -- print dataloader statistics --
def _safe_len(obj):
    """Return len(obj) if available; otherwise return None."""
    try:
        return len(obj)
    except TypeError:
        return None

def _dataset_len_from_loader(loader):
    """Return dataset length if loader has a map-style dataset with __len__; else None."""
    ds = getattr(loader, "dataset", None)
    if ds is None:
        return None
    try:
        return len(ds)
    except TypeError:
        return None

def print_dataset_info(name, dataset):
    """Print dataset-level info (number of samples)."""
    n = _safe_len(dataset)
    print(f"[{name}] dataset samples: {n if n is not None else 'unknown (no __len__)'}")

def print_loader_info(name, loader):
    """Print loader-level info: number of batches, dataset size, batch size, drop_last."""
    num_batches = _safe_len(loader)
    ds_len = _dataset_len_from_loader(loader)
    bs = getattr(loader, "batch_size", None)
    drop_last = getattr(loader, "drop_last", None)

    print(f"[{name}] loader type: {type(loader).__name__}")
    print(f"  batches per epoch: {num_batches if num_batches is not None else 'unknown (IterableDataset or no __len__)'}")
    print(f"  underlying dataset samples: {ds_len if ds_len is not None else 'unknown (dataset has no __len__)'}")
    if bs is not None:
        print(f"  batch_size: {bs} | drop_last: {drop_last}")