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
    
class SubsetWithTransform(Dataset):
    """Wrap an existing dataset with a fixed index list and an optional override transform.
    It defers image loading/transform until __getitem__, avoiding upfront materialization."""
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform

        # Expose targets for fast class index building if available
        base_targets = getattr(dataset, "targets", None)
        if base_targets is not None:
            if isinstance(base_targets, torch.Tensor):
                base_targets = base_targets.tolist()
            self.targets = [int(base_targets[i]) for i in self.indices]  # list[int]
        else:
            self.targets = None  # falls back to generic path in prepare_class_indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.dataset[int(self.indices[i])]
        if self.transform is not None:
            # Ensure PIL input for torchvision transforms
            x = Image.fromarray(x) if isinstance(x, np.ndarray) else (x if isinstance(x, Image.Image) else transforms.ToPILImage()(x))
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


# def sample_dataset(dataset, num_samples, class_uniform_sample=False, num_classes=None, seed=None):
#     if seed is not None:
#         np.random.seed(seed)

#     if class_uniform_sample:
#         if num_classes is None:
#             raise ValueError("num_classes must be provided when class_uniform_sample is True")

#         if not hasattr(dataset, 'class_indices'):
#             dataset.class_indices = prepare_class_indices(dataset)

#         samples_per_class = num_samples // num_classes
#         selected_indices = [np.random.choice(indices, samples_per_class, replace=False)
#                             for indices in dataset.class_indices.values()]
#         selected_indices = np.concatenate(selected_indices)
#     else:
#         selected_indices = np.random.choice(len(dataset), num_samples, replace=False)

#     to_tensor_transform = transforms.ToTensor()  
#     tensors, labels = [], []

#     for idx in selected_indices:
#         image, label = dataset[idx]
#         if not torch.is_tensor(image):
#             image = to_tensor_transform(image)
#         tensors.append(image)
#         labels.append(label)

#     return torch.stack(tensors), torch.tensor(labels)

# def sample_and_split_dataset(dataset, num_samples, class_uniform_sample=False, num_classes=None, seed=None):
#     """
#     Split a dataset into two parts:
#       1) a sampled subset (using the same sampling logic as `sample_dataset`)
#       2) the remaining subset (all items not selected in the sample)

#     Args:
#         dataset: A dataset implementing __len__ and __getitem__ -> (image, label).
#         num_samples (int): Number of samples to draw (without replacement).
#         class_uniform_sample (bool): If True, sample uniformly across classes.
#         num_classes (int or None): Total number of classes; required if class_uniform_sample is True.
#         seed (int or None): Random seed for reproducibility.

#     Returns:
#         ((sample_tensors, sample_labels), (rest_tensors, rest_labels)):
#             - sample_tensors: torch.Tensor of shape [N, C, H, W]
#             - sample_labels: torch.LongTensor of shape [N]
#             - rest_tensors: torch.Tensor of shape [M, C, H, W]
#             - rest_labels: torch.LongTensor of shape [M]
#           where N = number of sampled items, M = len(dataset) - N.

#     Notes:
#         - Sampling is performed without replacement.
#         - In class-uniform mode, this function uses `num_samples // num_classes` per class
#           (same behavior as the reference function). If `num_samples` is not divisible by
#           `num_classes`, the remainder is ignored.
#         - All images are converted to tensors using `transforms.ToTensor()` if they aren't already
#           torch tensors. Ensure all images have the same spatial size and channels so that
#           `torch.stack` succeeds.
#     """
#     if seed is not None:
#         np.random.seed(seed)

#     dataset_len = len(dataset)
#     if num_samples > dataset_len:
#         raise ValueError("num_samples cannot exceed dataset length when sampling without replacement")

#     # --- Determine sampled indices with/without class-uniform sampling ---
#     if class_uniform_sample:
#         if num_classes is None:
#             raise ValueError("num_classes must be provided when class_uniform_sample is True")

#         # Try to use a cached or helper-prepared index map; otherwise build it here.
#         if not hasattr(dataset, 'class_indices'):
#             # If a helper exists in the global scope, use it for parity with the reference function.
#             if 'prepare_class_indices' in globals() and callable(globals()['prepare_class_indices']):
#                 dataset.class_indices = prepare_class_indices(dataset)
#             else:
#                 # Build {label: np.ndarray of indices} by scanning the dataset once.
#                 tmp = {}
#                 for i in range(dataset_len):
#                     _, lbl = dataset[i]
#                     tmp.setdefault(lbl, []).append(i)
#                 dataset.class_indices = {k: np.asarray(v, dtype=np.int64) for k, v in tmp.items()}

#         # Basic sanity checks
#         if len(dataset.class_indices) < num_classes:
#             raise ValueError(
#                 f"Found {len(dataset.class_indices)} classes, but num_classes={num_classes} was provided."
#             )

#         samples_per_class = num_samples // num_classes
#         if samples_per_class == 0:
#             raise ValueError(
#                 "num_samples is smaller than num_classes; cannot draw at least one per class with uniform sampling."
#             )

#         selected_chunks = []
#         for lbl, indices in dataset.class_indices.items():
#             if len(indices) < samples_per_class:
#                 raise ValueError(
#                     f"Not enough items in class {lbl} to draw {samples_per_class} without replacement."
#                 )
#             chosen = np.random.choice(indices, samples_per_class, replace=False)
#             selected_chunks.append(chosen)

#         selected_indices = np.concatenate(selected_chunks)
#     else:
#         selected_indices = np.random.choice(dataset_len, num_samples, replace=False)

#     # --- Compute remainder indices ---
#     selected_indices = np.asarray(selected_indices, dtype=np.int64)
#     # Keep remainder in ascending order to preserve dataset order
#     rest_indices = np.setdiff1d(np.arange(dataset_len, dtype=np.int64), selected_indices, assume_unique=False)

#     # --- Materialize tensors and labels for both splits ---
#     to_tensor_transform = transforms.ToTensor()
#     sample_imgs, sample_lbls = [], []
#     rest_imgs, rest_lbls = [], []

#     # Gather sampled subset
#     for idx in selected_indices:
#         img, lbl = dataset[int(idx)]
#         if not torch.is_tensor(img):
#             img = to_tensor_transform(img)
#         sample_imgs.append(img)
#         sample_lbls.append(lbl)

#     # Gather remaining subset
#     for idx in rest_indices:
#         img, lbl = dataset[int(idx)]
#         if not torch.is_tensor(img):
#             img = to_tensor_transform(img)
#         rest_imgs.append(img)
#         rest_lbls.append(lbl)

#     # Stack into tensors (will fail if shapes are inconsistent across items)
#     sample_tensors = torch.stack(sample_imgs) if sample_imgs else torch.empty(0)
#     sample_labels = torch.tensor(sample_lbls, dtype=torch.long) if sample_lbls else torch.empty(0, dtype=torch.long)

#     rest_tensors = torch.stack(rest_imgs) if rest_imgs else torch.empty(0)
#     rest_labels = torch.tensor(rest_lbls, dtype=torch.long) if rest_lbls else torch.empty(0, dtype=torch.long)

#     return (sample_tensors, sample_labels), (rest_tensors, rest_labels)

def sample_indices(dataset, num_samples, class_uniform_sample=False, num_classes=None, seed=None):
    """Return only the selected indices; do not materialize images."""
    if seed is not None:
        np.random.seed(seed)

    if class_uniform_sample:
        if num_classes is None:
            raise ValueError("num_classes must be provided when class_uniform_sample is True")
        class_map = getattr(dataset, 'class_indices', None)
        if class_map is None:
            class_map = prepare_class_indices(dataset)  # fast-path uses .targets when available
            dataset.class_indices = class_map
        samples_per_class = num_samples // num_classes
        selected = []
        for indices in class_map.values():
            indices = np.asarray(indices, dtype=np.int64)
            if len(indices) < samples_per_class:
                raise ValueError("Not enough items for uniform sampling in one class")
            selected.append(np.random.choice(indices, samples_per_class, replace=False))
        selected_indices = np.concatenate(selected)
    else:
        selected_indices = np.random.choice(len(dataset), num_samples, replace=False)
    return selected_indices.astype(np.int64)

def sample_and_split_indices(dataset, num_samples, class_uniform_sample=False, num_classes=None, seed=None):
    """Return (selected_indices, rest_indices) without materializing pixel tensors."""
    if seed is not None:
        np.random.seed(seed)
    if num_samples > len(dataset):
        raise ValueError("num_samples cannot exceed dataset length when sampling without replacement")
    selected = sample_indices(dataset, num_samples, class_uniform_sample, num_classes, seed)
    selected = np.asarray(selected, dtype=np.int64)
    rest = np.setdiff1d(np.arange(len(dataset), dtype=np.int64), selected, assume_unique=False)
    return selected, rest

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

# def set_loader(opt, augment_type='weak', twoviews=False):
#     """
#     Build dataloaders for (base, train) with memory-safe index-based sampling.

#     Key points:
#       - Avoids materializing large tensors during sampling (no torch.stack of whole subsets).
#       - Uses index-only samplers: `sample_and_split_indices` / `sample_indices`.
#       - Wraps subsets with `SubsetWithTransform` so images are loaded/transformed lazily in __getitem__.
#       - Keeps default collate_fn so when `twoviews=True`, each batch yields:
#           images: [ tensor(B, C, H, W), tensor(B, C, H, W) ]
#           labels: tensor(B)
#         which matches `images = torch.cat([images[0], images[1]], dim=0)` in train loops.
#       - Supports (optional) DistributedSampler when `opt.distributed` is True.
#       - Return signatures are unchanged.
#     """
#     import numpy as np
#     import torch
#     from torch.utils.data.distributed import DistributedSampler
#     from torchvision import datasets

#     # ----- Dataset config -----
#     # Expect `datasets_setting` dict and its factory functions in the current module.
#     if opt.dataset in ('cifar10', 'cifar100', 'mnist', 'fashion_mnist'):
#         dataset_config = datasets_setting.__dict__[opt.dataset]()
#     else:
#         raise ValueError('dataset not supported: {}'.format(opt.dataset))

#     weak_transformation = dataset_config['weak_transformation']
#     strong_transformation = dataset_config['strong_transformation']
#     eval_transformation  = dataset_config['eval_transformation']
#     num_classes          = dataset_config['num_classes']

#     # ----- Choose transform by augment_type -----
#     if augment_type == 'no':
#         base_transform = eval_transformation
#     elif augment_type == 'weak':
#         base_transform = weak_transformation
#     else:
#         base_transform = strong_transformation

#     # For SimCLR: wrap transform to produce two views if requested
#     # TwoCropTransform is expected to be defined in utils.py and returns [view1, view2]
#     transform_for_train = TwoCropTransform(base_transform) if twoviews else base_transform

#     # ----- Build torchvision datasets (transform=None; we will apply transforms lazily) -----
#     if opt.dataset == 'cifar10':
#         train_dataset = datasets.CIFAR10(root=opt.data_folder, transform=None, train=True,  download=True)
#         test_dataset  = datasets.CIFAR10(root=opt.data_folder, transform=None, train=False, download=True)
#     elif opt.dataset == 'cifar100':
#         train_dataset = datasets.CIFAR100(root=opt.data_folder, transform=None, train=True,  download=True)
#         test_dataset  = datasets.CIFAR100(root=opt.data_folder, transform=None, train=False, download=True)
#     elif opt.dataset == 'mnist':
#         train_dataset = datasets.MNIST(root=opt.data_folder, transform=None, train=True,  download=True)
#         test_dataset  = datasets.MNIST(root=opt.data_folder, transform=None, train=False, download=True)
#     elif opt.dataset == 'fashion_mnist':
#         train_dataset = datasets.FashionMNIST(root=opt.data_folder, transform=None, train=True,  download=True)
#         test_dataset  = datasets.FashionMNIST(root=opt.data_folder, transform=None, train=False, download=True)
#     else:
#         raise ValueError(opt.dataset)

#     # ----- Optional downsample wrapper for the base dataset (unchanged behavior) -----
#     if int(opt.ds_stepsize) > 1:
#         # DSCustomDataset should act like a Dataset; SubsetWithTransform will wrap it later.
#         train_dataset = DSCustomDataset(train_dataset, int(opt.ds_stepsize))

#     # Determine number of labeled training samples
#     if opt.num_train is None or opt.num_train > len(train_dataset):
#         num_train = len(train_dataset)
#     else:
#         num_train = int(opt.num_train)

#     # ----- Per-GPU batch size -----
#     # Keep original behavior but ensure valid positive integers.
#     train_batch_size = int(opt.batch_size)
#     test_batch_size  = int(opt.test_batch_size)
#     # Proportional labeled batch size; clamp to [1, train_batch_size]
#     label_train_batch_size = max(1, min(train_batch_size, int(num_train / len(train_dataset) * train_batch_size)))
#     unlabel_train_batch_size = max(0, train_batch_size - label_train_batch_size)

#     # ----- Distributed knobs -----
#     use_dist   = bool(getattr(opt, 'distributed', False))
#     world_size = int(getattr(opt, 'world_size', 1))
#     rank       = int(getattr(opt, 'rank', 0))

#     # =========================
#     # Train split (labeled / unlabeled) — index-based sampling
#     # =========================
#     sel_idx, rest_idx = sample_and_split_indices(
#         train_dataset, num_train,
#         class_uniform_sample=opt.class_uni_sample,
#         num_classes=num_classes,
#         seed=opt.seed
#     )
#     sel_idx  = np.asarray(sel_idx,  dtype=np.int64)
#     rest_idx = np.asarray(rest_idx, dtype=np.int64)

#     # Labeled train dataset (lazy transforms; two views if requested)
#     label_train_dataset = SubsetWithTransform(
#         train_dataset, sel_idx, transform=transform_for_train
#     )

#     # Unlabeled train loader (if there are remaining indices and a positive batch size)
#     if len(rest_idx) > 0 and unlabel_train_batch_size > 0:
#         unlabel_train_dataset = SubsetWithTransform(
#             train_dataset, rest_idx, transform=transform_for_train
#         )
#         unlabel_train_sampler = DistributedSampler(unlabel_train_dataset,
#                                                 num_replicas=world_size,
#                                                 rank=rank,
#                                                 shuffle=True) if use_dist else None
#         unlabel_train_loader = torch.utils.data.DataLoader(
#             unlabel_train_dataset,
#             batch_size=unlabel_train_batch_size,
#             shuffle=(unlabel_train_sampler is None),
#             num_workers=opt.num_workers,
#             pin_memory=True,
#             sampler=unlabel_train_sampler,
#             drop_last=True,
#             collate_fn=None
#         )
#     else:
#         unlabel_train_loader = None

#     # ----- score training dataset loader (same labeled subset, but wrapped with DatasetWithScore) -----
#     label_train_dataset_score = DatasetWithScore(label_train_dataset, scores=None)
#     train_sampler_score = DistributedSampler(label_train_dataset_score,
#                                             num_replicas=world_size,
#                                             rank=rank,
#                                             shuffle=True) if use_dist else None
#     label_train_loader_score = torch.utils.data.DataLoader(
#         label_train_dataset_score,
#         batch_size=label_train_batch_size,
#         shuffle=(train_sampler_score is None),
#         num_workers=opt.num_workers,
#         pin_memory=True,
#         sampler=train_sampler_score,
#         drop_last=True,
#         collate_fn=None
#     )

#     # =========================
#     # Evaluation loaders
#     # =========================
#     if opt.num_base_data >= num_train:
#         raise ValueError("num_base_data must be smaller than num_train")

#     # Base (prototype) set is sampled from the labeled pool, with eval transforms
#     labeled_pool_for_eval = SubsetWithTransform(
#         train_dataset, sel_idx, transform=eval_transformation
#     )
#     base_rel_idx = sample_indices(
#         labeled_pool_for_eval, opt.num_base_data,
#         class_uniform_sample=opt.class_uni_sample,
#         num_classes=num_classes,
#         seed=opt.seed
#     )
#     base_rel_idx = np.asarray(base_rel_idx, dtype=np.int64)
#     base_abs_idx = sel_idx[base_rel_idx]

#     base_dataset = SubsetWithTransform(
#         train_dataset, base_abs_idx, transform=eval_transformation
#     )
#     # NOTE: If opt.num_base_data is large and you hit GPU OOM during evaluation,
#     # reduce this batch size from len(base_dataset) to a smaller value.
#     eval_base_loader = torch.utils.data.DataLoader(
#         base_dataset,
#         batch_size=len(base_dataset),
#         shuffle=True,
#         num_workers=opt.num_workers,
#         pin_memory=True,
#         sampler=None
#     )

#     # Test & train evaluation datasets as lazy full-index subsets
#     eval_test_dataset = SubsetWithTransform(
#         test_dataset,
#         indices=np.arange(len(test_dataset), dtype=np.int64),
#         transform=eval_transformation
#     )
#     eval_test_loader = torch.utils.data.DataLoader(
#         eval_test_dataset,
#         batch_size=test_batch_size,
#         shuffle=True,
#         num_workers=opt.num_workers,
#         pin_memory=True,
#         sampler=None
#     )

#     eval_train_dataset = SubsetWithTransform(
#         train_dataset,
#         indices=np.arange(len(train_dataset), dtype=np.int64),
#         transform=eval_transformation
#     )
#     eval_train_loader = torch.utils.data.DataLoader(
#         eval_train_dataset,
#         batch_size=train_batch_size,
#         shuffle=True,
#         num_workers=opt.num_workers,
#         pin_memory=True,
#         sampler=None
#     )

#     return (label_train_dataset_score, label_train_loader_score, unlabel_train_loader), \
#         (eval_base_loader, eval_train_loader, eval_test_loader)


def set_loader(opt, augment_type='weak', twoviews=False, return_full=False):
    """
    Build dataloaders for (train) with memory-safe index-based sampling.

    Changes requested:
      - Remove base/prototype loader logic entirely.
      - Replace eval_base_loader and eval_train_loader with:
          * eval_labeled_train_loader     (evaluates label_train_dataset with eval_transformation)
          * eval_unlabeled_train_loader   (evaluates unlabel_train_dataset with eval_transformation; may be None)
      - Keep eval_test_loader unchanged.
      - NEW: if `return_full=True`, also return:
          * full_train_loader        (entire training set with train-time transform)
          * full_eval_train_loader   (entire training set with eval transform)

    Key points (unchanged otherwise):
      - Avoids materializing large tensors during sampling (no torch.stack of whole subsets).
      - Uses index-only samplers: `sample_and_split_indices` / `sample_indices`.
      - Wraps subsets with `SubsetWithTransform` so images are loaded/transformed lazily in __getitem__.
      - Keeps default collate_fn so when `twoviews=True`, each batch yields:
          images: [ tensor(B, C, H, W), tensor(B, C, H, W) ]
          labels: tensor(B)
        which matches `images = torch.cat([images[0], images[1]], dim=0)` in train loops.
      - Supports (optional) DistributedSampler when `opt.distributed` is True.
    """
    import numpy as np
    import torch
    from torch.utils.data.distributed import DistributedSampler
    from torchvision import datasets

    # ----- Dataset config -----
    if opt.dataset in ('cifar10', 'cifar100', 'mnist', 'fashion_mnist', 'emnist'):
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
    transform_for_train = TwoCropTransform(base_transform) if twoviews else base_transform

    # ----- Build torchvision datasets (transform=None; transforms applied lazily) -----
    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder, transform=None, train=True,  download=True)
        test_dataset  = datasets.CIFAR10(root=opt.data_folder, transform=None, train=False, download=True)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder, transform=None, train=True,  download=True)
        test_dataset  = datasets.CIFAR100(root=opt.data_folder, transform=None, train=False, download=True)
    elif opt.dataset == 'mnist':
        train_dataset = datasets.MNIST(root=opt.data_folder, transform=None, train=True,  download=True)
        test_dataset  = datasets.MNIST(root=opt.data_folder, transform=None, train=False, download=True)
    elif opt.dataset == 'fashion_mnist':
        train_dataset = datasets.FashionMNIST(root=opt.data_folder, transform=None, train=True,  download=True)
        test_dataset  = datasets.FashionMNIST(root=opt.data_folder, transform=None, train=False, download=True)
    elif opt.dataset == 'emnist':
        train_dataset = datasets.EMNIST(root=opt.data_folder, split='balanced', transform=None, train=True,  download=True)
        test_dataset  = datasets.EMNIST(root=opt.data_folder, split='balanced', transform=None, train=False, download=True)
    else:
        raise ValueError(opt.dataset)

    # Optional downsample wrapper for the base dataset (unchanged behavior)
    if int(opt.ds_stepsize) > 1:
        train_dataset = DSCustomDataset(train_dataset, int(opt.ds_stepsize))

    # Determine number of labeled training samples
    if opt.num_train is None or opt.num_train > len(train_dataset):
        num_train = len(train_dataset)
    else:
        num_train = int(opt.num_train)

    # ----- Per-GPU batch sizes -----
    train_batch_size = int(opt.batch_size)
    test_batch_size  = int(opt.test_batch_size)
    label_train_batch_size = max(1, min(train_batch_size, int(num_train / len(train_dataset) * train_batch_size)))
    unlabel_train_batch_size = max(0, train_batch_size - label_train_batch_size)

    # ----- Distributed knobs -----
    use_dist   = bool(getattr(opt, 'distributed', False))
    world_size = int(getattr(opt, 'world_size', 1))
    rank       = int(getattr(opt, 'rank', 0))

    # =========================
    # Train split (labeled / unlabeled) — index-based sampling
    # =========================
    sel_idx, rest_idx = sample_and_split_indices(
        train_dataset, num_train,
        class_uniform_sample=opt.class_uni_sample,
        num_classes=num_classes,
        seed=opt.seed
    )
    sel_idx  = np.asarray(sel_idx,  dtype=np.int64)
    rest_idx = np.asarray(rest_idx, dtype=np.int64)

    # Labeled train dataset (lazy transforms; two views if requested)
    label_train_dataset = SubsetWithTransform(
        train_dataset, sel_idx, transform=transform_for_train
    )

    # Unlabeled train dataset/loader (may be None)
    if len(rest_idx) > 0 and unlabel_train_batch_size > 0:
        unlabel_train_dataset = SubsetWithTransform(
            train_dataset, rest_idx, transform=transform_for_train
        )
        unlabel_train_sampler = DistributedSampler(
            unlabel_train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        ) if use_dist else None
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
    else:
        unlabel_train_dataset = None
        unlabel_train_loader = None

    # ----- score training dataset loader (same labeled subset, but wrapped with DatasetWithScore) -----
    label_train_dataset_score = DatasetWithScore(label_train_dataset, scores=None)
    train_sampler_score = DistributedSampler(
        label_train_dataset_score,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    ) if use_dist else None
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

    # =========================
    # Evaluation loaders (updated)
    # =========================
    # Evaluate labeled subset with eval transforms
    eval_labeled_train_dataset = SubsetWithTransform(
        train_dataset, sel_idx, transform=eval_transformation
    )
    eval_labeled_train_loader = torch.utils.data.DataLoader(
        eval_labeled_train_dataset,
        batch_size=label_train_batch_size if label_train_batch_size > 0 else 1,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        sampler=None
    )

    # Evaluate unlabeled subset with eval transforms (None-safe)
    if unlabel_train_dataset is not None:
        # Re-wrap using eval transform on the same indices (rest_idx)
        eval_unlabeled_train_dataset = SubsetWithTransform(
            train_dataset, rest_idx, transform=eval_transformation
        )
        eval_unlabeled_train_loader = torch.utils.data.DataLoader(
            eval_unlabeled_train_dataset,
            batch_size=unlabel_train_batch_size if unlabel_train_batch_size > 0 else 1,
            shuffle=True,
            num_workers=opt.num_workers,
            pin_memory=True,
            sampler=None
        )
    else:
        eval_unlabeled_train_loader = None

    # Test evaluation dataset (full test set with eval transforms)
    eval_test_dataset = SubsetWithTransform(
        test_dataset,
        indices=np.arange(len(test_dataset), dtype=np.int64),
        transform=eval_transformation
    )
    eval_test_loader = torch.utils.data.DataLoader(
        eval_test_dataset,
        batch_size=test_batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        sampler=None
    )

    # =========================
    # Optional: full-train loaders (entire training set)
    # =========================
    full_train_loader = None
    full_eval_train_loader = None
    if return_full:
        # Full training dataset with train-time transform (lazily applied)
        full_train_dataset = SubsetWithTransform(
            train_dataset,
            indices=np.arange(len(train_dataset), dtype=np.int64),
            transform=transform_for_train
        )
        full_train_sampler = DistributedSampler(
            full_train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        ) if use_dist else None
        full_train_loader = torch.utils.data.DataLoader(
            full_train_dataset,
            batch_size=train_batch_size,
            shuffle=(full_train_sampler is None),
            num_workers=opt.num_workers,
            pin_memory=True,
            sampler=full_train_sampler,
            drop_last=True,
            collate_fn=None
        )

        # Full training dataset with eval-time transform
        full_eval_train_dataset = SubsetWithTransform(
            train_dataset,
            indices=np.arange(len(train_dataset), dtype=np.int64),
            transform=eval_transformation
        )
        full_eval_train_loader = torch.utils.data.DataLoader(
            full_eval_train_dataset,
            batch_size=train_batch_size if train_batch_size > 0 else 1,
            shuffle=True,
            num_workers=opt.num_workers,
            pin_memory=True,
            sampler=None
        )

    # =========================
    # Return
    # =========================
    train_tuple = (label_train_dataset_score, label_train_loader_score, unlabel_train_loader)
    eval_tuple  = (eval_labeled_train_loader, eval_unlabeled_train_loader, eval_test_loader)

    # Append full loaders if requested
    if return_full:
        train_tuple = train_tuple + (full_train_loader,)
        eval_tuple  = eval_tuple + (full_eval_train_loader,)

    return train_tuple, eval_tuple




def set_model(opt):
    import argparse
    torch.serialization.add_safe_globals([argparse.Namespace])
    
    # Use a whitelist to keep error messages clean and avoid long chains
    _supported = {'cifar10', 'cifar100', 'mnist', 'fashion_mnist', 'emnist'}

    if opt.dataset in _supported:
        # Call the function with the same name as the dataset in datasets_setting
        dataset_config = datasets_setting.__dict__[opt.dataset]()
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    num_classes = dataset_config.pop('num_classes')

    if opt.model == 'customCNN' and (opt.dataset == 'mnist' or opt.dataset == 'fashion_mnist'):
        model = customCNN()
    else:
        is_contrastive_pretrain = getattr(opt, "pretrain_method", "") in ["SimCLR", "SupCon", "combined"] \
                                    and getattr(opt, 'distributed', False)

        in_channel = getattr(opt, "in_channel", None)
        if in_channel is None:
            in_channel = 1 if opt.dataset in {"mnist", "fashion_mnist", "emnist"} else 3

        model = buildnet(
            name=opt.model,
            head=opt.head_type,
            feat_dim=opt.embedding_dim,
            num_classes=num_classes,
            softmax=not opt.no_softmax,
            include_classifier=not is_contrastive_pretrain,  # disable classifier for SimCLR-style pretrain
            in_channel=in_channel,                           # <-- added: pass input channels to backbone
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

def test_network(model, base_loader, test_loader, opt, predictor='GL', return_per_class=False):
    """
    Directly use the network with Laplace learning layer to do the test.

    Additions:
      - Top-k accuracy controlled by `opt.top` (default Top-1).
      - Optional per-class accuracy via `return_per_class` (default: False).
        When True, returns (overall_acc, per_class_acc) where per_class_acc is
        a NumPy array of shape (num_classes,) in percentages; classes absent in
        the test set are reported as np.nan.
    """
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    lap = LaplaceLearningSparseHard.apply
    model.eval()

    # Prepare base batch if using GL predictor
    if predictor == "GL":
        base_images, base_labels = next(iter(base_loader))
        if torch.cuda.is_available() and (opt.dev != 'cpu'):
            base_images = base_images.cuda(non_blocking=True)
            base_labels = base_labels.cuda(non_blocking=True)

        # Infer number of classes for one-hot (prefer opt.num_classes if provided)
        if hasattr(opt, "num_classes") and opt.num_classes is not None:
            num_classes_gl = int(opt.num_classes)
        else:
            num_classes_gl = int(base_labels.max().item()) + 1
        label_matrix = F.one_hot(base_labels, num_classes=num_classes_gl).float()
    elif predictor != "MLP":
        raise ValueError(predictor)

    # Top-k setting
    k = int(getattr(opt, 'top', 1))
    total_count = 0
    correct_count = 0

    # Will be initialized after first forward when num_classes is known
    per_class_correct = None
    per_class_total = None

    for idx, (images, labels) in enumerate(test_loader):
        if torch.cuda.is_available() and (opt.dev != 'cpu'):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        if predictor == "GL":
            # Concatenate base + current test images, then compute features once
            images_cat = torch.cat((base_images, images), dim=0)
            _, features = model(images_cat)
            pred_all = lap(features, label_matrix, opt.temp, opt.epsilon)

            # Keep only the predictions corresponding to the current test batch
            bsz = labels.shape[0]
            pred = pred_all[-bsz:]
        else:  # MLP
            pred, _ = model(images)

        # Determine num_classes from the model output of this batch
        num_classes = pred.shape[1]
        k_eff = max(1, min(k, num_classes))  # clamp k

        # Initialize per-class accumulators once we know num_classes
        if per_class_correct is None:
            per_class_correct = torch.zeros(num_classes, dtype=torch.long, device=labels.device)
            per_class_total = torch.zeros(num_classes, dtype=torch.long, device=labels.device)

        # Compute correctness flags for Top-1 or Top-k
        if k_eff == 1:
            pred_labels = torch.argmax(pred, dim=1)
            correct_flags = (pred_labels == labels)
        else:
            # True if the ground-truth label index is among the top-k indices
            topk_vals, topk_idx = torch.topk(pred, k=k_eff, dim=1)
            # Compare each row's label to its top-k indices
            correct_flags = (topk_idx == labels.unsqueeze(1)).any(dim=1)

        # Update overall counts
        correct_count += int(correct_flags.sum().item())
        total_count += pred.shape[0]

        # Update per-class counts
        for c in range(num_classes):
            mask_c = (labels == c)
            cnt_c = int(mask_c.sum().item())
            if cnt_c > 0:
                per_class_total[c] += cnt_c
                per_class_correct[c] += int(correct_flags[mask_c].sum().item())

    overall_acc = 100.0 * correct_count / max(1, total_count)

    # Pretty name for Top-k
    top_name = f"Top-{k if k <= num_classes else num_classes}"

    print('Test set: Accuracy for {} predictor ({}): {}/{} ({:.2f}%)\n'.format(
        predictor, top_name, correct_count, total_count, overall_acc))

    if not return_per_class:
        return overall_acc

    # Convert per-class to NumPy percentages; classes with zero samples -> NaN
    per_class_total_np = per_class_total.cpu().numpy()
    per_class_correct_np = per_class_correct.cpu().numpy()
    per_class_acc = np.full((per_class_total_np.shape[0],), np.nan, dtype=float)
    nonzero_mask = per_class_total_np > 0
    per_class_acc[nonzero_mask] = (
        100.0 * per_class_correct_np[nonzero_mask] / per_class_total_np[nonzero_mask]
    )

    return overall_acc, per_class_acc


def test_GL_NP(model, train_loader_ss, test_loader, opt, unlabel_train_loader=None, return_per_class=False):
    """
    Transform to numpy and do standard Laplace learning test.

    Additions:
      - Print counts of test/labeled/unlabeled data.
      - Support Top-k accuracy via `opt.top` (defaults to Top-1 if missing).
      - New flag `return_per_class` (default: False). If True, also return per-class accuracy
        as a NumPy array of shape (num_classes,), with percentages. Classes absent in the test
        set are reported as np.nan.

    Returns:
      - If return_per_class is False:
          acc_overall
      - If return_per_class is True:
          acc_overall, per_class_acc
    """
    import numpy as np

    model.eval()

    # Convert loaders to numpy (features + labels)
    test_data,  test_label  = loader_to_numpy(test_loader, opt, model)
    train_data, train_label = loader_to_numpy(train_loader_ss, opt, model)

    # Track counts
    test_count      = len(test_data)
    labeled_count   = len(train_data)
    unlabeled_count = 0

    # Optional unlabeled data
    if unlabel_train_loader is not None:
        unlabeled_train_data, _ = loader_to_numpy(unlabel_train_loader, opt, model)
        unlabeled_count = len(unlabeled_train_data)
        all_data = np.concatenate((train_data, unlabeled_train_data, test_data), axis=0)
    else:
        all_data = np.concatenate((train_data, test_data), axis=0)

    # Run Graph Laplacian predictor on the whole pool; only labeled targets are provided
    U = laplace(
        all_data,
        train_label,
        knn_num=50,
        epsilon=getattr(opt, 'epsilon', 0.0),
        n_classes='auto',
        tau=getattr(opt, 'tau', 1.0)
    )

    # Determine Top-k setting; default to Top-1 if opt.top is missing
    k = int(getattr(opt, 'top', 1))
    num_classes = U.shape[1]
    k = max(1, min(k, num_classes))  # clamp to [1, num_classes]

    # Slice out test logits (the last `test_count` rows correspond to test samples)
    U_test = U[-test_count:]

    # Compute predictions and overall correctness flags for Top-1 or Top-k
    if k == 1:
        pred = np.argmax(U_test, axis=1)
        correct_flags = (pred == test_label)
        correct_num = int(np.sum(correct_flags))
    else:
        kth = U_test.shape[1] - k  # index to partition at (keeps k largest in the tail)
        topk_idx = np.argpartition(U_test, kth=kth, axis=1)[:, -k:]  # shape (N_test, k)
        correct_flags = np.fromiter(
            (test_label[i] in topk_idx[i] for i in range(test_count)),
            count=test_count,
            dtype=bool
        )
        correct_num = int(np.sum(correct_flags))

    total_test_num = test_count
    acc_overall = 100.0 * correct_num / max(1, total_test_num)

    # Pretty name for Top-k
    top_name = f"Top-{k}"

    # Print detailed counts and accuracy
    print(
        'GL predictor evaluation:\n'
        f'  Test samples       : {test_count}\n'
        f'  Labeled train      : {labeled_count}\n'
        f'  Unlabeled train    : {unlabeled_count}\n'
        f'  {top_name} Accuracy : {correct_num}/{total_test_num} ({acc_overall:.2f}%)\n'
    )

    if not return_per_class:
        return acc_overall

    # Compute per-class accuracy (percent). Use np.nan for classes not present in test set.
    per_class_acc = np.full((num_classes,), np.nan, dtype=float)
    for c in range(num_classes):
        mask = (test_label == c)
        denom = int(np.sum(mask))
        if denom > 0:
            per_class_acc[c] = 100.0 * float(np.sum(correct_flags[mask])) / denom

    return acc_overall, per_class_acc




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

    def select_base_data(self, num_samples, class_uniform_sample=False, seed=None, mode='random', transform=None):
        """Return a memory-friendly, index-only subset. No pre-stacked tensors."""
        import random, numpy as np, torch

        if seed is not None:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

        # Build or reuse class->indices map using prepare_class_indices
        class_map = getattr(self, "class_indices", None)
        if not class_map:
            class_map = prepare_class_indices(self.original_dataset)
            self.class_indices = class_map

        # Flatten scores for sorting if needed
        scores = self.scores.detach().cpu().tolist() if isinstance(self.scores, torch.Tensor) else list(self.scores)

        selected = []
        if class_uniform_sample:
            per_cls = max(1, num_samples // max(1, len(class_map)))
            for _, idxs in class_map.items():
                if mode == 'score':
                    idxs = sorted(idxs, key=lambda i: scores[i], reverse=True)[:min(per_cls, len(idxs))]
                else:
                    idxs = random.sample(idxs, k=min(per_cls, len(idxs)))
                selected.extend(idxs)
            if len(selected) > num_samples:
                selected = random.sample(selected, k=num_samples)
        else:
            if mode == 'score':
                all_idx = sorted(range(len(self.original_dataset)), key=lambda i: scores[i], reverse=True)
                selected = all_idx[:num_samples]
            else:
                selected = random.sample(range(len(self.original_dataset)), k=num_samples)

        # Choose transform: prefer provided override
        if transform is None and hasattr(self.original_dataset, "transform"):
            transform = self.original_dataset.transform

        # Return an index-based lazy subset; images are loaded/transformed in __getitem__
        return SubsetWithTransform(self.original_dataset, selected, transform=transform)

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