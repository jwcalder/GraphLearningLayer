import torchvision.transforms as transforms
from .augmentations import RandAugment, GrayRandAugment
from .utils import export
import os

@export
def mnist():
    channel_stats = dict(mean=[0.1307],
                         std=[0.3081])

    weak_transformation = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomCrop(28, padding=4),
        RandAugment(1),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    strong_transformation = transforms.Compose([
        transforms.RandomRotation(20),
        transforms.RandomCrop(28, padding=4),
        RandAugment(2),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    eval_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    data_dir = 'data-local/images/mnist'

    return {
        'weak_transformation': weak_transformation,
        'strong_transformation': strong_transformation,
        'eval_transformation': eval_transformation,
        'datadir': data_dir,
        'num_classes': 10
    }

@export
def fashion_mnist():
    channel_stats = dict(mean=[0.2860],
                         std=[0.3530])

    weak_transformation = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomCrop(28, padding=4),
        RandAugment(1),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    strong_transformation = transforms.Compose([
        transforms.RandomRotation(20),
        transforms.RandomCrop(28, padding=4),
        RandAugment(2),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    eval_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    data_dir = 'data-local/images/fashion_mnist'

    return {
        'weak_transformation': weak_transformation,
        'strong_transformation': strong_transformation,
        'eval_transformation': eval_transformation,
        'datadir': data_dir,
        'num_classes': 10
    }

@export
def cifar10():
    channel_stats = dict(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470,  0.2435,  0.2616])
    
    weak_transformation = transforms.Compose([        
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4,padding_mode="reflect"),
        RandAugment(1),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])
    
    strong_transformation = transforms.Compose([        
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4,padding_mode="reflect"),
        RandAugment(2),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])
    
    eval_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])


    # myhost = os.uname()[1]
    data_dir = 'data-local/images/cifar/cifar10/by-image'

    # print("Using CIFAR-10 from", data_dir)

    return {
        'weak_transformation': weak_transformation,
        'strong_transformation': strong_transformation,
        'eval_transformation': eval_transformation,
        'datadir': data_dir,
        'num_classes': 10
    }


@export
def cifar100():
    channel_stats = dict(mean=[0.5071, 0.4867, 0.4408],
                         std=[0.2675,  0.2565,  0.2761]) 
    
    weak_transformation = transforms.Compose([        
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4,padding_mode="reflect"),
        RandAugment(1),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])
    
    strong_transformation = transforms.Compose([        
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4,padding_mode="reflect"),
        RandAugment(2),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])
    
    eval_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])


    # myhost = os.uname()[1]
    data_dir = 'data-local/images/cifar/cifar100/by-image'

    # print("Using CIFAR-100 from", data_dir)

    return {
        'weak_transformation': weak_transformation,
        'strong_transformation': strong_transformation,
        'eval_transformation': eval_transformation,
        'datadir': data_dir,
        'num_classes': 100
    }



@export
def miniimagenet():
    mean_pix = [x/255.0 for x in [120.39586422,  115.59361427, 104.54012653]]
    std_pix = [x/255.0 for x in [70.68188272,  68.27635443,  72.54505529]]

    channel_stats = dict(mean=mean_pix, std=std_pix) 

    weak_transformation = transforms.Compose([        
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(84, padding=8,padding_mode="reflect"),
        RandAugment(1),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])
    
    strong_transformation = transforms.Compose([        
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(84, padding=8,padding_mode="reflect"),
        RandAugment(2),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    eval_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    data_dir = 'data-local/images/miniimagenet'
    

    # print("Using mini-imagenet from", data_dir)


    return {
        'weak_transformation': weak_transformation,
        'strong_transformation': strong_transformation,
        'eval_transformation': eval_transformation,
        'datadir': data_dir,
        'num_classes': 100
    }


# @export
# def emnist():
#     # Use MNIST-like stats; replicate to 3 channels because we convert grayscale -> RGB-like
#     channel_stats = dict(mean=[0.1307, 0.1307, 0.1307],
#                          std=[0.3081, 0.3081, 0.3081])

#     # Convert to 3-channel PIL first so RGB-oriented augmentations (e.g., Cutout) are safe
#     weak_transformation = transforms.Compose([
#         transforms.Grayscale(num_output_channels=3),   # 1->3 channels (PIL 'RGB'-like)
#         transforms.RandomRotation(10),
#         transforms.RandomCrop(28, padding=4),
#         RandAugment(1),
#         transforms.ToTensor(),
#         transforms.Normalize(**channel_stats)
#     ])

#     strong_transformation = transforms.Compose([
#         transforms.Grayscale(num_output_channels=3),   # keep pipeline RGB-compatible
#         transforms.RandomRotation(20),
#         transforms.RandomCrop(28, padding=4),
#         RandAugment(2),
#         transforms.ToTensor(),
#         transforms.Normalize(**channel_stats)
#     ])

#     eval_transformation = transforms.Compose([
#         transforms.Grayscale(num_output_channels=3),   # ensure eval is also 3-channel
#         transforms.ToTensor(),
#         transforms.Normalize(**channel_stats)
#     ])

#     # Point to EMNIST (balanced) directory; keep the structure consistent with MNIST
#     data_dir = 'data-local/images/emnist/balanced'

#     return {
#         'weak_transformation': weak_transformation,
#         'strong_transformation': strong_transformation,
#         'eval_transformation': eval_transformation,
#         'datadir': data_dir,
#         'num_classes': 47  # EMNIST Balanced has 47 classes
#     }


@export
def emnist():
    # Grayscale statistics (EMNIST / MNIST-like)
    channel_stats = dict(mean=[0.1307], std=[0.3081])

    # NOTE:
    # - Entire pipeline stays single-channel ('L' for PIL; [1, H, W] as tensor).
    # - GrayRandAugment applies only grayscale-safe ops (no RGB conversions).
    # - For small 28x28 digits, keep geometry magnitudes modest to avoid label leakage.

    weak_transformation = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # ensure single-channel
        transforms.RandomCrop(28, padding=4),         # light spatial jitter
        GrayRandAugment(n=1, m=10, magnitude_std=0.0, cutout_fill=0),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats),
    ])

    strong_transformation = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomCrop(28, padding=4),
        transforms.RandomRotation(15),               # extra geometry on top
        GrayRandAugment(n=2, m=14, magnitude_std=3.0, cutout_fill=0),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats),
    ])

    eval_transformation = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats),
    ])

    data_dir = 'data-local/images/emnist/balanced'

    return {
        'weak_transformation': weak_transformation,
        'strong_transformation': strong_transformation,
        'eval_transformation': eval_transformation,
        'datadir': data_dir,
        'num_classes': 47  # EMNIST Balanced
    }
