# GLL: Graph Learning Layer

We provide codes for our graph learning layer (GLL). GLL is data-dependent without any trainable parameters. The forward
propagation of GLL is [graph Laplace learning](https://cdn.aaai.org/ICML/2003/ICML03-118.pdf). We have also implemented
its backpropagation.

![GLL Flowchart](GLL%20Flowchart.png)

# Get Started

To get started, make sure you are using Python 3.9 or higher. Then, install the required packages by running the
following command:

```bash
pip install -r requirements.txt
```

# Fully Supervised Training

Use `FullySup.py` for fully supervised training. Here are examples:

**MLP classifier and train from scratch for CIFAR-10 dataset**

```bash
python3 FullySup.py --model resnet18 --dataset cifar10 --plot_freq_ss 100 --sup_train_type mlp --cosine
```

**GLL classifier and train with a checkpoint for CIFAR-10 dataset**

```bash
python3 FullySup.py --model resnet18 --dataset cifar10 --plot_freq_ss 100 --cosine \
--cp_load_path checkpoints/resnet18_SimCLR_mlp.pth --sup_train_type gl
```

Note:

1. All training results are saved to the folder `save`. You can use `--print_all_parameters` to print out all
   parameters.
2. `--dataset` can be `mnist`, `fashion_mnist` or `cifar10`. For `cifar10` dataset, we recommend to use the pretrained
   checkpoint `checkpoints/resnet18_SimCLR_mlp.pth`. This checkpoint is trained by
   the [SimCLR](https://arxiv.org/abs/2002.05709) contrastive
   learning ([GitHub](https://github.com/HobbitLong/SupContrast/tree/master)). You can use `--cp_load_path no` to
   training from scratch.
3. `--model` can be the keys in `model_dict` in `networks/BuildNet.py` or `customCNN`. `customCNN` is used for detailed
   comparison between GLL and MLP classifiers for `mnist` or `fashion_mnist` dataset.
4. For the GLL training `--sup_train_type gl`, we can use `--epsilon` to choose the normalization parameter in graph
   learning. The default is `--epsilon 1`. `--epsilon auto` means to use an adaptive epsilons. The gradients are tracked
   on this adaptive part.
5. Parameters for updating the base datasets in GLL-based fully-supervised training.
    - `--gl_update_base_epochs` is the number of epochs to update the base dataset for GL.
    -  `--gl_update_base_mode` is the way to update the base dataset, can be either 'score' or 'random'. Here 'score'
       means to choose the base dataset as the most uncertain or worst predicted data points.
    -  `--gl_score_type` is the type of score, can be either 'entropy' or 'l2'.
6. For the `cifar10` dataset, a checkpoint named `checkpoints/fullysup_ckpt.pth` is the GLL-trained fully-supervised
   resnet-18 with more than 96% accuracy.

# Adversarial Training and Attacks

## Training a Model

To train a model, use the following syntax:

```bash
python3 train_and_adversarial.py {classifier_type} {training_method} {dataset}
```

- **classifier_type**: Can be ``gl``, ``mlp``, or ``both``.
    - ``gl`` uses our graph learning layer (GLL) at the end of a neural network as a classifier.
    - ``mlp`` uses a linear layer (to go from d feature dimensions to k dimensions, where k is the number of classes)
      and softmax to classify.
    - ``both`` trains a ``gl`` then a ``mlp`` model sequentially.
- **training_method**: Can be ``natural`` or ``robust``.
    - ``robust`` runs 5 iterations of PGD training.
- **dataset**: Can be ``mnist`` or ``fashionmnist``.
    - ``mnist`` will use a small CNN.
    - ``fashionmnist`` will use ResNet-18.

The output will be the weights of the trained model, saved to `/models`.

## Running Adversarial Attacks

To run attacks on trained models, use the following syntax:

```bash
python3 adversarial.py {attack} {classifier_type} {training_method} {dataset}
```

- **attack**: Can be ``fgsm``, ``ifgsm``, or ``cw``.

The other arguments are the same as above.

The output will be the accuracies, and a grid of images demonstrating successful adversarial attacks saved to `/images`.

# Using GLL in Your Own Network

GLL, as a PyTorch-based layer, has its main code in `GLL.py`. You can use this layer by importing it
with 
```bash
from GLL import LaplaceLearningSparseHard
```
To use this layer, please note the following points:

1. In addition to the usual `data = (image, label)` for training data, you will need a portion
   of `base_data = (base_image, base_label)` as "labeled nodes" for graph Laplace learning. Here, `data` is a minibatch,
   similar to training a standard CNN network. For `base_data`, you can either fix them throughout the training process
   or change them every few epochs. This changing strategy slightly improves training performance, as detailed
   in `FullySup.py`.

2. The output of your neural network `model` should be feature vectors, not predictions.

3. Use the following code to compute the loss:
    ```python
    lap = LaplaceLearningSparseHard.apply
    label_matrix = nn.functional.one_hot(base_labels, num_classes=10).float()
    images = torch.cat((base_images, images), dim=0)  
    features = model(images)
    pred = lap(features, label_matrix, tau, epsilon)
    loss = criterion(pred, labels)
    ```
   Please ensure that when combining `base_images` and `images`, `base_images` are placed first.

4. Remarks:
    - The output `pred` is only for the `images` portion, so you can directly calculate the loss between `pred` and the
      labels (training targets) corresponding to `images`.
    - Do not apply softmax to the `pred` output by GLL. Avoid using loss functions like `CrossEntropyLoss` that include
      a softmax operation.
