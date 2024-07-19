import matplotlib.pyplot as plt
import umap
from sklearn.manifold import TSNE
import torch
import numpy as np
from networks.BuildNet import buildnet
from torchvision import transforms, datasets
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score


def umap_visualization(encoded_data, labels, data_name='', n_neighbors=20, sel_labels=None, base=None, base_labels=None,
                       save_dir=None):
    reducer = umap.UMAP(n_neighbors=n_neighbors)
    umap_embedded_data = reducer.fit_transform(encoded_data)

    plt.figure(figsize=(10, 10))

    plt.scatter(umap_embedded_data[:, 0], umap_embedded_data[:, 1], c=labels, s=.5)
    if sel_labels is not None:
        plt.scatter(umap_embedded_data[sel_labels, 0], umap_embedded_data[sel_labels, 1], c='r', edgecolors='black',
                    s=1, marker='*')
    if base is not None:
        b = reducer.transform(base)
        plt.scatter(b[:, 0], b[:, 1], c=base_labels, edgecolors='red', s=20, marker='*', linewidth=0.5)
    plt.title("UMAP Embedding of " + data_name + " Data")
    if save_dir is not None:
        plt.savefig("{}_UMAP_{}.png".format(save_dir, data_name))
    plt.close()


def tsne_visualization(encoded_data, labels, data_name='', sel_labels=None, base=None, base_labels=None, save_dir=None):
    if base is not None:
        encoded_data = np.concatenate((encoded_data, base), axis=0)
        l = base.shape[0]
    tsne_embedded_data = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(encoded_data)
    plt.figure(figsize=(10, 10))
    if base is None:
        plt.scatter(tsne_embedded_data[:, 0], tsne_embedded_data[:, 1], c=labels, s=.5)
    if sel_labels is not None:
        plt.scatter(tsne_embedded_data[sel_labels, 0], tsne_embedded_data[sel_labels, 1], c='r', edgecolors='black',
                    s=1, marker='*')
    if base is not None:
        plt.scatter(tsne_embedded_data[:-l, 0], tsne_embedded_data[:-l, 1], c=labels, s=.5)
        plt.scatter(tsne_embedded_data[-l:, 0], tsne_embedded_data[-l:, 1], c=base_labels, edgecolors='red', s=20,
                    marker='*', linewidth=0.5)
    plt.title("TSNE Embedding of " + data_name + " Data")
    if save_dir is not None:
        plt.savefig("{}_TSNE_{}.png".format(save_dir, data_name))
    plt.close()


def visualize(model_path, encoder, base=None, TSNE=True, head=False, svm=False, save_dir=None, dev="cuda", feat_dim=128, num_classes=10, head_type='no'):
    if TSNE:
        print("Use both UMAP and TSNE embedding.")
    else:
        print("Use only UMAP embedding.")

    model = buildnet(name=encoder, head=head_type,
                         feat_dim=feat_dim, num_classes=num_classes)
    # model = SupConResNet(name=encoder)
    device = torch.device("cuda")
    if torch.cuda.is_available() & (dev != 'cpu'):
        # if torch.cuda.device_count() > 1: #check for multiple GPU
        #    model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        # criterion = criterion.cuda()
        # cudnn.benchmark = True

    stuff = torch.load(model_path)
    try:
        model.load_state_dict(stuff["model"])
    except:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in stuff["model"].items():
            name = k.replace(".module", "")  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

    model.eval()

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    normalize = transforms.Normalize(mean=mean, std=std)

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    data_folder = './datasets/'
    val_dataset = datasets.CIFAR10(root=data_folder,
                                   train=True,  # This is training data - default is True
                                   transform=val_transform)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=len(val_dataset), shuffle=False,
        num_workers=8, pin_memory=True)

    data, labels = next(iter(val_loader))

    encoded_data = None
    batch_size = 256
    with torch.no_grad():
        for idx in range(0, len(data), batch_size):
            data_batch = data[idx:idx + batch_size]
            if head:
                if encoded_data is None:
                    encoded_data = model(data_batch.to(device))[1].cpu().numpy()
                else:
                    encoded_data = np.vstack((encoded_data, model(data_batch.to(device))[1].cpu().numpy()))
            else:
                if encoded_data is None:
                    encoded_data = model.encoder(data_batch.to(device))[1].cpu().numpy()
                else:
                    encoded_data = np.vstack((encoded_data, model.encoder(data_batch.to(device))[1].cpu().numpy()))


    base_data = None
    base_labels = None
    if base is not None:
        with torch.no_grad():
            for idx, (images, base_labels) in enumerate(base):
                if head:
                    if base_data is None:
                        base_data = model(images.to(device))[1].cpu().numpy()
                    else:
                        base_data = np.vstack((base_data, model(images.to(device))[1].cpu().numpy()))
                else:
                    if base_data is None:
                        base_data = model.encoder(images.to(device))[1].cpu().numpy()
                    else:
                        base_data = np.vstack((base_data, model.encoder(images.to(device))[1].cpu().numpy()))

    if head:
        d_name = "CIFAR10_head_"
    else:
        d_name = "CIFAR10_"

    umap_visualization(encoded_data, labels, data_name=d_name + "Train", base=base_data, base_labels=base_labels,
                       save_dir=save_dir)
    if TSNE:
        tsne_visualization(encoded_data, labels, data_name=d_name + "Train", base=base_data, base_labels=base_labels,
                           save_dir=save_dir)
    plt.show()

    val_dataset = datasets.CIFAR10(root=data_folder,
                                   train=False,
                                   transform=val_transform)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=len(val_dataset), shuffle=False,
        num_workers=8, pin_memory=True)

    data, labels = next(iter(val_loader))

    encoded_data = None
    batch_size = 256
    with torch.no_grad():
        for idx in range(0, len(data), batch_size):
            data_batch = data[idx:idx + batch_size]
            if head:
                if encoded_data is None:
                    encoded_data = model(data_batch.to(device))[1].cpu().numpy()
                else:
                    encoded_data = np.vstack((encoded_data, model(data_batch.to(device))[1].cpu().numpy()))
            else:
                if encoded_data is None:
                    encoded_data = model.encoder(data_batch.to(device))[1].cpu().numpy()
                else:
                    encoded_data = np.vstack((encoded_data, model.encoder(data_batch.to(device))[1].cpu().numpy()))

    umap_visualization(encoded_data, labels, data_name=d_name + "Test", save_dir=save_dir)
    if TSNE:
        tsne_visualization(encoded_data, labels, data_name=d_name + "Test", save_dir=save_dir)

    if svm:
        # Train linear classifier on training data real quick
        clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))
        clf.fit(encoded_data, labels)
        # End training

        val_dataset = datasets.CIFAR10(root=data_folder,
                                       train=False,
                                       transform=val_transform)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=len(val_dataset), shuffle=False,
            num_workers=8, pin_memory=True)

        data, labels = next(iter(val_loader))

        encoded_data = None
        batch_size = 256
        with torch.no_grad():
            for idx in range(0, len(data), batch_size):
                data_batch = data[idx:idx + batch_size]
                if head:
                    if encoded_data is None:
                        encoded_data = model(data_batch.to(device))[1].cpu().numpy()
                    else:
                        encoded_data = np.vstack((encoded_data, model(data_batch.to(device))[1].cpu().numpy()))
                else:
                    if encoded_data is None:
                        encoded_data = model.encoder(data_batch.to(device))[1].cpu().numpy()
                    else:
                        encoded_data = np.vstack((encoded_data, model.encoder(data_batch.to(device))[1].cpu().numpy()))

        y_pred = clf.predict(encoded_data)
        print('Linear Classifier Accuracy is {:.2f}%'.format(100 * accuracy_score(labels, y_pred)))

        umap_visualization(encoded_data, labels,
                           data_name=d_name + "Test LC Acc {:.1f}%".format(100 * accuracy_score(labels, y_pred)),
                           save_dir=save_dir)
        if TSNE:
            tsne_visualization(encoded_data, labels,
                               data_name=d_name + "Test LC Acc {:.1f}%".format(100 * accuracy_score(labels, y_pred)),
                               save_dir=save_dir)
        plt.show()

    # Test linear classifier real quick
