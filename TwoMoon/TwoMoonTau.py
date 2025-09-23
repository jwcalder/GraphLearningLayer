from HardGL import LaplaceLearningSparseHard
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import graphlearning as gl
import os as os
from util import AverageMeter
import argparse

def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    # tau
    parser.add_argument('--tau', type=float, default=0,
                        help='tau for regularization function')
    opt = parser.parse_args()
    return opt

# Define the MLP model
class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        return self.fc3(x)

def ce_loss(softmax_logits, targets):
    # the input logits should sum to 1 (each row)
    # Convert targets to one-hot encoding
    batch_size, num_classes = softmax_logits.shape
    one_hot_targets = F.one_hot(targets, num_classes=num_classes).to(softmax_logits.dtype)

    # Calculate CE loss manually
    loss = -torch.sum(one_hot_targets * torch.log(softmax_logits + 1e-8)) / batch_size
    return loss


def save_plot(encoded_data, y, base_labels, pred, X, fname, tau):
  encoded_data = model(X).detach().cpu().numpy()

  cmap = plt.cm.get_cmap('bwr')

  plt.scatter(encoded_data[:,0],encoded_data[:,1], c = y, cmap=cmap, alpha=0.75)
  plt.scatter(encoded_data[:10,0],encoded_data[:10,1], c = 'w', marker='*', s=800, zorder=3)
  plt.scatter(encoded_data[:10,0],encoded_data[:10,1], c = 'k', marker='*', s=600, zorder=4)
  plt.scatter(encoded_data[:10,0],encoded_data[:10,1], c = y[:10], cmap=cmap, marker='*', s=150, zorder=5)

  fname1 = 'save/tau_new/{}/{}.png'.format(tau,fname+'_Emb')
  plt.axis('off')
  plt.savefig(fname1)
  plt.close()

def save_loss(loss_list, tau):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_list, label='Train Loss GL')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('save/tau_new/{}/train_loss_plot.png'.format(tau))
    plt.close()



if __name__ == '__main__':
    opt = parse_option()
    seed=3
    torch.manual_seed(seed)

    # Set model parameters
    input_size = 2
    hidden_size = 64
    output_size = 2

    # Instantiate the model, loss function, and optimizer
    model = MLPClassifier(input_size, hidden_size, output_size)
    #criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)


    X,y = make_moons(n_samples=500, noise = 0.1, random_state=1)
    X, y = torch.tensor(X).float(), torch.tensor(y)

    # Training the model
    num_epochs = 150000
    base_idx = np.arange(10)
    lap = LaplaceLearningSparseHard.apply
    plot_list = [1,500,1000,2000,4000,10000,20000,30000,50000,75000,100000,125000,150000]
    loss_list = []

    save_folder = 'save/tau_new/{}'.format(opt.tau)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for epoch in np.arange(num_epochs)+1:
        losses_gl = AverageMeter()
        # Forward pass
        outputs = model(X)

        base_labels = y[:10]
        train_labels = y[10:]

        labels_tensor = nn.functional.one_hot(base_labels, num_classes=2).float()
        #print(labels_tensor.shape)
        pred = lap(outputs, labels_tensor, opt.tau, 1)
        loss = ce_loss(pred,train_labels)
        losses_gl.update(loss.item(), 500)
        loss_list.append(losses_gl.avg)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch) in plot_list:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
            encoded_data = model(X).detach().cpu().numpy()
            fname = '{}'.format(epoch)
            save_plot(encoded_data, y, base_labels, pred, X, fname, opt.tau)
    save_loss(loss_list, opt.tau)
            