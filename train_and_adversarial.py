import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Subset, ConcatDataset

#newest version copied from github
import torch
import graphlearning as gl
import scipy.sparse as sparse
import numpy as np
import umap
import scipy.sparse.csgraph as csgraph
import warnings
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh

from networks.BuildNet import buildnet, model_dict
from networks.preact_resnet import PreActResNet18
import sys

class LaplaceLearningSparseHard(torch.autograd.Function):
    # Assume that the first k entries are the labelled data vectors and the rest are unlabelled

    @staticmethod
    def forward(ctx, X, label_matrix, tau=0, epsilon='auto'):
        if X.is_cuda:
            dev = 'cuda'
        else:
            dev='cpu'
        # X is the upstream feature vectors coming from a pytorch neural network
        # label_matrix
        # tau is an optional parameter that adds regularity to the expression. tau=0 indicates no regularizer
        # In this hard label problem, we are only receiving gradients for the unlabeled samples

        # W, V = knn_connected(X.detach().cpu().numpy(),k=25) #Note - similarity is euclidean. If you want angular, maybe normalize them before passing in because that's annoying
        #W, V = knn_connected(X.detach().cpu().numpy(), label_matrix.shape[0], k=25)

        W, V, mod_V, C, knn_ind = knn_sym_dist(X.detach().cpu().numpy(), k=25, epsilon=epsilon)

        L = sparse.csgraph.laplacian(W).tocsr()
        label_matrix = label_matrix.detach().cpu().numpy()

        k = label_matrix.shape[0]
        n = X.shape[0]

        # W = W[k:, k:]
        # V = V[k:, k:]
        Luu = L[k:, k:]  # Lower Right Corner - unlabelled with unlabelled
        Lul = L[k:, :k]  # Lower Left Rectangle - Labelled and Unlabelled

        # print(Luu.shape, Lul.shape, label_matrix.shape)
        # print(type(Luu))
        # print(type(Lul@label_matrix))

        m = Luu.shape[0]
        # eig = eigsh(Luu, k=1, which='SA', return_eigenvectors=False)
        # print('min eigenvalue is ', eig[0])

        Luu = Luu + sparse.spdiags(tau * np.ones(m), 0, m, m).tocsr()

        # checking for well posed
        # print(torch.ones(n-k))

        Pred = sparse.linalg.spsolve(Luu,-Lul@label_matrix) # sparse solver

        # M = Luu.diagonal()
        # M = sparse.spdiags(1 / np.sqrt(M + 1e-10), 0, m, m).tocsr()

        # Pred = sparse.linalg.spsolve(M * Luu * M, -M * Lul @ label_matrix)

        # Pred = M*Pred
        
        # Pred = stable_conjgrad(M * Luu * M,
        #                       -M * Lul @ label_matrix)  # could push things to GPU for conjgrad - label matrix should have 0's for unlabeled terms
        # Pred = M * Pred

        Pred = torch.from_numpy(Pred)

        # Save tensors for backward and return output
        ctx.W, ctx.V, ctx.Luu, ctx.label_matrix, ctx.mod_V, ctx.C, ctx.knn_ind = W, V, Luu, label_matrix, mod_V, C, knn_ind # all non-pytorch tensors
        ctx.save_for_backward(X, Pred)  # L here is sparse - Only takes pytorch tensors
        # print('endforward')

        return Pred.to(device=dev)

    @staticmethod
    def backward(ctx, grad_output):
        # print('backward1')

        # Retrieve saved tensors.
        X, Pred = ctx.saved_tensors
        W, V, Luu, label_matrix, mod_V, C, knn_ind = ctx.W, ctx.V, ctx.Luu, ctx.label_matrix, ctx.mod_V, ctx.C, ctx.knn_ind

        n = X.shape[0] # number of data points
        m = Pred.shape[0]  # number of unlabelled samples
        l = Pred.shape[1]  # number of classes/labels
        k = label_matrix.shape[0]  # number of labeled samples

        # print('backward2')
        device = grad_output.device
        grad_output = grad_output.detach().cpu().numpy()  # Should be mxl and dense
        # print(np.linalg.norm(grad_output))

        w = sparse.linalg.spsolve(Luu, grad_output) # SP SOLVE

        # m = Luu.shape[0]
        # M = Luu.diagonal()
        # M = sparse.spdiags(1 / np.sqrt(M + 1e-10), 0, m, m).tocsr()
        # w = stable_conjgrad(M * Luu * M,
        #                    M * grad_output)  # not backpropagating through conjgrad. Just ensure that the output is sparse.
        # w = M * w

        # print(np.linalg.norm(grad_output), np.linalg.norm(w))

        w = np.concatenate((np.zeros_like(label_matrix), w), axis=0)  # Pad zeros as in GL paper
        # In Soft case, Pred contained all the labels. But since in hard case, they only represent the labels for the
        # unlabelled samples. We have to append the  actual labels back onto pred so that it can continue
        # backpropagating towards the feature vectors.

        Pred = np.concatenate((label_matrix, Pred.numpy()), axis=0)

        graph = gl.graph(-V)
        for i in range(l):
            if i == 0:
                w_sp = graph.gradient(w[:, i]).transpose()
                pred_sp = graph.gradient(Pred[:, i])
                G = w_sp.multiply(pred_sp)
            else:
                w_sp = graph.gradient(w[:, i]).transpose()
                pred_sp = graph.gradient(Pred[:, i])
                G += w_sp.multiply(pred_sp)

        ##### self tuning eps grad

        if not isinstance(C, int):
          
          b = G.multiply(mod_V).dot(np.ones(n))
          
          T = sparse.csgraph.laplacian(C.multiply(b),symmetrized=True) #coo matrix

          values = T.data
          indices = np.vstack((T.row, T.col))

          i = torch.LongTensor(indices)
          v = torch.FloatTensor(values)
          shape = T.shape

          T = torch.sparse_coo_tensor(i, v, torch.Size(shape), device=device)

          extra_grad = -T@X #note the sign

        else:
          extra_grad = 0

        ##### rest of gradient

        G = G.multiply(V)

        G = sparse.csgraph.laplacian(G)
        # convert to torch and push to device since it needs to be there anyway
        values = G.data
        indices = np.vstack((G.row, G.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = G.shape

        G = torch.sparse_coo_tensor(i, v, torch.Size(shape), device=device)
        
        out = G @ X + extra_grad

        if torch.linalg.matrix_norm(out) > 10:
            print("possible exploding gradient")
            print("grad norm: ", np.linalg.norm(grad_output))
            print("w norm: ", np.linalg.norm(w))
            print("out norm: ", torch.linalg.norm(out))
            print("term 1 norm: ", torch.linalg.norm(G@X))
            print("term 2 norm: ", torch.linalg.norm(extra_grad))

        # test:
        # array = Luu.toarray()
        # try:
        #     chol_A = np.linalg.cholesky(array)
        #     print("successfully computed cholesky")
        # except:
        #     print('cholesky FAILED, checking Luu for NAN: ')
        #     print("Contains NAN? ", np.isnan(array).any())
        # end test

        return out, None, None, None  # G@X gives gradients for data, None, None, give gradients for label_matrix and tau

import scipy.sparse as sparse
def knn_sym_dist(data, k=25, epsilon='auto'):

    method = 'annoy'

    knn_ind, knn_dist = gl.weightmatrix.knnsearch(data, k, similarity='euclidean',method=method)

    # Restrict to k nearest neighbors
    n = knn_ind.shape[0]
    k = np.minimum(knn_ind.shape[1], k)
    knn_ind = knn_ind[:, :k]
    knn_dist = knn_dist[:, :k]

    # Self indices
    self_ind = np.ones((n, k)) * np.arange(n)[:, None]
    self_ind = self_ind.flatten()

    # Construct sparse matrix and convert to Compressed Sparse Row (CSR) format
    Dist = sparse.coo_matrix((knn_dist.flatten(), (self_ind, knn_ind.flatten())), shape=(n, n)).tocsr()
    Dist = Dist + Dist.T.multiply(Dist.T > Dist) - Dist.multiply(Dist.T > Dist)
    rows, cols, values = sparse.find(Dist)

    if epsilon == 'auto':

        # eps = Dist.max(axis=1).toarray().flatten() # NOTE: this is not in general = d_k(x_i) because of the symmetrization above

        # Compute epsilon = d_k(x_i)
        eps = Dist[knn_ind[:,0],[knn_ind[:,-1]]].toarray().flatten() 

        # Construct C, # C_ij = 1 when i is kth nn of j, and -1 on the diag
        # C = np.diag(-np.ones(n))
        C = np.zeros((n,n))

        C[knn_ind[:,-1],knn_ind[:,0]] = 1 

        C = sparse.csr_matrix(C)

        # Compute W, V, and mod_V
        W_values = np.exp(-4 * values * values / eps[rows] / eps[cols])
        V_values = -8 * np.exp(-4 * values * values / eps[rows] / eps[cols]) / eps[rows] / eps[cols]
        mod_V_values = values*values*V_values / (eps[rows]**2) / 2

        # Construct sparse matrix and convert to Compressed Sparse Row (CSR) format
        W = sparse.coo_matrix((W_values, (rows, cols)), shape=(n, n)).tocsr()
        V = sparse.coo_matrix((V_values, (rows, cols)), shape=(n, n)).tocsr()
        mod_V = sparse.coo_matrix((mod_V_values, (rows, cols)), shape=(n, n)).tocsr()

    else:
        eps = epsilon * np.ones(n)

        # placeholders - don't need these quantities when eps is not a function of x
        mod_V = 0
        C = 0

        # compute W and V
        W_values = np.exp(-4 * values * values / eps[rows] / eps[cols])
        V_values = -8 * np.exp(-4 * values * values / eps[rows] / eps[cols]) / eps[rows] / eps[cols]

        # Construct sparse matrix and convert to Compressed Sparse Row (CSR) format
        W = sparse.coo_matrix((W_values, (rows, cols)), shape=(n, n)).tocsr()
        V = sparse.coo_matrix((V_values, (rows, cols)), shape=(n, n)).tocsr()

    if (eps < 1e-10).any():
        warnings.warn("Epsilon in KNN is very close to zero.", UserWarning)
    eps = np.maximum(eps, 1e-6)

    return W, V, mod_V, C, knn_ind

# def stable_conjgrad(A, b, x0=None, max_iter=1e5, tol=1e-10):
#     if x0 is None:
#         x = np.zeros_like(b)
#     else:
#         x = x0

#     r = b - A @ x
#     p = r
#     rsold = np.sum(r ** 2, axis=0)

#     err = 1
#     i = 0
#     while (err > tol) and (i < max_iter):
#         i += 1
#         Ap = A @ p
#         alpha = np.zeros_like(rsold)
#         alpha[rsold > tol ** 2] = rsold[rsold > tol ** 2] / np.sum(p * Ap, axis=0)[rsold > tol ** 2]
#         x += alpha * p
#         r -= alpha * Ap
#         rsnew = np.sum(r ** 2, axis=0)
#         err = np.max(np.sqrt(rsnew))
#         beta = np.zeros_like(rsold)
#         beta[rsnew > tol ** 2] = rsnew[rsnew > tol ** 2] / rsold[rsnew > tol ** 2]
#         p = r + beta * p
#         rsold = rsnew

#     if err > tol:
#         print('max iter reached: ', i, ' iters')

#     return x

# "Small-CNN" network from Osher/Wang:

from collections import OrderedDict

class CNN(nn.Module):
    def __init__(self, drop=0.5):
        super(CNN, self).__init__()
        self.num_channels = 1
        self.num_labels = 10
        activ = nn.ReLU(True)
        
        self.feature_extractor = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(self.num_channels, 32, 3)),
            ('relu1', activ),
            ('conv2', nn.Conv2d(32, 32, 3)),
            ('relu2', activ),
            ('maxpool1', nn.MaxPool2d(2, 2)),
            ('conv3', nn.Conv2d(32, 64, 3)),
            ('relu3', activ),
            ('conv4', nn.Conv2d(64, 64, 3)),
            ('relu4', activ),
            ('maxpool2', nn.MaxPool2d(2, 2)),
        ]))
        
        self.fc1 = nn.Linear(64*4*4, 200)
        self.relu1 = activ
        self.drop = nn.Dropout(drop)
        
        self.fc = nn.Linear(200, 200)
        self.relu2 = activ
        self.linear = nn.Linear(200, self.num_labels)
        
        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # nn.init.constant(self.linear.weight, 0)
        # nn.init.constant(self.linear.bias, 0)
    
    def forward(self, x):
        '''
        x = self.conv1(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = x.view(-1, 4*4*50)
        '''
        x = self.feature_extractor(x)
        x = x.view(-1, 64*4*4)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.drop(x)
        x = self.fc(x)  

        # final steps for normal CNN. This will be replaced with graph learning in our method
        x = self.relu2(x)
        x = self.linear(x)
        output = F.log_softmax(x, dim=1)

        return output

class CNN_GL(nn.Module):
    def __init__(self, drop=0.5):
        super(CNN_GL, self).__init__()
        self.num_channels = 1
        self.num_labels = 10
        activ = nn.ReLU(True)
        
        self.feature_extractor = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(self.num_channels, 32, 3)),
            ('relu1', activ),
            ('conv2', nn.Conv2d(32, 32, 3)),
            ('relu2', activ),
            ('maxpool1', nn.MaxPool2d(2, 2)),
            ('conv3', nn.Conv2d(32, 64, 3)),
            ('relu3', activ),
            ('conv4', nn.Conv2d(64, 64, 3)),
            ('relu4', activ),
            ('maxpool2', nn.MaxPool2d(2, 2)),
        ]))
        
        self.fc1 = nn.Linear(64*4*4, 200)
        self.relu1 = activ
        self.drop = nn.Dropout(drop)
        
        self.fc = nn.Linear(200, 200)

        # replace final layer with graph learning (note we also remove the relu)
        
        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        '''
        x = self.conv1(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = x.view(-1, 4*4*50)
        '''
        x = self.feature_extractor(x)
        x = x.view(-1, 64*4*4)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.drop(x)
        x = self.fc(x)  
        
        # we normalize the output before feeding into graph learning, which we found
        # helps with vanishing gradients

        output = F.normalize(x, p=2, dim=1)

        return output

import time

def get_base_samples_new(dataset, rate=10, num_class=10, seed=None):
    # Set the random seed for reproducibility if provided
    if seed is not None:
        torch.manual_seed(seed)

    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=2)
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
    base_samples_indices = torch.cat(base_samples_indices)

    return features[base_samples_indices], labels[base_samples_indices]

def custom_ce_loss(softmax_logits, targets):
    # the input logits should sum to 1 (each row)
    # Convert targets to one-hot encoding
    batch_size, num_classes = softmax_logits.shape
    one_hot_targets = F.one_hot(targets, num_classes=num_classes).to(softmax_logits.dtype)

    # Calculate CE loss manually
    loss = -torch.sum(one_hot_targets * torch.log(softmax_logits + 1e-8)) / batch_size
    return loss

def train(model, device, train_loader, optimizer, epoch, dataset):
    if epoch == 1:
        print("natural training")
    model.train()
    total_loss = 0
    correct = 0
    i = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # smallcnn
        if dataset == 'mnist':
            output = model(data)

        # resnet
        else:
            output, _ = model(data)
        
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        loss = nn.functional.nll_loss(output, target)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        i+=1
    print("Avg Loss: ", total_loss/i)
    print("Accuracy: ", correct/(i*len(target))) 
    return total_loss/i

def test(model, device, test_loader, dataset):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            if dataset == 'mnist':
                output = model(data)

            else:
                output, _ = model(data)

            test_loss += nn.functional.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"Test loss: {test_loss}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)\n")
    return test_loss, accuracy

def train_supervised(model, device, train_loader, dataset_train, optimizer, epoch, base_sample_rate, dataset):

    if epoch == 1:
        print("natural training")
        
    model.train()
    i = 0
    acc = 0
    total_loss = 0

    base_data, base_labels = get_base_samples_new(dataset_train, rate=base_sample_rate)

    for batch_idx, (data, labels) in enumerate(train_loader):

        if base_data.dim() == 3:
          base_data = base_data[:,None,:,:]

        data = torch.vstack((base_data, data))
        target = torch.cat((base_labels, labels))

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        lap = LaplaceLearningSparseHard.apply
        label_matrix = torch.nn.functional.one_hot(target, num_classes=10)

        if dataset == 'mnist':
            features = model(data)
        else:
            _, features = model(data)
            
        output = lap(features, label_matrix[:len(base_labels),:])

        pred_labels = torch.argmax(output,dim=1)

        acc += torch.sum(pred_labels == target[len(base_labels):]) / len(target[len(base_labels):])
        i +=1

        loss = custom_ce_loss(output, target[len(base_labels):])

        loss.backward()

        optimizer.step()
        total_loss += loss.item()

    print("Train Loss: ", total_loss/i)
    print('Train Accuracy: ', acc.item()/i)

    return total_loss / i

def test_GL(model, device, test_loader, dataset_train, dataset):
    model.eval()
    test_loss = 0
    correct = 0
    i = 0

    base_data, base_labels = get_base_samples_new(dataset_train, rate=5)

    with torch.no_grad():
        for data, target in test_loader:
          data = torch.vstack((base_data, data))
          target = torch.cat((base_labels,target))

          data, target = data.to(device), target.to(device)
          lap = LaplaceLearningSparseHard.apply

          label_matrix = torch.nn.functional.one_hot(target, num_classes=10)
          k = len(base_labels)

          if dataset == 'mnist':
              features = model(data)
          else:
              _, features = model(data)
              
          output = lap(features, label_matrix[:k,:])

          pred = output.argmax(dim=1, keepdim=True)

          target = target[len(base_labels):]
          correct += pred.eq(target.view_as(pred)).sum().item()

          loss = custom_ce_loss(output, target)

          test_loss += loss.item()
          i+=1

    print("\nTest Loss: ", test_loss/i)
    print('Test set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss/i, 100. * correct / len(test_loader.dataset)

def train_pgd(model, device, train_loader, optimizer, epoch, min_val, max_val, dataset, epsilon=0.3, alpha=.01, num_iters=40):
    '''
    epsilon is the range of the random initial perturbation, alpha is the step
    size of the attack, num_iters if number of iterations of IFGSM. Defaults taken
    from Wang/Osher
    '''
    
    if epoch == 1:
        print("PGD Training")
    
    model.train()
    total_loss = 0
    i = 0
    for batch_idx, (data, target) in enumerate(train_loader):
      data = data.to(device)
        
      data_perturbed = data + torch.zeros_like(data).uniform_(-epsilon,epsilon).to(device)
        
      data_perturbed, target = data_perturbed.to(device), target.to(device)
      if dataset == 'cifar10':
          min_val, max_val = min_val.to(device), max_val.to(device) # for cifar these are tensors, so need to be on same device
        
      data_perturbed = torch.clamp(data_perturbed, min_val, max_val) # ensure input is in valid range (normalized dataset range)
      
      for j in range(num_iters):
        data_perturbed.requires_grad_()

        if dataset == 'mnist':
            output = model(data_perturbed)
        else:
            output, _ = model(data_perturbed)
        
        loss = nn.functional.nll_loss(output, target)

        grad = torch.autograd.grad(loss, [data_perturbed])[0]
        data_perturbed = data_perturbed.detach() + alpha*torch.sign(grad.detach())
        data_perturbed = torch.clamp(data_perturbed,data-epsilon,data+epsilon) # ensure attack is within epsilon
        data_perturbed = torch.clamp(data_perturbed, min_val, max_val) # ensure input is in valid range

      optimizer.zero_grad()
        
      if dataset == 'mnist':
          output = model(data_perturbed)
      else:
          output, _ = model(data_perturbed)
        
      loss = nn.functional.nll_loss(output, target)
      total_loss += loss.item()
      loss.backward()
      optimizer.step()
      i+=1

    print("Avg Loss: ", total_loss/i)
    return total_loss/i

def train_GL_pgd(model, device, train_loader, dataset_train, optimizer, epoch, base_sample_rate, min_val, max_val, dataset,
                 epsilon=0.3, alpha=.01, num_iters=40):
    '''
    epsilon is the range of the random initial perturbation, alpha is the step
    size of the attack, num_iters if number of iterations of IFGSM. Defaults taken
    from Wang/Osher
    '''
                     
    if epoch == 1:
        print("PGD Training")
                     
    model.train()
    total_loss = 0
    i = 0

    base_data, base_labels = get_base_samples_new(dataset_train, rate=base_sample_rate)
    base_data, base_labels = base_data.to(device), base_labels.to(device)
    
    for batch_idx, (train_data, train_target) in enumerate(train_loader):

      train_data, train_target = train_data.to(device), train_target.to(device)
      
      # initial perturbation (first step of PGD training):
      data_perturbed = train_data + torch.zeros_like(train_data).uniform_(-epsilon,epsilon).to(device)

      data_perturbed = data_perturbed.to(device) # redundant?

      if dataset == 'cifar10':
          min_val, max_val = min_val.to(device), max_val.to(device) # for cifar these are tensors, so need to be on same device
      
      data_perturbed = torch.clamp(data_perturbed, min_val, max_val) # ensure input is in valid range

      for j in range(num_iters):

        data = torch.vstack((base_data, data_perturbed))
        target = torch.cat((base_labels, train_target))

        data, target = data.to(device), target.to(device) 
        
        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # laplace learning set up
        lap = LaplaceLearningSparseHard.apply
        label_matrix = torch.nn.functional.one_hot(target, num_classes=10)
        k = len(base_labels)

        # pass through model, compute loss
        if dataset == 'mnist':
            features = model(data)
        else:
            _, features = model(data)
            
        output = lap(features, label_matrix[:k,:])
        loss = custom_ce_loss(output, target[len(base_labels):])

        # apply an iteration of fgsm
        grad = torch.autograd.grad(loss, [data])[0]
        grad = grad[len(base_labels):]
        data_perturbed = data_perturbed.detach() + alpha*torch.sign(grad.detach())
        data_perturbed = torch.clamp(data_perturbed,train_data-epsilon,train_data+epsilon) # ensure attack is within epsilon
        data_perturbed = torch.clamp(data_perturbed, min_val, max_val) # ensure input is in valid range

      data = torch.vstack((base_data, data_perturbed))
      data = data.to(device)
        
      # laplace learning set up
      lap = LaplaceLearningSparseHard.apply

      # pass through model, compute loss
      optimizer.zero_grad()

      if dataset == 'mnist':
          features = model(data)
      else:
          _, features = model(data)
    
      output = lap(features, label_matrix[:k,:])
      loss = custom_ce_loss(output, target[len(base_labels):])

      total_loss += loss.item()
      loss.backward()
      optimizer.step()
      i+=1

    print("Avg Loss: ", total_loss/i)
    return total_loss/i

model_type = sys.argv[1]
if model_type not in ['gl','mlp','both']:
    raise ValueError("argument must be gl, mlp, or both") # both means we will train a model with gl and a model with softmax classifiers
print("Classifier: ", model_type)

if model_type == 'gl':
    isgraph = [True]
elif model_type == 'mlp':
    isgraph = [False]
elif model_type == 'both':
    isgraph = [False, True]

rob_or_nat = sys.argv[2]
if rob_or_nat not in ['robust', 'natural']:
    raise ValueError("dataset must be mnist or fashionmnist")
print("Training method: ", rob_or_nat)

dataset = sys.argv[3]
if dataset not in ['mnist', 'fashionmnist', 'cifar10']:
    raise ValueError("dataset must be mnist, fashionmnist, or cifar10")

use_cuda = torch.cuda.is_available()
if use_cuda:
  device = torch.device("cuda")
else:
  print("no cuda available")

if dataset == 'mnist':
    # # MNIST:
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    dataset_train = datasets.MNIST('../data', train=True, download=True,
                        transform=transform)
    dataset_test = datasets.MNIST('../data', train=False,
                        transform=transform)

    # these are the min and max over the pixels for the normalized images
    # we save them for clamping during adversarial training
    min_val = -.425 
    max_val = 2.822

    pgd_epsilon = 0.3

    # mnist
    batch_size = 1000
    base_sample_rate = 10
    test_batch_size = 10000
    lr = .01
    gamma = .1
    step_size = 25
    epochs = 100

    network = 'SmallCNN'

elif dataset == 'fashionmnist':
    # FashionMNIST:
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
        ])
    
    dataset_train = datasets.FashionMNIST('../data', train=True, download=True,
                        transform=transform)
    dataset_test = datasets.FashionMNIST('../data', train=False,
                        transform=transform)

    # these are the min and max over the pixels for the normalized images
    # we save them for clamping during adversarial training
    min_val = -.8102 
    max_val = 2.0227
    
    pgd_epsilon = 0.05

    # fmnist training
    batch_size = 2000
    base_sample_rate = 20
    test_batch_size = 2000
    lr = .01
    gamma = .5
    step_size = 10
    epochs = 100

    network = 'resnet18'
    
elif dataset == 'cifar10':
    # Cifar10:
    transform=transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
    
    dataset_train = datasets.CIFAR10('../data', train=True, download=True,
                        transform=transform)
    dataset_test = datasets.CIFAR10('../data', train=False,
                        transform=transform)

    # these are the min and max over the pixels for the normalized images
    # we save them for clamping during adversarial training
    min_val = torch.tensor([-1.9895,-1.9803,-1.7068]).reshape((1,3,1,1))
    max_val = torch.tensor([2.0591,2.1265,2.1158]).reshape((1,3,1,1))
    
    pgd_epsilon = 0.05

    batch_size = 200
    base_sample_rate = 10
    test_batch_size = 200
    lr = .1
    epochs = 150

    network = 'PreActResNet18'

train_kwargs = {'batch_size': batch_size}
test_kwargs = {'batch_size': test_batch_size}

train_loader = torch.utils.data.DataLoader(dataset_train,**train_kwargs, num_workers=2,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs, num_workers=2,shuffle=False)

print("Number of training points: ", len(dataset_train))
print("Number of batches: ", len(train_loader))

print('Network: ', network)

if rob_or_nat == 'robust':
    pgd_iters = 5
    print("PGD epsilon: ", pgd_epsilon)
    print("PGD iters: ", pgd_iters)

for graph in isgraph:
  print("TRAINING")

  if graph:
    print("GRAPH Classifier")
    if dataset == 'mnist':
        print("Dataset: ", dataset)
        print("Initializing Small CNN")
        model_GL = CNN_GL().to(device)
        optimizer = optim.Adam(model_GL.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        
    elif dataset == 'fashionmnist':
        print("Dataset: ", dataset)
        print("Initializing ResNet")
        model_GL = buildnet(name=network, head='linear', feat_dim=128, num_classes=10, softmax=True).to(device)
        optimizer = optim.Adam(model_GL.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        
    elif dataset == 'cifar10':
        print("Dataset: ", dataset)
        print("Initializing PreAct ResNet")
        model_GL = PreActResNet18().to(device)
        optimizer = optim.SGD(model_GL.parameters(), lr=lr,
                      momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)
    else:
        raise ValueError("Dataset not supported")
      
  else:
    print("MLP/Softmax Classifier")
    if dataset == 'mnist':
        print("Dataset: ", dataset)
        print("Initializing Small CNN")   
        model = CNN().to(device)
        #optimizer = optim.Adam(model.parameters(), lr=lr)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4) #osher/wang settings
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        
    elif dataset == 'fashionmnist':
        print("Dataset: ", dataset)
        print("Initializing ResNet")  
        model = buildnet(name=network, head='linear', feat_dim=128, num_classes=10, softmax=True).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    elif dataset == 'cifar10':
        print("Dataset: ", dataset)
        print("Initializing PreAct ResNet")
        model = PreActResNet18().to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr,
                      momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)   
    else:
        raise ValueError("Dataset not supported")

  if graph:
    GL_train_loss_vec = []
    GL_test_loss_vec = []
    GL_test_acc_vec = []
  else:
    MLP_train_loss_vec = []
    MLP_test_loss_vec = []
    MLP_test_acc_vec = []

  for epoch in range(1,epochs+1):
      print(f"Epoch {epoch}")
      if graph:
        if rob_or_nat == 'natural':
            avg_loss = train_supervised(model_GL, device, train_loader, dataset_train, optimizer, epoch, base_sample_rate, dataset)
        elif rob_or_nat == 'robust':
            avg_loss = train_GL_pgd(model_GL, device, train_loader, dataset_train, optimizer, epoch, base_sample_rate, min_val, max_val, dataset, epsilon = pgd_epsilon, num_iters=pgd_iters)
        test_loss, test_acc = test_GL(model_GL, device, test_loader, dataset_train, dataset)       
        GL_train_loss_vec.append(avg_loss)
        GL_test_loss_vec.append(test_loss)
        GL_test_acc_vec.append(test_acc)
      else:
        if rob_or_nat == 'natural':
            avg_loss = train(model, device, train_loader, optimizer, epoch, dataset)
        elif rob_or_nat == 'robust':
            avg_loss = train_pgd(model, device, train_loader, optimizer, epoch, min_val, max_val, dataset, epsilon = pgd_epsilon, num_iters=pgd_iters)
        test_loss, test_acc = test(model, device, test_loader, dataset)
        MLP_train_loss_vec.append(avg_loss)
        # MLP_test_loss_vec.append(test_loss)
        # MLP_test_acc_vec.append(test_acc)

      scheduler.step()
      
  if graph:
      if rob_or_nat == 'natural':
          torch.save(model_GL.state_dict(), f"models/{dataset}_{network}_GL_natural_weights.pth")
      elif rob_or_nat == 'robust':
          torch.save(model_GL.state_dict(), f"models/{dataset}_{network}_GL_pgd_{pgd_epsilon}_{pgd_iters}_weights.pth")
      
  else:
      if rob_or_nat == 'natural':
          torch.save(model.state_dict(), f"models/{dataset}_{network}_natural_weights.pth")
      elif rob_or_nat == 'robust':
          torch.save(model.state_dict(), f"models/{dataset}_{network}_pgd_{pgd_epsilon}_{pgd_iters}_weights.pth")
