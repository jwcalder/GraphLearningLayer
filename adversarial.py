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

import sys
from torch.autograd import Variable

from networks.BuildNet import buildnet, model_dict
from networks.preact_resnet import PreActResNet18

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

        # if torch.linalg.matrix_norm(out) > 10000:
        #     print("possible exploding gradient")
        #     print("grad norm: ", np.linalg.norm(grad_output))
        #     print("w norm: ", np.linalg.norm(w))
        #     print("out norm: ", torch.linalg.norm(out))

        # test:
        # array = Luu.toarray()
        # try:
        #     chol_A = np.linalg.cholesky(array)
        #     print("successfully computed cholesky")
        # except:
        #     print('cholesky FAILED, checking Luu for NAN: ')
        #     print("Contains NAN? ", np.isnan(array).any())
        # end test

        return out, None, None, None  # G@X gives gradients for data, None, None, give gradients for label_matrix and ta

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
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)
    
    def forward(self, x):
        """
        Forward propagate the information.
        #Argument:
            x: the data to be transformed by DNN, the whole dataset, training + testing.
            target: the label of the whole dataset.
            numTrain: the number of data in the training set.
            numTest: the number of data in the testing set.
            stage_flag: our training algorithm has two stages:
                1. The stage to train the regular DNN.
                2. The stage to train the WNLL activated DNN.
            train_flag: training or testing.
                0: test process.
                1: training process.
        """
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
        """
        Forward propagate the information.
        #Argument:
            x: the data to be transformed by DNN, the whole dataset, training + testing.
            target: the label of the whole dataset.
            numTrain: the number of data in the training set.
            numTest: the number of data in the testing set.
            stage_flag: our training algorithm has two stages:
                1. The stage to train the regular DNN.
                2. The stage to train the WNLL activated DNN.
            train_flag: training or testing.
                0: test process.
                1: training process.
        """
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

# (i)FGSM attack code
def fgsm_attack(image, epsilon, data_grad, attack, min_val, max_val, alpha=0.05):
    
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    
    # Create the perturbed image by adjusting each pixel of the input image
    if attack == 'fgsm': 
        perturbed_image = image + epsilon*sign_data_grad
    elif attack == 'ifgsm':
        perturbed_image = image + alpha*sign_data_grad
        
    # Adding clipping to maintain range
    perturbed_image = torch.clamp(perturbed_image, min_val, max_val) #  -.425, 2.822 for mnist,  -.8102,2.0227 for fmnist
    
    # Return the perturbed image
    return perturbed_image
    
def test_fastgrad_attack(model, device, test_loader, dataset_train, epsilon, attack, is_gl, min_val, max_val, dataset):

    print(f"Epsilon: {epsilon}")
    
    # Accuracy counter
    correct = 0
    start_loss = 0
    end_loss = 0
    adv_examples = []

    model.eval()

    if attack == 'fgsm':
        num_iters = 1
    elif attack == 'ifgsm':
        num_iters = int(5*(epsilon/.05))
        print("Num iters ifgsm: ", num_iters) 
    else:
        raise ValueError("attack must be fgsm or ifgsm")
        
    if is_gl:
        if dataset=='mnist':
            base_label_rate = 1000
        elif dataset == 'fashionmnist':
            base_label_rate = 50
        elif dataset == 'cifar10':
            base_label_rate = 50
        base_data, base_labels = get_base_samples_new(dataset_train, rate=base_label_rate)
        base_data, base_labels = base_data.to(device), base_labels.to(device)

    # Loop over all examples in test set
    for data, target in test_loader:

        upper_bound = data+epsilon #for clamping later
        lower_bound = data-epsilon
        upper_bound, lower_bound = upper_bound.to(device), lower_bound.to(device)
        
        for i in range(num_iters):

            # Send the data and label to the device
            data, target = data.to(device), target.to(device)  

            if is_gl:     
              data_full = torch.vstack((base_data, data))
              target_full = torch.cat((base_labels, target))
      
              data_full, target_full = data_full.to(device), target_full.to(device)
              data_full.requires_grad = True
              lap = LaplaceLearningSparseHard.apply
              label_matrix = torch.nn.functional.one_hot(target_full, num_classes=10)
              k = len(base_labels)

              if dataset == 'mnist':
                  output = lap(model(data_full), label_matrix[:k,:])
    
              else:
                  _, features = model(data_full)
                  output = lap(features, label_matrix[:k,:])
                
              loss = custom_ce_loss(output, target)
            
            else: 
              data = data.clone().detach()
              data.requires_grad = True
              
              if dataset == 'mnist':
                output = model(data)
              else:
                output = model(data)[0]

              loss = F.nll_loss(output, target)

            if i==0:
                start_loss += loss.item()
                init_pred = output.max(1, keepdim=True)[1]

            # if attack == 'ifgsm':
            #     print("Iter: ", i, ", Loss: ", loss.item())
    
            # Collect ``datagrad``
            if is_gl:
              data_grad = torch.autograd.grad(loss, [data_full])[0][k:,:]
            else:
              data_grad = torch.autograd.grad(loss, [data])[0]
    
            # Call FGSM Attack
            data = fgsm_attack(data, epsilon, data_grad, attack, min_val, max_val)

            data = torch.clamp(data,lower_bound,upper_bound) # ensure attack is within epsilon

            # adv_examples.append( (target[0], init_pred[0].item(), data[0,:].squeeze().detach().cpu().numpy(), 
            #                       upper_bound[0,:].squeeze().detach().cpu().numpy()-epsilon) )

        # Re-classify the perturbed image
        if is_gl:
          data_full = torch.vstack((base_data, data))
            
          if dataset == 'mnist':
              output = lap(model(data_full), label_matrix[:k,:])
    
          else:
              _, features = model(data_full)
              output = lap(features, label_matrix[:k,:])
            
          loss = custom_ce_loss(output, target)

        else: 
          if dataset == 'mnist':
            output = model(data)
          else:
            output = model(data)[0]
          loss = F.nll_loss(output, target)
        
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        correct += final_pred.eq(target.view_as(final_pred)).sum().item()
        end_loss += loss.item()

        true_data = upper_bound-epsilon

        # cifar10 inverse normalize
        inv_normalize = transforms.Normalize(
          mean=[-0.4914/0.2470, -0.4822/0.2435, -0.4465/0.2616],
          std=[1/0.2470, 1/0.2435, 1/0.2616])
        
        for i in range(len(target)):
          if final_pred[i].item() == target[i].item():
              # Special case for saving 0 epsilon examples
              if epsilon == 0 and len(adv_examples) < 1:
                  if dataset == 'cifar10':
                    adv_ex = inv_normalize(data[i,:]).permute((1,2,0)).squeeze().detach().cpu().numpy()
                    true_ex = inv_normalize(true_data[i,:]).permute((1,2,0)).squeeze().detach().cpu().numpy()
                  else:
                    adv_ex = data[i,:].squeeze().detach().cpu().numpy()
                    true_ex = true_data[i,:].squeeze().detach().cpu().numpy()
                  adv_examples.append( (init_pred[i].item(), final_pred[i].item(), adv_ex,true_ex) )
          else:
              # Save some adv examples for visualization later
              if len(adv_examples) < 1:
                  if dataset == 'cifar10':
                    adv_ex = inv_normalize(data[i,:]).permute((1,2,0)).squeeze().detach().cpu().numpy()
                    true_ex = inv_normalize(true_data[i,:]).permute((1,2,0)).squeeze().detach().cpu().numpy()
                  else:
                    adv_ex = data[i,:].squeeze().detach().cpu().numpy()
                    true_ex = true_data[i,:].squeeze().detach().cpu().numpy()
                  adv_examples.append( (init_pred[i].item(), final_pred[i].item(), adv_ex,true_ex) )

    # Calculate final accuracy for this epsilon
    final_acc = correct / 10000
    print(f"Average starting loss: {start_loss/len(test_loader)}")
    print(f"Average ending loss: {end_loss/len(test_loader)}")
    print(" ")
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / 10000 = {final_acc}")
    print(" ")

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples

def test_cw_attack(model, device, test_loader, dataset_train, is_gl, min_val, max_val, dataset, cw_lr=0.005, iters=100, c=50):

    print("Value of c: ", c)

    model.eval()
    correct = 0
    adv_examples = []
    end_loss = 0
    total_loss1 = 0
    total_loss2 = 0

    if is_gl:
        if dataset=='mnist':
            base_label_rate = 100
        elif dataset=='fashionmnist':
            base_label_rate = 50
        elif dataset == 'cifar10':
            base_label_rate = 50
        base_data, base_labels = get_base_samples_new(dataset_train, rate=base_label_rate)
        base_data.requires_grad = False
        base_data, base_labels = base_data.to(device), base_labels.to(device)

    for data, target in test_loader:

        data, target = data.to(device), target.to(device)

        if is_gl:
          data_full = torch.vstack((base_data, data))
          target = torch.cat((base_labels, target))

          data_full, target = data_full.to(device), target.to(device)

          lap = LaplaceLearningSparseHard.apply
          label_matrix = torch.nn.functional.one_hot(target, num_classes=10)
          k = len(base_labels)

          if dataset == 'mnist':
              output = lap(model(data_full), label_matrix[:k,:])
    
          else:
              _, features = model(data_full)
              output = lap(features, label_matrix[:k,:])

        else:
          if dataset == 'mnist':
            output = model(data)
          else:
            output = model(data)[0]

        idx = torch.arange(output.shape[0]) # we need this a couple times
        init_pred = output.max(1, keepdim=False)[1]

        output[idx,init_pred] = -1000000
        next_pred = output.max(1, keepdim=False)[1] # we need this for the 2nd term of the cw loss

        # this line just ensures we start with the natural images (ie l2 loss = 0) when we compute w_tanh
        w = torch.atanh(((2/(max_val-min_val))*(data-min_val))-1).clone().detach()      
        w.requires_grad_(True) 

        optimizer = optim.Adam([w], lr=cw_lr)

        for i in range(iters):

          optimizer.zero_grad()

          # scale to be in (normalized) range of data, irrespective of previous optimization step
          w_tanh = .5*(torch.tanh(w) + 1)*(max_val-min_val) + min_val

          loss1 = torch.pow(w_tanh-data, 2).sum()
            
          if i == iters-1:
              total_loss1 += loss1.item()
              
          loss1 *= (1/len(target))

          if is_gl:
            data_full = torch.vstack((base_data, w_tanh))
            data_full = data_full.to(device)

            if dataset == 'mnist':
              output = lap(model(data_full), label_matrix[:k,:])
    
            else:
              _, features = model(data_full)
              output = lap(features, label_matrix[:k,:])
        
            loss2 = c*torch.clamp(torch.max(output,1)[0] - output[idx,next_pred],min=0).sum()
              
          else:
            # For fair comparison, we take the exp of the output of CNN in this term. This is bc GL naturally outputs probabilities,
            # whereas our CNN outputs log probabilities (log softmax).
            
            if dataset == 'mnist':
              output = model(w_tanh)
            else:
              output = model(w_tanh)[0]

            loss2 = c*torch.clamp(torch.max(torch.exp(output),1)[0] - torch.exp(output[idx,next_pred]),min=0).sum()  

          if i == iters-1:
              total_loss2 += loss2.item()
              
          loss2 *= (1/len(target))

          loss = loss1+loss2

          # print("Iter: ", i, " Loss1: ", loss1.item(), " Loss2: ", loss2.item())
          # print(" ")
          # print("Total Loss: ", loss.item())
          # print(" ")

          loss.backward()
          optimizer.step()
        
        # Re-classify the perturbed image
        w_tanh = .5*(torch.tanh(w) + 1)*(max_val-min_val) + min_val

        if is_gl:
            data_full = torch.vstack((base_data, w_tanh))
            data_full = data_full.to(device)

            if dataset == 'mnist':
              output = lap(model(data_full), label_matrix[:k,:])
    
            else:
              _, features = model(data_full)
              output = lap(features, label_matrix[:k,:])
            
            loss = custom_ce_loss(output, target[k:])
            final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += final_pred.eq(target[k:].to(device).view_as(final_pred)).sum().item()
        else:
            if dataset == 'mnist':
                output = model(w_tanh)
            else:
                output = model(w_tanh)[0]
            loss = F.nll_loss(output, target)
            final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += final_pred.eq(target.view_as(final_pred)).sum().item()

        # print("Final Loss: ", loss.item())
        end_loss += loss.item()

        # cifar10 inverse normalize
        inv_normalize = transforms.Normalize(
          mean=[-0.4914/0.2470, -0.4822/0.2435, -0.4465/0.2616],
          std=[1/0.2470, 1/0.2435, 1/0.2616])
        
        for i in range(len(final_pred)):
            if len(adv_examples) < 1:
                if final_pred[i] != init_pred[i]:
                  if dataset == 'cifar10':
                      adv_ex = inv_normalize(w_tanh[i,:]).permute((1,2,0)).squeeze().detach().cpu().numpy()
                      true_ex = inv_normalize(data[i,:]).permute((1,2,0)).squeeze().detach().cpu().numpy()
                  else:
                      adv_ex = w_tanh[i,:].squeeze().detach().cpu().numpy()
                      true_ex = data[i,:].squeeze().detach().cpu().numpy()
                  adv_examples.append( (init_pred[i].item(), final_pred[i].item(), adv_ex,true_ex) )

    print("Average Distance Squared: ", total_loss1/10000)
    print("Average Adversary's Loss: ", total_loss2/c/10000)
    
    # Calculate final accuracy for this epsilon
    final_acc = correct / 10000
    print(f"Average final loss: {end_loss/len(test_loader)}")
    print(" ")
    print(f"CW \tTest Accuracy = {correct} / 10000 = {final_acc}")
    print(" ")

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples, total_loss1/10000

attack = sys.argv[1]
if attack not in ['fgsm', 'ifgsm', 'cw']:
    raise ValueError("attack must be fgsm or ifgsm or cw")
print("Attack style: ", attack)

model_type = sys.argv[2]
if model_type not in ['gl','mlp','both']:
    raise ValueError("argument must be gl, mlp, or both")
print("Classifier: ", model_type)

rob_or_nat = sys.argv[3]
if rob_or_nat not in ['robust', 'natural']:
    raise ValueError("dataset must be mnist or fashionmnist")
print("Training paradigm: ", rob_or_nat)

dataset = sys.argv[4]
if dataset not in ['mnist', 'fashionmnist', 'cifar10']:
    raise ValueError("dataset must be mnist or fashionmnist")
print("Dataset: ", dataset)

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

    network = 'SmallCNN'
    print("Network: ", network)

    if rob_or_nat == 'natural':
        model_GL = CNN_GL().to(device)
        model_GL.load_state_dict(torch.load(f"models/{dataset}_model_GL_natural_weights.pth"))
        
        model = CNN().to(device)
        model.load_state_dict(torch.load("model_natural_weights.pth"))
        
        model_encoder = CNN_GL().to(device)
        model_encoder.load_state_dict(torch.load("model_natural_weights.pth"),strict=False)
        
    elif rob_or_nat == 'robust':
        model_GL = CNN_GL().to(device)
        model_GL.load_state_dict(torch.load(f"models/{dataset}_model_GL_pgd_0.3_5_weights.pth"))
        
        model = CNN().to(device)
        model.load_state_dict(torch.load("model_pgd_0.3_weights.pth"))
        
        model_encoder = CNN_GL().to(device)
        model_encoder.load_state_dict(torch.load("model_pgd_0.3_weights.pth"),strict=False)

    min_val = -.425 # for clamping later
    max_val = 2.822
    if attack == 'cw':
        test_batch_size = 1000 #just have to do this for memory purposes.
    else:
        test_batch_size = 10000

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

    network = 'resnet18'
    print("Network: ", network)
    
    if rob_or_nat == 'natural':
        model_GL = buildnet(name=network, head='linear', feat_dim=128, num_classes=10, softmax=True).to(device)
        model_GL.load_state_dict(torch.load(f"models/fashionmnist_model_GL_natural_weights.pth"))
        
        model = buildnet(name=network, head='linear', feat_dim=128, num_classes=10, softmax=True).to(device)
        model.load_state_dict(torch.load("fmnist_model_natural_weights.pth"))
        
    elif rob_or_nat == 'robust':
        model_GL = buildnet(name=network, head='linear', feat_dim=128, num_classes=10, softmax=True).to(device)
        model_GL.load_state_dict(torch.load("models/fashionmnist_model_GL_pgd_0.05_5_weights.pth"))
        
        model = buildnet(name=network, head='linear', feat_dim=128, num_classes=10, softmax=True).to(device)
        model.load_state_dict(torch.load("fmnist_model_pgd_0.05_5_weights.pth"))

    min_val = -.8102 # for clamping later
    max_val = 2.0227

    test_batch_size = 250

elif dataset == 'cifar10':
    # Cifar10:
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
    
    dataset_train = datasets.CIFAR10('../data', train=True, download=True,
                        transform=transform)
    dataset_test = datasets.CIFAR10('../data', train=False,
                        transform=transform)

    network = 'PreActResNet18'
    print("Network: ", network)

    if rob_or_nat == 'natural':
        model_GL = PreActResNet18().to(device)
        model_GL.load_state_dict(torch.load(f"models/{dataset}_{network}_GL_natural_weights.pth"))
        
        model = PreActResNet18().to(device)
        model.load_state_dict(torch.load(f"models/{dataset}_{network}_natural_weights.pth"))
        
    elif rob_or_nat == 'robust':
        model_GL = PreActResNet18().to(device)
        model_GL.load_state_dict(torch.load(f"models/{dataset}_{network}_GL_pgd_0.05_5_weights.pth"))
        
        model = PreActResNet18().to(device)
        model.load_state_dict(torch.load(f"models/{dataset}_{network}_pgd_0.05_5_weights.pth"))

    # these are the min and max over the pixels for the normalized images
    # we save them for clamping during adversarial training
    min_val = torch.tensor([-1.9895,-1.9803,-1.7069]).reshape((1,3,1,1)).to(device)
    max_val = torch.tensor([2.0592,2.1265,2.1159]).reshape((1,3,1,1)).to(device)

    test_batch_size = 250
    if attack == 'cw':
        test_batch_size = 200

print("Attacking in batches of ", test_batch_size)

if attack == 'fgsm':
    epsilons = [0,0.05,0.1,.2,.3,.4,.5,0.6,0.7,0.8,0.9,1] # the 0 is just to easily check the natural accuracy
elif attack == 'ifgsm':
    epsilons = [0.05,0.1,.2,.3,.4,.5,0.6,0.7,0.8,0.9,1]
elif attack == 'cw':
    c_values = [1,5,10,20,50,100,200,500,1000]

if model_type == 'gl':
    isgraph = [True]
elif model_type == 'mlp':
    isgraph = [False]
elif model_type == 'both':
    isgraph = [True, False]

for graph in isgraph:

  accuracies = []
  examples = []
  distances = []

  if graph:
    print("GRAPH Classifier")
  else:
    print("MLP/Softmax Classifier")
    
  test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=test_batch_size, num_workers=8) #osher/wang use 2000

  if attack in ['fgsm', 'ifgsm']:
    num_subfigs = 3
    # Run test for each epsilon
    for epsilon in epsilons:
        if graph:
          acc, ex = test_fastgrad_attack(model_GL, device, test_loader, dataset_train, epsilon, attack, graph, min_val, max_val, dataset)
          accuracies.append(acc)
          examples.append(ex)
        else:
          acc, ex = test_fastgrad_attack(model, device, test_loader, dataset_train, epsilon, attack, graph, min_val, max_val, dataset)
          accuracies.append(acc)
          examples.append(ex)
            
  else:
    for c in c_values:
        num_subfigs = 3
        if dataset == 'cifar10':
            cw_iters = 50
        else:
            cw_iters = 100
        print("CW Iters: ", cw_iters)
        if graph:
            acc, ex, dist = test_cw_attack(model_GL, device, test_loader, dataset_train, graph, min_val, max_val, dataset, cw_lr=0.005, c=c, iters=cw_iters)
            distances.append(dist)
            accuracies.append(acc)
            examples.append(ex)
        else:
            acc, ex, dist = test_cw_attack(model, device, test_loader, dataset_train, graph, min_val, max_val, dataset, cw_lr=0.005, c=c, iters=cw_iters)
            distances.append(dist)
            accuracies.append(acc)
            examples.append(ex)
      
  print(accuracies)
  print(distances)

  cnt = 0
  plt.figure()
  for i in range(len(examples)):
    for j in range(len(examples[i])):
        
        orig,adv,ex,true_ex = examples[i][j]
        
        cnt += 1
        plt.subplot(len(examples),num_subfigs,cnt)   
        plt.title("Original data",fontsize=6)
        plt.imshow(true_ex,interpolation='none')
        plt.colorbar(fraction=0.046, pad=0.04)
  
        cnt += 1
        plt.subplot(len(examples),num_subfigs,cnt)
        plt.title("Adversarial Attack",fontsize=6)
        plt.imshow(ex-true_ex,interpolation='none')
        plt.colorbar(fraction=0.046, pad=0.04)
        
        cnt += 1
        plt.subplot(len(examples),num_subfigs,cnt)
        plt.title(f"{orig} -> {adv}",fontsize=6)
        plt.imshow(ex,interpolation='none')
        plt.colorbar(fraction=0.046, pad=0.04)
  plt.tight_layout()

  # cnt = 0
  # plt.figure(figsize=(16,20))
  # for i in range(len(examples)): # number of epsilons
  #   for j in range(len(examples[i])): # number of adv examples saved
        
  #       orig,adv,ex,true_ex = examples[i][j]

  #       if j == 0:
  #         cnt += 1
  #         plt.subplot(len(examples[i]),3,cnt)   
  #         plt.title("Original data",fontsize=6)
  #         plt.imshow(true_ex,interpolation='none')
  #         plt.colorbar()
    
  #         cnt += 1
  #         plt.subplot(len(examples[i]),3,cnt)
  #         plt.title("Adversarial Attack",fontsize=6)
  #         plt.imshow(ex-true_ex,interpolation='none')
  #         plt.colorbar()
          
  #         cnt += 1
  #         plt.subplot(len(examples[i]),3,cnt)
  #         plt.title(f"{orig} -> {adv}",fontsize=6)
  #         plt.imshow(ex,interpolation='none')
  #         plt.colorbar()
          
  #       if j != 0:
  #         cnt += 1
  #         plt.subplot(len(examples[i]),3,cnt)   
  #         plt.title("Original data",fontsize=6)
  #         plt.imshow(examples[i][j-1][2],interpolation='none')
  #         plt.colorbar()
    
  #         cnt += 1
  #         plt.subplot(len(examples[i]),3,cnt)
  #         plt.title("Adversarial Attack",fontsize=6)
  #         plt.imshow(ex-examples[i][j-1][2],interpolation='none')
  #         plt.colorbar()
          
  #         cnt += 1
  #         plt.subplot(len(examples[i]),3,cnt)
  #         plt.title(f"{orig} -> {adv}",fontsize=6)
  #         plt.imshow(ex,interpolation='none')
  #         plt.colorbar()
        
  if graph:
      plt.savefig(f"images/Adv_Ex_GL_{rob_or_nat}_{dataset}_{network}_{attack}.png")
  else:
      plt.savefig(f"images/Adv_Ex_CNN_{rob_or_nat}_{dataset}_{network}_{attack}.png")
