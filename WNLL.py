import scipy.sparse
import scipy.sparse.linalg
import numpy as np
import copy
import torch
from torch.autograd import Variable
import sys
sys.path.insert(0, "/GraphLearningLayer/pyflann")
from pyflann import *
# from pyflann.pyflann.index import FLANN


def weight_ann(data, num_s=15, num_s_normal=8):
    """
    This function is used to compute the weight matrix.
    Input: data.
           num_s: number of nearest neighbors used.
           num_s_normal: the index of the distant used to normalize the data.
    Output: sparse weight matrix.
    """
    [m, n] = data.shape
    fea = data.T

    # Build a KD tree for KNN search
    flann = FLANN()
    idx, dist = flann.nn(
                         fea, fea, num_s, algorithm="kmeans",
                         branching=32, iterations=500, checks=512
                        )

    # Construct the sparse matrix
    row_sigma = range(n)
    col_sigma = range(n)
    diag_sigma = 1./(dist[:, num_s_normal-1] + 1.e-10)
    sigma = scipy.sparse.coo_matrix((diag_sigma, (row_sigma, col_sigma))).tocsc()

    dist_scipy = scipy.sparse.coo_matrix(dist.T).tocsc()
    tmp = -((dist_scipy*sigma).power(2))
    #tmp = -dist_scipy*sigma

    [m1, n1] = tmp.shape
    row_w = []; col_w = []; val_w = []

    item = tmp.nonzero()
    row_w = list(item[0]); col_w = list(item[1])
    for iter1 in range(len(row_w)):
         val_w.append(np.exp(tmp[row_w[iter1], col_w[iter1]]))

    w = scipy.sparse.coo_matrix((val_w, (row_w, col_w))).tocsc()
    id_row = np.matlib.repmat(np.arange(n), num_s, 1)
    id_col = idx.T

    size1 = max(m1, n1)
    m2, n2 = id_row.shape

    id_row = np.array(id_row)
    id_col = np.array(id_col)
    w = np.array(w.todense())
    # print('w shape: ', w.shape)
    id_row_vector = list(np.reshape(id_row, m2*n2))
    id_col_vector = list(np.reshape(id_col, m2*n2))
    w_vector = list(np.reshape(w, m2*n2))
    for i in range(n):
         id_row_vector.append(i)
         id_col_vector.append(i)
         w_vector.append(1.0)

    y = scipy.sparse.coo_matrix((w_vector, (id_row_vector, id_col_vector))).tocsc()
    return y


def weight_GL(W, g, id_o, id_c, flag):
    """
    This function is used to find the ID of an instance by WNLL.
    W: weight matrix.
    g: labeled value.
    id_o: index of the labeled points.
    id_c: index of the unlabeled points.
    flag: 1: WNLL; 0: GL.
    """
    n, m = W.shape
    uf = np.zeros((n,))
    uf[id_o] = g
    u = uf[id_c]

    W_Laplace_full = W + W.T
    if flag == 0:
        gamma =0            #GL
    else:
        gamma = n/len(id_o) #WNLL

    W_Laplace = W_Laplace_full[id_c, :]
    W_Laplace = W_Laplace[:, id_c]

    W_Ls = W[id_o, :]
    W_Ls = W_Ls[:, id_c]
    W_Ls = W_Ls.T

    W_Rs = W_Laplace_full[id_c, :]
    W_Rs = W_Rs[:, id_o]
    W_RHs = (W_Rs + gamma*W_Ls)
    rhs = W_RHs * g

    tmpMat1 = W_Laplace_full[id_c, :]
    sumvec1 = tmpMat1.sum(axis=1)
    sumvec2 = W_Ls.sum(axis=1)
    sumvec2 = sumvec2 * gamma
    sumvec3 = sumvec1 + sumvec2

    sumvec = np.zeros((np.prod(sumvec3.shape),))
    for idx in range(len(sumvec3)):
        sumvec[idx] = float(sumvec3[idx].item())

    Mat1 = scipy.sparse.diags(sumvec).tocsc()
    coef_mat = Mat1 - W_Laplace

    #u = scipy.sparse.linalg.spsolve(coef_mat, rhs)
    u = scipy.sparse.linalg.bicgstab(coef_mat, rhs, atol=1e-7, maxiter=int(1e5))

    #print('Shape of u: ', u[0].shape)
    uf[id_c] = u[0]
    return uf

def WNLL(x, target, numTrain, train_flag=1):
        """
        WNLL Interpolation
        # Argument:
            x: the entire data to be transformed by the DNN.
            target: the label of the entire data, x.
            numTrain: the number of data in the training set.
            train_flag: 1 for training, 0 for testing
        """
        # xdata: numpy array representation of the whole data
        # x_whole: the whole data (features)
        # x_Unknown: the features of the unknown instances
        xdata = x.clone().cpu().data.numpy()
        x_whole = copy.deepcopy(xdata)

        # targetdata: numpy array representation of the label for the whole data
        targetdata = target.clone().cpu().data.numpy()

        # f: total number of instances.
        # dim: the total number of classes.
        f, dim = int(xdata.shape[0]), int(np.max(targetdata)+1)

        Predict = np.zeros((f, dim))

        #----------------------------------------------------------------------
        # Perform the nearest neighbor search and solve WNLL to find the predicted
        #labels: Predict
        #----------------------------------------------------------------------
        num_classes = dim
        k = num_classes
        ndim = xdata.shape[1]
        idx_fidelity = range(numTrain)

        fidelity = np.asarray([idx_fidelity, targetdata[idx_fidelity]]).T

        # Compute the similarity matrix, exp(dist(kNN)).
        # Use num_s nearest neighbors, with the num_s_normal-th neighbor
        #to normalize the weights.
        W = weight_ann(x_whole.T, num_s=15, num_s_normal=8)

        # Solve the graph Laplacian to get the prior label for each class.
        for i in range(k):
            g = np.zeros((fidelity.shape[0],))
            tmp = fidelity[:, 1]
            tmp = tmp - i*np.ones((fidelity.shape[0],))
            subset1 = np.where(tmp == 0)[0]
            g[subset1] = 1
            idx_fidelity = fidelity[:, 0]
            total = range(0, f)
            idx_diff = [x1 for x1 in total if x1 not in idx_fidelity]
            # Convert idx_fidelity, idx_diff to integer.
            idx_fidelity = list(map(int, idx_fidelity))
            idx_diff = list(map(int, idx_diff))
            tmp = weight_GL(W, g, idx_fidelity, idx_diff, 1)
            # Assign the estimated prior label for ith class.
            Predict[:, i] = tmp

        Predict = Variable(torch.Tensor(Predict).cuda())
        x.data = Predict.data

        if train_flag == 1: # Training: only backprop the loss of misclassified data
          xdata = x.cpu().data.numpy()
          targetdata = target.cpu().data.numpy()

          xargmax = np.argmax(xdata, axis=1)

          idx_Wrong = []
          for iter1 in range(len(xargmax)):
              if int(xargmax[iter1]) != int(targetdata[iter1]):
                  idx_Wrong.append(iter1)

          xdata_Wrong = xdata[idx_Wrong]
          targetdata_Wrong = targetdata[idx_Wrong]

          idx_Right = []
          for iter1 in range(len(xargmax)):
              if int(xargmax[iter1]) == int(targetdata[iter1]):
                  idx_Right.append(iter1)

          targetdata_Right = targetdata[idx_Right]
          xdata_Right = np.zeros((len(idx_Right), xdata_Wrong.shape[1]))
          for i in range(len(idx_Right)):
              xdata_Right[i, int(xargmax[i])] = 1

          xdata_Whole = np.append(xdata_Wrong, xdata_Right, axis=0)
          targetdata_Whole = np.append(targetdata_Wrong, targetdata_Right, axis=0)

          xdata_Whole = Variable(torch.Tensor(xdata_Whole).cuda())
          targetdata_Whole = Variable(torch.Tensor(targetdata_Whole).long().cuda())

          x.data = xdata_Whole.data
          target.data = targetdata_Whole.data

        return x