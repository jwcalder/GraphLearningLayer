import torch
import graphlearning as gl
import numpy as np
import scipy.sparse.csgraph as csgraph
import warnings
from scipy.sparse.linalg import eigsh
import scipy.sparse as sparse


class LaplaceLearningSparseHard(torch.autograd.Function):
    # Assume that the first k entries are the labelled data vectors and the rest are unlabelled

    @staticmethod
    def forward(ctx, X, label_matrix, tau=0, epsilon='auto'):
        if X.is_cuda:
            dev = 'cuda'
        else:
            dev = 'cpu'
        # X is the upstream feature vectors coming from a pytorch neural network
        # label_matrix
        # tau is an optional parameter that adds regularity to the expression. tau=0 indicates no regularizer
        # In this hard label problem, we are only receiving gradients for the unlabeled samples

        # W, V = knn_connected(X.detach().cpu().numpy(),k=25) #Note - similarity is euclidean. If you want angular, maybe normalize them before passing in because that's annoying
        # W, V = knn_connected(X.detach().cpu().numpy(), label_matrix.shape[0], k=25)

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

        Pred = sparse.linalg.spsolve(Luu, -Lul @ label_matrix)  # sparse solver

        # M = Luu.diagonal()
        # M = sparse.spdiags(1 / np.sqrt(M + 1e-10), 0, m, m).tocsr()

        # Pred = sparse.linalg.spsolve(M * Luu * M, -M * Lul @ label_matrix)

        # Pred = M*Pred

        # Pred = stable_conjgrad(M * Luu * M,
        #                       -M * Lul @ label_matrix)  # could push things to GPU for conjgrad - label matrix should have 0's for unlabeled terms
        # Pred = M * Pred

        Pred = torch.from_numpy(Pred)

        # Save tensors for backward and return output
        ctx.W, ctx.V, ctx.Luu, ctx.label_matrix, ctx.mod_V, ctx.C, ctx.knn_ind = W, V, Luu, label_matrix, mod_V, C, knn_ind  # all non-pytorch tensors
        ctx.save_for_backward(X, Pred)  # L here is sparse - Only takes pytorch tensors
        # print('endforward')

        return Pred.to(device=dev)

    @staticmethod
    def backward(ctx, grad_output):
        # print('backward1')

        # Retrieve saved tensors.
        X, Pred = ctx.saved_tensors
        W, V, Luu, label_matrix, mod_V, C, knn_ind = ctx.W, ctx.V, ctx.Luu, ctx.label_matrix, ctx.mod_V, ctx.C, ctx.knn_ind

        n = X.shape[0]  # number of data points
        m = Pred.shape[0]  # number of unlabelled samples
        l = Pred.shape[1]  # number of classes/labels
        k = label_matrix.shape[0]  # number of labeled samples

        # print('backward2')
        device = grad_output.device
        grad_output = grad_output.detach().cpu().numpy()  # Should be mxl and dense
        # print(np.linalg.norm(grad_output))

        w = sparse.linalg.spsolve(Luu, grad_output)  # SP SOLVE

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

            T = sparse.csgraph.laplacian(C.multiply(b), symmetrized=True)  # coo matrix

            values = T.data
            indices = np.vstack((T.row, T.col))

            i = torch.LongTensor(indices)
            v = torch.FloatTensor(values)
            shape = T.shape

            T = torch.sparse_coo_tensor(i, v, torch.Size(shape), device=device)

            extra_grad = -T @ X  # note the sign

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


def knn_sym_dist(data, k=25, epsilon='auto'):
    method = 'annoy'

    knn_ind, knn_dist = gl.weightmatrix.knnsearch(data, k, similarity='euclidean', method=method)

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
        eps = Dist[knn_ind[:, 0], [knn_ind[:, -1]]].toarray().flatten()

        # Construct C, # C_ij = 1 when i is kth nn of j, and -1 on the diag
        # C = np.diag(-np.ones(n))
        C = np.zeros((n, n))

        C[knn_ind[:, -1], knn_ind[:, 0]] = 1

        C = sparse.csr_matrix(C)

        # Compute W, V, and mod_V
        W_values = np.exp(-4 * values * values / eps[rows] / eps[cols])
        V_values = -8 * np.exp(-4 * values * values / eps[rows] / eps[cols]) / eps[rows] / eps[cols]
        mod_V_values = values * values * V_values / (eps[rows] ** 2) / 2

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


def stable_conjgrad(A, b, x0=None, max_iter=1e5, tol=1e-10):
    if x0 is None:
        x = np.zeros_like(b)
    else:
        x = x0

    r = b - A @ x
    p = r
    rsold = np.sum(r ** 2, axis=0)

    err = 1
    i = 0
    while (err > tol) and (i < max_iter):
        i += 1
        Ap = A @ p
        alpha = np.zeros_like(rsold)
        alpha[rsold > tol ** 2] = rsold[rsold > tol ** 2] / np.sum(p * Ap, axis=0)[rsold > tol ** 2]
        x += alpha * p
        r -= alpha * Ap
        rsnew = np.sum(r ** 2, axis=0)
        err = np.max(np.sqrt(rsnew))
        beta = np.zeros_like(rsold)
        beta[rsnew > tol ** 2] = rsnew[rsnew > tol ** 2] / rsold[rsnew > tol ** 2]
        p = r + beta * p
        rsold = rsnew

    if err > tol:
        print('max iter reached: ', i, ' iters')

    return x
