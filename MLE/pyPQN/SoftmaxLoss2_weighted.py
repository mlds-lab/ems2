from __future__ import division
import numpy as np


def SoftmaxLoss2_weighted(w, X, y, k, sample_weight):
    # w(feature*class,1) - weights for last class assumed to be 0
    # X(instance,feature)
    # y(instance,1)
    #
    # version of SoftmaxLoss where weights for last class are fixed at 0
    #   to avoid overparameterization

    n, p = X.shape
    w = w.reshape((p, k - 1))
    w = np.hstack((w, np.zeros((p, 1))))

    Z = np.exp(X.dot(w)).sum(axis=1)
    nll = -(((X * w[:, y.astype(int)].T).sum(axis=1) - np.log(Z)) * sample_weight).sum()

    g = np.zeros((p, k - 1))

    for c in range(k - 1):
        g[:, c] = -((X * ((y == c) - np.exp(X.dot(w[:, c])) / Z)
                        [:, np.newaxis]) * np.tile(np.expand_dims(sample_weight, axis=0), (p, 1)).T).sum(axis=0)
    g = np.ravel(g)

    return nll, g
