
from MLE.pyPQN.minConF_PQN import minConf_PQN
from MLE.pyPQN.projectRandom2 import randomProject
# from MLE.pyPQN.groupl1_makeGroupPointers import groupl1_makeGroupPointers
# from MLE.pyPQN.auxGroupL2Project import groupL2Proj
import numpy as np
import torch
from numpy import linalg as LA
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim



def MSELossWeighted(input, target, weight = None, size_average=False):
    loss =  (input - target)**2
    if weight is not None:
        loss = loss * weight
    if size_average:
        return loss.mean()
    else:
        return loss.sum()
#Define a basic MLP with pyTorch Logistic Regression with sparse group lasso using different Optimizer
class Net(nn.Module):
    def __init__(self, feature, hiddenLayer = [], output_dim =1,  lossFnc = 'mse', sizeAverage = False, seed = 0):
        super(Net, self).__init__()
        self.k = feature
        self.lossFnc = lossFnc
        self.sizeAverage = sizeAverage

        self.nb_layers = len(hiddenLayer) 
        self.hiddenLayer = hiddenLayer
        torch.manual_seed(0)
        np.random.seed(0)
        if self.nb_layers>0:
            fc = []
            input_dim = self.k
            for i in range(self.nb_layers):
                fc.append(nn.Linear(input_dim, self.hiddenLayer[i]))
                input_dim = self.hiddenLayer[i]
            self.fc = nn.ModuleList (fc)
            self.out = nn.Linear(input_dim, 1)
        else:
            self.out = nn.Linear(self.k, 1)


    def _predict_proba(self, X):

        if self.nb_layers>0:
            for i in range(self.nb_layers):
                X = F.relu(self.fc[i](X))
        if self.lossFnc == 'cross-entropy':
            yhat = F.sigmoid(self.out(X))
            return yhat
        yhat = self.out(X)
        return yhat
        
    def forward(self, X, y, weight_ = None):
        yhat = self._predict_proba(X)
        if self.lossFnc == 'cross-entropy':
            #Add weighted loss for cross-entropy
            #ceriation = F.binary_cross_entropy_with_logits( size_average = True)
            #loss = ceriation(yhat,y)
            loss = F.binary_cross_entropy( yhat, y)
            if weight_ is not None:
                loss = loss*  weight_
            if self.sizeAverage:
                return loss.mean()
            else:
                return loss.sum()
        else:
            if weight_ is None:
                ceriation = nn.MSELoss(size_average =  self.sizeAverage)
                loss = ceriation(yhat,y)
            else:
                loss = MSELossWeighted(input = yhat, target = y, weight = weight_, size_average = self.sizeAverage)
        return loss  
    
    def predict_proba(self, X):
        X = Variable(torch.from_numpy(X).float())
        yhat = self._predict_proba(X)
        return yhat.data.numpy().reshape(-1)

class PqnLassoNet():
    def __init__(self, layers = [10] , groups = None, reg = 'lasso', tau = 2., lbda = 10.,\
     lossFnc = 'mse', sizeAverage = False, useProjection = False,  output_dim = 1):
        """
        Wrapper for Multilayer Group Lasso (using PQN).
        Main method is fit which return weigh vector.

        Args:
            layers (list) : list with number of neurons in each hidden layer. Input layer dimension should be the first one.
            groups (list): marker indicators for each feature/column of X
            reg (str) : 'group' or 'lasso' whether group lasso or normal lasso is applied
            tau (float): Lasso Regularizer. (if not group lasso)
            lbda (float): Group Lasso ball radius (default single layer) for projection.
            lossFnc (str) : 'mse' or 'cross-entropy' provided.
            sizeAverage (bool) : whether to apply size average in loss function
            useProjection (bool) : whether to use projection or penalty for group lasso minimization

        """
        #Setup Layer architecture
        assert ( len(layers) <= 10), 'Maximum ten hiddenLayer Supported!'
        assert ( reg in  {'lasso', 'group'}), 'Only group lasso \'group\' or lasso \'lasso\' provided !'
        assert ( lossFnc in  {'mse', 'cross-entropy'}), 'Only \'mse\' or \'cross-entropy\' provided !'
        self.isDeep = False
        self.nb_layers = len(layers) -1
        self.isGroupLasso =  reg == 'group'
        self.h1 = 1
        if len(layers) >= 2:
            self.isGroupLasso =  True
            self.d, self.hiddenLayer = layers[0], layers[1:]
            self.isDeep = True
            self.h1 = layers[1]
        else:
            self.d, self.hiddenLayer = layers[0], []

        self.total_w =  0#(self.d+1)*self.h1 + (self.h1+1)
        input_dim = self.d
        for i in range(self.nb_layers):
            self.total_w +=  (input_dim+1)*layers[i+1] 
            input_dim = layers[i+1]
        self.total_w +=  input_dim+1
        self.total_w = int(self.total_w)

        #If group lasso(marker present) or lasso implicit groups
        if self.isGroupLasso:
            if groups == None:
                groups = np.arange(self.d)

            img_h = int(self.h1)
            self.groups = np.array(groups)
            self.nGroups = int(np.max(self.groups) + 1)
            #Set groupStart and GroupPtr
            self.groupStart = np.zeros(self.nGroups+1).astype(int)
            self.groupPtr = np.zeros(self.d* img_h)
            #If multiple outgoing edges
            self.groups = np.tile(self.groups,(img_h,1)).reshape(-1)
            start = 0
            indexes = np.arange(self.d*img_h)
            for i in range(self.nGroups):
                subGroup = indexes[self.groups == i]
                subLen = len(subGroup)
                self.groupStart[i] = start
                self.groupPtr[start: start+subLen] = subGroup
                start +=subLen
            self.groupStart[self.nGroups] = start 
            self.groupStart.astype(int)
            self.groupPtr.astype(int)
        self.lbda = lbda
        self.useProjection = useProjection
        self.tau = tau
        self.reg = reg
        self.lossFnc = lossFnc
        self.weight = None
        self.output_dim = int(output_dim)
        self.model = Net(feature = self.d, hiddenLayer = self.hiddenLayer, output_dim = self.output_dim, \
            lossFnc = lossFnc, sizeAverage = sizeAverage)

    def predict(self, X):
        return self.model.predict_proba(X)
    
    def fit(self, X, y, weight = None):
        """
        Fit methods for PqnLasso

        Args:
            X (ndarray): of shape (NxD)
            y (ndarray): of shape (N,)
            weight (optional ndarray):  of shape (N,)
        Returns:
            w (ndarray) : the weigh vector of length (D+1)x Len(Layer[0]) + Len(layer[0]) +1 of deep. O/W, D+1.
        """
        assert ( X.shape[1] == self.d), 'Input Layer dimension not matched!'
        if weight is not None:
            assert ( weight.shape[0] == X.shape[0]), 'Weight should have same number of instances!'
            self.weight = torch.from_numpy(weight).float()

        self.X2 = torch.from_numpy(X).float()
        if self.lossFnc == 'cross-entropy':
            self.y2 = torch.from_numpy(y).float()    
        else:
            self.y2 = torch.from_numpy(y[:,np.newaxis]).float()
        # print ('\nComputing optimal Lasso parameters...')
        torch.manual_seed(0)
        np.random.seed(0)
        if self.isDeep:
            wL1 = 0.1 * np.random.randn(self.total_w)# all one will have same grad
        else:#One layer Group Lass
            wL1 = 0.1* np.ones(self.total_w)
        if self.isGroupLasso:
            if self.useProjection:
                wL1 = minConf_PQN(self.funObj, wL1, self.funProjL12, verbose = 0)[0]

            else:
                alpha = self.getInitialGroupNorm(wL1)
                original_len = wL1.shape[0]
                x0 = np.concatenate((np.ravel(wL1), alpha))
                x0 = minConf_PQN(self.funObjGroup, x0, self.funProjGroup, verbose = 0)[0]
                wL1 = x0[:original_len]
        else:# Lasso
            wL1 = minConf_PQN(self.funObj, wL1, self.funProj, verbose = 0)[0]

        wL1[np.fabs(wL1) < 1e-4] = 0
        self.model_set_params(wL1)
        return wL1

    def getInitialGroupNorm(self, W):
        norm_loss = np.zeros(self.nGroups)
        for i in range(self.nGroups):
            groupInd = self.groupPtr[self.groupStart[i]:self.groupStart[i + 1]]
            norm_loss[i] = np.linalg.norm(W[groupInd])
        return norm_loss
    
    def model_set_params(self, W):
        assert ( self.total_w == W.shape[0]), 'Shape mismatch!'
        dtype = torch.FloatTensor
        i = 0
        start = 0
        input_dim = self.d
        for param in self.model.parameters():
            if i/2 <self.nb_layers:
                output_dim = self.hiddenLayer[i//2]
            else:
                output_dim = 1
            if i%2 == 0:
                param.data = torch.from_numpy(W[start:start+input_dim*output_dim].reshape\
                    ((output_dim,input_dim))).type(dtype)
                i += 1
                start += input_dim*output_dim
            else:
                param.data = torch.from_numpy(W[start: start+output_dim].\
                    reshape(-1)).type(dtype)
                i += 1
                start += output_dim
                input_dim = output_dim
    
    def model_get_params(self):
        grad = np.zeros(self.total_w)
        start = 0
        for param in self.model.parameters():
            g = param.grad.data.numpy().reshape(-1)
            grad_len = g.shape[0]
            grad[start:start+grad_len] = g
            start = start+grad_len
        return grad

    
    # Set up Objective Function For Lasso
    def funObj(self,W):
        self.model.zero_grad()
        self.model_set_params(W)
        if self.weight is not None:
            loss = self.model(Variable(self.X2), Variable(self.y2), Variable(self.weight))
        else:
            loss = self.model(Variable(self.X2), Variable(self.y2))
        loss.backward()
        f = loss.data.numpy()
        g = self.model_get_params()

        return f, g

    # Set up L1-Ball Projection for Lasso
    def funProj(self, w):
        if self.isDeep:
            return np.hstack( (np.sign(w[0:self.d*self.h1]) * randomProject(np.fabs(w[:self.d*self.h1]), self.tau), w[self.d*self.h1:])) 
        else:
            return np.hstack( (np.sign(w[:-1]) * randomProject(np.fabs(w[:-1]), self.tau), w[-1])) 
    
    # Set up L12-Ball Projection for GroupLasso
    def funProjL12(self, w):
        normedW, alpha = self.getGroupNorm(w)
        alpha_proj = randomProject(alpha, self.lbda)
        wOut = normedW.copy()
        for i in range(self.nGroups):
            groupInd = self.groupPtr[self.groupStart[i]:self.groupStart[i + 1]].astype(int)
            wOut[groupInd] *= alpha_proj[i]
        return wOut
        #if self.isDeep:
        #    return np.hstack( (np.sign(w[0:self.d*self.h1]) * randomProject(np.fabs(w[:self.d*self.h1]), self.tau), w[self.d*self.h1:])) 
        #else:
        #    return np.hstack( (np.sign(w[:-1]) * randomProject(np.fabs(w[:-1]), self.tau), w[-1])) 

    def getGroupNorm(self, W):
        alpha = np.zeros(self.nGroups)
        normedW = W.copy()
        for i in range(self.nGroups):
            groupInd = self.groupPtr[self.groupStart[i]:self.groupStart[i + 1]].astype(int)
            alpha[i] = np.linalg.norm(W[groupInd])
            if  alpha[i] >1e-10:
                normedW[groupInd] = W[groupInd]/alpha[i]

        return normedW, alpha
     # Set up Objective Function for GroupLasso
    def funObjGroup(self,x0):
        self.model.zero_grad()
        p = x0.shape[0]
        nGroups = self.nGroups
        self.model_set_params(x0[:p-nGroups])
        
        if self.weight is not None:
            loss = self.model(Variable(self.X2), Variable(self.y2), Variable(self.weight))
        else:
            loss = self.model(Variable(self.X2), Variable(self.y2))
        loss.backward()
        f1 = loss.data.numpy()
        g1 = self.model_get_params()
        f = f1 + self.lbda* np.sum(x0[p-nGroups:])
        g = np.concatenate((g1, self.lbda * np.ones(nGroups)))
        return f, g
    
    # Set up group L2-Ball Projection for unconstrained group lasso
    def funProjGroup(self, x0):
        p = x0.shape[0]
        nGroups = self.nGroups
        alpha = x0[p-nGroups:].copy()
        w = x0[:p-nGroups].copy()

        for i in range(nGroups):
            groupInd = self.groupPtr[self.groupStart[i]:self.groupStart[i + 1]]
            w[groupInd], alpha[i] = self.projectAux(w[groupInd], alpha[i])

        return np.concatenate((w, alpha))

    def projectAux(self, w, alpha):
        nw = np.linalg.norm(w)
        if nw <= -alpha:
            w[:] = 0
            alpha = 0
        elif nw >= alpha:
            scale = 0.5 * (1 + alpha / nw)
            w = scale * w
            alpha = scale * nw
        return w, alpha







