
import numpy as np
from numpy import linalg as LA
from sklearn.metrics import mean_squared_error
from scipy.optimize import fmin_l_bfgs_b
from MLE.pyPQN.minConF_PQN import minConf_PQN
from MLE.PqnLasso import PqnLassoNet
from MLE.pyPQN.SquaredError import SquaredError
from MLE.pyPQN.projectRandom2 import randomProject
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import pearsonr


class NNMixedRegressionOne:
    def __init__(self, optim = 'l-bfgs', ridge_layers = [], lasso_layers = [10], alpha_lasso= 0.01,\
     alpha_ridge=.001, tol=0.001, random_state=0, epoch = 1000, lamda = 0.5):
        """
        Wrapper for Multilayer Regression model combining  Torch Group Lasso (using PQN) and Ridge penalty.
        This wrapper is compatible with GridSearchCV of Sci-Kit-Learn. Main methods are fit and predict.

        Args:
            alpha_ridge (float): Ridge regularizer
            alpha_lasso (float): Lasso regularizer
            optim (str) : optimization methods for multilayer regression. 'l-bfgs' and 'adam' supported
            ridge_layers (list) : list with number of neurons in each layer. First one is None to adapt any subset of features.
            lasso_layers (list) : list with number of neurons in each layer. First one is None to adapt any subset of features.
            epoch (int) : number of iterations for 'adam' optimizer.

        """
        self.alpha_lasso = alpha_lasso
        self.alpha_ridge = alpha_ridge
        self.model_lasso = None
        self.model_ridge = None
        self.feature_support = None
        self.lasso_coef = None
        self.ridge_coef = None
        self.tol = tol
        self.random_state = random_state
        
        self.ridge_layers = ridge_layers
        self.lasso_layers = lasso_layers
        self.epoch = epoch
        self.lamda = lamda
        self.optim = optim
        self.group_lasso_strategy = 'PQN'

    def fit(self, X, y):
        y_class = y.copy()
        y_class[y_class>0.] = 1

        d1, d2 = X.shape
        l_layers =  self.lasso_layers.copy()
        l_layers.insert(0,d2)

        #Union features of two lasso net
        """
        model_lasso_1 =  PqnLassoNet (layers = l_layers , reg = 'lasso', lossFnc ='mse', useProjection = True,\
         lbda = self.alpha_lasso)
        model_lasso_2 =  PqnLassoNet (layers = l_layers , reg = 'lasso', lossFnc ='cross-entropy', useProjection = True,\
         lbda = self.alpha_lasso)
        w_2 = model_lasso_2.fit(X,y_class.reshape((-1,1)))
        w_1 = model_lasso_1.fit(X,y.reshape((-1,1)))
        self.model_lasso = model_lasso_1
        if len(l_layers)>1:
            norm_w_1 =  np.linalg.norm(w_1[0: d2*l_layers[1]].reshape((l_layers[1],d2)), axis= 0)
            norm_w_2 =  np.linalg.norm(w_2[0: d2*l_layers[1]].reshape((l_layers[1],d2)), axis= 0)
        else:
            norm_w_1 =  np.linalg.norm(w_1[0: d2].reshape((1,d2)), axis= 0)
            norm_w_2 =  np.linalg.norm(w_2[0: d2].reshape((1,d2)), axis= 0)
        norm_w = norm_w_1+ norm_w_2
        """

        #Singla Lasso Net
        model_lasso_1 =  MixedLassoNet (layers = l_layers ,lbda = self.alpha_lasso , lamda = 0.5)
        w_1 = model_lasso_1.fit(X,y.reshape((-1,1)))
        if len(l_layers)>1:
            norm_w =  np.linalg.norm(w_1[0: d2*l_layers[1]].reshape((l_layers[1],d2)), axis= 0)
        else:
            norm_w =  np.linalg.norm(w_1[0: d2].reshape((1,d2)), axis= 0)

        feature_support = np.where(norm_w > self.tol)[0]
        self.lasso_coef = norm_w
        self.model_lasso = model_lasso_1
        
        # feature selection        
        self.feature_support = feature_support

        r_layers = self.ridge_layers.copy()
        r_layers.insert(0,  len(feature_support))
        if len(feature_support) == 0:
            r_layers[0] = 1

        self.ridge_coef = np.zeros(X.shape[1])

        ##Till this point same. Create MixedNet
        model = MixedNet(alpha=self.alpha_ridge, lamda = self.lamda,  optim = self.optim, layers = r_layers, epoch = self.epoch)
        if len(self.feature_support) > 0:
            w2 = model.fit(X[:, feature_support], y.reshape((-1,1)))
            if len(r_layers)>1:
                norm_w_1 =  np.linalg.norm(w2[0: r_layers[0]*r_layers[1]].reshape((r_layers[1],r_layers[0])), axis= 0)
                norm_w_2 =  np.linalg.norm(w2[r_layers[0]*r_layers[1]+ r_layers[1]: r_layers[1]+ r_layers[0]*r_layers[1]*2].reshape((r_layers[1],r_layers[0])), axis= 0)
            else:
                norm_w_1 =  np.linalg.norm(w2[0: r_layers[0]].reshape((1,r_layers[0])), axis= 0)
                norm_w_2 =  np.linalg.norm(w2[r_layers[0]+ 1: 1+r_layers[0]*2].reshape((1,r_layers[0])), axis= 0)
            norm_w = norm_w_2+ norm_w_2
            self.ridge_coef[feature_support] = norm_w
        else:
            X2 = np.ones((X.shape[0], 1))
            model.fit(X2, y.reshape((-1,1)))
        self.model_ridge = model

        return self

    def score(self, X, y):

        self.fit(X, y)
        y_predict = self.predict(X)
        #score_ = mean_squared_error(y, y_predict)
        score_ = pearsonr(y, y_predict)

        return score_


    def get_params(self, deep=True):

        params = {"alpha_lasso": self.alpha_lasso, "alpha_ridge": self.alpha_ridge, "ridge_layers": self.ridge_layers,\
        "lasso_layers": self.lasso_layers, "lamda": self.lamda}

        return params

    def set_params(self, **params):

        for parameter, value in params.items():
            setattr(self, parameter, value)

        return self

    def predict(self, X):

        model = self.model_ridge
        feature_support = self.feature_support

        if feature_support is None:
            raise ValueError('Feature indices cannot be None!')

        if len(self.feature_support) > 0:
            y_predict = model.predict(X[:, feature_support])
        else:
            X2 = np.ones((X.shape[0], 1))
            y_predict = model.predict(X2)
        return y_predict.reshape(-1)

    def predict_class(self, X):

        model = self.model_ridge
        feature_support = self.feature_support

        if feature_support is None:
            raise ValueError('Feature indices cannot be None!')

        if len(self.feature_support) > 0:
            y_predict = model.predict_class(X[:, feature_support])
        else:
            X2 = np.ones((X.shape[0], 1))
            y_predict = model.predict_class(X2)
        return y_predict.reshape(-1)




def MSELossWeighted(input, target, weight = None, size_average=True):
    
    loss =  (input - target)**2
    if weight is not None:
        loss = loss * weight
    if size_average:
        return loss.mean()
    else:
        return loss.sum()


#Define a basic MLP with pyTorch with Ridge regression
class Net(nn.Module):
    def __init__(self, feature, hiddenLayer = [], lamda = 0.5):
        super(Net, self).__init__()
        self.k = feature
        self.nb_layers = len(hiddenLayer) 
        self.hiddenLayer = hiddenLayer
        self.lamda = lamda
        torch.manual_seed(0)
        np.random.seed(0)
        if self.nb_layers>0:
            fc = []
            input_dim = self.k
            for i in range(self.nb_layers):
                fc.append(nn.Linear(input_dim, self.hiddenLayer[i]))
                fc.append(nn.Linear(input_dim, self.hiddenLayer[i]))
                input_dim = self.hiddenLayer[i]
            fc.append(nn.Linear(input_dim, 1))
            fc.append(nn.Linear(input_dim, 1))
            self.fc = nn.ModuleList (fc)
        else:
            fc = []
            fc.append(nn.Linear(self.k, 1))
            fc.append(nn.Linear(self.k, 1))
            self.fc = nn.ModuleList (fc)

    def _predict_proba(self, X):
        if self.nb_layers>0:
            for i in range(self.nb_layers):
                if i<1:
                    o_1 = F.relu(self.fc[2*i](X))
                    o_2 = F.relu(self.fc[2*i+1](X))
                else:
                    o_1 = F.relu(self.fc[2*i](o_1))
                    o_2 = F.relu(self.fc[2*i+1](o_2))
            o_1 = (self.fc[2*self.nb_layers](o_1))
            o_2 = (self.fc[2*self.nb_layers+1](o_2))
        else:
            o_1 = (self.fc[2*self.nb_layers](X))
            o_2 = (self.fc[2*self.nb_layers+1](X))
        yclass = F.sigmoid(o_1)
        return yclass, o_2
        
    def forward(self, X, y, y_class, weight_ = None):
        yc_hat, yr_hat = self._predict_proba(X)
        reg_ceriation = nn.MSELoss()
        reg_loss = reg_ceriation(yr_hat*yc_hat,y)
        class_loss = F.binary_cross_entropy( yc_hat, y_class, weight = weight_)
        loss =  self.lamda * reg_loss + (1.-self.lamda)* class_loss
        return loss  
    
    def predict_proba(self, X):
        X = Variable(torch.from_numpy(X).float())
        yc_hat, yr_hat = self._predict_proba(X)
        return yc_hat.data.numpy() * yr_hat.data.numpy()
    def predict_class(self, X):
        X = Variable(torch.from_numpy(X).float())
        yc_hat, yr_hat = self._predict_proba(X)
        return yc_hat.data.numpy() 

class MixedNet():
    def __init__(self, layers = [] , alpha = 1.,lamda = 0.5, sizeAverage = False, optim = 'l-bfgs', epoch = 1000):
        #Setup Layer architecture
        assert ( len(layers) <= 10), 'Maximum ten hiddenLayer Supported for your safety!'
        assert ( len(layers) >= 1), 'Please provide atleast feature dimension in the list!'
        assert ( optim in  {'l-bfgs', 'adam'}), 'Only slow \'l-bfgs\'  or \'adam\' provided !'
        self.isDeep = False
        self.optim = optim
        self.epoch = epoch
        self.nb_layers = len(layers) -1
        if len(layers) >= 2:
            self.d, self.hiddenLayer = layers[0], layers[1:]
            self.isDeep = True
        else:
            self.d, self.hiddenLayer = layers[0], []
        self.total_w =  0#(self.d+1)*self.h1 + (self.h1+1)
        input_dim = self.d
        for i in range(self.nb_layers):
            self.total_w +=  (input_dim+1)*layers[i+1] 
            input_dim = layers[i+1]
        self.total_w +=  input_dim+1
        self.total_w *= 2
        self.alpha = alpha
        self.weight = None
        self.model = Net(feature = self.d, hiddenLayer = self.hiddenLayer, lamda=lamda)

    def predict(self, X):
        return self.model.predict_proba(X)
    def predict_class(self, X):
        return self.model.predict_class(X)

    
    def fit(self, X, y, weight = None):     
        assert ( X.shape[1] == self.d), 'Input Layer dimension not matched!'
        y_class = y.copy()
        y_class[y_class>0.] = 1
        weight = np.ones((y_class.shape[0],1))
        weight[y_class == 0.] =     1.
        weight = Variable(torch.from_numpy(weight.reshape((-1,1))).float(), requires_grad = False)
        X2 = Variable(torch.from_numpy(X).float(),requires_grad=False)
        Y2 = Variable(torch.from_numpy(y).float(),requires_grad=False)
        Y3 = Variable(torch.from_numpy(y_class).float(),requires_grad=False)
                
        # print ('\nComputing optimal Lasso parameters...')
        torch.manual_seed(0)
        np.random.seed(0)
        if self.isDeep:
            wL1 = 0.1 * np.random.randn(self.total_w)# all one will have same grad
        else:#One layer regression
            wL1 = 0.1* np.ones(self.total_w)
        if self.optim == 'l-bfgs':
            wL1 = fmin_l_bfgs_b(self.objective, wL1, args = (X2,Y2,Y3, weight), disp = 0)
            #wL1[np.fabs(wL1) < 1e-4] = 0
            self.model_set_params(wL1[0])
            return wL1[0]
        else: #Adam
            optimizer = optim.Adam(self.model.parameters(), lr=0.1, weight_decay= self.alpha)
            for epoch in range(self.epoch):  # 
                self.model.zero_grad()

                loss= self.model.forward(X2, Y2, Y3, weight)
                epoch_loss = loss.data.numpy()
                loss.backward()
                optimizer.step()
                if epoch%500 == 0:
                    print ("Iteraion: ", epoch, ",loss: ", epoch_loss)
            return self.model_get_weights()


    def objective(self, x0, X2, Y2,Y3,  weight ):
        self.model.zero_grad()
        self.model_set_params(x0)
        loss = self.model(X2,Y2,Y3, weight)
        #l2_crit = nn.MSELoss(size_average=False)
        bias_flag = True
        for param in self.model.parameters():##bias is also getting penalized. Maybe add counter to remove bias
            if bias_flag:
                loss += self.alpha*(param**2).sum()
                bias_flag = False
            else:
                bias_flag = True
        loss.backward()
        f = loss.data.numpy()
        g = self.model_get_params()
        return f,g


    def model_set_params(self, W):
        assert ( self.total_w == W.shape[0]), 'Shape mismatch!'
        dtype = torch.FloatTensor
        i = 0
        start = 0
        input_dim = self.d
        for param in self.model.parameters():
            if i/4 <self.nb_layers:
                output_dim = self.hiddenLayer[i//4]
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
            if i%4 == 3:
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

    def model_get_weights(self):
        grad = np.zeros(self.total_w)
        start = 0
        for param in self.model.parameters():
            g = param.data.numpy().reshape(-1)
            grad_len = g.shape[0]
            grad[start:start+grad_len] = g
            start = start+grad_len
        return grad



class MixedLassoNet():
    def __init__(self, layers = [10] , lbda = 10., lamda = 0.5):
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
        self.isDeep = False
        self.nb_layers = len(layers) -1
        self.isGroupLasso =  False
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
        self.total_w *= 2
        self.total_w = int(self.total_w)

        #If group lasso(marker present) or lasso implicit groups
        if self.isGroupLasso:
            groups = np.arange(self.d)

            img_h = int(self.h1)
            self.groups = np.array(groups)
            self.nGroups = int(np.max(self.groups) + 1)
            #Set groupStart and GroupPtr
            self.groupStart = np.zeros(self.nGroups+1).astype(int)
            self.groupPtr = np.zeros(self.d* img_h*2)
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
                self.groupPtr[start: start+subLen] = (subGroup + (self.d+1)* img_h)
                start +=subLen
            self.groupStart[self.nGroups] = start 
            self.groupStart.astype(int)
            self.groupPtr.astype(int)
        self.lbda = lbda

        self.model = Net(feature = self.d, hiddenLayer = self.hiddenLayer, lamda = lamda)

    def predict(self, X):
        return self.model.predict_proba(X)
    def predict_class(self, X):
        return self.model.predict_class(X)
    
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
        y_class = y.copy()
        y_class[y_class>0.] = 1
        weight = np.ones((y_class.shape[0],1))
        weight[y_class == 0.] =     1.
        self.weight = Variable(torch.from_numpy(weight.reshape((-1,1))).float(), requires_grad = False)
        self.X2 = Variable(torch.from_numpy(X).float(),requires_grad=False)
        self.Y2 = Variable(torch.from_numpy(y).float(),requires_grad=False)
        self.Y3 = Variable(torch.from_numpy(y_class).float(),requires_grad=False)
        # print ('\nComputing optimal Lasso parameters...')
        torch.manual_seed(0)
        np.random.seed(0)
        if self.isDeep:
            wL1 = 0.1 * np.random.randn(self.total_w)# all one will have same grad
        else:#One layer Group Lass
            wL1 = 0.1* np.ones(self.total_w)
        if self.isGroupLasso:
            wL1 = minConf_PQN(self.funObj, wL1, self.funProjL12, verbose = 0)[0]
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
            if i/4 <self.nb_layers:
                output_dim = self.hiddenLayer[i//4]
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
            if i%4 == 3:
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
        loss = self.model(self.X2, self.Y2,self.Y3, self.weight)
        loss.backward()
        f = loss.data.numpy()
        g = self.model_get_params()

        return f, g

    
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









    












