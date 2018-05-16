
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
import copy


class NNRegressionOne:
    def __init__(self, bounds=[-np.inf,np.inf], optim = 'l-bfgs', ridge_layers = [8], lasso_layers = [8], alpha_lasso=None,\
     alpha_ridge=None, tol=0.00001, random_state=0, epoch = 10):
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
        self.optim = optim
        self.group_lasso_strategy = 'PQN'
        self.bounds = bounds

    def clip(self,y):
        y[y<self.bounds[0]] = self.bounds[0] 
        y[y>self.bounds[1]] = self.bounds[1]
        return(y)  
        
        
    def fit(self, X, y):


        self.scale=1+np.std(y)
        self.mean = np.mean(y)

        d1, d2 = X.shape
        l_layers =  copy.copy(self.lasso_layers)
        
        l_layers.insert(0,d2)
        model_lasso =  PqnLassoNet (layers = l_layers , reg = 'lasso', lossFnc ='mse', useProjection = True, lbda = self.alpha_lasso)
        w2 = model_lasso.fit(X,(y-self.mean)/self.scale)
        self.model_lasso = model_lasso
        if len(l_layers)>1:
            norm_w =  np.linalg.norm(w2[0: d2*l_layers[1]].reshape((l_layers[1],d2)), axis= 0)
        else:
            norm_w =  np.linalg.norm(w2[0: d2].reshape((1,d2)), axis= 0)
        feature_support = np.where(norm_w > self.tol)[0]
        self.lasso_coef = norm_w
        
        # feature selection        
        self.feature_support = feature_support
        # print("Feature Supported :", feature_support, " Lasso params is :", self.alpha_lasso)

        r_layers = copy.copy(self.ridge_layers)
        r_layers.insert(0,  len(feature_support))
        if len(feature_support) == 0:
            r_layers[0] = 1

        self.ridge_coef = np.zeros(X.shape[1])
        model = RidgeNet(alpha=self.alpha_ridge, optim = self.optim, layers = r_layers, epoch = self.epoch)
        if len(self.feature_support) > 0:
            w2 = model.fit(X[:, feature_support], (y.reshape((-1,1))-self.mean)/self.scale)
            if len(r_layers)>1:
                norm_w =  np.linalg.norm(w2[0: r_layers[0]*r_layers[1]].reshape((r_layers[1],r_layers[0])), axis= 0)
            else:
                norm_w =  np.linalg.norm(w2[0: r_layers[0]].reshape((1,r_layers[0])), axis= 0)
            self.ridge_coef[feature_support] = norm_w
        else:
            X2 = np.ones((X.shape[0], 1))
            model.fit(X2, y)
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
        "lasso_layers": self.lasso_layers}

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
            y_predict = self.clip(self.mean + self.scale*model.predict(X[:, feature_support]))
        else:
            X2 = np.ones((X.shape[0], 1))
            y_predict = self.clip(self.mean + self.scale*model.predict(X2))
        return y_predict.reshape(-1)




def MSELossWeighted(input, target, weight = None, size_average=False):
    
    loss =  (input - target)**2
    if weight is not None:
        loss = loss * weight
    if size_average:
        return loss.mean()
    else:
        return loss.sum()
#Define a basic MLP with pyTorch with Ridge regression
class Net(nn.Module):
    def __init__(self, feature, hiddenLayer = [],  lossFnc = 'mse', sizeAverage = False):
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
            ceriation = nn.BCELoss(size_average =  self.sizeAverage)
            loss = ceriation(yhat,y)
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
        return yhat.data.numpy()

class RidgeNet():
    def __init__(self, layers = [] , groups = 0, alpha = 10.,\
     lossFnc = 'mse', sizeAverage = False, optim = 'l-bfgs', epoch = 1000):
        #Setup Layer architecture
        assert ( len(layers) <= 10), 'Maximum ten hiddenLayer Supported for your safety!'
        assert ( len(layers) >= 1), 'Please provide atleast feature dimension in the list!'
        assert ( lossFnc in  {'mse', 'cross-entropy'}), 'Only \'mse\' or \'cross-entropy\' provided !'
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
        self.alpha = alpha
        self.lossFnc = lossFnc
        self.weight = None
        self.model = Net(feature = self.d, hiddenLayer = self.hiddenLayer, lossFnc = lossFnc, sizeAverage = sizeAverage)

    def predict(self, X):
        return self.model.predict_proba(X)

    
    def fit(self, X, y, weight = None):     
        assert ( X.shape[1] == self.d), 'Input Layer dimension not matched!'
        if weight is not None:
            assert ( weight.shape[0] == X.shape[0]), 'Weight should have same number of instances!'
            weight = Variable(torch.from_numpy(weight.reshape((-1,1))).float())
            if self.lossFnc != 'mse':
                print("Weight is only supported for MSE loss. Resorting to default! ")
        X2 = Variable(torch.from_numpy(X).float(),requires_grad=False)
        Y2 = Variable(torch.from_numpy(y).float(),requires_grad=False)
                
        # print ('\nComputing optimal Lasso parameters...')
        torch.manual_seed(0)
        np.random.seed(0)
        if self.isDeep:
            wL1 = 0.1 * np.random.randn(self.total_w)# all one will have same grad
        else:#One layer regression
            wL1 = 0.1* np.ones(self.total_w)
        if self.optim == 'l-bfgs':
            wL1 = fmin_l_bfgs_b(self.objective, wL1, args = (X2,Y2,weight), disp = 0)
            #wL1[np.fabs(wL1) < 1e-4] = 0
            self.model_set_params(wL1[0])
            return wL1[0]
        else: #Adam
            optimizer = optim.Adam(self.model.parameters(), lr=0.0075, weight_decay= self.alpha)
            for epoch in range(self.epoch):  # 
                self.model.zero_grad()

                loss= self.model.forward(X2, Y2, weight)
                epoch_loss = loss.data.numpy()
                loss.backward()
                optimizer.step()
                if epoch%10 == 0:
                    print ("Iteraion: ", epoch, ",loss: ", epoch_loss)

    def objective(self, x0, X2, Y2, weight ):
        self.model.zero_grad()
        self.model_set_params(x0)
        loss = self.model(X2,Y2,weight)
        #l2_crit = nn.MSELoss(size_average=False)
        for param in self.model.parameters():##bias is also getting penalized. Maybe add counter to remove bias
            loss += self.alpha*(param**2).sum()
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
    












