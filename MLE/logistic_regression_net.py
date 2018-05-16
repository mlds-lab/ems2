import numpy as np
from numpy import linalg as LA
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from MLE.pyPQN.minConF_PQN import minConf_PQN
from MLE.pyPQN.SquaredError import SquaredError
from MLE.pyPQN.SoftmaxLoss2 import SoftmaxLoss2
from MLE.pyPQN.projectRandom2 import randomProject
from MLE.PqnLasso import PqnLassoNet
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from scipy.optimize import fmin_l_bfgs_b

class LogisticRegressionNet:

    def __init__(self, marker_groups, C= 1., lbda= 10., optim = 'l-bfgs', layers = [None, 10, 4 ], \
        epoch = 10, lasso_layer = []):
        """
        Wrapper for Logistic Regression model combining Group Lasso (using PQN) and multilayer Logistic Regression.
        This wrapper is compatible with GridSearchCV of Sci-Kit-Learn. Main methods are fit and predict.

        Args:
            marker_groups (list): marker indicators for each feature/column of X
            C (int): regularizer
            lbda (int): Group Lasso regularizer
        """

        self.marker_groups = marker_groups
        self.C = C
        self.lbda = lbda
        self.model = None
        self.layers = layers
        self.epoch = epoch
        self.optim = optim
        self.feature_support = None
        self.feature_indices = None
        self.group_lasso_strategy = 'PQN'  # 'PQN' or 'None'
        self.sample_weight = None
        self.lasso_layer = lasso_layer

    def fit(self, X, y, sample_weight=None):

        feature_support = self.group_lasso_PQN(X, y, sample_weight, self.lasso_layer)
        print("feature_support", feature_support)
        markers, repeats = np.unique(self.marker_groups, return_counts=True)
        feature_indices = []
        for i in markers:
            if i in feature_support:
                feature_indices.extend(np.repeat(True, repeats[i]))
            else:
                feature_indices.extend(np.repeat(False, repeats[i]))
        layers = self.layers
        layers[0] = feature_indices.count(True)
        max_y = np.max(y)+1
        model = LogisticNet(C=self.C, optim = self.optim, layers = layers, epoch = self.epoch, output_dim = max_y)
        model.fit(X[:, feature_indices], y.reshape((-1,1)), weight=sample_weight)
        self.model = model
        self.feature_support = feature_support
        self.feature_indices = feature_indices

    def score(self, X, y, sample_weight=None):

        self.fit(X, y, sample_weight)
        y_predict = self.predict(X)
        score_ = accuracy_score(y, y_predict)

        return score_

    def get_params(self, deep=True):

        params = {"C": self.C, "lbda": self.lbda, "marker_groups": self.marker_groups, "layers": self.layers}

        return params

    def set_params(self, **params):

        for parameter, value in params.items():
            setattr(self, parameter, value)

        return self

    def predict(self, X):

        model = self.model
        feature_indices = self.feature_indices

        if feature_indices is None:
            raise ValueError('Feature indices cannot be None!')

        y_predict = model.predict(X[:, feature_indices])

        return y_predict

    def group_lasso_PQN(self, X, y, sample_weight, lasso_layer):

        d1, d2 = X.shape
        markers, repeats = np.unique(self.marker_groups, return_counts=True)
        gr = np.asarray(self.marker_groups)#np.zeros(d2)
        max_y = np.max(y)+1
        layers = [d2]
        if lasso_layer:
            layers.append(lasso_layer[0])
        net =  PqnLassoNet (layers = layers ,groups = gr, reg = 'group', lossFnc ='cross-entropy', useProjection = \
            True, lbda = self.lbda, output_dim = max_y)
        w2 = net.fit(X,y, sample_weight)

        feature_support = []
        for j in markers:
            indices = np.where(self.marker_groups == j)[0]
            if LA.norm(w2[indices], 2) != 0:
                feature_support.append(int(j))

        return feature_support

#Define a basic MLP with pyTorch with Logistic regression
class Net(nn.Module):
    def __init__(self, feature, hiddenLayer = [],  lossFnc = 'mse', output_dim = 1, sizeAverage = False):
        super(Net, self).__init__()
        self.k = feature
        self.lossFnc = lossFnc
        self.sizeAverage = sizeAverage
        self.nb_layers = len(hiddenLayer) 
        self.hiddenLayer = hiddenLayer
        torch.manual_seed(0)
        np.random.seed(0)
        self.output_dim = int(output_dim)
        if self.nb_layers>0:
            fc = []
            input_dim = self.k
            for i in range(self.nb_layers):
                fc.append(nn.Linear(input_dim, self.hiddenLayer[i]))
                input_dim = self.hiddenLayer[i]
            self.fc = nn.ModuleList (fc)
            self.out = nn.Linear(input_dim, self.output_dim)
        else:
            self.out = nn.Linear(self.k, self.output_dim)


    def _predict_proba(self, X):
        if self.nb_layers>0:
            for i in range(self.nb_layers):
                X = F.relu(self.fc[i](X))
        yhat = self.out(X)
        return yhat
        
    def forward(self, X, y, weight_ = None):
        yhat = self._predict_proba(X)
        if self.lossFnc == 'cross-entropy':
            #Add weighted loss for cross-entropy
            ceriation = nn.CrossEntropyLoss(size_average = False, reduce= False)
            loss = ceriation(yhat,y)
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
        return np.argmax(yhat.data.numpy(), axis = 1)


class LogisticNet():
    def __init__(self, layers = [] , groups = 0, C = 10.,output_dim = 1,\
     lossFnc = 'cross-entropy', sizeAverage = False, optim = 'adam', epoch = 1000):
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
        self.total_w +=  (input_dim+1)*output_dim
        self.total_w = int(self.total_w)
        self.alpha = C
        self.lossFnc = lossFnc
        self.output_dim = output_dim
        self.weight = None
        self.model = Net(feature = self.d, hiddenLayer = self.hiddenLayer, output_dim = self.output_dim, \
            lossFnc = lossFnc, sizeAverage = sizeAverage)

    def predict(self, X):
        return self.model.predict_proba(X)

    
    def fit(self, X, y, weight = None):     
        assert ( X.shape[1] == self.d), 'Input Layer dimension not matched!'
        if weight is not None:
            assert ( weight.shape[0] == X.shape[0]), 'Weight should have same number of instances!'
            weight = Variable(torch.from_numpy(weight.reshape((-1,1))).float())
        X2 = Variable(torch.from_numpy(X).float(),requires_grad=False)
        Y2 = Variable(torch.from_numpy(y.reshape(-1)).long(),requires_grad=False)
                
        # print ('\nComputing optimal Lasso parameters...')
        torch.manual_seed(0)
        np.random.seed(0)
        if self.isDeep:
            wL1 = 0.1 * np.random.randn(self.total_w)# all one will have same grad
        else:#One layer regression
            wL1 = 0.1* np.ones(self.total_w)
        if self.optim == 'l-bfgs':
            wL1 = fmin_l_bfgs_b(self.objective, wL1, args = (X2,Y2,weight))
            #wL1[np.fabs(wL1) < 1e-4] = 0
            self.model_set_params(wL1[0])
            return
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
        input_dim = int(self.d)
        for param in self.model.parameters():
            if i/2 <self.nb_layers:
                output_dim = int(self.hiddenLayer[i//2])
            else:
                output_dim = int(self.output_dim)
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
    

