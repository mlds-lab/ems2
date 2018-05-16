
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from scipy.stats import pearsonr


class LinearRegressionOne:

    def __init__(self, bounds=[-np.inf,np.inf], alpha_lasso=None, alpha_ridge=None, tol=0.00001, random_state=0):
        """
        Wrapper for Linear Regression model combining Lasso and Ridge.
        This wrapper is compatible with GridSearchCV of Sci-Kit-Learn. Main methods are fit and predict.

        Args:
            alpha_ridge (float): Ridge regularizer
            alpha_lasso (float): Lasso regularizer
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
        self.bounds = bounds

    def clip(self,y):
        
        #y=y-min(y)
        #if(not np.isinf(self.bounds[1])):
        #    y=y/(1e-4+max(y))        
        #    y=self.bounds[0]+ y*(self.bounds[1]-self.bounds[0])
        #else:
        #    y=self.bounds[0]+ y
        
        y[y<self.bounds[0]] = self.bounds[0] 
        y[y>self.bounds[1]] = self.bounds[1]
        return(y)        

    def fit(self, X, y, sample_weight=None):

        np.random.seed(42)
        np.random.seed(10)

        if sample_weight is None:
            sample_weight = np.ones(y.shape)

        # Lasso
        model_lasso = Lasso(alpha=self.alpha_lasso, random_state=self.random_state, max_iter=1000,warm_start=True)
        
        self.scale=float(self.bounds[1] - self.bounds[0]) 
        self.mean = np.mean(y)
        
        model_lasso.fit(X, (y-self.mean)/float(self.scale))
        self.lasso_coef = model_lasso.coef_
        self.model_lasso = model_lasso

        # feature selection        
        feature_support = np.where(abs(model_lasso.coef_)/(1e-10+np.sum(abs(model_lasso.coef_))) > self.tol)[0]
        if(len(feature_support)<5):
            ind = np.argsort(-1*np.abs(model_lasso.coef_))
            feature_support  = ind[:5]

        if(len(feature_support)>0.5*X.shape[0]):
            ind = np.argsort(-1*np.abs(model_lasso.coef_))
            maxd = int(0.5*X.shape[0])
            feature_support  = ind[:maxd]            

        self.feature_support = feature_support


        self.ridge_coef = np.zeros(X.shape[1])
        if len(self.feature_support) > 0:
            # Ridge
            model_ridge = Ridge(alpha=self.alpha_ridge)
            model_ridge.fit(X[:, feature_support], (y-self.mean)/self.scale, sample_weight=sample_weight)
            self.ridge_coef[feature_support] = model_ridge.coef_ 
            self.model_ridge = model_ridge
        else:
            X2 = np.ones((X.shape[0], 1))
            model_ridge = Ridge(alpha=self.alpha_ridge)
            model_ridge.fit(X2, y, sample_weight=sample_weight)
                              
        self.model_ridge = model_ridge            

        return self

    def score(self, X, y, sample_weight=None):

        if sample_weight is None:
            sample_weight = np.ones(y.shape)

        self.fit(X, y, sample_weight=sample_weight)
        y_predict = self.predict(X)
        score_ = pearsonr(y, y_predict)

        return score_

    def get_params(self, deep=True):

        params = {"alpha_lasso": self.alpha_lasso, "alpha_ridge": self.alpha_ridge}

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

        return y_predict
