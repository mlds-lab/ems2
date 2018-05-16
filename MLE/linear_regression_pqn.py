
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from MLE.pyPQN.minConF_PQN import minConf_PQN
from MLE.pyPQN.SquaredError import SquaredError
from MLE.pyPQN.projectRandom2 import randomProject


class LinearRegressionPQN:

    def __init__(self, bounds=[-np.inf,np.inf], alpha_ridge=None, alpha_lasso=None, tol=0.00001, random_state=0):
        """
        Wrapper for Linear Regression model combining Lasso (using PQN) and Ridge.
        This wrapper is compatible with GridSearchCV of Sci-Kit-Learn. Main methods are fit and predict.

        Args:
            alpha_ridge (int): Ridge regularizer
            alpha_lasso (int): Group Lasso regularizer
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
        self.sample_weight = None
        self.X = None
        self.y = None

    def clip(self, y):
        y[y < self.bounds[0]] = self.bounds[0]
        y[y > self.bounds[1]] = self.bounds[1]
        return y

    def fit(self, X, y, sample_weight=None):

        if sample_weight is None:
            sample_weight = np.ones(y.shape)

        self.sample_weight = sample_weight
        self.X = X
        self.scale = 1 + np.std(y)
        self.mean = np.mean(y)
        self.y = (y - self.mean) / self.scale

        # Lasso
        self.lasso_coef = self.lasso_PQN(X)

        # feature selection
        feature_support = np.where(abs(self.lasso_coef) / (1e-10 + np.sum(abs(self.lasso_coef))) > self.tol)[0]
        self.feature_support = feature_support

        # Ridge
        self.ridge_coef = np.zeros(X.shape[1])
        if len(self.feature_support) > 0:
            model_ridge = Ridge(alpha=self.alpha_ridge)
            model_ridge.fit(X[:, feature_support], self.y, sample_weight=sample_weight)
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

        self.fit(X, y, sample_weight)
        y_predict = self.predict(X)
        score_ = pearsonr(y, y_predict)

        return score_

    def get_params(self, deep=True):

        params = {"alpha_ridge": self.alpha_ridge, "alpha_lasso": self.alpha_lasso}

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

    def lasso_PQN(self, X):

        w1 = np.zeros((X.shape[1],))

        w2 = minConf_PQN(self.fun_object, w1, self.fun_project, verbose=0)[0]

        return w2

    def fun_project(self, w):

        d1, d2 = self.X.shape

        v = np.zeros((d2,))
        v1 = np.zeros(d2)
        for i in range(d2):
            v[i] = abs(w[i])
            if v[i] != 0:
                v1[i] = w[i] / v[i]

        p = randomProject(v, self.alpha_lasso)
        test = v1 * p

        return test

    def fun_object(self, w):

        test = SquaredError(w, np.tile(np.reshape(np.sqrt(self.sample_weight), (np.size(self.X, 0), 1)),
                            (1, np.size(self.X, 1))) * self.X, np.sqrt(self.sample_weight) * self.y)

        return test
