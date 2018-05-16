
import numpy as np
from numpy import linalg as LA
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from MLE.pyPQN.minConF_PQN import minConf_PQN
from MLE.pyPQN.SoftmaxLoss2_weighted import SoftmaxLoss2_weighted
from MLE.pyPQN.projectRandom2 import randomProject


class LogisticRegressionPQN:

    def __init__(self, marker_groups, C=None, lbda=None):
        """
        Wrapper for Logistic Regression model combining Group Lasso (using PQN) and regular Logistic Regression.
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
        self.feature_support = None
        self.feature_indices = None
        self.group_lasso_strategy = 'PQN'  # 'PQN' or 'None'
        self.coef = None
        self.sample_weight = None
        self.X = None
        self.y = None
        self.n_classes = 3

    def fit(self, X, y, sample_weight=None):

        if sample_weight is None:
            sample_weight = np.ones(y.shape)

        self.sample_weight = sample_weight
        self.X = X
        self.y = y

        if self.group_lasso_strategy == 'PQN':
            feature_support = self.group_lasso_PQN(X, y, sample_weight)
        else:
            markers, repeats = np.unique(self.marker_groups, return_counts=True)
            feature_support = markers

        markers, repeats = np.unique(self.marker_groups, return_counts=True)
        feature_indices = []
        for i in markers:
            if i in feature_support:
                feature_indices.extend(np.repeat(True, repeats[i]))
            else:
                feature_indices.extend(np.repeat(False, repeats[i]))

        model = LogisticRegression(random_state=0, C=self.C)
        model.fit(X[:, feature_indices], y, sample_weight=sample_weight)
        self.coef = model.coef_

        self.model = model
        self.feature_support = feature_support
        self.feature_indices = feature_indices

        return self

    def score(self, X, y, sample_weight=None):

        self.fit(X, y, sample_weight)
        y_predict = self.predict(X)
        score_ = accuracy_score(y, y_predict)

        return score_

    def get_params(self, deep=True):

        params = {"C": self.C, "lbda": self.lbda, "marker_groups": self.marker_groups}

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

    def group_lasso_PQN(self, X, y, sample_weight):

        d1, d2 = X.shape
        markers, repeats = np.unique(self.marker_groups, return_counts=True)

        w1 = np.ravel(np.zeros((d2, self.n_classes-1)))

        w2 = minConf_PQN(self.fun_object, w1, self.fun_project, verbose=0)[0]

        feature_support = []
        for j in markers:
            indices = np.where(self.marker_groups == j)[0]
            if LA.norm(w2[indices], 2) != 0:
                feature_support.append(int(j))

        return feature_support

    def fun_project(self, w):

        markers, repeats = np.unique(self.marker_groups, return_counts=True)
        marker_groups_ravel = np.repeat(markers, repeats*(self.n_classes-1))

        v = np.zeros((markers.shape[0],))
        v1 = np.zeros(w.shape)

        for i in markers:
            indices1 = np.where(marker_groups_ravel == i)[0]
            w_group = w[indices1]
            v[i] = LA.norm(w_group, 2)

            if v[i] != 0:
                v1[indices1] = w_group / v[i]

        p = randomProject(v, self.lbda)

        test = v1 * np.repeat(p, repeats*(self.n_classes-1))

        return test

    def fun_object(self, w):

        test = SoftmaxLoss2_weighted(w, self.X, self.y, self.n_classes, self.sample_weight)

        return test
