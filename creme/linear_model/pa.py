import collections

import numpy as np

from .. import base
from .. import optim
from .. import utils


__all__ = ['PAClassifier', 'PARegressor']


class BasePA:

    def __init__(self, C, mode, fit_intercept, data, learning_rate, rho):
        self.C = C
        self.calc_tau = {0: self._calc_tau_0, 1: self._calc_tau_1, 2: self._calc_tau_2}[mode]
        self.fit_intercept = fit_intercept
        self.weights_x = collections.defaultdict(float)
        self.weights_y = collections.defaultdict(float)
        self.intercept_x = 0.
        self.intercept_y = 0.
        self.data=data
        self.learning_rate = learning_rate
        self.rho = rho
        self.momentum_x = collections.defaultdict(float)
        self.momentum_y = collections.defaultdict(float)

    @classmethod
    def _calc_tau_0(cls, x, loss):
        norm = utils.math.norm(x, order=2) ** 2
        if norm > 0:
            return loss / utils.math.norm(x, order=2) ** 2
        return 0

    def _calc_tau_1(self, x, loss):
        norm = utils.math.norm(x, order=2) ** 2
        if norm > 0:
            return min(self.C, loss / norm)
        return 0

    def _calc_tau_2(self, x, loss):
        return loss / (utils.math.norm(x, order=2) ** 2 + 0.5 / self.C)


class PARegressor(BasePA, base.Regressor):
    """Passive-aggressive learning for regression.

    Example:

        The following example is taken from `this blog post <https://www.bonaccorso.eu/2017/10/06/ml-algorithms-addendum-passive-aggressive-algorithms/>`_.

        ::

            >>> from creme import linear_model
            >>> from creme import metrics
            >>> from creme import stream
            >>> import numpy as np
            >>> from sklearn import datasets

            >>> np.random.seed(1000)
            >>> X, y = datasets.make_regression(n_samples=500, n_features=4)

            >>> model = linear_model.PARegressor(
            ...     C=0.01,
            ...     mode=2,
            ...     eps=0.1,
            ...     fit_intercept=False
            ... )
            >>> metric = metrics.MAE() + metrics.MSE()

            >>> for xi, yi in stream.iter_array(X, y):
            ...     y_pred = model.predict_one(xi)
            ...     model = model.fit_one(xi, yi)
            ...     metric = metric.update(yi, y_pred)

            >>> print(metric)
            MAE: 10.123199, MSE: 843.816135

        References:
            1. `Online Passive-Aggressive Algorithms <http://jmlr.csail.mit.edu/papers/volume7/crammer06a/crammer06a.pdf>`_

    """

    def __init__(self, C=1.0, mode=1, eps=0.1, fit_intercept=True, data=[], learning_rate=0.1, rho=0.9):
        super().__init__(C=C, mode=mode, fit_intercept=fit_intercept, data=data, learning_rate=learning_rate, rho=rho)
        self.loss = optim.losses.EpsilonInsensitiveHinge(eps=eps)
        self.x_act = None
        self.y_act = None

    def fit_one(self, X, x, y):

        x_pred, y_pred = self.predict_one(X, False, -1, -1)
        tau_x = self.calc_tau(X, self.loss(x, x_pred))
        tau_y = self.calc_tau(X, self.loss(y, y_pred))
        step_x = tau_x * np.sign(x - x_pred)
        step_y = tau_y * np.sign(y - y_pred)
       	
        for i, xi in X.items():
            self.momentum_x[i] = self.rho * self.momentum_x[i] + (1 - self.rho) * step_x ** 2
            self.momentum_y[i] = self.rho * self.momentum_y[i] + (1 - self.rho) * step_y ** 2
            self.weights_x[i] += step_x * xi 
            self.weights_y[i] += step_y * xi

        if self.fit_intercept:
            self.intercept_x += step_x
            self.intercept_y += step_y

        return self

    def fit_n(self, frames):
        for k in frames:
            [X, x, y] = self.data[k]
            x_pred, y_pred = self.predict_one(X, False, -1, -1)
            tau_x = self.calc_tau(X, self.loss(x, x_pred))
            tau_y = self.calc_tau(X, self.loss(y, y_pred))
            step_x = self.learning_rate * tau_x * np.sign(x - x_pred)
            step_y = self.learning_rate * tau_y * np.sign(y - y_pred)
        
            # x_pred, y_pred = self.predict_one(X)
            # loss_x = self.loss(x, x_pred)
            # loss_y = self.loss(y, y_pred)
            # sign_x = np.sign(x - x_pred)
            # sign_y = np.sign(y - y_pred)
        

            for i, xi in X.items():
                self.momentum_x[i] = self.rho * self.momentum_x[i] + (1 - self.rho) * step_x ** 2
                self.momentum_y[i] = self.rho * self.momentum_y[i] + (1 - self.rho) * step_y ** 2
                self.weights_x[i] += step_x * xi
                self.weights_y[i] += step_y * xi

            if self.fit_intercept:
                self.intercept_x += step_x
                self.intercept_y += step_y

        return self

    def predict_one(self, x, use_momentum, x_act, y_act):
            return utils.math.dot(x, self.weights_x) + self.intercept_x, utils.math.dot(x, self.weights_y) + self.intercept_y


class PAClassifier(BasePA, base.BinaryClassifier):
    """Passive-aggressive learning for classification.

    Example:

        The following example is taken from `this blog post <https://www.bonaccorso.eu/2017/10/06/ml-algorithms-addendum-passive-aggressive-algorithms/>`_.

        ::

            >>> from creme import linear_model
            >>> from creme import metrics
            >>> from creme import stream
            >>> import numpy as np
            >>> from sklearn import datasets
            >>> from sklearn import model_selection

            >>> np.random.seed(1000)
            >>> X, y = datasets.make_classification(
            ...     n_samples=5000,
            ...     n_features=4,
            ...     n_informative=2,
            ...     n_redundant=0,
            ...     n_repeated=0,
            ...     n_classes=2,
            ...     n_clusters_per_class=2
            ... )

            >>> X_train, X_test, y_train, y_test = model_selection.train_test_split(
            ...     X,
            ...     y,
            ...     test_size=0.35,
            ...     random_state=1000
            ... )

            >>> model = linear_model.PAClassifier(
            ...     C=0.01,
            ...     mode=1
            ... )

            >>> for xi, yi in stream.iter_array(X_train, y_train):
            ...     y_pred = model.fit_one(xi, yi)

            >>> metric = metrics.Accuracy() + metrics.LogLoss()

            >>> for xi, yi in stream.iter_array(X_test, y_test):
            ...     metric = metric.update(yi, model.predict_proba_one(xi))

            >>> print(metric)
            Accuracy: 0.884571, LogLoss: 0.325727

    References:
        1. `Online Passive-Aggressive Algorithms <http://jmlr.csail.mit.edu/papers/volume7/crammer06a/crammer06a.pdf>`_

    """

    def __init__(self, C=1.0, mode=1, fit_intercept=True):
        super().__init__(C=C, mode=mode, fit_intercept=fit_intercept)
        self.loss = optim.losses.Hinge()

    def fit_one(self, x, y):

        y_pred = utils.math.dot(x, self.weights) + self.intercept
        tau = self.calc_tau(x, self.loss(y, y_pred))
        step = tau * (y or -1)  # y == False becomes -1

        for i, xi in x.items():
            self.weights[i] += step * xi
        if self.fit_intercept:
            self.intercept += step

        return self

    def predict_proba_one(self, x):
        y_pred = utils.math.sigmoid(utils.math.dot(x, self.weights) + self.intercept)
        return {False: 1. - y_pred, True: y_pred}
