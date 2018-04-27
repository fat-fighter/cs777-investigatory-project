import sys
import numpy as np

from math import ceil
from includes.utils import folds

from sklearn.neural_network import MLPRegressor


class Logistic:
    def __init__(self, data, l_rate=0.1, l_decay=1.0, eta=1.0, eta_decay=0.9, n_iters=200, batch_size=50, tol=0.0001, verbose=False):
        self.data = data

        self.coefficients = np.zeros(data.shape[1] - 1, dtype=float)
        self.intercept = 0.0

        self.tol = tol
        self.l_rate = l_rate
        self.l_decay = l_decay
        self.verbose = verbose
        self.n_epochs = n_iters
        self.batch_size = batch_size

        self.weak_regressor_parameter = {
            "hidden_layer_sizes": (1),
            "activation": "identity",
            "solver": "sgd",
            "alpha": 0,
            "batch_size": self.batch_size,
            "learning_rate_init": 0.001,
            "max_iter": 200,
            "tol": 0.00005,
            "momentum": 0,
            "verbose": verbose
        }

        self.models = list()

    def predict(self, X):
        yhat = \
            self.intercept + \
            np.sum(self.coefficients * X, axis=1) + \
            np.sum(np.array([
                weight * np.array(model.predict(X)) for weight, model in self.models
            ]), axis=0)

        return 1 / (1 + np.exp(-yhat))

    def fit(self, weight=1.0, boosted=False):
        prev_loss = float("inf")

        if boosted:
            np.random.shuffle(self.data)
            yhat = self.predict(self.data[:, :-1])
            gradients = self.data[:, -1] - yhat

            weak_model = MLPRegressor()
            weak_model.set_params(**self.weak_regressor_parameter)

            weak_model.fit(self.data[:, :-1], gradients)

            self.models.append((self.l_rate, weak_model))

        else:
            for n_epoch in range(self.n_epochs):
                np.random.shuffle(self.data)

                for batch in folds(self.data, ceil(self.data.shape[0] / float(self.batch_size))):
                    batch_X, batch_Y = batch[:, :-1], batch[:, -1]

                    yhat = self.predict(batch_X)
                    error = (batch_Y - yhat) * yhat * (1.0 - yhat)

                    self.intercept += weight * self.l_rate * sum(error)
                    self.coefficients += weight * self.l_rate * np.sum(
                        error[:, None] * batch_X, axis=0
                    )

                self.l_rate = max(self.l_rate * self.l_decay, 0.01)

                yhat = self.predict(self.data[:, :-1])
                loss = -1 * np.mean(
                    self.data[:, -1] * np.log(yhat) +
                    (1 - self.data[:, -1]) * np.log(1 - yhat)
                )
                if self.verbose:
                    print "\tEpoch %d, Loss = %f" % (n_epoch, loss)

                if abs(prev_loss - loss) < self.tol:
                    if self.verbose:
                        print "Training loss did not improve more than tol=%f for two consecutive epochs. Stopping." % self.tol
                    return

                prev_loss = loss

        if self.verbose:
            print
