import sys
import numpy as np

from math import ceil
from includes.utils import folds

from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier


class Softmax:
    def __init__(self, data, n_classes, l_rate=0.1, l_decay=0.95, eta=1.0, eta_decay=0.9, n_epochs=200, batch_size=50, tol=0.0001, verbose=False):
        self.X = data[:, :-1]
        self.Y = np.eye(n_classes)[data[:, -1].astype(int)]

        self.data = np.concatenate((self.X, self.Y), axis=1)

        self.n_classes = n_classes

        # self.activation = lambda x: x                     # IDENTITY
        self.activation = lambda x: np.maximum(x, 0)      # RELU
        # self.activation = lambda x: np.tanh(x)            # TANH

        self.eta = eta
        self.eta_decay = eta_decay

        self.verbose = verbose

        # self.tol = tol
        # self.l_rate = l_rate
        # self.l_decay = l_decay
        # self.max_iters = n_epochs
        # self.n_classes = n_classes
        # self.batch_size = batch_size

        self.init_model = MLPClassifier(
            hidden_layer_sizes=(1),
            activation="relu",
            max_iter=n_epochs,
            tol=tol,
            learning_rate_init=l_rate,
            verbose=verbose
        )

        self.weak_regressor_parameters = {
            "hidden_layer_sizes": (1),
            "activation": "relu",
            "solver": "sgd",
            "alpha": 0,
            "batch_size": batch_size,
            "learning_rate_init": 0.001,
            "max_iter": 50,
            "tol": 0.00001,
            "momentum": 0,
            "verbose": verbose
        }

        self.models = list()

    def predict(self, X):
        yhat = (
            self.bias_hidden +
            np.matmul(
                self.activation(
                    self.bias_input +
                    np.matmul(X, self.coefficients_input.transpose())
                ), self.coefficients_hidden.transpose()
            ) +
            np.sum(np.array([
                weight * np.array(model.predict(X)) for weight, model in self.models
            ]), axis=0)
        )

        yhat = np.exp(yhat)

        return yhat / np.sum(yhat, axis=1)[:, None]

    def get_best_weight(self, model):
        np.random.shuffle(self.data)

        val_X, val_Y = np.split(self.data[:1000], [-self.n_classes], axis=1)
        val_Y = np.argmax(val_Y, axis=1)

        best_weight = -1
        best_accuracy = 0

        for weight in [0.01, 0.03, 0.1, 0.3, 0.5, 0.6, 0.9, 1.0]:
            self.models.append((weight, model))

            predictions = self.predict(val_X)
            predictions = np.argmax(predictions, axis=1)

            accuracy = np.mean(predictions == val_Y)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_weight = weight

            self.models.pop()

        return best_weight

    def fit(self, weight=1.0, boosted=False):
        if boosted:
            yhat = self.predict(self.X)
            gradients = self.Y - yhat

            weak_model = MLPRegressor()
            weak_model.set_params(**self.weak_regressor_parameters)

            weak_model.fit(self.X, gradients)

            eta = self.get_best_weight(weak_model)

            print eta
            self.models.append((eta, weak_model))
            # self.eta = max(self.eta * self.eta_decay, 0.1)

        else:
            self.init_model.fit(self.X, self.Y)

            self.bias_input = np.array(self.init_model.intercepts_[0])
            self.bias_hidden = np.array(self.init_model.intercepts_[1])

            self.coefficients_input = np.array(
                self.init_model.coefs_[0]
            ).transpose()
            self.coefficients_hidden = np.array(
                self.init_model.coefs_[1]
            ).transpose()

            if self.verbose:
                print
