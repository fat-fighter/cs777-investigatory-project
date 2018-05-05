import sys
import numpy as np

from math import ceil
from includes.utils import folds

from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier


class Softmax:
    def __init__(self, data, n_classes, l_rate=0.1, l_decay=0.95, eta=1.0, eta_decay=0.9, n_epochs=200, batch_size=50, tol=0.0001, verbose=False, direct=False, activation="relu"):
        if n_classes > 1:
            self.X = data[:, :-1]
            self.Y = np.eye(n_classes)[data[:, -1].astype(int)]

            self.data = np.concatenate((self.X, self.Y), axis=1)
        else:
            self.X = data[:, :-1]
            self.Y = data[:, -1]

            self.data = np.copy(data)

        self.n_classes = n_classes

        self.converged = False

        self.eta = eta
        self.eta_decay = eta_decay

        self.verbose = verbose

        self.direct = direct

        if direct == True:
            hidden_layer_sizes = ()
        else:
            hidden_layer_sizes = (2)

            if activation == "relu":
                self.activation = lambda x: np.maximum(x, 0)
            elif activation == "identity":
                self.activation = lambda x: x
            elif activation == "tanh":
                self.activation = lambda x: np.tanh(x)
            elif activation == "logistic":
                self.activation = lambda x: 1.0 / (1.0 + np.exp(-x))
            else:
                assert(False)

        self.init_model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            max_iter=n_epochs,
            tol=tol,
            learning_rate_init=l_rate,
            verbose=verbose
        )

        self.weak_regressor_parameters = {
            "hidden_layer_sizes": hidden_layer_sizes,
            "activation": activation,
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

    def predict(self, X, linear=False):
        if self.direct == True:
            assert(linear == False)

            return self.init_model.predict_proba(X)

        yhat_l = (
            self.bias_hidden +
            np.matmul(
                self.activation(
                    self.bias_input +
                    np.matmul(X, self.coefficients_input)
                ), self.coefficients_hidden
            ) +
            np.sum(np.array([
                weight * np.array(model.predict(X)) for weight, model in self.models
            ]), axis=0)
        )

        if linear == True:
            return yhat_l

        yhat = np.exp(yhat_l)

        return yhat / np.sum(yhat, axis=1)[:, None]

    def get_best_weight(self, model):
        np.random.shuffle(self.data)

        val = np.split(self.data[:1000], [-self.n_classes], axis=1)
        assert(len(val) == 2)

        val_X, val_Y = val
        val_Y = np.argmax(val_Y, axis=1)

        best_weight = -1
        best_accuracy = 0

        yhatl = self.predict(val_X, linear=True)
        model_yhatl = np.array(model.predict(val_X))

        weights = [0.01, 0.03] + [x * 0.1 for x in range(0, 20)]
        for weight in weights:
            yhat = np.exp(yhatl + weight * model_yhatl)
            yhat = yhat / np.sum(yhat, axis=1)[:, None]

            accuracy = np.mean(np.argmax(yhat, axis=1) == val_Y)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_weight = weight

        return best_weight

    def fit(self, weight=1.0, boosted=False):
        if boosted == True:
            eta = 0

            self.converged = True
            for _ in range(0, 5):
                yhat = self.predict(self.X)
                gradients = self.Y - yhat

                weak_model = MLPRegressor()
                weak_model.set_params(**self.weak_regressor_parameters)

                weak_model.fit(self.X, gradients)

                eta = self.get_best_weight(weak_model)

                if self.verbose == True:
                    print
                    print "eta: %f" % eta
                    print

                if eta != 0:
                    self.converged = False

                if eta >= 0.1:
                    break

            self.models.append((eta, weak_model))

        else:
            self.init_model.fit(self.X, self.Y)

            if self.direct == False:
                self.bias_input = np.array(self.init_model.intercepts_[0])
                self.bias_hidden = np.array(self.init_model.intercepts_[1])

                self.coefficients_input = np.array(
                    self.init_model.coefs_[0]
                )
                self.coefficients_hidden = np.array(
                    self.init_model.coefs_[1]
                )

            if self.verbose == True:
                print
