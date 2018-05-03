import sys
import utils
import numpy as np

from math import ceil
from tqdm import tqdm

from utils import bold

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc_params_from_file("includes/mlprc")

from models.softmax import Softmax
from models.logistic import Logistic
from sklearn.neural_network import MLPClassifier as MLP


def softmax_regression(train_set, test_set, verbose, l_rate,
                       n_classes, n_epochs, batch_size, tol):
    model = Softmax(
        data=train_set,
        n_classes=n_classes,
        l_rate=l_rate,
        n_epochs=n_epochs,
        tol=tol,
        batch_size=batch_size,
        verbose=verbose,
        direct=True
    )
    model.fit()

    predictions = model.predict(test_set[:, :-1])
    predictions = np.argmax(predictions, axis=1)

    return predictions, model


def boosted_softmax_regression(train_set, test_set, verbose,
                               l_rate, n_classes, n_epochs,
                               batch_size, m_stop, eta, activation,
                               plot_file):
    model = Softmax(
        data=train_set,
        n_classes=n_classes,
        l_rate=l_rate,
        n_epochs=n_epochs,
        batch_size=batch_size,
        activation=activation,
        verbose=verbose
    )

    if verbose:
        print bold("Stop 0")
    model.fit()

    accuracies = list()

    predictions = model.predict(test_set[:, :-1])
    predictions = np.argmax(predictions, axis=1)

    accuracies.append(np.mean(test_set[:, -1] == predictions) * 100)

    print bold("Stop 0") + ", Test Accuracy: % f\n" % accuracies[-1]

    for stop in range(m_stop - 1):
        if verbose:
            print "\033[1mStop %d\033[0m" % (stop + 1)

        model.fit(boosted=True)

        predictions = model.predict(test_set[:, :-1])
        predictions = np.argmax(predictions, axis=1)

        accuracies.append(np.mean(test_set[:, -1] == predictions) * 100)

        print bold("Stop %d" % (stop+1)) + \
            ", Test Accuracy: % f\n" % accuracies[-1]

        if model.converged:
            break

    np.save(plot_file, accuracies)
    plt.plot(accuracies)
    plt.savefig("plots/boosted-sft-plot.png")
    plt.close()

    return predictions, model


def mlp(train_set, test_set, verbose, l_rate, n_classes,
        activation, n_epochs, batch_size, hidden_size, tol):
    model = MLP(
        hidden_layer_sizes=hidden_size,
        learning_rate_init=l_rate,
        batch_size=batch_size,
        max_iter=n_epochs,
        solver="sgd",
        activation=activation,
        nesterovs_momentum=False,
        verbose=verbose,
        tol=tol,
    )

    train_X = train_set[:, :-1]
    train_Y = np.eye(n_classes)[train_set[:, -1].astype(int)]

    model.fit(train_X, train_Y)

    predictions = model.predict_proba(test_set[:, :-1])
    predictions = np.argmax(predictions, axis=1)

    return predictions, model


def pretrained_mlp(train_set, test_set, verbose, l_rate,
                   n_classes, activation, n_epochs, batch_size,
                   hidden_size, tol, bm):
    model = MLP(
        hidden_layer_sizes=hidden_size,
        learning_rate_init=l_rate,
        batch_size=batch_size,
        max_iter=1,
        solver="sgd",
        activation=activation,
        nesterovs_momentum=False,
        verbose=verbose,
        tol=tol,
        warm_start=True
    )

    sample = np.asarray(train_set[0], dtype=int)
    sample_x = sample[:-1]
    sample_y = np.zeros(n_classes)
    sample_y[sample[-1]] = 1
    model.fit([sample_x], [sample_y], )

    model.intercepts_[0][0] = bm.bias_input[0]
    model.intercepts_[1][:] = bm.bias_hidden[:]

    for hn in range(1, hidden_size):
        model.intercepts_[0][hn] = bm.models[hn - 1][1].intercepts_[0][0]
        model.intercepts_[1][:] += bm.models[hn - 1][0] * \
            bm.models[hn - 1][1].intercepts_[1][:]

    model.coefs_[0][:, 0] = bm.coefficients_input[:, 0]
    model.coefs_[1][0, :] = bm.coefficients_hidden[:, 0]

    for hn in range(1, hidden_size):
        model.coefs_[0][:, hn] = bm.models[hn - 1][1].coefs_[0][:, 0]
        model.coefs_[1][hn, :] = bm.models[hn - 1][0] * \
            bm.models[hn - 1][1].coefs_[1]

    if verbose == True:
        predictions = model.predict_proba(test_set[:, :-1])
        predictions = np.argmax(predictions, axis=1)

        accuracy = np.mean(test_set[:, -1] == predictions)

        print "Pretrained MLP, Activation = %s", activation
        print "Initial Test Accuracy: %f" % accuracy

    for _ in range(n_epochs):
        model.fit(train_set[:, :-1], np.eye(n_classes)
                  [np.asarray(train_set[:, -1], dtype=int)])

    predictions = model.predict_proba(test_set[:, :-1])
    predictions = np.argmax(predictions, axis=1)

    accuracy = np.mean(predictions == test_set[:, -1])

    return accuracy, model
