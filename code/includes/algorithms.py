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


def softmax_regression(train_set, test_set, verbose, l_rate, n_classes, n_epochs, batch_size, tol):
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


def boosted_softmax_regression(
        train_set, test_set, verbose, l_rate, n_classes,
        n_epochs, batch_size, m_stop, eta, activation, plot_file
):
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
    plt.clf()

    return predictions, model


def mlp(train_set, test_set, verbose, l_rate, n_classes, n_epochs, batch_size, hidden_size, tol):
    model = MLP(
        hidden_layer_sizes=hidden_size,
        learning_rate_init=l_rate,
        batch_size=batch_size,
        max_iter=n_epochs,
        solver="sgd",
        activation="tanh",
        nesterovs_momentum=False,
        verbose=verbose,
        tol=tol,
    )

    train_X = train_set[:, :-1]
    train_Y = np.eye(n_classes)[train_set[:, -1].astype(int)]

    model.fit(train_X, train_Y)

    predictions = model.predict(test_set[:, :-1])
    predictions = np.argmax(predictions, axis=1)

    return predictions, model
