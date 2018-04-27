import sys
import utils
import numpy as np

from tqdm import tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc_params_from_file("includes/mlprc")

from models.softmax import Softmax
from models.logistic import Logistic
from sklearn.neural_network import MLPClassifier as MLP

from math import ceil


def logistic_regression(train_set, test_set, verbose, l_rate, n_epochs, batch_size, tol):
    model = Logistic(
        data=train_set,
        l_rate=l_rate,
        n_iters=n_epochs,
        batch_size=batch_size,
        tol=tol,
        verbose=verbose
    )
    model.fit()

    predictions = model.predict(test_set[:, :-1])
    predictions = np.round(predictions)

    return predictions


def boosted_logistic_regression(train_set, test_set, verbose, l_rate, eta, eta_decay, n_epochs, batch_size, tol, m_stop):
    model = Logistic(
        data=train_set,
        l_rate=l_rate,
        eta=eta,
        eta_decay=eta_decay,
        n_iters=n_epochs,
        batch_size=batch_size,
        tol=tol,
        verbose=verbose
    )
    model.fit()

    with tqdm(range(m_stop)) as bar:
        bar.set_postfix(**{"Accuracy": 0})
        for _ in bar:
            model.fit(boosted=True)

            predictions = model.predict(test_set[:, :-1])
            predictions = np.round(predictions)

            accuracy = np.mean(test_set[:, -1] == predictions) * 100
            bar.set_postfix(**{"Accuracy": accuracy})

    return predictions


def mlp(train_set, test_set, verbose, l_rate, n_epochs, batch_size, hidden_size):
    model = MLP(
        hidden_layer_sizes=hidden_size,
        learning_rate_init=0.1,
        batch_size=batch_size,
        max_iter=n_epochs,
        activation='relu',
        solver='sgd',
        nesterovs_momentum=False,
        verbose=verbose
    )

    model.fit(train_set[:, :-1], train_set[:, -1])

    predictions = model.predict(test_set[:, :-1])

    return predictions


def softmax_regression(train_set, test_set, verbose, l_rate, n_classes, n_epochs, batch_size, tol):
    model = Softmax(
        data=train_set,
        n_classes=n_classes,
        l_rate=l_rate,
        n_epochs=n_epochs,
        tol=tol,
        batch_size=batch_size,
        verbose=verbose
    )
    model.fit()

    predictions = model.predict(test_set[:, :-1])
    predictions = np.argmax(predictions, axis=1)

    return predictions


def boosted_softmax_regression(train_set, test_set, verbose, l_rate, n_classes, n_epochs, batch_size, m_stop, eta):
    model = Softmax(
        data=train_set,
        n_classes=n_classes,
        l_rate=l_rate,
        n_epochs=n_epochs,
        batch_size=batch_size,
        verbose=verbose
    )

    print "\033[1mStop 0\033[0m"
    model.fit()

    accuracies = list()

    predictions = model.predict(test_set[:, :-1])
    predictions = np.argmax(predictions, axis=1)

    accuracies.append(np.mean(test_set[:, -1] == predictions) * 100)
    print "Test Accuracy: %f\n" % accuracies[-1]

    for stop in range(m_stop - 1):
        print "\033[1mStop %d\033[0m" % (stop + 1)
        model.fit(boosted=True)

        predictions = model.predict(test_set[:, :-1])
        predictions = np.argmax(predictions, axis=1)

        accuracies.append(np.mean(test_set[:, -1] == predictions) * 100)
        print "Test Accuracy: %f\n" % accuracies[-1]

    plt.plot(accuracies)
    plt.savefig("plots/boosted-sft-plot.png")
    plt.clf()

    return predictions, model


def multi_mlp(train_set, test_set, verbose, l_rate, n_classes, n_epochs, batch_size, hidden_size, tol):
    model = MLP(
        hidden_layer_sizes=hidden_size,
        learning_rate_init=l_rate,
        batch_size=batch_size,
        max_iter=n_epochs,
        solver="sgd",
        activation="relu",
        nesterovs_momentum=False,
        verbose=verbose,
        tol=tol,
    )

    train_X = train_set[:, :-1]
    train_Y = np.eye(n_classes)[train_set[:, -1].astype(int)]

    model.fit(train_X, train_Y)

    predictions = model.predict(test_set[:, :-1])
    predictions = np.argmax(predictions, axis=1)

    return predictions
