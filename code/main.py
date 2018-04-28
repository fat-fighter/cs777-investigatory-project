%load_ext autoreload
%autoreload 2

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams = mpl.rc_params_from_file("includes/mlprc")

import os
if not os.path.exists("plots"):
    os.makedirs("plots")

import numpy as np

from includes import utils
from includes import config
from includes import algorithms


def evaluate_algorithm(train_set, test_set, algorithm, verbose, *args):
    if len(test_set) == 0:
        np.random.shuffle(train_set)

        train_set, test_set = np.split(
            train_set, [int(train_set.shape[0] * 0.7)])

    predicted, model = algorithm(train_set, test_set, verbose, *args)

    accuracy = np.mean(test_set[:, -1] == predicted) * 100

    return accuracy, model


train_set, test_set, n_classes = utils.get_data("mnist")

conf = config.Config(verbose=False)

conf.set_algorithm("boosted-sft-reg")
for activation in ["logistic", "relu", "tanh", "identity"]:
    accuracy, model = evaluate_algorithm(
        train_set,
        test_set,
        algorithms.boosted_softmax_regression,
        conf.verbose,
        conf.lr,
        n_classes,
        conf.n_epochs,
        conf.batch_size,
        conf.m_stop,
        conf.eta,
        activation,
        "plots/sft-data-%s.npy" % activation
    )

    print "Boosted Softmax Regression, Activation = %s" % activation
    print "Final Test Accuracy: %f" % accuracy

# conf.set_algorithm("multi-mlp")
# accuracy, model = evaluate_algorithm(
#     train_set,
#     test_set,
#     algorithms.mlp,
#     conf.verbose,
#     conf.lr,
#     n_classes,
#     conf.n_epochs,
#     conf.batch_size,
#     conf.hidden_size,
#     conf.tol
# )

# print "Multi Layer Perceptron"
# print "Accuracy: %f" % accuracy

# colors = ["blue", "red", "green", "orange"]
# activations = ["logistic", "relu", "tanh", "identity"]

# for color, activation in zip(colors, activations):
#     data = np.load("plots/sft-data-%s.npy" % activation)
#     plt.plot(data, label=activation, color=color)
#     plt.xlabel("m_stop")

# plt.legend()
# plt.show()
