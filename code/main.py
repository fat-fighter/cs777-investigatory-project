import numpy as np

from includes import utils
from includes import algorithms
from includes.config import config


eval_seed = np.random.randint(0, 100)


def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    np.random.seed(eval_seed)
    folds = utils.folds(dataset, n_folds)
    np.random.seed(np.random.randint(0, 100))

    scores = list()
    for n_fold in range(n_folds):
        indices = [True] * n_folds
        indices[n_fold] = False

        train_set = np.concatenate(folds[indices], axis=0)

        test_set = folds[n_fold]

        predicted = algorithm(train_set, test_set, *args)

        accuracy = np.mean(test_set[:, -1] == predicted) * 100
        scores.append(accuracy)

    return scores


def main():
    dataset = utils.get_data("mnist")

    config.set_algorithm("log-reg")
    scores = evaluate_algorithm(
        dataset,
        algorithms.logistic_regression,
        config.n_folds,
        config.lr,
        config.n_epochs,
        config.batch_size
    )

    print "Logistic Regression"
    print "\tScores: %s" % scores
    print "\tMean Accuracy: %.3f%%" % (sum(scores)/float(len(scores)))
    print ""

    config.set_algorithm("boosted-log-reg")
    scores = evaluate_algorithm(
        dataset,
        algorithms.boosted_logistic_regression,
        config.n_folds,
        config.lr,
        config.n_epochs,
        config.batch_size,
        config.m_stop,
        config.eta
    )

    print "Boosted Logistic Regression"
    print "\tScores: %s" % scores
    print "\tMean Accuracy: %.3f%%" % (sum(scores)/float(len(scores)))
    print ""

    config.set_algorithm("mlp")
    scores = evaluate_algorithm(
        dataset,
        algorithms.mlp,
        config.n_folds,
        config.lr,
        config.n_epochs,
        config.batch_size,
        config.hidden_size
    )

    print "Multi Layer Perceptron"
    print "\tScores: %s" % scores
    print "\tMean Accuracy: %.3f%%" % (sum(scores)/float(len(scores)))
    print ""


if __name__ == "__main__":
    main()
