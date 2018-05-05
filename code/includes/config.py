import os
import random


class Config:
    def __init__(self, batch_size=100, verbose=True):
        self.n_folds = 5
        self.verbose = verbose

        self.batch_size = batch_size

        self.eval_seed = random.randint(0, 100)

        if not os.path.exists("plots"):
            os.makedirs("plots")

    def set_lr(self, lr):
        self.lr = lr

    def set_algorithm(self, algorithm):
        # Softmax Regression
        if algorithm == "sft-reg":
            self.n_epochs = 200
            self.tol = 0.00001
            self.lr = 0.01

        # Boosted Softmax Regression
        elif algorithm == "boosted-sft-reg":
            self.n_epochs = 100
            self.tol = 0.00001
            self.lr = 0.01

            self.m_stop = 25
            self.eta = 1.0
            self.eta_decay = 0.95

        # Multi Class Multi Layered Perceptron
        elif algorithm == "multi-mlp":
            self.n_epochs = 500
            self.lr = 0.005
            self.tol = 0.00001

            self.hidden_size = 50
