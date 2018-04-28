import random


class Config:
    def __init__(self, batch_size=100, verbose=True):
        self.n_folds = 5
        self.verbose = verbose

        self.batch_size = batch_size

        self.eval_seed = random.randint(0, 100)

    def set_lr(self, lr):
        self.lr = lr

    def set_algorithm(self, algorithm):
        # Logistic Regression
        if algorithm == "log-reg":
            self.n_epochs = 500
            self.tol = 0.0001
            self.lr = 0.1

        # Gradient Boosted Logistic Regression
        elif algorithm == "boosted-log-reg":
            self.m_stop = 200
            self.n_epochs = 200
            self.lr = 0.1
            self.eta = 0.5
            self.eta_decay = 0.9
            self.tol = 0.000001

        # Single Class Multi Layered Perceptron
        elif algorithm == "mlp":
            self.n_epochs = 200
            self.lr = 0.1

            self.hidden_size = (50)

        # Softmax Regression
        elif algorithm == "sft-reg":
            self.n_epochs = 200
            self.tol = 0.00001
            self.lr = 0.01

        # Boosted Softmax Regression
        elif algorithm == "boosted-sft-reg":
            self.n_epochs = 50
            self.tol = 0.00001
            self.lr = 0.01

            self.m_stop = 100
            self.eta = 1.0
            self.eta_decay = 0.95

        # Multi Class Multi Layered Perceptron
        elif algorithm == "multi-mlp":
            self.batch_size = 50
            self.n_epochs = 200
            self.lr = 0.01
            self.tol = 0.00001

            self.hidden_size = (60)
