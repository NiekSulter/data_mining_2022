# python 3.10.6
from dataclasses import dataclass
from numpy import np


#https://dafriedman97.github.io/mlbook/content/c6/s2/boosting.html
@dataclass
class AdaBoost:
    weights: int
    trees: list
    alphas: list
    def fit(self, X_train, y_train, T, stub_depth):
        self.y_train = y_train
        self.X_train = X_train
        self.N, self.D = X_train.shape
        self.T = T
        self.stub_depth = stub_depth