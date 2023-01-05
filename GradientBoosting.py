from sklearn.tree import DecisionTreeRegressor
from DecisionTreeRegression import DecisionTreeReg
import numpy as np


class GradientBoosting:
    def __init__(self, n_estimators, learning_rate, max_depth) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        
        self.trees = []
        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            self.trees.append(tree)
            
        self.loss = CrossEntropy()
        

    def fit(self, X, y):
        """Fit data to the trees generated in the constructor. 
        

        Args:
            X (_type_): _description_
            y (_type_): _description_
        """
        class_0 = np.sum(y[:, 0], dtype='int')
        class_1 = np.sum(y[:, 1], dtype='int')
        
        log_odds = np.log(class_1 / class_0)
        
        probability = np.exp(log_odds) / (1 + np.exp(log_odds))
        
        pa = np.array([probability, 1 - probability])
    
        y_pred = np.full(np.shape(y), pa)
        
        for i in range(self.n_estimators):
            gradient = self.loss.gradient(y, y_pred)
            self.trees[i].fit(X, gradient)
            update = self.trees[i].predict(X)
            y_pred -= np.multiply(self.learning_rate, update)
                
    def predict(self, X):
        y_pred = np.array([])
        for tree in self.trees:
            update = np.multiply(self.learning_rate, tree.predict(X))
            if not y_pred.any():
                y_pred = -update
            else:
                y_pred = y_pred - update
            
        y_pred = np.exp(y_pred) / np.expand_dims(np.sum(np.exp(y_pred), axis=1), axis=1)
        
        y_pred = np.argmax(y_pred, axis=1)
            
        return y_pred
    

class CrossEntropy:
    def __init__(self) -> None:
        pass
    
    def loss(self, y, y_pred):
        return -np.sum(y * np.log(y_pred)) / len(y)
    
    def accuracy(self, y, y_pred):
        return np.sum(y == y_pred) / len(y)
    
    def gradient(self, y, y_pred):
        return - (y / y_pred) / len(y)
    

class ConvertToOneHot:
    def __init__(self) -> None:
        pass
    
    def to_categorical(self, y):
        y = np.array(y, dtype='int')
        n_values = np.max(y) + 1
        return np.eye(n_values)[y]
    
    def to_labels(self, y):
        return np.argmax(y, axis=1)