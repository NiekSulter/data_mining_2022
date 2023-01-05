from sklearn.tree import DecisionTreeRegressor, plot_tree
from DecisionTreeRegression import DecisionTreeReg
import numpy as np
import time
import matplotlib.pyplot as plt

class GradientBoosting:
    def __init__(self, n_estimators, learning_rate, max_depth) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        
        self.trees = []
        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_depth=self.max_depth, max_leaf_nodes=3)
            self.trees.append(tree)
            
        self.loss = CrossEntropy()
        

    def fit(self, X, y):
        """Fit data to the trees generated in the constructor. 
        

        Args:
            X (_type_): _description_
            y (_type_): _description_
        """
        class_0 = np.count_nonzero(y == 0)
        class_1 = np.count_nonzero(y == 1)
        
        log_odds = np.log(class_1 / class_0)
        
        initial_prediction = np.round(np.exp(log_odds) / (1 + np.exp(log_odds)), 1)
    
        #calculated array with the residuals
        residuals = np.full(np.shape(y), np.round(y - initial_prediction, 1))
        
        print(residuals)
        print(initial_prediction)
        
        for i in range(self.n_estimators):
            #train regression tree on residuals
            self.trees[i].fit(X, residuals)
            
            plt.figure()
            plot_tree(self.trees[i])
            plt.savefig(f"tree_{i}.png")
            
            node_loc = self.trees[i].apply(X)
            
            leaves = dict((i, []) for i in np.unique(node_loc))
            output_values = dict((i, []) for i in np.unique(node_loc))
            
            for j in range(len(residuals)):
                leaves[node_loc[j]].append(residuals[j])
                
            print(leaves)
            
            for key in leaves:
                # output_value = (np.sum(leaves[key]) / np.sum([np.multiply(p, 1 - p) for p in leaves[key]]))
                
                output_values[key] = np.round((np.sum(leaves[key]) / np.sum([np.multiply(initial_prediction, 1 - initial_prediction) for p in leaves[key]])), 1)
                
                
            print(output_values)
            
            updated_probabilities = []
            
            for j in range(len(residuals)):
                print(f"{initial_prediction} + ({self.learning_rate} * {output_values[node_loc[j]]})")
                updated_probabilities.append(np.round(self.log_odds_to_prob(initial_prediction + np.multiply(self.learning_rate, output_values[node_loc[j]])), 1))
                
            print(updated_probabilities)
            #update residuals with the predictions of the regression tree
            # update = self.trees[i].predict(X)
            
            # # update_2 = np.array([(i / np.multiply(update[i], 1 - update[i])) for i in update])
            
            # update_2 = []
            # for i in range(len(update)):
            #     x = i / np.multiply(update[i], 1 - update[i])
            #     update_2.append(x)
            # update_2 = np.array(update_2)
            
            # #multiply the new residuals with the learning rate
            # residuals += np.multiply(self.learning_rate, update_2)
            
            # new_probabilities = np.exp(residuals) / (1 + np.exp(residuals))
            
            # print(new_probabilities)
            
            
                
    def predict(self, X):
        y_pred = np.array([])
        for tree in self.trees:
            update = np.multiply(self.learning_rate, tree.predict(X))
            if not y_pred.any():
                y_pred = -update
            else:
                y_pred = np.subtract(y_pred, update)
            
        y_pred = np.exp(y_pred) / np.expand_dims(np.sum(np.exp(y_pred), axis=1), axis=1)
        
        y_pred = np.argmax(y_pred, axis=1)
            
        return y_pred
    
    def log_odds_to_prob(self, y):
        return np.exp(y) / (1 + np.exp(y))
    

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