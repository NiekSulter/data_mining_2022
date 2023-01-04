# python 3.10.6
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

class Node:
    """Single node of a decision tree
    """
    # a single node of a decision tree

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, mse=None):
        self.left = left
        self.right = right
        self.feature = feature
        self.threshold = threshold
        self.value = value
        self.mse = mse

class DecisionTreeReg:
    def __init__(self, min_sample_split=2, max_depth=float("inf")):
        # inf value for depth unless otherwise mentioned
        self.min_samples_split = min_sample_split
        self.max_depth = max_depth
        self.root = None

    def _calc_mse(self, y):
        """Calculates the mean squared error of a list of values.
        Args:
            y (numpy.ndarray): values
        Returns:
            float: mean squared error
        """
        return np.mean((y - np.mean(y)) ** 2)

    def _information_gain(self, parent, left, right):
        """Calculates the information gain of a split.
        Args:
            parent (numpy.ndarray): parent values
            left (numpy.ndarray): left child values
            right (numpy.ndarray): right child values
        Returns:
            float: information gain
        """
        return self._calc_mse(parent) - (len(left) / len(parent) * self._calc_mse(left) + len(right) / len(parent) * self._calc_mse(right))

    def _best_split(self, X, y):
        """Calculate the best split for X and y via the mean squared error.

        Args:
            X (_type_): _description_
            y (_type_): _description_
        """
        best_gain = 0
        best_feature = None
        best_threshold = None
        parent_mse = self._calc_mse(y)
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left = y[X[:, feature] < threshold]
                right = y[X[:, feature] >= threshold]
                if len(left) == 0 or len(right) == 0:
                    continue
                # gain = parent_mse - (len(left) / len(y) * self._calc_mse(left) + len(right) / len(y) * self._calc_mse(right))
                # calculatig again parent mse of y
                gain = self._information_gain(y, left, right)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        return best_gain, best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        """Builds a decision tree recursively.

        Args:
            X (_type_): _description_
            y (_type_): _description_
            depth (int, optional): _description_. Defaults to 0.
        """
        gain, feature, threshold = self._best_split(X, y)
        if gain == 0 or depth >= self.max_depth:
            value = np.mean(y)
            return Node(value=value, mse=self._calc_mse(y))
        left_idx = np.where(X[:, feature] < threshold)
        right_idx = np.where(X[:, feature] >= threshold)
        left = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right = self._build_tree(X[right_idx], y[right_idx], depth + 1)
        return Node(feature, threshold, left, right)
    
    def fit(self, X, y):
        """Trains the decision tree.

        Args:
            X (_type_): _description_
            y (_type_): _description_
        """
        self.root = self._build_tree(X, y)
    
    def _predict(self, x, node):
        """Predicts a single sample.

        Args:
            x (_type_): _description_
            node (_type_): _description_
        """
        if node.value is not None:
            return node.value
        if x[node.feature] < node.threshold:
            return self._predict(x, node.left)
        return self._predict(x, node.right)

    def predict(self, X):
        """Predicts a list of samples.

        Args:
            X (_type_): _description_
        """
        y_pred = [self._predict(x, self.root) for x in X]
        return np.array(y_pred)

def main():
    # load the dataset
    X, y = load_diabetes(return_X_y=True)
    # split the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # train the model
    model = DecisionTreeReg()
    model.fit(X_train, y_train)
    # predict
    y_pred = model.predict(X_test)
    # evaluate
    mse = mean_squared_error(y_test, y_pred)
    print(f"MSE: {mse:.3f}")

main()