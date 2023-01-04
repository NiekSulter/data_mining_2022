# python 3.10.6
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split


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

class DecTreeReg:
    # decision tree regression
    def __init__(self, max_depth=float("inf"), min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def _calc_mse(self, y):
        """Calculates the mean squared error of a list of values.
        Args:
            y (numpy.ndarray): values
        Returns:
            float: mean squared error
        """
        return np.mean((y - np.mean(y)) ** 2)

    def _split(self, X, y, feature, threshold):
        """Splits a dataset based on a feature and threshold.
        Args:
            X (numpy.ndarray): dataset
            y (numpy.ndarray): labels
            feature (int): feature to split on
            threshold (float): threshold value
        Returns:
            tuple: left and right splits
        """
        left = np.where(X[:, feature] < threshold)
        right = np.where(X[:, feature] >= threshold)
        # print(f"X: {X.shape}, y: {y.shape}, left: {left[0].shape}, right: {right[0].shape}")
        # Xl = X[left]
        # Xr = X[right]
        # yl = y[left,]
        # yr = y[right]
        df = np.concatenate((X, y.reshape(1, -1).T), axis=1)
        left = np.array([row for row in df if row[feature] <= threshold])
        right = np.array([row for row in df if row[feature] > threshold])
        return X[left], X[right], y[left], y[right]
    
    def _find_best_split(self, X, y):
        """Finds the best split for a dataset.
        Args:
            X (numpy.ndarray): features
            y (numpy.ndarray): target
        Returns:
            dict: best split
        """
        best_split = {}
        best_mse = float("inf")
        print(f"X: {X.shape}, y: {y.shape}")
        for feature in range(X.shape[1]):
            for threshold in np.unique(X[:, feature]):
                # left, right, y_left, y_right = self._split(X, y, feature, threshold)
                df = np.concatenate((X, y.reshape(1, -1).T), axis=1)
                left = np.array(
                    [row for row in df if row[feature] <= threshold])
                right = np.array(
                    [row for row in df if row[feature] > threshold])
                if len(left) > 0 and len(right) > 0:
                    # mse = self._calc_mse(y_left) + self._calc_mse(y_right)
                    mse = self._calc_mse(left[:, -1]) + self._calc_mse(right[:, -1])
                    if mse < best_mse:
                        best_split["feature"] = feature
                        best_split["threshold"] = threshold
                        best_split["mse"] = mse
                        best_split["left"] = left
                        best_split["right"] = right
                        best_mse = mse
        return best_split
    
    def _build_tree(self, X, y, depth=0):
        """Builds the decision tree by finding the best split for a 
        dataset and then recursively building the left and right child
        nodes.
        Args:
            X (numpy.ndarray): features
            y (numpy.ndarray): target
            depth (int, optional): current_depth. Defaults to 0.
        Returns:
            Node: node
        """
        # find the best split
        if X.shape[0] >= self.min_samples_split and depth <= self.max_depth:
            print(f"build_tree: X: {X.shape}, y: {y.shape}")
            split = self._find_best_split(X, y)
            if split["mse"] < float("inf"):
                left = self._build_tree(split["left"], split["right"], depth + 1)
                right = self._build_tree(split["right"], split["right"], depth + 1)
                return Node(feature=split["feature"], threshold=split["threshold"], left=left, right=right, mse=split["mse"])
        # if no split is found, return a leaf node
        return Node(value=np.mean(y), mse=self._calc_mse(y))

    def fit(self, X, y):
        """Builds the decision tree.
        Args:
            X (numpy.ndarray): features
            y (numpy.ndarray): target
        """
        self.root = self._build_tree(X, y)

    def _predict(self, x, node):
        """Predicts a value for a single sample.
        Args:
            x (numpy.ndarray): sample
            node (Node): node
        Returns:
            float: predicted value
        """
        if node.value is not None:
            return node.value
        if x[node.feature] < node.threshold:
            return self._predict(x, node.left)
        return self._predict(x, node.right)
    
    def predict(self, X):
        """Predicts values for a dataset.
        Args:
            X (numpy.ndarray): features
        Returns:
            numpy.ndarray: predicted values
        """
        y_pred = []
        for x in X:
            y_pred.append(self._predict(x, self.root))
        return np.array(y_pred)

    def _print_tree(self, node, spacing=""):
        """Prints the decision tree.
        Args:
            node (Node): node
            spacing (str, optional): spacing. Defaults to "".
        """
        if node.value is not None:
            print(spacing + "Predict", node.value)
            return
        print(spacing + "X" + str(node.feature) + " < " + str(node.threshold) + "?")
        print(spacing + "--> True:")
        self._print_tree(node.left, spacing + "  ")
        print(spacing + "--> False:")
        self._print_tree(node.right, spacing + "  ")

    def _score(self, X, y):
        """Calculate the R^2 score of the model.
        Args:
            X (_type_): X_test
            y (_type_): y_test
        """
        y_pred = self.predict(X)
        u = ((y - y_pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        print(f"u: {u}, v: {v}, 1 - (u / v): {1 - u / v}")
        return 1 - (u / v)
        
if __name__ == "__main__":
    # create a dataset
    X = np.array([[1, 2, 3], [2, 3, 3], [3, 4, 3], [4, 5, 3], [5, 6, 3], [6, 7, 3], [7, 8, 3], [8, 9, 3], [9, 10, 3], [10, 11, 3]])
    y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # create and fit the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecTreeReg(max_depth=3, min_samples_split=2)
    model.fit(X_train, y_train)
    # make predictions
    y_pred = model.predict(X_test)
    print(f"R^2: {model._score(X_test, y_test)}")
    print(y_pred)
    print(f"X: {X.shape}, y: {y.shape}")


    X, y = load_diabetes(return_X_y=True)
    print(f"X: {X.shape}, y: {y.shape}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    model = DecTreeReg()
    model.fit(X_train, y_train)