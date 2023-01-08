# python 3.10.6
import numpy as np

# https://levelup.gitconnected.com/building-a-decision-tree-from-scratch-in-python-machine-learning-from-scratch-part-ii-6e2e56265b19
# https://medium.com/analytics-vidhya/regression-trees-decision-tree-for-regression-machine-learning-e4d7525d8047
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
# https://austindavidbrown.github.io/post/2019/01/regression-decision-trees-in-python/
# https://betterdatasciefnce.com/mml-decision-trees/


class Node:
    """Single node of a decision tree
    """
    # a single node of a decision tree

    def __init__(self, feature=None, threshold=None, left=None,
                 right=None, value=None, loss=None):
        self.left = left
        self.right = right
        self.feature = feature
        self.threshold = threshold
        self.value = value
        self.loss = loss  # future expansion to print the tree


class DecisionTreeReg:
    """
    Regression decision tree for continuous data. 
    Give the max depth and the min number of samples to split the 
    default is 2 for min_samples_split and inf for max_depth
    t"""

    def __init__(self, min_sample_split=2, max_depth=float("inf")):
        # inf value for depth unless otherwise mentioned
        self.min_samples_split = min_sample_split
        self.max_depth = max_depth
        self.root = None

    def _calc_mse(self, y, y_pred=None):
        """Calculates the mean squared error of a list of values.
        Args:
            y (numpy.ndarray): values
        Returns:
            float: mean squared error
        """
        if y_pred is not None:  # to predict the training data
            # return np.sum(np.mean((y - y_pred) ** 2))
            return np.mean((y - y_pred) ** 2)
        return np.mean((y - np.mean(y)) ** 2)

    def _cmse(self, left, right):
        """Adds the left and right and combines the mse value.
        Args:
            left (numpy.ndarray): values
            right (numpy.ndarray): values
        Returns:
            float: added mean squared error
        """
        return self._calc_mse(left) + self._calc_mse(right)

    def _best_split(self, X, y):
        """Calculate the best split for X and y via difference in
        the mean squared error.
        Args:
            X (numpy.ndarray): features
            y (numpy.ndarray): target
        Returns:
            dict: best split
        """
        # Greedy algorithm to find the best split
        best_loss = 0
        best_split_mse = {
            "loss": 0
        }
        parent_mse = self._calc_mse(y)
        for feature in range(X.shape[1]):  # col
            # Returns the sorted unique elements of an array
            for treshold in np.unique(X[:, feature]):
                # create dataset and split it into left and right
                # left contains all the values less than the threshold
                # right all the values greater/ equal than the threshold
                df = np.concatenate((X, y.reshape(1, -1).T), axis=1)
                left = np.array(
                    [row for row in df if row[feature] < treshold])
                right = np.array(
                    [row for row in df if row[feature] >= treshold])
                if len(left) > 0 and len(right) > 0:
                    y = df[:, -1]
                    # mse = self._cmse(left[:, -1], right[:, -1])
                    loss = parent_mse - 1 / y.shape[0] * (self._calc_mse(
                        left[:, -1]) * left[:, -1].shape[0] +
                        self._calc_mse(right[:, -1]) *
                        right[:, -1].shape[0])
                    if loss >= best_loss:
                        best_loss = loss
                        best_split_mse["feature"] = feature
                        best_split_mse["threshold"] = treshold
                        best_split_mse["loss"] = best_loss
                        best_split_mse["left"] = left
                        best_split_mse["right"] = right
                        # best_split_mse["mse"] = # use self._cmse()
        return best_split_mse

    def _build_tree(self, X, y, depth=0):
        """Builds a decision tree recursively.
        Args:
            X (numpy.ndarray): features
            y (numpy.ndarray): target
            depth (int, optional): depth. Defaults to 0.
        Returns:
            Node: decision tree
        """
        # recursive function to build the tree
        if X.shape[0] >= self.min_samples_split and depth <= self.max_depth:
            split = self._best_split(X, y)
            # print(split["mse"], "split")
            if split["loss"] > 0:  # to prevent None
                left_n = self._build_tree(
                    X=split["left"][:, :-1],
                    y=split["left"][:, -1], depth=depth+1)
                right_n = self._build_tree(
                    X=split["right"][:, :-1],
                    y=split["right"][:, -1], depth=depth+1)
                return Node(feature=split["feature"],
                            threshold=split["threshold"],
                            left=left_n, right=right_n, loss=split["loss"])
        return Node(value=np.mean(y))

    def fit(self, X, y):
        """Trains the decision tree.
        Args:
            X (numpy.ndarray): features
            y (numpy.ndarray): target
        """
        self.root = self._build_tree(X, y)

    def _predict(self, x, node):
        """Predicts a single sample.
        Args:
            x (np.array): single observation
            node (Node): decision tree
        Returns:
            float: prediction
        """
        if node.value is not None:
            return node.value
        if x[node.feature] < node.threshold:
            return self._predict(x, node.left)
        return self._predict(x, node.right)

    def predict(self, X):
        """Predicts a list of samples.
        Args:
            X (numpy.ndarray): features
        Returns:
            numpy.ndarray: predictions
        """
        # y_pred = [self._predict(x, self.root) for x in X]
        y_pred = []  # add predictions
        for x in X:  # check for each sample in X
            y_pred.append(self._predict(x, self.root))
        return np.array(y_pred)

    def _score(self, X, y):
        """
        Calculates the R^2 score of the model, print with 3 decimals.
        Args:
            X (numpy.ndarray): features
            y (numpy.ndarray): target
        Returns:
            float: R^2 score"""
        y_pred = self.predict(X)
        u = ((y - y_pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        # print(f"u: {u}, v: {v}, 1 - (u / v): {round(1 - u / v, 3)}")
        print(f"R^2: {round(1 - u / v, 3)}")  # 3 decimals
        return 1 - (u / v)
