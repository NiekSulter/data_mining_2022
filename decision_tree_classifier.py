# python 3.10.6
from pyedflib import highlevel
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import accuracy_score


def read_edf(path):
    signals, signal_headers, header = highlevel.read_edf(path)
    # signal is a list containing a numpy array for each signal

# write a decision tree classifier
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
# https://betterdatascience.com/mml-decision-trees/


class Node:
    """Single node of a decision tree
    """
    # a single node of a decision tree

    def __init__(self, feature=None, threshold=None, left=None, right=None, gain=None, value=None):
        self.left = left
        self.right = right
        self.feature = feature
        self.threshold = threshold
        self.gain = gain
        self.value = value


class DecisionTree:
    def __init__(self, min_sample_split=2, max_depth=float("inf")):
        # inf value for depth unless otherwise mentioned
        self.min_samples_split = min_sample_split
        self.max_depth = max_depth
        self.root = None

    @staticmethod
    def _entropy(s):
        """Calculates the entropy of a list of values. The entropy is
        calculated using the formula: -sum(p(x)log(p(x))) where p(x) is
        the probability of each unique value in the list.
        Args:
            s (numpy.ndarray): values
        Returns:
            float: entropy
        """
        counts = np.bincount(np.array(s, dtype=np.int64))
        # bincount returns the number of occurrences of each value in an array of non-negative ints
        perct = counts / len(s)
        # perct is the percentage in the array e.g. [0.7, 0.3]
        entropy = 0
        for pct in perct:
            if pct > 0:
                entropy -= pct * np.log2(pct)
        return entropy

    def _information_gain(self, parent, left, right):
        """Calculates the information gain. From the parent and two
        child nodes, the information gain is calculated using the 
        formula: entropy(parent) - ((left_c * entropy(left) + right_c * entropy(right)))
        Args:
            parent (numpy.ndarray): parent node
            left (numpy.ndarray): left child node
            right (numpy.ndarray): right child node
        Returns:
            float: information gain
        """
        # parent is the entropy of the parent node
        # left and right are the entropy of the left and right child nodes
        # parent - (left + right) is the information gain
        left_c = len(left) / len(parent)
        right_c = len(right) / len(parent)
        gain = self._entropy(
            parent) - ((left_c * self._entropy(left)) + right_c * self._entropy(right))
        return gain  # can one be a one liner.

    def _best_split(self, X, y):
        """Calculates the best split for a for a feature and target.
        Args:
            X (numpy.ndarray): features
            y (numpy.ndarray): target

        Returns:
            dict: best split dictionary with features, treshold,
                gain, left and right
        """
        best_split = {
            "feature": None,
            "threshold": None,
            "gain": None,
            "left": None,
            "right": None
        }
        best_gain = -1
        for feature in range(X.shape[1]):  # in columns
            # every unique of the feature
            # uniq_f = np.unique(X[:, feature])
            for threshold in np.unique(X[:, feature]):
                # create a new df and split it into left and right
                # right includes records higher than the threshold
                df = np.concatenate((X, y.reshape(1, -1).T), axis=1)
                left = np.array(
                    [row for row in df if row[feature] <= threshold])
                right = np.array(
                    [row for row in df if row[feature] > threshold])

                # only calculate the information gain if \
                # there are records in both left and right
                if len(left) > 0 and len(right) > 0:
                    y = df[:, -1]
                    gain = self._information_gain(y, left[:, -1], right[:, -1])
                    if gain > best_gain:
                        best_split["feature"] = feature
                        best_split["threshold"] = threshold
                        best_split["gain"] = gain
                        best_split["left"] = left
                        best_split["right"] = right
                        best_gain = gain
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
            split = self._best_split(X, y)
            if split["gain"] > 0:  # to prevent None
                left_n = self._build_tree(
                    X=split["left"][:, :-1], y=split["left"][:, -1], depth=depth+1)
                right_n = self._build_tree(
                    X=split["right"][:, :-1], y=split["right"][:, -1], depth=depth+1)
                return Node(feature=split["feature"], threshold=split["threshold"], left=left_n, right=right_n, gain=split["gain"])
        return Node(value=Counter(y).most_common(1)[0][0])

    def fit(self, X, y):
        """Used to fit the decision tree to the training data.
        Args:
            X (np.array): features
            y (np.array or list): target
        """
        self.root = self._build_tree(X, y)

    def _predict(self, x, tree):
        """Used to predict the class of a single observation.
        Args:
            x (np.array): single observation
            tree (Node): decision tree
        Returns:
            _type_: _description_
        """
        if tree.value is not None:
            return tree.value

        # x[tree.feature] is the value of the feature in the observation
        if x[tree.feature] <= tree.threshold:
            return self._predict(x, tree.left)
        else:   # to right tree
            return self._predict(x, tree.right)

    def predict(self, X):
        """Used to predict the class of multiple observations.
        Args:
            X (np.array): features
        Returns:
            list: predicted classes
        """
        preds = []
        for x in X:
            # calls itself recursively
            # TODO list comprehension
            preds.append(self._predict(x, self.root))
        return preds
        # return [self._predict(x, self.root) for x in X]


def main():
    # file_edf = "/Users/lean/Library/CloudStorage/OneDrive-Persoonlijk/School/Master/Pre-Master_DataScience_(Minor_Computing_Science)/2223_Data_Mining_(KW1_V)/data_project/aaaaaaff_s002_t000.edf"
    # read_edf(file_edf)

    # tested on the Iris dataset
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris["data"], iris["target"], test_size=0.2, random_state=669)
        # accuracy needs to be 0.9 on the iris dataset

    model = DecisionTree(max_depth=5)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print(accuracy_score(y_test, preds))


main()
