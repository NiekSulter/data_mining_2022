# python 3.10.6
from pyedflib import highlevel
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import accuracy_score

#https://dafriedman97.github.io/mlbook/content/c6/s2/boosting.html


def read_edf(path):
    signals, signal_headers, header = highlevel.read_edf(path)
    # signal is a list containing a numpy array for each signal

# write a decision tree classifier
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html


# https://betterdatascience.com/mml-decision-trees/

class Node:
    # a single node of a decision tree
    def __init__(self, feature=None, threshold=None, left=None, right=None, gain=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
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
        formula: entropy(parent) - (left_c * entropy(left) + right_c * entropy(right))
        Args:
            parent (_type_): _description_
            left (_type_): _description_
            right (_type_): _description_

        Returns:
            _type_: _description_
        """
        # parent is the entropy of the parent node
        # left and right are the entropy of the left and right child nodes
        # parent - (left + right) is the information gain
        left_c = len(left) / len(parent)
        right_c = len(right) / len(parent)
        gain = self._entropy(parent) - (left_c * self._entropy(left)) + (right_c * self._entropy(right))
        return gain # can one be a one liner. 


    def _best_split(self, X, y):
        best_split = {
            "feature": None,
            "threshold": None, 
            "gain": None,
            "left": None,
            "right": None
        }
        best_gain = -1
        for feature in range(X.shape[1]):
            # every unique of the feature
            uniq_f = np.unique(X[:, feature])
            for threshold in uniq_f:
                # create a new df and split it into left andright
                # right includes records higher than the threshold
                df = np.concatenate((X, y.reshape(1, -1).T), axis=1)
                left = np.array([row for row in df if row[feature] <= threshold])
                right = np.array([row for row in df if row[feature] > threshold])

                # only calculate the information gain if \
                # there are records in both left and right
                if len(left) > 0 and len(right) > 0:
                    y = df[:, -1]
                    y_left = left[:, -1]
                    y_right = right[:, -1]
                    gain = self._information_gain(y, y_left, y_right)
                    if gain > best_gain:
                        best_gain = gain
                        best_split["feature"] = feature
                        best_split["threshold"] = threshold
                        best_split["gain"] = gain
                        best_split["left"] = left
                        best_split["right"] = right
        return best_split

    def _build_tree(self, X, y, depth=0):
        """Builds the decision tree by finding the best split for a 
        dataset and then recursively building the left and right child
        nodes.

        Args:
            X (_type_): _description_
            y (_type_): _description_
            depth (int, optional): _description_. Defaults to 0.

        Returns:
            _type_: _description_
        """
        # find the best split
        split = self._best_split(X, y)
        if X.shape[0] >= self.min_samples_split and depth <= self.max_depth:
            if split["gain"] > 0: # to prevent None
                left = self._build_tree(X=split["left"][:, :-1], y=split["left"][:, -1], depth=depth+1)
                right = self._build_tree(X=split["right"][:, :-1], y=split["right"][:, -1], depth=depth+1)
                return Node(feature=split["feature"], threshold=split["threshold"], left=left, right=right, gain=split["gain"])
        return Node(value=Counter(y).most_common(1)[0][0])

    def fit(self, X, y):
        self.root = self._build_tree(X, y)


    def _predict(self, x, tree):
        '''
        Helper recursive function, used to predict a single instance (tree traversal).
        
        :param x: single observation
        :param tree: built tree
        :return: float, predicted class
        '''
        # Leaf node
        if tree.value != None:
            return tree.value
        feature_value = x[tree.feature]
        
        # Go to the left
        if feature_value <= tree.threshold:
            return self._predict(x=x, tree=tree.left)
        
        # Go to the right
        if feature_value > tree.threshold:
            return self._predict(x=x, tree=tree.right)
        
    def predict(self, X):
        '''
        Function used to classify new instances.
        
        :param X: np.array, features
        :return: np.array, predicted classes
        '''
        # Call the _predict() function for every observation
        return [self._predict(x, self.root) for x in X]


def main():
    # file_edf = "/Users/lean/Library/CloudStorage/OneDrive-Persoonlijk/School/Master/Pre-Master_DataScience_(Minor_Computing_Science)/2223_Data_Mining_(KW1_V)/data_project/aaaaaaff_s002_t000.edf"
    # read_edf(file_edf)
    iris = load_iris()
    X = iris["data"]
    y = iris["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = DecisionTree()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(accuracy_score(y_test, preds))


main()
