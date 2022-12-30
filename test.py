from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import accuracy_score
import numpy as np

class Node:
    '''
    Helper class which implements a single tree node.
    '''
    def __init__(self, feature=None, threshold=None, data_left=None, data_right=None, gain=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.data_left = data_left
        self.data_right = data_right
        self.gain = gain
        self.value = value

class DecisionTree:
    '''
    Class which implements a decision tree classifier algorithm.
    '''
    def __init__(self, min_samples_split=2, max_depth=5):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None
        
    @staticmethod
    def _entropy(s):
        '''
        Helper function, calculates entropy from an array of integer values.
        
        :param s: list
        :return: float, entropy value
        '''
        # Convert to integers to avoid runtime errors
        counts = np.bincount(np.array(s, dtype=np.int64))
        # Probabilities of each class label
        percentages = counts / len(s)

        # Caclulate entropy
        entropy = 0
        for pct in percentages:
            if pct > 0:
                entropy += pct * np.log2(pct)
        return -entropy
    
    def _information_gain(self, parent, left_child, right_child):
        '''
        Helper function, calculates information gain from a parent and two child nodes.
        
        :param parent: list, the parent node
        :param left_child: list, left child of a parent
        :param right_child: list, right child of a parent
        :return: float, information gain
        '''
        # print(parent)
        num_left = len(left_child) / len(parent)
        num_right = len(right_child) / len(parent)
        
        # One-liner which implements the previously discussed formula
        return self._entropy(parent) - (num_left * self._entropy(left_child) + num_right * self._entropy(right_child))
    def _informationbn_gain(self, parent, left, right):
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
        print(parent)
        # parent is the entropy of the parent node
        # left and right are the entropy of the left and right child nodes
        # parent - (left + right) is the information gain
        left_c = len(left) / len(parent)
        right_c = len(right) / len(parent)
        gain = self._entropy(parent) - (left_c * self._entropy(left)) + (right_c * self._entropy(right))
        return gain # can one be a one liner.        
    
    # def _best_split(self, X, y):
    #     '''
    #     Helper function, calculates the best split for given features and target
        
    #     :param X: np.array, features
    #     :param y: np.array or list, target
    #     :return: dict
    #     '''
    #     best_split = {}
    #     best_info_gain = -1
    #     # print(X.shape)
    #     n_rows, n_cols = X.shape
    #     # print(X, y)
    #     # For every dataset feature
    #     for f_idx in range(n_cols):
    #         X_curr = X[:, f_idx]
    #         # For every unique value of that feature
    #         for threshold in np.unique(X_curr):
    #             # Construct a dataset and split it to the left and right parts
    #             # Left part includes records lower or equal to the threshold
    #             # Right part includes records higher than the threshold
    #             df = np.concatenate((X, y.reshape(1, -1).T), axis=1)
    #             df_left = np.array([row for row in df if row[f_idx] <= threshold])
    #             df_right = np.array([row for row in df if row[f_idx] > threshold])
    #             # print(df_left, y, "df_left")
    #             # Do the calculation only if there's data in both subsets
    #             if len(df_left) > 0 and len(df_right) > 0:
    #                 # Obtain the value of the target variable for subsets
    #                 y = df[:, -1]
    #                 y_left = df_left[:, -1]
    #                 y_right = df_right[:, -1]

    #                 # Caclulate the information gain and save the split parameters
    #                 # if the current split if better then the previous best
    #                 # print(y, "y")
    #                 gain = self._information_gain(y, y_left, y_right)
    #                 if gain > best_info_gain:
    #                     best_split = {
    #                         'feature_index': f_idx,
    #                         'threshold': threshold,
    #                         'df_left': df_left,
    #                         'df_right': df_right,
    #                         'gain': gain
    #                     }
    #                     best_info_gain = gain
    #     return best_split


    def _best_split(self, X, y):
        best_split = {
            "feature_index": None,
            "threshold": None, 
            "gain": None,
            "df_left": None,
            "df_right": None
        }
        best_gain = -1
        # print(X.shape)
        for feature in range(X.shape[1]): # in columns
            # every unique of the feature
            uniq_f = np.unique(X[:, feature])
            # for threshold in uniq_f:
            for threshold in np.unique(X[:, feature]):
                # create a new df and split it into left and right
                # right includes records higher than the threshold
                df = np.concatenate((X, y.reshape(1, -1).T), axis=1)
                left = np.array([row for row in df if row[feature] <= threshold])
                right = np.array([row for row in df if row[feature] > threshold])

                # only calculate the information gain if \
                # there are records in both left and right
                if len(left) > 0 and len(right) > 0:
                    y = df[:, -1]
                    # y_left = left[:, -1]
                    # y_right = right[:, -1]
                    gain = self._information_gain(y, left[:,-1], right[:,-1])
                    if gain > best_gain:
                        best_split["feature_index"] = feature
                        best_split["threshold"] = threshold
                        best_split["gain"] = gain
                        best_split["df_left"] = left
                        best_split["df_right"] = right
                        best_gain = gain
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
        # print(split)
        print(X.shape)
        if X.shape[0] >= self.min_samples_split and depth <= self.max_depth:
            split = self._best_split(X, y)
            # print(split)
            if split["gain"] > 0: # to prevent None
                left_n = self._build_tree(X=split["df_left"][:, :-1], y=split["df_left"][:, -1], depth=depth+1)
                right_n = self._build_tree(X=split["df_right"][:, :-1], y=split["df_right"][:, -1], depth=depth+1)
                return Node(feature=split["feature_index"], threshold=split["threshold"], data_left=left_n, data_right=right_n, gain=split["gain"])
        return Node(value=Counter(y).most_common(1)[0][0])
    
    def fit(self, X, y):
        '''
        Function used to train a decision tree classifier model.
        
        :param X: np.array, features
        :param y: np.array or list, target
        :return: None
        '''
        # Call a recursive function to build the tree
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
            return self._predict(x, tree.data_left)
        else:   # to right tree
            return self._predict(x, tree.data_right)
        
    
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


def main():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris["target"], test_size=0.2, random_state=42)
    # X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris["target"], test_size=0.2)
    model = DecisionTree()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)


    print(accuracy_score(y_test, preds))

main()