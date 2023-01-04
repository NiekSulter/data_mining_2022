# python 3.10.6
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np

# https://levelup.gitconnected.com/building-a-decision-tree-from-scratch-in-python-machine-learning-from-scratch-part-ii-6e2e56265b19

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

    def _find_score(self, left, right):
        """Calculates the weighted mean squared error of two lists of values.
        Args:
            left (numpy.ndarray): values
            right (numpy.ndarray): values
        Returns:
            float: weighted mean squared error
        """
        left_std = np.std(left)
        right_std = np.std(right)
        left_mse = self._calc_mse(left)
        right_mse = self._calc_mse(right)
        # return left_std * left.sum() + right_std * right.sum()
        return left_mse + right_mse # gives the best i
        

    def _best_split(self, X, y):
        """Calculate the best split for X and y via the mean squared error.

        Args:
            X (_type_): _description_
            y (_type_): _description_
        """
        best_score = -1
        best_feature = None
        best_threshold = None
        parent_mse = self._calc_mse(y)
        best_split = {
            "feature": None,
            "threshold": None,
            "gain": None,
            "left": None,
            "right": None
        }
        print(X.shape, y.shape)
        for feature in range(X.shape[1]):
            # thresholds = np.unique(X[:, feature])
            for treshold in np.unique(X[:, feature]):
                # df = np.concatenate((X, y.reshape(-1, 1).T), axis=1)
                df = np.concatenate((X, y.reshape(1, -1).T), axis=1)
                left = np.array([row for row in df if row[feature] <= treshold])
                right = np.array([row for row in df if row[feature] > treshold])
                if len(left) > 0 and len(right) > 0:
                    y = df[:,-1]
                    # gain = self._information_gain(y, left[:,-1], right[:,-1])
                    score = self._find_score(left[:,-1], right[:,-1])
                    
                    if score > best_score:
                        best_score = score
                        best_feature = feature
                        best_threshold = treshold
                        best_split["feature"] = feature
                        best_split["threshold"] = treshold
                        best_split["gain"] = score
                        best_split["left"] = left
                        best_split["right"] = right
        return best_score, best_feature, best_threshold, best_split
            # for threshold in thresholds:
            #     left = y[X[:, feature] < threshold]
            #     right = y[X[:, feature] >= threshold]
                # if len(left) == 0 or len(right) == 0:
                    # continue
                # gain = parent_mse - (len(left) / len(y) * self._calc_mse(left) + len(right) / len(y) * self._calc_mse(right))
                # calculatig again parent mse of y
        #         print(left)
        #         gain = self._information_gain(y, left[:,-1], right[:,-1])
        #         if gain > best_gain:
        #             best_gain = gain
        #             best_feature = feature
        #             best_threshold = threshold
        # return best_gain, best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        """Builds a decision tree recursively.

        Args:
            X (_type_): _description_
            y (_type_): _description_
            depth (int, optional): _description_. Defaults to 0.
        """
        # gain, feature, threshold = self._best_split(X, y)
        # if gain == 0 or depth >= self.max_depth:
        #     value = np.mean(y)
        #     return Node(value=value, mse=self._calc_mse(y))
         
        # left_idx = np.where(X[:, feature] < threshold)
        # right_idx = np.where(X[:, feature] >= threshold)
        # left = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        # right = self._build_tree(X[right_idx], y[right_idx], depth + 1)
        # return Node(feature, threshold, left, right)
        if X.shape[0] >= self.min_samples_split and depth <= self.max_depth:
            gain, feature, threshold, split = self._best_split(X, y)
            if gain > 0: # to prevent None
                left_n = self._build_tree(X=split["left"][:,:-1], y=split["left"][:,-1], depth=depth+1)
                right_n = self._build_tree(X=split["right"][:,:-1], y=split["right"][:,-1], depth=depth+1)
                return Node(feature=split["feature"], threshold=split["threshold"], left=left_n, right=right_n)
        return Node(value=np.mean(y))


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


def main():
    # load the dataset
    X, y = load_diabetes(return_X_y=True)
    # split the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    # train the model
    # model = DecisionTreeReg(max_depth=5)
    model = DecisionTreeReg()
    model.fit(X_train, y_train)
    # predict
    y_pred = model.predict(X_test)
    print(f"y_pred: {y_pred}")
    print(f"R^2: {model._score(X_test, y_test):.3f}")
    # print(DecisionTreeRegressor.predict(X_test))

    # sklearn
    model_sk = DecisionTreeRegressor(random_state=44)
    model_sk.fit(X_train, y_train)
    predictions = model_sk.predict(X_test)
    print(f"y_pred_sl: {predictions}")
    print(f"R^2: {model_sk.score(X_test, y_test)}")


main()
