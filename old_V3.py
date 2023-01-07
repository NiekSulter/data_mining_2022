# python 3.10.6
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
# has a shit R^2 of -0.012 whereas sklearn's is around 0.12

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


class DecisionTreeRegression:
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

    def _best_split(self, X, y):
        best_mse = float("inf")
        best_split = {}
        for feature in range(X.shape[1]):
            for threshold in np.unique(X[:, feature]):
                # create a new df and split it into left and right
                df = np.concatenate((X, y.reshape(1, -1).T), axis=1)
                left = np.array(
                    [row for row in df if row[feature] < threshold])
                right = np.array(
                    [row for row in df if row[feature] >= threshold])
                if len(left) > 0 and len(right) > 0:
                    # calculate mse for left and right
                    left_mse = self._calc_mse(left[:, -1])
                    right_mse = self._calc_mse(right[:, -1])
                    mse = left_mse + right_mse
                    # calculate mse for split
                    # split_mse = (len(left) / len(df)) * left_mse + \
                    #     (len(right) / len(df)) * right_mse
                    # print(f"mse: {mse}, split_mse: {split_mse}")
                    # print(f"{left_mse + right_mse}, {best_mse}")
                    if mse < best_mse:
                        best_mse = (left_mse + right_mse)
                        best_split = {"feature": feature,
                                      "threshold": threshold,
                                      "left": left,
                                      "right": right,
                                      "mse": best_mse}
        return best_split 

    def _build_tree(self, X, y, depth=0):
        if X.shape[0] >= self.min_samples_split and depth <= self.max_depth:
            best_split = self._best_split(X, y)
            if best_split["mse"] < float("inf"):
                left = self._build_tree(best_split["left"][:, :-1],
                                        best_split["left"][:, -1], depth + 1)
                right = self._build_tree(best_split["right"][:, :-1],
                                         best_split["right"][:, -1], depth + 1)
                return Node(feature=best_split["feature"],
                            threshold=best_split["threshold"],
                            left=left,
                            right=right,
                            mse=best_split["mse"])
        return Node(value=np.mean(y), mse=self._calc_mse(y))

    def fit(self, X, y):
        """Used to fit the decision tree to the training data.
        Args:
            X (np.array): features
            y (np.array): target
        """
        self.root = self._build_tree(X, y)
    
    def _predict(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] < node.threshold:
            return self._predict(x, node.left)
        return self._predict(x, node.right)

    def predict(self, X):
        """Used to predict the target values for a given set of features.
        Args:
            X (np.array): features
        Returns:
            np.array: predicted target values
        """
        return np.array([self._predict(x, self.root) for x in X])

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
    # load data
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # fit model
    model = DecisionTreeRegression()
    model.fit(X_train, y_train)
    # predict
    y_pred = model.predict(X_test)
    # evaluate
    mse = np.mean((y_test - y_pred) ** 2)
    print(f"MSE: {mse}")
    print(f"R^2: {model._score(X_test, y_test)}")

    model_sk = DecisionTreeRegressor()
    model_sk.fit(X_train, y_train)
    y_pred_sk = model_sk.predict(X_test)
    print(f"R^2 sklearn: {model_sk.score(X_test, y_test)}")