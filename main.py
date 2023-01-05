from DecisionTreeRegression import DecisionTreeReg
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    # load data
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # fit model
    model = DecisionTreeReg()
    DecisionTreeReg
    model.fit(X_train, y_train)
    model._score(X_test, y_test)
    # predict
    y_pred = model.predict(X_test)

    # compare with sklearn
    model_sk = DecisionTreeRegressor(random_state=78)
    # model_sk = DecisionTreeRegressor()
    model_sk.fit(X_train, y_train)
    print(f"compared R^2 sklearn: {model_sk.score(X_test, y_test)}.")
    # compared R^2 sklearn: 0.12360152082629405.