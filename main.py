from DecisionTreeRegression import DecisionTreeReg
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
# example of the decision tree algorithm for regression
# using the load_diabetes dataset from sklearn

if __name__ == "__main__":
    # load data
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = DecisionTreeReg()
    model.fit(X_train, y_train)  # fitting the model
    model._score(X_test, y_test)  # calculating the R^2 score
    y_pred = model.predict(X_test)
    print(f"Result MSE on training set: "
          f"{model._calc_mse(y_train, model.predict(X_train))}")
    print(f"Result MSE on test set: "
          f"{model._calc_mse(y_test, model.predict(X_test))}")

    # comparde with sklearn
    model_sk = DecisionTreeRegressor(random_state=78)
    # model_sk = DecisionTreeRegressor()
    model_sk.fit(X_train, y_train)
    print(f"compared R^2 sklearn: {model_sk.score(X_test, y_test)}.")
