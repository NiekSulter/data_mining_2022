from DecisionTreeRegression import DecisionTreeReg
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import time
# example of the decision tree algorithm for regression
# using the load_diabetes dataset from sklearn

if __name__ == "__main__":
    # load data
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = DecisionTreeReg()
    start_time = time.time()
    model.fit(X_train, y_train)  # fitting the model
    model._score(X_test, y_test)  # calculating the R^2 score
    y_pred = model.predict(X_test)
    print(f"Result MSE on training set: "
          f"{model._calc_mse(y_train, model.predict(X_train))}")
    print(f"Result MSE on test set: "
          f"{model._calc_mse(y_test, model.predict(X_test))}")
    print(f"Total time: {time.time() - start_time} seconds.\n")
    
    # compare with sklearn
    start_time = time.time()
    model_sk = DecisionTreeRegressor(random_state=78)
    # model_sk = DecisionTreeRegressor()
    model_sk.fit(X_train, y_train)
    print(f"compared R^2 sklearn: {model_sk.score(X_test, y_test)}.")
    print(f"Result MSE on training set single run: "
          f"{model._calc_mse(y_train, model_sk.predict(X_train))}")
    print(f"Result MSE on test set single run: "
          f"{model._calc_mse(y_test, model_sk.predict(X_test))}")
    print(f"Total time sklearn: {time.time() - start_time} seconds.\n")

    
    # compare with sklearn without a random state # not used
    avg_MSE = []
    avg_R = []
    for run in range(10):
        start_time = time.time()
        # without random state
        model_skr = DecisionTreeRegressor()
        model_skr.fit(X_train, y_train)
        avg_MSE.append(model._calc_mse(y_test, model_skr.predict(X_test)))
        avg_R.append(model_skr.score(X_test, y_test))
    print(f"Average MSE_sk: {sum(avg_MSE)/len(avg_MSE)}")
    print(f"Average R_sk: {sum(avg_R)/len(avg_R)}")