# data_mining_2022
By: Niek Sülter and Lean Schoonveld
## Improving decision tree performance with Gradient Boosting

Data mining project 2022 'Algorithm'

Most notable software/ packages used:
- Python 3.10.6, Numpy and scikit-learn.

### Run the DecisionTreeRegression model
- Get the score and the MSE score of the model
```model = DecisionTreeReg()
    model.fit(X_train, y_train)  # fitting the model
    model._score(X_test, y_test)  # calculating the R^2 score
    y_pred = model.predict(X_test)
    print(f"Result MSE on training set: "
          f"{model._calc_mse(y_train, model.predict(X_train))}")
    print(f"Result MSE on test set: "
          f"{model._calc_mse(y_test, model.predict(X_test))}")
```