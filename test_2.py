from GradientBoosting import GradientBoosting, ConvertToOneHot
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, matthews_corrcoef, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
import numpy as np
from data_prep import data_p

# features_list = []
# classes = []


# with open('datasets/insurance.txt') as f:
#     next(f)
#     for line in f:
#         features_list.append([x for x in line.split(',')[:-1]])
#         classes.append(int(line.split(',')[-1]))

# # features = np.array(features_list)
# # classes = np.array(classes)
        
# classes_encoded = preprocessing.LabelEncoder().fit_transform(classes)

# print(features_list)
        
# X_train, X_test, y_train, y_test = train_test_split(features_list, classes_encoded, test_size=0.2)

X_train, y_train, X_test, y_test = data_p()

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

GRB = GradientBoosting(n_estimators=100, learning_rate=0.1, max_depth=3)
DTC = DecisionTreeClassifier(max_depth=3)

CTOH = ConvertToOneHot()

y_train = CTOH.to_categorical(y_train)

DTC.fit(X_train, y_train)
GRB.fit(X_train, y_train)

DTCpred = CTOH.to_labels(DTC.predict(X_test))
GRBpred = GRB.predict(X_test)

print("Non-Boosting:", confusion_matrix(y_test, DTCpred, labels=[0, 1]))
print("Boosting:", confusion_matrix(y_test, GRBpred, labels=[0, 1]))

