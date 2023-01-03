from GradientBoosting import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import numpy as np

features_list = []
classes = []



with open('data_banknote_authentication.txt') as f:
    for line in f:
        features_list.append([float(x) for x in line.split(',')[:-1]])
        classes.append(int(line.split(',')[-1]))
        
features = np.array(features_list)
classes = np.array(classes)
        
classes_encoded = preprocessing.LabelEncoder().fit_transform(classes)
        
X_train, X_test, y_train, y_test = train_test_split(features, classes_encoded, test_size=0.2)

GRB = GradientBoostingClassifier(n_estimators=10, learning_rate=0.5, max_depth=3, min_info_gain=0.0)

GRB.fit(X_train, y_train)

preds = GRB.predict(X_test)
print(preds)
