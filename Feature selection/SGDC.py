from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

class skl:
    def __init__(self):

        self.scaler = MinMaxScaler()
        self.class_weight = 'balanced' # default: None

    def training(self, X, y):
        X_normalized = self.scaler.fit_transform(X)

        loo = LeaveOneOut()

        accuracies = []  

        for train_index, test_index in loo.split(X_normalized):
            X_train, X_test = X_normalized[train_index], X_normalized[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model = OneVsRestClassifier(SGDClassifier(n_jobs=-1,random_state=0, class_weight=self.class_weight),n_jobs = -1)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)

        mean_accuracy = np.mean(accuracies)

        return mean_accuracy

