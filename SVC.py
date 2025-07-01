from sklearn import datasets
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd

class skl:
    def __init__(self):

        self.scaler = MinMaxScaler()
        self.kernel = 'poly'
        self.C_parameter = 1
        self.class_weight = 'balanced' 

    def training(self, X, y):
        X_normalized = self.scaler.fit_transform(X)

        loo = LeaveOneOut()

        accuracies = []  

        for train_index, test_index in loo.split(X_normalized):
            X_train, X_test = X_normalized[train_index], X_normalized[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model = OneVsOneClassifier(SVC(random_state=0,kernel=self.kernel,C=self.C_parameter,class_weight=self.class_weight))
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)

        mean_accuracy = np.mean(accuracies)

        return mean_accuracy

data = pd.read_excel('training_dataset.xlsx')

X = data.iloc[:,2:].apply(lambda x: round(x, 4)).values
y = data.iloc[:,1].values

validation_model = skl()
validation_accuracy = validation_model.training(X, y)

print("validation accuracy:", validation_accuracy)

model = make_pipeline(
    MinMaxScaler(),
    OneVsOneClassifier(SVC(random_state=0,kernel='poly',C=1,class_weight='balanced'))
)

model.fit(X, y)

y_pred = model.predict(X)

accuracy = accuracy_score(y, y_pred)
print("accuracy:", accuracy)
precision = precision_score(y, y_pred, average='macro')
print("precision:", precision)
recall = recall_score(y, y_pred, average='macro')
print("recall:", recall)
f1 = f1_score(y, y_pred, average='macro')
print("f1:", f1)