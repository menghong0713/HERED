from sklearn import datasets
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd

data = pd.read_excel('train_dataset.xlsx')

X = data.iloc[:,2:].apply(lambda x: round(x, 4)).values
y = data.iloc[:,1].values

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