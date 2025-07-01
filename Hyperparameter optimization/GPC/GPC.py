from sklearn.multiclass import OneVsRestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

class skl:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.model = OneVsRestClassifier(GaussianProcessClassifier(random_state=0),n_jobs = -1)
               
        self.cv =KFold(n_splits=92, shuffle=True, random_state=1000)
        self.parameters = {
                            'estimator__n_restarts_optimizer': np.round(np.arange(0, 10, 1), 0).tolist(),
                            'estimator__max_iter_predict':np.round(np.arange(100, 501, 100), 0).tolist(), 
                           }
        self.njobs = -1
        self.scoring = 'accuracy'

    def training(self, X, y):
        X_normalized = self.scaler.fit_transform(X)
        
        grid_search = GridSearchCV(self.model, 
                                    param_grid=self.parameters,
                                   scoring=self.scoring, 
                                   n_jobs = self.njobs,
                                   cv=self.cv
                                   )
        grid_search.fit(X_normalized, y)

        best_parameters = grid_search.best_params_

        scores = grid_search.best_score_

        return scores, best_parameters

data = pd.read_excel('train_dataset.xlsx')   
X = data.iloc[:,2:].apply(lambda x: round(x, 4))  
# print(X)
y = data.iloc[:,1].values
# print(y)

model_cv = skl()
scores, best_parameters = model_cv.training(X, y)

print("scores:", scores)
print("best_parameters:", best_parameters)
