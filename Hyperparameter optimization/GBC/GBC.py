from sklearn.pipeline import make_pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

class skl:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.model = OneVsRestClassifier(GradientBoostingClassifier(random_state=0),n_jobs = -1)
               
        self.cv =KFold(n_splits=92, shuffle=True, random_state=1000)
        self.parameters = {
                            'estimator__loss': ['log_loss', 'exponential'], # 'squared_error'
                            'estimator__learning_rate':[0.01,0.05,0.1,0.15,0.2,0.5,1], # 0.1,[0.0, inf)
                            'estimator__n_estimators': [1,50,100,150,200,500,1000], # 100,[1, inf)
                            'estimator__min_samples_split': [2,5,10,15,20,50,100], # 2, [2, inf)
                            'estimator__max_depth':np.round(np.arange(3, 11, 1), 0).tolist(), # 3, [1, inf)
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
