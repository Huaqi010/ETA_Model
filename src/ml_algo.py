from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_percentage_error
import pandas as pd
from math import sqrt
import pickle
import os
import h5py
import numpy as np
from time import time

y = h5py.File('./temp4_y.H5')
y_train = np.array(y['y_train'])
y_test = np.array(y['y_test'])
y.close()

data = h5py.File('./temp4_imputation.H5')
df_train = np.array(data['df_train'])
df_test = np.array(data['df_test'])
data.close()

def MLpipe(X, y, ML_algo, param_grid):
    res = []
    test_scores = []
    best_models = []

    for i in range(10):
        cv = KFold(n_splits=4, shuffle=True, random_state=42 * i)

        grid = GridSearchCV(ML_algo, param_grid=param_grid, scoring='neg_mean_squared_log_error',
                            cv=cv, return_train_score=True)
        grid.fit(X, y)

        result = grid.cv_results_
        res.append(result)
        test_score = grid.score(df_test, y_test)
        test_scores.append(test_score)
        best_models.append(grid.best_params_)
    return best_models, test_scores, res

t0 = time()

model_ridge = Ridge()
param_grid = {'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]}
Ridge_best, Ridge_scores, Ridge_res = MLpipe(df_train, y_train, model_ridge, param_grid)

print('Running time for Ridge:', int(time() - t0))
with open("Ridge_best_temp5.txt", "wb") as fp:  # Pickling
    pickle.dump(Ridge_best, fp)
with open("Ridge_socres_temp5.txt", "wb") as fp:  # Pickling
    pickle.dump(Ridge_scores, fp)
with open("Ridge_res_temp5.txt", "wb") as fp:  # Pickling
    pickle.dump(Ridge_res, fp)

alpha = Ridge_best[0]['alpha']
ridge = Ridge(alpha=alpha)
ridge.fit(df_train,y_train)
ridge_pred = ridge.predict(df_test)

with open("Ridge_pred_temp5.txt", "wb") as fp:   #Pickling
    pickle.dump(ridge_pred5, fp)
    
t1 = time()

RF = RandomForestRegressor()
param_grid = {'max_depth': [10, 30, 50],
              'n_estimators': [100]}
RF_best, RF_scores, RF_res = MLpipe(df_train, y_train, RF, param_grid)

print('Running time for RF:', int(time() -t1))

with open("RF_best6.txt", "wb") as fp:  # Pickling
    pickle.dump(RF_best, fp)
with open("RF_socres6.txt", "wb") as fp:  # Pickling
    pickle.dump(RF_scores, fp)
with open("RF_res6.txt", "wb") as fp:  # Pickling
    pickle.dump(RF_res, fp)

max_depth = RF_best[0]['max_depth']
n_estimators = RF_best[0]['n_estimators']
rf = RandomForestRegressor(max_depth = max_depth, n_estimators=n_estimators)
rf.fit(df_train,y_train)
rf_pred = rf.predict(df_test)

with open("rf_pred6.txt", "wb") as fp:   #Pickling
    pickle.dump(rf_pred, fp)
