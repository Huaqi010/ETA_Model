from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
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

pca_30 = PCA(n_components=30).fit(df_train)
# use fit_transform to run PCA on our standardized matrix
X_train_pca30 = pca_30.transform(df_train)
X_test_pca30 = pca_30.transform(df_test)

def save_model(pca, output_file):
    try:
        with open(output_file,'wb') as outfile:
            pickle.dump({
                'pca_fit':pca
            },outfile)
        return True
    except:
        return False
save_model(pca_30, './pca_30.pkl')

print('pca finished')

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
        test_score = grid.score(X_test_pca30, y_test)
        test_scores.append(test_score)
        best_models.append(grid.best_params_)
    return best_models, test_scores, res

t0 = time()
from sklearn.linear_model import Lasso, Ridge, ElasticNet
model_ridge = Ridge()
param_grid = {'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]}
Ridge_best, Ridge_scores, Ridge_res = MLpipe(X_train_pca30, y_train, model_ridge, param_grid)

print('Running time for Ridge:', int(time()-t0))

with open("Ridge_pca30_best5.txt", "wb") as fp:   #Pickling
    pickle.dump(Ridge_best, fp)
with open("Ridge_pca30_socres5.txt", "wb") as fp:   #Pickling
    pickle.dump(Ridge_scores, fp)
with open("Ridge_pca30_res5.txt", "wb") as fp:   #Pickling
    pickle.dump(Ridge_res, fp)

alpha = Ridge_best[0]['alpha']
ridge = Ridge(alpha=alpha)
ridge.fit(X_train_pca30,y_train)
ridge_pred = ridge.predict(X_test_pca30)

with open("Ridge_pca30_pred5.txt", "wb") as fp:   #Pickling
    pickle.dump(ridge_pred, fp)


t1 = time()

RF = RandomForestRegressor()
param_grid = {'max_depth': [10,30,50],
              'n_estimators': [100]}
RF_best, RF_scores, RF_res = MLpipe(X_train_pca30, y_train, RF, param_grid)

print('Running time for R:', int(time() -t1))


with open("RF_pca30_best4.txt", "wb") as fp:   #Pickling
    pickle.dump(RF_best, fp)
with open("RF_pca30_socres4.txt", "wb") as fp:   #Pickling
    pickle.dump(RF_scores, fp)
with open("RF_pca30_res.txt4", "wb") as fp:   #Pickling
    pickle.dump(RF_res, fp)

max_depth = RF_best[0]['max_depth']
n_estimators = RF_best[0]['n_estimators']
rf = RandomForestRegressor(max_depth = max_depth, n_estimators=n_estimators)
rf.fit(X_train_pca30,y_train)
rf_pred = rf.predict(X_test_pca30)

with open("rf_pca30_pred4.txt", "wb") as fp:   #Pickling
    pickle.dump(rf_pred, fp)
