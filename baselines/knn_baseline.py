'''Author: Priyanka Raghavan
   Description: k-nearest neighbors regression baseline with hyperparameter optimization for randomly split data'''

path = "C:/Users/ChemeGrad2020/OneDrive/10.C51/Project"
import sys
sys.path.insert(1,path)
import numpy as np
from data.data_processing import read_randomsplit_data
from figs.plot_parity import plot_parity_chemprop
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score

def optimize_knn(X_train,y_train): #hyperparameter optimization for KNN
   knn_params = {'knn_model__n_neighbors': [5,20,50,100]}
   knn_model = KNeighborsRegressor()
   pipeline = Pipeline([('knn_model', knn_model)])
   knn_reg = GridSearchCV(pipeline,knn_params, cv=5, scoring = 'neg_mean_absolute_error')
   knn_reg.fit(X_train,y_train)
   knn_best_params = knn_reg.best_params_
   knn_best_score = knn_reg.best_score_
   print(knn_best_params, knn_best_score)
   return knn_best_params

def knn_reg(best_params,X_train,X_test,y_train,y_test):
   knn_model = KNeighborsRegressor(n_neighbors=best_params['knn_model__n_neighbors'])
   knn_model.fit(X_train,y_train)
   y_pred = knn_model.predict(X_test)

   # remove non-retained molecules
   #nrt_idx = [i for i in range(len(y_test)) if y_test[i] < 5]
   #y_test = [y_test[i] for i in range(len(y_test)) if i not in nrt_idx]
   #y_pred = [y_pred[i] for i in range(len(y_pred)) if i not in nrt_idx]

   mae = mean_absolute_error(y_test,y_pred)
   medae = median_absolute_error(y_test,y_pred)
   mre = mean_absolute_percentage_error(y_test,y_pred)
   r2 = r2_score(y_test,y_pred)
   #plot_parity_chemprop(y_test,y_pred,'knn_parity.png')

   return mae,medae,mre,r2
