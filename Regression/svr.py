import os
import numpy as np
import time
import logging

import pickle
from sklearn import svm
from scipy.stats import spearmanr

import sys
sys.path.append('../')

from settings import Settings

SRC_DIR = Settings.PROCESSED_DIR
RESULT_DIR = Settings.RESULT_DIR
FEATURES = Settings.FEATURES
CATEGORIES = Settings.CATEGORIES

def svr(x_train, y_train, x_test, y_test):
  """
  Support vector regression
  """
  start = time.time()
  reg = svm.SVR(kernel='rbf', C=0.01, epsilon=0.1, gamma=0.001).fit(x_train, y_train)
  mae, mse, rmse, rho, pval = score(reg, x_test, y_test)
  return rmse, rho, pval, mse, mae 

def score(reg, x_test, y_test):
  """
  Some of metric score for regression evaluation
  """
  y_predict = reg.predict(x_test)
  diff = y_test - y_predict
  mse = np.mean(diff*diff)
  rmse = np.sqrt(mse)
  mae = np.mean(abs(diff))

  # Compute spearmanr correlation score
  rho, pval = spearmanr(y_predict, y_test)
  return mae, mse, rmse, rho, pval

def grid_search_cv(x_train, y_train, x_test, y_test):
  start = time.time()
  from sklearn.grid_search import GridSearchCV
  tuned_parameters = [
    {'kernel': ['rbf'], 'gamma': [10**i for i in range(-4,3)], 'C': [10**i for i in range(-3, 4)], 'epsilon':[float(i)/10.0 for i in range(0, 11)] }]
    #{'kernel': ['linear'], 'C': [10**i for i in range(-4, 3)], 'epsilon':[float(i)/10.0 for i in range(0, 11)] } ]
  gscv = GridSearchCV(svm.SVR(), tuned_parameters, cv=5, scoring='mean_squared_error', n_jobs=-1)
  gscv.fit(x_train, y_train)

  # Worse and best score
  #params_min,_,_ = gscv.grid_scores_[np.argmin([x[1] for x in gscv.grid_scores_])]
  params_max,_,_ = gscv.grid_scores_[np.argmax([x[1] for x in gscv.grid_scores_])]
  #reg_min = svm.SVR(kernel=params_min['kernel'], C=params_min['C'], gamma=params_min['gamma'])
  reg_max = gscv.best_estimator_
  #params_max = reg_max.get_params()
  
  # Refit using all training data
  #reg_min.fit(x_train, y_train)
  reg_max.fit(x_train, y_train)
  #print 'reg_min ', score(reg_min, x_test, y_test), params_min
  print 'reg_max ', score(reg_max, x_test, y_test), params_max
  #params_max['kernel'], params_max['gamma'], params_max['C'], params_max['epsilon']
  print 'grid_search_cv in', time.time() - start

def load_src_data(site, index, ftype, data_dir = SRC_DIR):
  """
  Load data from new structure data
  """
  # Get partition of exp
  exp_file = '%s_partition.pickle' % site
  with open(os.path.join(data_dir, exp_file), 'rb') as handle:
    part = pickle.load(handle)

  # Load data file
  data_file = '%s_%s_features.npy' % (site, ftype)
  data = np.load(os.path.join(data_dir, data_file))
  train_index = part[index]['train']
  test_index = part[index]['test']

  x_train = data[train_index, 0:(-3)]
  x_test = data[test_index, 0:(-3)]

  y_train = data[train_index, -1]
  y_test  = data[test_index, -1]

  print data.shape
  print 'x_train', x_train.shape, 'y_train', y_train.shape, 'x_test', x_test.shape, 'y_test', y_test.shape
  return x_train, y_train, x_test, y_test

if __name__ == '__main__':
  # x_train, y_train, x_test, y_test = load_data('yelp', 1, 64, 0)
  # print x_train.shape, x_test.shape, y_train.shape, y_test.shape
  # grid_search_cv(x_train, y_train, x_test, y_test)

  log_file = '%s/log_svr_%s_%s.txt' % (RESULT_DIR, time.strftime('%Y-%m-%d_%H-%M-%S_'), str(time.time()).replace('.', '') )
  logging.basicConfig(
    filename= log_file,
    format='%(asctime)s : %(levelname)s : %(message)s', 
    level=logging.INFO)

  for site in CATEGORIES:
    for ftype in FEATURES:
      for index in range(50):
        print site, ftype, index
        x_train, y_train, x_test, y_test = load_src_data(site, index, ftype)
        #grid_search_cv(x_train[:5000], y_train[:5000], x_test[:1000], y_test[:1000])
        rmse, rho, pval, mse, mae = svr(x_train, y_train, x_test, y_test)
        msg = 'categ=%s, ftype=%s, index=%d, rmse=%.10f, rho=%.10f, pval=%.10f, mse=%.10f, mae=%.10f' %(site, ftype, index, rmse, rho, pval, mse, mae)
        print msg
        logging.info(msg)
