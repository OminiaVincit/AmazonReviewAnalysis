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
#FEATURES = ['STR', 'TOPICS_64', 'LIWC', 'GALC']
CATEGORIES = Settings.CATEGORIES
EPS = 1e-10

def svr(x_train, y_train, x_test):
  """
  Support vector regression
  """
  start = time.time()
  reg = svm.SVR(kernel='rbf', C=0.01, epsilon=0.1, gamma=0.001).fit(x_train, y_train)
  return reg.predict(x_test)

def score(y_predict, y_test, h_test, N_test):
  """
  Some of metric score for regression evaluation
  """
  diff = y_test - y_predict
  mse = np.mean(diff*diff)
  rmse = np.sqrt(mse)
  mae = np.mean(abs(diff))

  # Compute spearmanr correlation score
  rho, pval = spearmanr(y_predict, y_test)

  # Compute log-likelihood
  
  pre_logfit = np.mean(h_test * np.log(y_predict + EPS) + (N_test - h_test)*np.log(1.0 - y_predict + EPS) )
  
  return mae, mse, rmse, rho, pval, pre_logfit

def eval_conf(m_dat):
  """
  Remove outlier and take median
  """
  (nr, nc) = m_dat.shape
  rs = np.zeros((nc, ))
  for i in range(nc):
    tmp = m_dat[:, i]
    med = np.median(tmp)
    mad = 1.4826 * np.median(abs(tmp - med))
    rs[i] = np.mean( tmp[ abs(tmp - med) / mad < 1 ] )
  return rs

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
  y_train = data[train_index, -1]

  # Reduce number of training data
  redc = y_train.shape[0] / 2
  x_train = x_train[0:redc]
  y_train = y_train[0:redc]

  # Test data
  x_test = data[test_index, 0:(-3)]
  y_test  = data[test_index, -1]

  h_test = data[test_index, -3]
  N_test = data[test_index, -2]

  #print data.shape
  #print 'x_train', x_train.shape, 'y_train', y_train.shape, 'x_test', x_test.shape, 'y_test', y_test.shape
  return x_train, y_train, x_test, y_test, h_test, N_test

if __name__ == '__main__':
  # x_train, y_train, x_test, y_test = load_data('yelp', 1, 64, 0)
  # print x_train.shape, x_test.shape, y_train.shape, y_test.shape
  # grid_search_cv(x_train, y_train, x_test, y_test)

  log_file = '%s/log_conf_%s_%s.txt' % (RESULT_DIR, time.strftime('%Y-%m-%d_%H-%M-%S_'), str(time.time()).replace('.', '') )
  logging.basicConfig(
    filename= log_file,
    format='%(asctime)s : %(levelname)s : %(message)s', 
    level=logging.INFO)

  for site in CATEGORIES:
    for index in range(50):
      dat = []
      for ftype in FEATURES:
      #for ftype in ['STR', 'TOPICS_64', 'tfidf', 'LIWC', 'INQUIRER', 'GALC']:
        x_train, y_train, x_test, y_test, h_test, N_test = load_src_data(site, index, ftype)
        base_logfit = np.mean(h_test * np.log(y_test + EPS) + (N_test - h_test)*np.log(1.0 - y_test + EPS) )
        print base_logfit
        y_predict = svr(x_train, y_train, x_test)
        mae, mse, rmse, rho, pval, pre_logfit = score(y_predict, y_test, h_test, N_test)
        dat.append(y_predict)
        msg = 'categ=%s, ftype=%s, index=%d, rmse=%.10f, corr=%.10f, pval=%.10f, mse=%.10f, mae=%.10f, likelihood=%.10f, basehood=%.10f' %(site, ftype, index, rmse, rho, pval, mse, mae, pre_logfit, base_logfit)
        print msg
        logging.info(msg)
      dat = np.vstack(dat)
      t_predict = eval_conf(dat)
      mae, mse, rmse, rho, pval, pre_logfit = score(t_predict, y_test, h_test, N_test)
      msg = 'categ=%s, ftype=ConfAll, index=%d, rmse=%.10f, corr=%.10f, pval=%.10f, mse=%.10f, mae=%.10f, likelihood=%.10f, basehood=%.10f' %(site, index, rmse, rho, pval, mse, mae, pre_logfit, base_logfit)
      print msg
      logging.info(msg)

