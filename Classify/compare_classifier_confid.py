'''Compare classifier for different features design'''
import numpy as np
import os
import logging 
import argparse
import pickle
import time

import sys
sys.path.append('../')

from settings import Settings

SRC_DIR = Settings.PROCESSED_DIR
RESULT_DIR = Settings.RESULT_DIR
FEATURES = Settings.FEATURES
CATEGORIES = Settings.CATEGORIES

from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn import svm, metrics
from sklearn.svm import LinearSVC, SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier

# Not support for ver. 0.17
#from sklearn.neural_network import MLPClassifier
NUM_TEST = 10

classifiers = [
  #GaussianNB(),
  MultinomialNB(),
  LDA(),
  #QDA(),
  #DecisionTreeClassifier(),
  #RandomForestClassifier(n_estimators=10, n_jobs=-1),
  #ExtraTreesClassifier(n_estimators=10, n_jobs=-1),
  AdaBoostClassifier(n_estimators= 50, learning_rate = 1.0),
  #NearestCentroid(),
  #KNeighborsClassifier(),
  #LinearRegression(normalize=False, n_jobs=-1),
  #LinearRegression(normalize=True, n_jobs=-1),
  #LinearRegression(n_jobs=-1),
  LogisticRegression(),
  #SVC(kernel='rbf', gamma=2, C=1), # VERY SLOW
  #SVC(kernel='linear', C=0.025), # VERY SLOW
  OneVsRestClassifier( LinearSVC( penalty='l1', loss='squared_hinge', dual=False, tol=1e-4 ) ),
  OneVsOneClassifier( LinearSVC( penalty='l1', loss='squared_hinge', dual=False, tol=1e-4 ) ),
  OneVsRestClassifier( SVC(kernel='rbf', gamma=2, C=1), n_jobs=-1 ),
  OneVsOneClassifier( SVC(kernel='rbf', gamma=2, C=1), n_jobs=-1 )
]

names = [
  #'GaussianNB',
  'MultinomialNB',
  'LDA',
  #'QDA',
  #'DecisionTreeClassifier',
  #'RandomForestClassifier',
  #'ExtraTreesClassifier',
  'AdaBoostClassifier',
  #'NearestCentroid',
  #'KNeighborsClassifier',
  #'LinearRegression-UnNormalize',
  #'LinearRegression-Normalize',
  #'LinearRegression',
  'LogisticRegression',
  #'SVC_rbf',
  #'SVC_linear',
  'OneVsRestClassifier_LinearSVC',
  'OneVsOneClassifier_LinearSVC',
  'OneVsRestClassifier_RbfSVC',
  'OneVsOneClassifier_RbfSVC'
]

def load_src_data(site, index, ftype, data_dir):
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
  redc = y_train.shape[0] / 2

  # Reduce number of training samples
  x_train = x_train[0:redc]
  y_train = y_train[0:redc]

  y_train = y_train / 0.2
  y_train = y_train.astype(int)
  y_train[y_train == 5] = 4

  x_test = data[test_index, 0:(-3)]
  y_test  = data[test_index, -1] / 0.2
  y_test = y_test.astype(int)
  y_test[y_test == 5] = 4

  #print data.shape
  #print 'x_train', x_train.shape, 'y_train', y_train.shape, 'x_test', x_test.shape, 'y_test', y_test.shape
  return x_train, y_train, x_test, y_test

def classify(site, index, ftype, clf, data_dir):
  """
  Classify data with partition index and classifier clf
  """
  x_train, y_train, x_test, y_test = load_src_data(site, index, ftype, data_dir)
  print y_test[y_test==2].shape, y_test[y_test==4].shape, y_test[y_test==5].shape
  clf.fit(x_train, y_train)
  # predict = clf.predict(x_test)

  accuracy = clf.score(x_test, y_test)
  print accuracy

def eval_conf(m_dat):
  """
  Remove outlier and take median
  """
  (nr, nc) = m_dat.shape
  rs = np.zeros((nc, ), dtype=np.int32)
  for i in range(nc):
    tmp = m_dat[:, i]
    med = np.median(tmp)
    mad = 1.4826 * np.median(abs(tmp - med))
    if mad == 0:
      rs[i] = int(med)
    else:
      rs[i] = int(np.median( tmp[ abs(tmp - med) / mad < 2 ] ))
  return rs

def multi_classifier(site, i, data_dir):
  """
  Multi classifiers
  """

  num_test = NUM_TEST
  x_train, y_train, x_test, y_test = None, None, None, None
  acc_list = {}
  # for name in names:
  #   acc_list[name] = []

  for name, clf in zip(names, classifiers):
    predict_buff = []
    for ftype in FEATURES:
      x_train, y_train, x_test, y_test = load_src_data(site, i, ftype, data_dir)
      clf.fit(x_train, y_train)
      predicted = clf.predict( x_test )
      predict_buff.append( predicted )
      # accuracy = clf.score( x_test, y_test )
      accuracy = np.sum(predicted == y_test) / float(y_test.shape[0])
      # acc_list[name].append(accuracy)
      msg = 'Iter:\t categ={}, ftype={}, classifier={}, iter={}, accuracy={}'\
        .format(site, ftype, name, i, accuracy)
      logging.info(msg)
      print ('%s' % msg)

    predict_buff = np.vstack(predict_buff)
    predict_buff = np.array(predict_buff, dtype=np.int32)
    conf_pred = eval_conf(predict_buff)
    accuracy = np.sum(conf_pred == y_test) / float(y_test.shape[0])
    # Out to log statistic results
    msg = 'Iter:\t categ={}, ftype=ConfAll, classifier={}, iter={}, accuracy={}'\
      .format(site, name, i, accuracy)
    logging.info(msg)
    print ('%s' % msg)

if __name__ == '__main__':
  # parser = argparse.ArgumentParser()
  # parser.add_argument('--logfile', type=str, default='log.txt')
  # args = parser.parse_args()

  log_file = '%s/log_classify_confd_halftrain_%s_%s.txt' % (RESULT_DIR, time.strftime('%Y-%m-%d_%H-%M-%S_'), str(time.time()).replace('.', '') )
  logging.basicConfig(
    filename= log_file,
    format='%(asctime)s : %(levelname)s : %(message)s', 
    level=logging.INFO)

  for site in CATEGORIES:
    for i in range(NUM_TEST):
      print site, i
      multi_classifier(site, i, SRC_DIR)
  