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
  x_test = data[test_index, 0:(-3)]

  y_train = data[train_index, -1] / 0.2
  y_train = y_train.astype(int)
  y_train[y_train == 5] = 4

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
  accuracy = clf.score(x_test, y_test)
  print accuracy

def multi_classifier(site, ftype, data_dir):
  """
  Multi classifiers
  """
  classifiers = [
    GaussianNB(),
    MultinomialNB(),
    LDA(),
    #QDA(),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=10, n_jobs=-1),
    ExtraTreesClassifier(n_estimators=10, n_jobs=-1),
    AdaBoostClassifier(n_estimators= 10, learning_rate = 1.0),
    NearestCentroid(),
    KNeighborsClassifier(),
    #LinearRegression(normalize=False, n_jobs=-1),
    #LinearRegression(normalize=True, n_jobs=-1),
    LinearRegression(n_jobs=-1),
    LogisticRegression(),
    #SVC(kernel='rbf', gamma=2, C=1), # VERY SLOW
    #SVC(kernel='linear', C=0.025), # VERY SLOW
    OneVsRestClassifier( LinearSVC( penalty='l1', loss='squared_hinge', dual=False, tol=1e-4 ) ),
    OneVsOneClassifier( LinearSVC( penalty='l1', loss='squared_hinge', dual=False, tol=1e-4 ) ),
    OneVsRestClassifier( SVC(kernel='rbf', gamma=2, C=1), n_jobs=-1 ),
    OneVsOneClassifier( SVC(kernel='rbf', gamma=2, C=1), n_jobs=-1 )
  ]

  names = [
    'GaussianNB',
    'MultinomialNB',
    'LDA',
    #'QDA',
    'DecisionTreeClassifier',
    'RandomForestClassifier',
    'ExtraTreesClassifier',
    'AdaBoostClassifier',
    'NearestCentroid',
    'KNeighborsClassifier',
    #'LinearRegression-UnNormalize',
    #'LinearRegression-Normalize',
    'LinearRegression',
    'LogisticRegression',
    #'SVC_rbf',
    #'SVC_linear',
    'OneVsRestClassifier_LinearSVC',
    'OneVsOneClassifier_LinearSVC',
    'OneVsRestClassifier_RbfSVC',
    'OneVsOneClassifier_RbfSVC'
  ]
  num_test = NUM_TEST
  x_train, y_train, x_test, y_test = None, None, None, None
  acc_list = {}
  for name in names:
    acc_list[name] = []

  for i in range(num_test):
    x_train, y_train, x_test, y_test = load_src_data(site, i, ftype, data_dir)
    for name, clf in zip(names, classifiers):
      clf.fit(x_train, y_train)
      accuracy = clf.score(x_test, y_test)
      acc_list[name].append(accuracy)
      msg = 'Iter:\t categ={}, ftype={}, classifier={}, iter={}, accuracy={}'\
        .format(site, ftype, name, i, accuracy)
      logging.info(msg)
      print ('%s' % msg)

  # Out to log statistic results
  for name in names:
    mean_s = np.mean(acc_list[name])
    med_s = np.median(acc_list[name])

    msg = 'Stat:\t categ={}, ftype={}, classifier={}, num_test={}, mean={}, median={}'\
      .format(site, ftype, name, num_test, mean_s, med_s)
    logging.info(msg)
    print ('%s' % msg)

if __name__ == '__main__':
  # parser = argparse.ArgumentParser()
  # parser.add_argument('--logfile', type=str, default='log.txt')
  # args = parser.parse_args()

  log_file = '%s/log_classify_10_%s_%s.txt' % (RESULT_DIR, time.strftime('%Y-%m-%d_%H-%M-%S_'), str(time.time()).replace('.', '') )
  logging.basicConfig(
    filename= log_file,
    format='%(asctime)s : %(levelname)s : %(message)s', 
    level=logging.INFO)

  for site in CATEGORIES:
    for ftype in FEATURES:
      multi_classifier(site, ftype, SRC_DIR)
  