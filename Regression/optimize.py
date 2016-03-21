#!env python
# -*- coding:utf-8 -*-
"""
Optimization
"""

import os
import numpy as np
import time
import logging
import pickle

import sys
sys.path.append('../')

from settings import Settings

SRC_DIR = Settings.PROCESSED_DIR
RESULT_DIR = Settings.RESULT_DIR
FEATURES = Settings.FEATURES
CATEGORIES = Settings.CATEGORIES
EPS = 1e-10

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
    
    # for i in range(data.shape[0]):
    #     avg = np.mean(data[i, 0:(-3)])
    #     std = np.std(data[i, 0:(-3)])
    #     data[i, 0:(-3)] -= avg
    #     if std != 0:
    #         data[i, 0:(-3)] /=  std
    # print 'Finish normalize data'
            
    train_index = part[index]['train']
    test_index = part[index]['test']

    num_features = data.shape[1] - 3
    avg = np.zeros((num_features, ))
    std = np.zeros((num_features, ))
    for i in range(num_features):
        avg[i] = np.mean(data[train_index, i])
        std[i] = np.std(data[train_index, i])
        data[:, i] -= avg[i]
        if std[i] != 0:
            data[:, i] /= std[i]
    print 'Finish normalize data'

    # Add colum one to data
    n_r, n_c = data.shape
    X_it = np.ones(shape=(n_r, n_c + 1), dtype=data.dtype)
    X_it[:, 1:(n_c+1)] = data

    # Get training data
    x_train = X_it[train_index, 0:(-3)]
    y_train = X_it[train_index, -1]

    h_train = X_it[train_index, -3]
    N_train = X_it[train_index, -2]

    # # Reduce number of training data
    # redc = y_train.shape[0] / 2
    # x_train = x_train[0:redc]
    # y_train = y_train[0:redc]

    # Test data
    x_test  = X_it[test_index, 0:(-3)]
    y_test  = X_it[test_index, -1]

    h_test = X_it[test_index, -3]
    N_test = X_it[test_index, -2]

    #print 'x_train', x_train.shape, 'y_train', y_train.shape, 'x_test', x_test.shape, 'y_test', y_test.shape
    return x_train, y_train, x_test, y_test, h_train, N_train, h_test, N_test

class SGDOptimizer():
    u'''Stochastic Gradient Descent optimizer'''

    def __init__(self, Xtrain, Ytrain, Xtest, Ytest, Htrain, Ntrain, Htest, Ntest, \
                batch_size=128, base_lr = 0.01, alpha=0, beta = 0, max_epoch = 100):
        """
        Initialize
        """
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.Xtest = Xtest
        self.Ytest = Ytest

        self.Ntrain = Ntrain
        self.Htrain = Htrain
        self.Ntest = Ntest
        self.Htest = Htest

        num_pams = Xtrain.shape[1]
        self.weights = np.zeros(shape=(num_pams, ))
        self.base_lr = base_lr
        self.min_lr = 1e-8
        self.min_mag = 1e-10
        self.gamma = 0.8
        self.iter = 0
        self.epoch = 0
        self.alpha = alpha
        self.beta = beta
        self.max_epoch = max_epoch
        self.batch_size = batch_size

        self.cost = None
        self.test_cost = None
        self.gradient = np.zeros(shape=(num_pams, ))
        self.prev_update = np.zeros(shape=(num_pams, ))
        self.test_likelihood = - np.mean(Htest * np.log(Ytest + EPS) + (Ntest - Htest) * np.log(1.0 - Ytest + EPS))
        self.train_likelihood = - np.mean(Htrain * np.log(Ytrain + EPS) + (Ntrain - Htrain) * np.log(1.0 - Ytrain + EPS))

    def _sigmoid(self, X, w):
        q = 1.0 / (1.0 + np.exp(-X.dot(w)))
        return q

    def compute_cost(self, xtrain, ytrain, ntrain, htrain):
        """
        Calculate cost function
        """
        q = self._sigmoid(xtrain, self.weights)
        self.cost = np.mean(htrain * np.log(q + EPS) + (ntrain-htrain) * np.log(1.0-q + EPS)) \
            + self.alpha * self.weights.T.dot(self.weights) / 2.0
        self.cost *= -1

    def compute_test_cost(self):
        """
        Calculate cost function
        """
        q = self._sigmoid(self.Xtest, self.weights)
        self.test_cost = np.mean(self.Htest * np.log(q + EPS) + (self.Ntest-self.Htest) * np.log(1.0-q + EPS))
        self.test_cost *= -1
            #+ self.alpha * self.weights.T.dot(self.weights) / 2.0

    def compute_gradient(self, xtrain, ntrain, htrain):
        """
        Compute gradient of cost function
        """
        q = self._sigmoid(xtrain, self.weights)
        self.gradient = xtrain.T.dot(htrain - ntrain * q) / float(q.shape[0]) \
            + self.alpha * self.weights
        self.gradient *= -1

    def compute_gradient_cost(self, xtrain, ytrain, ntrain, htrain):
        """
        Compute gradient and cost function
        """
        q = self._sigmoid(xtrain, self.weights)
        self.cost = np.mean(htrain * np.log(q + EPS) + (ntrain-htrain) * np.log(1.0-q + EPS)) \
            + self.alpha * self.weights.T.dot(self.weights) / 2.0
        self.cost *= -1
        self.gradient = xtrain.T.dot(htrain - ntrain * q) / float(q.shape[0]) \
            + self.alpha * self.weights
        self.gradient *= -1   

    def update(self):
        u'''Update paramters by gradient descent'''

        # direction = gradient.T.dot(self.prev_gradient)
        #if self.iter % self.step == 0:
        #    self.base_lr *= self.gamma
        #if direction < 0:
        #    self.base_lr *= self.gamma

        if self.iter % 10000 == 0:
            self.base_lr *= self.gamma

        new_update = self.base_lr * self.gradient + self.beta * self.prev_update
        self.weights -= new_update
        self.prev_update = new_update
        self.iter += 1
        return True

    def forward(self, xtrain, ytrain, ntrain, htrain):
        self.compute_gradient_cost(xtrain, ytrain, ntrain, htrain)
        #print 'Iter', self.iter, self.cost
        self.update()
        
    def run_epoch(self):
        """
        Run optimizer by epoch
        """
        if self.epoch > self.max_epoch or self.base_lr < self.min_lr:
            return False
        
        self.compute_cost(self.Xtrain, self.Ytrain, self.Ntrain, self.Htrain)
        self.compute_test_cost()

        print 'Epoch', self.epoch, self.cost, self.test_cost, self.train_likelihood, self.test_likelihood, self.base_lr, np.sqrt(self.prev_update.T.dot(self.prev_update))
        #return False
        # gradient = self.gradient
        # magnitude = gradient.T.dot(gradient)
        # magnitude = np.sqrt(magnitude)
        # if magnitude < self.min_mag:
        #     return False

        num_samples = self.Ytrain.shape[0]
        perm = np.random.permutation(num_samples)
        batch_size = self.batch_size
        #acc_cost = []
        for i in range(0, num_samples, batch_size):
            xtrain = self.Xtrain[perm[i:(i+batch_size)]]
            ytrain = self.Ytrain[perm[i:(i+batch_size)]]
            ntrain = self.Ntrain[perm[i:(i+batch_size)]]
            htrain = self.Htrain[perm[i:(i+batch_size)]]
            self.forward(xtrain, ytrain, ntrain, htrain)
            #acc_cost.append(self.cost)

        self.epoch += 1
        return True

    def run(self):
        while self.run_epoch():
            pass

if __name__ == '__main__':
    for site in CATEGORIES[1:2]:
        for ftype in FEATURES[3:4]:
            for index in range(1):
                x_train, y_train, x_test, y_test, h_train, N_train, h_test, N_test = \
                    load_src_data(site, index, ftype)
                opt = SGDOptimizer(x_train, y_train, x_test, y_test, h_train, N_train, h_test, N_test, alpha = 0, base_lr = 0.01, beta = 0.9, max_epoch = 2000)
                opt.run()





