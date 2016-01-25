"""
Convert the data to strict json to import to mongodb
"""
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import json
import gzip
import os
import time
import numpy as np
import multiprocessing

from nltk.corpus import stopwords
from nltk.tokenize import regexp
from nltk import sent_tokenize
from nltk import pos_tag, map_tag
import re

from nltk.corpus import sentiwordnet as swn
from nltk.corpus.reader.wordnet import WordNetError
from settings import Settings

src_dir = Settings.PROCESSED_DIR
dst_dir = Settings.PROCESSED_DIR
categories = Settings.CATEGORIES

def load_INQUIRER_vocal(dim, vocal_file='inquirerbasic_remove_col.csv'):
  """
  Load vocabulary file (with dimension dim) for LIWC model
  """
  vocal = {}
  with open(os.path.join(src_dir, vocal_file), 'r') as in_f:
    index = 0
    for line in in_f:
      data = re.split(',|\r|\n|\r\n', line)
      N = len(data)
      if index == 0:
        index += 1
        continue

      # map between feature value and feature index
      assert(dim <= N-1)
      if N > 0:
        features = np.zeros((dim, ), dtype=np.int32)
        for i in range(dim):
          if len(data[i+1]) > 0:
            features[i] = 1
        token = data[0].lower()
        vocal[token] = features
  print 'Vocal', len(vocal)        
  return vocal

def inquirer_eval(vocal, token, dim):
  """
  Return index of emotion in INQUIRER data for token
  """
  if token in vocal:
    return vocal[token]
  else:
    tmp = token + '#1'
    if tmp in vocal:
      return vocal[tmp]
  return np.zeros((dim, ), dtype=np.int32)

def extract_INQUIRER_features(categ, dim):
  """
  Extract INQUIRER features for reviews with >= 10 votes
  """
  # Debug time
  start = time.time()
  done = 0

  # Load vocabulary file
  vocal = load_INQUIRER_vocal(dim)

  filename = 'reviews_%s' % categ
  done = 0
  start = time.time()
  data = []
  with open(os.path.join(src_dir, '%s_tokens.json' % filename), 'r') as g:
    for l in g:      
      review = json.loads(json.dumps(eval(l)))
      rvid = review['review_id']
      votes = review['votes']
      helpful = review['helpful']
      features = np.zeros((dim+3), )
      features[dim] = helpful
      features[dim+1] = votes
      features[dim+2] = helpful / float(votes)

      # Extract INQUIRER value
      for token in review['idf']:
        #token = token.lower()
        features[:dim] += inquirer_eval(vocal, token, dim) * review['freq'][token]
      
      data.append(features)
      #print done, features

      done += 1
      if done % 1000 == 0:
        tmp = time.time() - start
        print categ, 'INQUIRER reviews, Done ', done, ' in', tmp
        #break

  print 'Number of processed reviews ', done
  data = np.vstack(data)
  print 'Data shape', data.shape
  np.save('%s/%s_INQUIRER_features' % (dst_dir, categ), data)

#extract_INQUIRER_features(categories[0], dim=Settings.INQUIRER_DIM)

jobs = []
dim = Settings.INQUIRER_DIM
for categ in categories[:2]:
  _ps = multiprocessing.Process(target=extract_INQUIRER_features, args=(categ, dim))
  jobs.append(_ps)
  _ps.start()

for j in jobs:
  j.join()
  print '%s.exitcode = %s' % (j.name, j.exitcode)


