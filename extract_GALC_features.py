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

def load_GALC_vocal(vocal_file='GALC_0.csv'):
  """
  Load vocabulary file (with dimension dim) for GALC model
  """
  vocal = {}
  with open(os.path.join(src_dir, vocal_file), 'r') as in_f:
    index = 0
    for line in in_f:
      data = re.split(',|\r|\n|\r\n', line)
      N = len(data)
      # Skip first token (category name)
      for i in range(1, N):
        if len(data[i]) > 0:
          vocal[data[i]] = index
      index += 1
  return vocal

def galc_eval(vocal, token, outbound):
  """
  Return index of emotion in GALC data for token
  """
  if token in vocal:
    return vocal[token]
  else:
    strlen = len(token)
    for i in range(strlen):
      tmp = token[:(strlen-i)] + '*'
      if tmp in vocal:
        return vocal[tmp]
  return outbound

def extract_GALC_features(categ, dim):
  """
  Extract GALC features for reviews with >= 10 votes
  """
  # Debug time
  start = time.time()
  done = 0

  # Load vocabulary file
  vocal = load_GALC_vocal()
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

      # Extract GALC value
      for token in review['idf']:
        label = galc_eval(vocal, token, dim-1)
        features[label] += review['freq'][token]

      data.append(features)
      #print done, features

      done += 1
      if done % 1000 == 0:
        tmp = time.time() - start
        print categ, 'GALC reviews, Done ', done, ' in', tmp
        #break

  print 'Number of processed reviews ', done
  data = np.vstack(data)
  print 'Data shape', data.shape
  np.save('%s/%s_GALC_features' % (dst_dir, categ), data)

#extract_GALC_features(categories[0], dim=Settings.GALC_DIM)

jobs = []
dim = Settings.GALC_DIM
for categ in categories[:2]:
  _ps = multiprocessing.Process(target=extract_GALC_features, args=(categ, dim))
  jobs.append(_ps)
  _ps.start()

for j in jobs:
  j.join()
  print '%s.exitcode = %s' % (j.name, j.exitcode)


