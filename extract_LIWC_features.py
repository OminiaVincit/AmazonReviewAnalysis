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

def load_LIWC_vocal(vocal_file='LIWC2007.dic'):
  """
  Load vocabulary file (with dimension dim) for LIWC model
  """
  vocal = {}
  fet = {}
  with open(os.path.join(src_dir, vocal_file), 'r') as in_f:
    index = 0
    for line in in_f:
      data = re.split(' |\t|\r|\n|\r\n', line)
      N = len(data)
      # map between feature value and feature index
      if N > 0 and data[0].isdigit():
        fet[data[0]] = index
        index += 1
      elif N > 1:
        features = np.zeros((index, ), dtype=np.int32)
        for i in range(1, N):
          if len(data[i]) > 0 and data[i].isdigit():
            mdx = fet[data[i]]
            features[mdx] = 1
        vocal[data[0]] = features

  print 'vocal', len(vocal)        
  return vocal

def liwc_eval(vocal, token, dim):
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
  return np.zeros((dim, ), dtype=np.int32)

def extract_LIWC_features(categ, dim):
  """
  Extract LIWC features for reviews with >= 10 votes
  """
  # Debug time
  start = time.time()
  done = 0

  # Load vocabulary file
  vocal = load_LIWC_vocal()
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

      # Extract LIWC value
      for token in review['idf']:
        features[:dim] += liwc_eval(vocal, token, dim) * review['freq'][token]
      
      data.append(features)
      #print done, features

      done += 1
      if done % 1000 == 0:
        tmp = time.time() - start
        print categ, 'LIWC reviews, Done ', done, ' in', tmp
        #break

  print 'Number of processed reviews ', done
  data = np.vstack(data)
  print 'Data shape', data.shape
  np.save('%s/%s_LIWC_features' % (dst_dir, categ), data)

# extract_LIWC_features(categories[0], dim=Settings.LIWC_DIM)

jobs = []
dim = Settings.LIWC_DIM
for categ in categories[:2]:
  _ps = multiprocessing.Process(target=extract_LIWC_features, args=(categ, dim))
  jobs.append(_ps)
  _ps.start()

for j in jobs:
  j.join()
  print '%s.exitcode = %s' % (j.name, j.exitcode)


