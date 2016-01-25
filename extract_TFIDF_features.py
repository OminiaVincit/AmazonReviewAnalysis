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


def load_tfidf_vocal(categ, dim):
    """
    Load vocabulary file (with dimension dim) for tfidf model
    """
    filename = 'reviews_%s' % categ
    vocal = []    
    with open(os.path.join(src_dir, '%s_tfidf_vocabularies.json' % filename), 'r') as in_f:
      for line in in_f:
        data = json.loads(line)
        num_docs = data['NUMDOCS']
        threshold = num_docs / 128
        print num_docs, len(data)
        # count = 0
        # for token in data:
        #     freq = data[token]
        #     if freq >= threshold:
        #         count += 1
        #         print count, token, freq

        dlist = sorted(data.items(), key=lambda x: -x[1])
        for i in range(1, dim + 1):
          vocal.append(dlist[i])
    print 'Load vocal', len(vocal)
    return vocal

def extract_TFIDF_features(categ, dim):
  """
  Extract tfidf features for reviews with >= 10 votes
  """
  # Debug time
  start = time.time()
  done = 0

  # Load vocabulary file
  vocal = load_tfidf_vocal(categ, dim)

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

      # Extract tf-idf value
      for i in range(dim):
        token = vocal[i][0]
        if token in review['idf']:
          features[i] = review['idf'][token] * review['tf'][token]
      
      data.append(features)
      #print done, features

      done += 1
      if done % 1000 == 0:
        tmp = time.time() - start
        print categ, 'TFIDF reviews, Done ', done, ' in', tmp
        #break

  print 'Number of processed reviews ', done
  data = np.vstack(data)
  print 'Data shape', data.shape
  np.save('%s/%s_TFIDF_features' % (dst_dir, categ), data)

#extract_TFIDF_features(categories[0], dim=Settings.TFIDF_DIM)

jobs = []
dim = Settings.TFIDF_DIM
for categ in categories[:2]:
  _ps = multiprocessing.Process(target=extract_TFIDF_features, args=(categ, dim))
  jobs.append(_ps)
  _ps.start()

for j in jobs:
  j.join()
  print '%s.exitcode = %s' % (j.name, j.exitcode)


