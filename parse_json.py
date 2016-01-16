"""
Convert the data to strict json to import to mongodb
"""
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import json
import gzip
import os

from nltk.corpus import stopwords
from nltk.tokenize import regexp
from nltk import sent_tokenize
from nltk import pos_tag, map_tag
import re

from settings import Settings

src_dir = Settings.PROCESSED_DIR
dst_dir = Settings.PROCESSED_DIR
categories = Settings.CATEGORIES

def corpus_parse(categ):
  # Parse corpus format to import to mongo database
  filename = 'reviews_%s' % categ
  f = open(os.path.join(dst_dir, '%s_corpus2.json' % filename), 'w')
  with open(os.path.join(src_dir, '%s_corpus.json' % filename), 'r') as g:
    for l in g:
      corpus = json.loads(json.dumps(eval(l)))
      f.write(json.dumps(corpus, indent=1).replace('\n', ''))
      f.write('\n')

for categ in categories:
  print categ
  corpus_parse(categ)