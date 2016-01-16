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
from nltk.stem.wordnet import WordNetLemmatizer
import re

from nltk.corpus import sentiwordnet as swn
from nltk.corpus.reader.wordnet import WordNetError
from settings import Settings

src_dir = Settings.PROCESSED_DIR
dst_dir = Settings.PROCESSED_DIR
categories = Settings.CATEGORIES

TAGS = ['ADJ', 'ADV', 'ADP', 'CONJ', 'NOUN', 'DET', 'EX', 'FW', 'MOD',\
  'NP', 'PRON', 'PRT', 'TO', 'UH', 'VERB', 'VD', 'VG', 'VN', 'WH', 'NUM', 'X', '.']

# Tag Meaning Examples
# ADJ adjective new, good, high, special, big, local
# ADV adverb  really, already, still, early, now
# ADP
# CONJ conjunction and, or, but, if, while, although
# DET determiner  the, a, some, most, every, no
# EX  existential there, there's
# FW  foreign word  dolce, ersatz, esprit, quo, maitre
# MOD modal verb  will, can, would, may, must, should
# NOUN noun  year, home, costs, time, education
# NP  proper noun Alison, Africa, April, Washington
# NUM number  twenty-four, fourth, 1991, 14:24
# PRON pronoun he, their, her, its, my, I, us
# PRT preposition on, of, at, with, by, into, under
# TO  the word to to
# UH  interjection  ah, bang, ha, whee, hmpf, oops
# VERB verb  is, has, get, do, make, see, run
# VD  past tense  said, took, told, made, asked
# VG  present participle  making, going, playing, working
# VN  past participle given, taken, begun, sung
# WH  wh determiner who, which, when, what, where, how

EXCEPT_CHAR = [u'0', u'1', u'2', u'3', u'4', u'5', u'6', u'7', u'8', u'9',
u'!', u'"', u'#', u'$', u'%', u'&', u'\'', u'(', u')', u'-', u'=', u'^', u'~', 
u'\\', u'|', u'@', u'`', u'[', u'{', u';', u'+', u':', u'*', u']', u'}', 
u',', u'<', u'>', u'.', u'/', u'?', u'_']

from nltk.corpus import stopwords

def corpus_condition(word):
    """
    Condition of word to add to corpus
    """
    save_flag = False
    if word['pos'] in ['NOUN', 'ADJ', 'ADV']:
        if len(word['word']) > 2:
            save_flag = True
            for ch in word['word']:
                if ch in EXCEPT_CHAR:
                    save_flag = False
                    break
    return save_flag

def make_corpus(categ):
  """
  Extract STR features
  """
  filename = 'reviews_%s' % categ
  done = 0
  lem = WordNetLemmatizer()
  start = time.time()
  stopwds = stopwords.words('english')

  with open(os.path.join(src_dir, '%s_tags.json' % filename), 'r') as g:
    for l in g:      
      u = json.loads(json.dumps(eval(l)))
      words = [word for word in u['words'] if corpus_condition(word)]
      nouns = [lem.lemmatize(word['word'].lower()) for word in words]
      nouns = [word for word in nouns if word not in stopwds]
      u['words'] = nouns

      #print done, features
      done += 1
      if done % 1000 == 0:
        tmp = time.time() - start
        print categ, 'Make corpus, Done ', done, ' in', tmp
      yield str(u)

def write_to_file(categ):
  """
  Write to json file
  """
  filename = 'reviews_%s' % categ
  print filename
  f = open(os.path.join(dst_dir, '%s_corpus.json' % filename), 'w')
  for l in make_corpus(categ):
    f.write(l + '\n')

#write_to_file(categories[0])

jobs = []
dim = Settings.STR_DIM
for categ in categories:
  _ps = multiprocessing.Process(target=write_to_file, args=(categ, ))
  jobs.append(_ps)
  _ps.start()

for j in jobs:
  j.join()
  print '%s.exitcode = %s' % (j.name, j.exitcode)


