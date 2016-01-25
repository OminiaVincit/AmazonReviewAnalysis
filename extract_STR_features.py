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

def extract_STR(categ, dim):
  """
  Extract STR features
  """
  filename = 'reviews_%s' % categ
  done = 0
  start = time.time()
  data = []
  with open(os.path.join(src_dir, '%s_tags.json' % filename), 'r') as g:
    for l in g:      
      u = json.loads(json.dumps(eval(l)))
      helpful = u['helpful']
      votes = u['votes']
      features = np.zeros((dim+3), )
      features[dim  ] = int(helpful)
      features[dim+1] = int(votes)
      features[dim+2] = helpful / float(votes)

      pos_total = u['num_tokens']
      if pos_total == 0:
        data.append(features)
        continue

      freq = {}
      pos = {}
      for tag in TAGS:
        pos[tag] = 0

      for twrd in u['words']:
        tag = twrd['pos']
        word = twrd['word']
        if word not in freq:
          freq[word] = 1
        else:
          freq[word] += 1
        if tag not in pos:
          pos[tag] = 0
        else:
          pos[tag] += 1

      num_uniqs = 0
      for word in freq:
        if freq[word] == 1:
          num_uniqs += 1

      features[0]  = u['sent_len']
      features[1]  = u['num_sent']
      features[2]  = u['num_tokens']
      features[3]  = pos['NOUN']   / float(pos_total)
      features[4]  = pos['ADJ'] / float(pos_total)
      features[5]  = pos['ADV'] / float(pos_total)
      features[6]  = (pos['VERB'] + pos['VD'] + pos['VG'] + pos['VN'] + pos['MOD']) / float(pos_total)
      features[7]  = pos['NUM'] / float(pos_total)
      features[8]  = pos['FW'] / float(pos_total)
      features[9]  = u['num_pos'] / float(pos_total)
      features[10] = u['num_neg'] / float(pos_total)
      features[11] = num_uniqs / float(pos_total)
      
      data.append(features)
      #print done, features
      done += 1
      if done % 1000 == 0:
        tmp = time.time() - start
        print categ, 'STR reviews, Done ', done, ' in', tmp
        
  # Write to file
  print 'Number of processed reviews ', done
  data = np.vstack(data)
  print 'Data shape', data.shape
  np.save('%s/%s_STR_features' % (dst_dir, categ), data)

#extract_STR(categories[0], dim=Settings.STR_DIM)

jobs = []
dim = Settings.STR_DIM
for categ in categories[:2]:
  _ps = multiprocessing.Process(target=extract_STR, args=(categ, dim))
  jobs.append(_ps)
  _ps.start()

for j in jobs:
  j.join()
  print '%s.exitcode = %s' % (j.name, j.exitcode)


