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

EXCEPT_CHAR = [u'0', u'1', u'2', u'3', u'4', u'5', u'6', u'7', u'8', u'9',
u'!', u'"', u'#', u'$', u'%', u'&', u'\'', u'(', u')', u'-', u'=', u'^', u'~', 
u'\\', u'|', u'@', u'`', u'[', u'{', u';', u'+', u':', u'*', u']', u'}', 
u',', u'<', u'>', u'.', u'/', u'?', u'_']

# Tag Meaning Examples
# ADJ adjective new, good, high, special, big, local
# ADV adverb  really, already, still, early, now
# CNJ conjunction and, or, but, if, while, although
# DET determiner  the, a, some, most, every, no
# EX  existential there, there's
# FW  foreign word  dolce, ersatz, esprit, quo, maitre
# MOD modal verb  will, can, would, may, must, should
# N noun  year, home, costs, time, education
# NP  proper noun Alison, Africa, April, Washington
# NUM number  twenty-four, fourth, 1991, 14:24
# PRO pronoun he, their, her, its, my, I, us
# P preposition on, of, at, with, by, into, under
# TO  the word to to
# UH  interjection  ah, bang, ha, whee, hmpf, oops
# V verb  is, has, get, do, make, see, run
# VD  past tense  said, took, told, made, asked
# VG  present participle  making, going, playing, working
# VN  past participle given, taken, begun, sung
# WH  wh determiner who, which, when, what, where, how

def corpus_condition(word):
    """
    Condition of word to add to corpus
    """
    save_flag = False
    if word['pos'] in ['NOUN', 'ADJ', 'ADV', 'VERB', 'VD', 'VG', 'VN']:
        if len(word['word']) > 2:
            save_flag = True
            for ch in word['word']:
                if ch in EXCEPT_CHAR:
                    save_flag = False
                    break
    return save_flag

def freq(word, tokens):
    return tokens.count(word)

def word_count(tokens):
    return len(tokens)

def tf(word, tokens):
    return (freq(word, tokens) / float(word_count(tokens)))

def make_tokens(categ):
  """
  Make tf-idf tokens
  """
  filename = 'reviews_%s' % categ
  done = 0
  start = time.time()
  lem = WordNetLemmatizer()
  stopwds = stopwords.words('english')

  # Compute the frequency for each term
  vocabulary = {} # number of docs where where word w appeared
  docs = {}

  with open(os.path.join(src_dir, '%s_tags.json' % filename), 'r') as g:
    for l in g:      
      review = json.loads(json.dumps(eval(l)))

      votes = int(review['votes'])
      helpful = int(review['helpful'])
      rvid = review['review_id']
      docs[rvid] = {'freq':{}, 'tf':{}}

      words = [word for word in review['words'] if corpus_condition(word)]
      tokens = [lem.lemmatize(word['word'].lower()) for word in words]
      tokens = [word for word in tokens if word not in stopwds]

      for token in set(tokens):
        # The frequency computed for each review
        docs[rvid]['freq'][token] = freq(token, tokens)
        # The true-frequency (normalized)
        docs[rvid]['tf'][token] = tf(token, tokens)
        if token not in vocabulary:
            vocabulary[token] = 1
        else:
            vocabulary[token] += 1

      #print done, features
      done += 1
      if done % 1000 == 0:
        tmp = time.time() - start
        print categ, 'Get vocal, Done ', done, ' in', tmp
        #break
        
  # Number of processed documents
  vocabulary['NUMDOCS'] = done
  with open(os.path.join(src_dir, '%s_tfidf_vocabularies.json' % filename), 'w') as _vfile:
    _vfile.write(json.dumps(vocabulary, indent=1).replace('\n', ''))
  print 'Dump vocal file'

  done = 0
  start = time.time()
  with open(os.path.join(src_dir, '%s_corpus.json' % filename), 'r') as cg:
    for l in cg:      
      review = json.loads(json.dumps(eval(l)))
      rvid = review['review_id']
      features = {}
      features['review_id'] = rvid
      features['item_id'] = review['item_id']
      features['user_id'] = review['user_id']
      features['votes'] = review['votes']
      features['helpful'] = review['helpful']
      features['freq']  = docs[rvid]['freq']
      features['tf']  = docs[rvid]['tf']
      features['idf'] = {}

      for token in docs[rvid]['tf']:
        # The inverse-document-frequency
        features['idf'][token] = np.log(vocabulary['NUMDOCS'] / float(vocabulary[token]) )

      done += 1
      if done % 1000 == 0:
        tmp = time.time() - start
        print categ, 'Get tfidf, Done ', done, ' in', tmp
        #break

      yield str(features)


def write_to_file(categ):
  """
  Write to json file
  """
  filename = 'reviews_%s' % categ
  print filename
  f = open(os.path.join(dst_dir, '%s_tokens.json' % filename), 'w')
  for l in make_tokens(categ):
    f.write(l + '\n')

#write_to_file(categories[0])

jobs = []
for categ in categories[:2]:
  _ps = multiprocessing.Process(target=write_to_file, args=(categ, ))
  jobs.append(_ps)
  _ps.start()

for j in jobs:
  j.join()
  print '%s.exitcode = %s' % (j.name, j.exitcode)


