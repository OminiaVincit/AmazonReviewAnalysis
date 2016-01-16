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

EXCEPT_CHAR = [u'!', u'"', u'#', u'$', u'%', u'&', u'\'', u'(', u')', u'-', u'=', u'^', u'~', 
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

def token_parse(categ, path):
  done = 0
  start = time.time()
    # Load stopwords and tokenizer
  stopwds = stopwords.words('english')
  tokenizer = regexp.RegexpTokenizer("[\w']+", flags=re.UNICODE)

  with open(path, 'r') as g:
    for l in g:
      u = json.loads(json.dumps(eval(l)))
      if not (u.get('reviewerID') and u.get('asin') and u.get('reviewerName') \
              and u.get('helpful') and u.get('reviewText')):
        continue
      if u['helpful'][1] < 10:
        continue
      sentences = sent_tokenize(u['reviewText'])
      num_sent = len(sentences)
      num_tokens = 0
      num_pos = 0
      num_neg = 0
      sent_len = 0
      words = []
      for sentence in sentences:
        sent_len += len(sentence)
        tokens = tokenizer.tokenize(sentence)
        num_tokens += len(tokens)

        pos_tagged = pos_tag(tokens)
        simplified_tags = [(word, map_tag('en-ptb', 'universal', tag)) for word, tag in pos_tagged]
        for word, tag in simplified_tags:
          words.append({'word': word, 'pos': tag})
          tf = tag[0].lower()
          if tag == 'ADV':
            tf = 'r'
          if tag == 'NP' or tag == 'NUM':
            tf = tag # No need to calculate positive score

          if tf in ['a', 'v', 'r', 'n']:
              try:
                  sen_ls = swn.senti_synsets(word, tf)
                  if len(sen_ls) != 0:
                      sen_score = sen_ls[0]
                      pos_score = sen_score.pos_score()
                      neg_score = sen_score.neg_score()
                      # obj_score = sen_score.obj_score()
                      if pos_score > neg_score:
                          num_pos += 1
                      if pos_score < neg_score:
                          num_neg += 1
              except WordNetError:
                  pass

      if num_sent != 0:
        sent_len = sent_len / num_sent

      tag = {}
      tag['num_sent'] = num_sent
      tag['sent_len'] = sent_len      
      tag['num_tokens'] = num_tokens
      tag['num_pos'] = num_pos
      tag['num_neg'] = num_neg
      tag['words'] = words
      tag['review_id'] = u['reviewerID']
      tag['user_id'] = u['reviewerName']
      tag['item_id'] = u['asin']
      tag['votes'] = int(u['helpful'][1])
      tag['helpful'] = int(u['helpful'][0])
      done += 1
      if done % 1000 == 0:
        tmp = time.time() - start
        print categ, 'Tagging reviews, Done ', done, ' in', tmp
      yield str(tag)

def write_to_file(categ):
  """
  Write to json file
  """
  filename = 'reviews_%s' % categ
  print filename
  f = open(os.path.join(dst_dir, '%s_tags.json' % filename), 'w')
  for l in token_parse(categ, os.path.join(src_dir, '%s.json' % filename)):
    f.write(l + '\n')

#write_to_file(categories[1])

jobs = []
for categ in categories:
  _ps = multiprocessing.Process(target=write_to_file, args=(categ,))
  jobs.append(_ps)
  _ps.start()

for j in jobs:
  j.join()
  print '%s.exitcode = %s' % (j.name, j.exitcode)


