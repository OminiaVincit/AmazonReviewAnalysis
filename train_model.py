#!env python
# -*- coding:utf-8 -*-
'''Feeds the reviews corpus to the gensim LDA model
'''

import logging
import gensim
import json
from gensim.corpora import BleiCorpus
from gensim.models import LdaModel
from gensim import corpora
import os.path
from settings import Settings
from pymongo import MongoClient

import multiprocessing

src_dir = Settings.PROCESSED_DIR
dst_dir = Settings.PROCESSED_DIR
categories = Settings.CATEGORIES

from settings import Settings

class GenCollection(object):
    u'''Holds a general collection'''
    
    def __init__(self, collection_name, connection_dir=Settings.MONGO_CONNECTION_STRING, \
                 database_name=Settings.DATABASE):
        '''Init Reviews collection'''
        self.collection = MongoClient(connection_dir)[database_name][collection_name]
        self.cursor = None
        self.count = 0
    
    def load_all_data(self):
        '''Load cursor'''
        self.cursor = self.collection.find()
        self.count = self.cursor.count()

class Corpus(object):
    u'''Corpus class'''
    def __init__(self, cursor, corpus_dictionary, corpus_path):
        u'''Initialize corpus'''
        self.cursor = cursor
        self.corpus_dictionary = corpus_dictionary
        self.corpus_path = corpus_path

    def __iter__(self):
        u'''Corpus iterator'''
        self.cursor.rewind()
        for corpus in self.cursor:
            yield self.corpus_dictionary.doc2bow(corpus['words'])

    def serialize(self):
        u'''Serialize corpus'''
        BleiCorpus.serialize(self.corpus_path, self, \
            id2word=self.corpus_dictionary)
        return self


class Dictionary(object):
    u'''Dictionary class'''
    def __init__(self, cursor, dictionary_path):
        u'''Initialize Dictionary class'''
        self.cursor = cursor
        self.dictionary_path = dictionary_path

    def build(self):
        u'''Build dictionary'''
        self.cursor.rewind()
        dictionary = corpora.Dictionary(review['words'] \
            for review in self.cursor)
        dictionary.filter_extremes(keep_n=10000)
        dictionary.compactify()
        corpora.Dictionary.save(dictionary, self.dictionary_path)

        return dictionary


class Train:
    u'''Training class'''
    def __init__(self):
        pass

    @staticmethod
    def run(lda_model_path, corpus_path, num_topics, id2word):
        u'''Training to create LDA model'''
        corpus = corpora.BleiCorpus(corpus_path)
        lda = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word=id2word, iterations=200)
        lda.save(lda_model_path)
        return lda

def make_model(categ, lda_num_topics):
    u'''Main function'''
    logging.basicConfig(format='%(asctime)s: %(levelname)s :%(message)s', level=logging.INFO)
    dictionary_path = os.path.join(dst_dir, 'models/dictionary_' + categ + '.dict')
    corpus_path = os.path.join(dst_dir, 'models/corpus_' + categ + '.lda-c')

    lda_model_path = os.path.join(dst_dir, 'models/lda_model_' + str(lda_num_topics) +'_topics_' + categ + '.lda')

    collection_name = '%s_corpus' % categ
    corpus_collection = GenCollection(collection_name=collection_name)
    corpus_collection.load_all_data()
    corpus_cursor = corpus_collection.cursor

    dictionary = Dictionary(corpus_cursor, dictionary_path).build()
    Corpus(corpus_cursor, dictionary, corpus_path).serialize()
    Train.run(lda_model_path, corpus_path, lda_num_topics, dictionary)

def test():
    pass

def display(categ, lda_num_topics):
    u'''Display hidden topics'''
    lda_model_path = os.path.join(dst_dir, 'models/lda_model_' + str(lda_num_topics) +'_topics_' + categ + '.lda')
    lda = LdaModel.load(lda_model_path)
    top_list = lda.show_topics(num_topics=lda_num_topics, num_words=20, log=False, formatted=True)
    index = 0
    for top in top_list:
        index += 1
        print index,
        #scores = []
        #words = []
        topwords = top.split(' + ')
        for topword in topwords:
            member = topword.split('*')
            print member[1],
            #words.append(member[1])
            #scores.append(member[0])
        print ''
        

if __name__ == '__main__':
    #make_model(categories[0], lda_num_topics=64)
    #display(categories[0], lda_num_topics=64)

    # jobs = []
    # dim = Settings.TOPICS_DIM
    # for categ in categories[1:]:
    #   _ps = multiprocessing.Process(target=make_model, args=(categ, dim))
    #   jobs.append(_ps)
    #   _ps.start()

    # for j in jobs:
    #   j.join()
    #   print '%s.exitcode = %s' % (j.name, j.exitcode)

    for categ in categories:
        print categ
        #make_model(categ, lda_num_topics=Settings.TOPICS_DIM)
        display(categ, lda_num_topics=Settings.TOPICS_DIM)