#!env python
# -*- coding:utf-8 -*-
'''Extract topics features
'''

import os
import time
import sys
import json
import numpy as np
import argparse
import logging

from gensim.models import LdaModel
from gensim import corpora
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import regexp
import re

from settings import Settings
from train_model import GenCollection

src_dir = Settings.PROCESSED_DIR
dst_dir = Settings.PROCESSED_DIR
categories = Settings.CATEGORIES


EXCEPT_CHAR = [u'0', u'1', u'2', u'3', u'4', u'5', u'6', u'7', u'8', u'9',
u'!', u'"', u'#', u'$', u'%', u'&', u'\'', u'(', u')', u'-', u'=', u'^', u'~', 
u'\\', u'|', u'@', u'`', u'[', u'{', u';', u'+', u':', u'*', u']', u'}', 
u',', u'<', u'>', u'.', u'/', u'?', u'_']

EPSILON = 1e-5

def corpus_condition(word):
    """
    Condition of word to add to corpus
    """
    save_flag = False
    if len(word) > 2:
        save_flag = True
        for ch in word:
            if ch in EXCEPT_CHAR:
                save_flag = False
                break
    return save_flag

class Predict(object):
    u'''Predict class'''

    def __init__(self, categ, lda_num_topics):
        """
        Initialize Predict class
        """
        collection_name = '%s_corpus' % categ
        dictionary_path = os.path.join(src_dir, 'models/dictionary_' + categ + '.dict')
        lda_model_path = os.path.join(dst_dir, 'models/lda_model_' + str(lda_num_topics) +'_topics_' + categ + '.lda')

        self.categ = categ
        self.collection_name = collection_name
        self.lda_num_topics = lda_num_topics
        self.dictionary = corpora.Dictionary.load(dictionary_path)
        self.lda = LdaModel.load(lda_model_path)
        self.stopwords = stopwords.words('english')
        self.lem = WordNetLemmatizer()
        self.tokenizer = regexp.RegexpTokenizer("[\w’]+", flags=re.UNICODE)

    def extract_lemmatized_nouns(self, new_review):
        u'''Extract lemmatize nouns'''
        
        nouns = []
        tokens = self.tokenizer.tokenize(new_review.lower())
        tokens = [token for token in tokens if corpus_condition(token)]
        tokens = [token for token in tokens if token not in self.stopwords]

        tagged_text = nltk.pos_tag(tokens)
        for word, tag in tagged_text:
          if tag in ['NN', 'NNS', 'JJ', 'JJR', 'JJS', 'RBR', 'RBS']:
            nouns.append(self.lem.lemmatize(word))
        
        return nouns

    def run_single_sentence(self, new_review):
        u'''Run model for predict topic'''
        nouns = self.extract_lemmatized_nouns(new_review)
        new_review_bow = self.dictionary.doc2bow(nouns)
        new_review_lda = self.lda[new_review_bow]

        return new_review_lda

    def run_single_word(self, new_word):
        u'''Run model for predict topic'''
        new_review_bow = self.dictionary.doc2bow([new_word])
        new_review_lda = self.lda[new_review_bow]

        return new_review_lda

    def extract_vocal_topics_features(self, data_dir=src_dir):
        """
        Extract topics features historgram from reviews (using corpus)
        """
        # Debug time
        import pickle

        start = time.time()

        # For categ name
        categ = self.categ
        lda_num_topics = self.lda_num_topics
        rvs = GenCollection(collection_name=self.collection_name)

        # Create vocabulary correspond to model if not exist
        vocal = {}
        vocal_file = '%s_%d_topics_vocabularies.pickle' % (categ, lda_num_topics)
        if not os.path.isfile(os.path.join(data_dir, vocal_file)):
            # Create vocabulary
            start = time.time()
            rvs.cursor = rvs.collection.find()
            N = rvs.cursor.count()
            done = 0
            for review in rvs.cursor:
                for word in review['words']:
                    if word not in vocal:
                        # Predict topic score
                        topics = self.run_single_word(word)
                        top_arr = process_topics(topics, self.lda_num_topics)
                        vocal[word] = top_arr
              
                done += 1
                if done % 1000 == 0:
                    end = time.time()
                    print categ + ' get vocabulary: Done ' + str(done) + \
                        ' out of ' + str(N) + ' reviews in ' + \
                        ('%.2f' % (end - start)) + ' sec ~ ' + \
                        ('%.2f' % (done / (end - start))) + '/sec'
                    sys.stdout.flush()

            # Write to pickle file
            with open(os.path.join(data_dir, vocal_file), 'wb') as handle:
                pickle.dump(vocal, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print 'Dumped vocal file'
        else:
            # Load vocabulary from pickle
            with open(os.path.join(data_dir, vocal_file), 'rb') as handle:
                vocal = pickle.load(handle)
                print 'Loaded vocal file'

        print 'Size of vocabulary set', len(vocal)
        # Sum up to form features fector for each review
        data = []
        done = 0
        dim = self.lda_num_topics
        rvs.cursor = rvs.collection.find()
        N = rvs.cursor.count()

        for review in rvs.cursor:
            rvid = review['review_id']
            votes = review['votes']
            if votes < 10:
                continue
            helpful = review['helpful']
            features = np.zeros((dim+3), )
            features[dim] = helpful
            features[dim+1] = votes
            features[dim+2] = helpful / float(votes)

            for word in review['words']:
                if word in vocal:
                    features[:dim] += vocal[word]

            data.append(features)
            #print done, features
            done += 1
            if done % 1000 == 0:
                end = time.time()
                print categ + ' TOPICS features: Done ' + str(done) + \
                    ' out of ' + str(N) + ' reviews in ' + \
                    ('%.2f' % (end - start)) + ' sec ~ ' + \
                    ('%.2f' % (done / (end - start))) + '/sec'
                sys.stdout.flush()
                #break

        print 'Number of processed reviews ', done
        data = np.vstack(data)
        print 'Data shape', data.shape
        np.save('%s/%s_TOPICS_%d_features' % (data_dir, categ, lda_num_topics), data)

def process_topics(topics, dim=Settings.TOPICS_DIM):
    """
    Process of topics vector, remove irrelevant members
    """
    top_arr = np.zeros((dim, ))
    for topic in topics:
        top_arr[topic[0]] = topic[1]
    top_arr *= np.var(top_arr)
    top_arr[top_arr < EPSILON] = 0
    total = np.sum(top_arr)
    if total > 0:
        top_arr /= total
    return top_arr

def _test():
    u'''Main function'''
    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    new_review = "It's like eating with a big Italian family. " \
        "Great, authentic Italian food, good advice when asked, and terrific service. " \
        "With a party of 9, last minute on a Saturday night, we were sat within 15 minutes. " \
        "The owner chatted with our kids, and made us feel at home. " \
        "They have meat-filled raviolis, which I can never find. " \
        "The Fettuccine Alfredo was delicious. We had just about every dessert on the menu. " \
        "The tiramisu had only a hint of coffee, the cannoli was not overly sweet, " \
        "and they had this custard with wine that was so strangely good. " \
        "It was an overall great experience!"

    new_review2 = 'I arrived around 10 am on a Saturday morning.  '\
        'I was greeted by the hostess,who was polite and friendly, '\
        'and I informed her that I was alone.  She had just arrived, '\
        'as well, and wasn\'t exactly sure what was going on yet, '\
        'so she asked (the manager?) where she should seat me.  '\
        'After receiving guidance, she brought me to a table towards '\
        'the back.  It had not yet been cleaned after the previous'\
        ' guests had dined.  So, she quickly got it cleared off, '\
        'grabbed a rag, and wiped it down.\n\nThe waitress came '\
        'rather quickly to take my drink order.  She was extremely '\
        'friendly.  I ordered coffee and water, which she went and '\
        'got while I looked over the menu.  She returned, and took '\
        'my order.  I ordered the Cinnamon Swirl French Toast Special'\
        ' - Three slices of French toast made with Pittsburgh\'s '\
        'own Jenny Lee® Cinnamon Swirl, two farm-fresh eggs, with '\
        'bacon or sausage (I chose sausage), and your choice of potato '\
        '(I went with the cheesy hash brown casserole).  I also added '\
        'an orange juice.  She went and put my order in, while I waited, '\
        'and came back with it after not too long.   The eggs were cooked '\
        'exactly how I wanted them, the cheesy hash brown casserole '\
        'and the french toast were both delicious.  I also enjoyed the '\
        'sausage which was pretty typical.\n\nKings Family Restaurant '\
        'featured a very friendly staff, great prices, and tasty food.  '\
        'I am pleased and will definitely come back again.'

    new_review3 = 'Sashimi here is wonderful and melt in your mouth.' \
        'I often get the sashimi tasting here.'\
        'I do not think the rolls here are that great.'\
        'I think the udon noodle sets and other items on the menu are very good but it changes from time to time.'\
        'If you have a group over 4 then this may not be a place for you. Agashi tofu was great here.'

    new_review4 = '####'
    predict = Predict(categ=categories[0], lda_num_topics=64)
    topics = predict.run_single_sentence(str(new_review).decode('utf-8'))
    print topics
    topics = predict.run_single_sentence(str(new_review2).decode('utf-8'))
    print topics
    topics = predict.run_single_sentence(str(new_review3).decode('utf-8'))
    print topics
    topics = predict.run_single_sentence(str(new_review4).decode('utf-8'))
    print topics
    top_arr = process_topics(topics)
    print 'Top arr: ', top_arr

if __name__ == '__main__':
    #_test()
    # predict = Predict(categ=categories[0], lda_num_topics=64)
    # predict.extract_vocal_topics_features(src_dir)

    for categ in categories:
        print categ
        predict = Predict(categ, lda_num_topics=Settings.TOPICS_DIM)
        predict.extract_vocal_topics_features(src_dir)
    # predict = Predict(collection_name=Settings.TRIPADVISOR_CORPUS_COLLECTION, lda_num_topics=64)
    # predict.extract_vocal_topics_features()      
