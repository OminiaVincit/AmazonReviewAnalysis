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

from settings import Settings

src_dir = Settings.SOURCE_DIR
dst_dir = Settings.PROCESSED_DIR
categories = Settings.CATEGORIES


def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield json.dumps(eval(l))

def subset_parse(categ, path):
  g = gzip.open(path, 'r')
  done = 0
  start = time.time()
  data = {}
  
  # Get list of top products
  for l in g:
    try:
      u = json.loads(json.dumps(eval(l)))
    except:
      continue
    if not (u.get('reviewerID') and u.get('asin') and u.get('reviewerName') \
            and u.get('helpful') and u.get('reviewText')):
      continue
    item_id = u['asin']
    if item_id not in data:
      data[item_id] = 1
    else:
      data[item_id] += 1
    done += 1
    if done % 50000 == 0:
      tmp = time.time() - start
      print categ, 'Create product list Done ', done, ' in', tmp
      # if done == 1550000:
      #   break
      
  print categ, 'Number of reviews: ', done
  print categ, 'Number of products: ', len(data)

  # List of top 500 products with most reviews
  dlist = sorted(data.items(), key=lambda x: -x[1])
  plist = [ d[0] for d in dlist[:500] ]

  # Get the reviews in top products
  done = 0
  start = time.time()
  g = gzip.open(path, 'r')
  for l in g:
    try:
      u = json.loads(json.dumps(eval(l)))
    except:
      continue
    if not (u.get('reviewerID') and u.get('asin') and u.get('reviewerName') \
            and u.get('helpful') and u.get('reviewText')):
      continue
    item_id = u['asin']
    if item_id not in plist:
      continue
    done += 1
    if done % 50000 == 0:
      tmp = time.time() - start
      print categ, 'Create subset reviews, Done ', done, ' in', tmp
      # if done == 250000:
      #   break
    yield str(u)
        
def write_to_file(categ):
  """
  Write to json file
  """
  filename = 'reviews_%s' % categ
  print filename
  f = open(os.path.join(dst_dir, '%s.json' % filename), 'w')
  for l in subset_parse(categ, os.path.join(src_dir, '%s.json.gz' % filename)):
    f.write(l + '\n')

write_to_file(categories[1])
# jobs = []
# for categ in categories:
#   _ps = multiprocessing.Process(target=write_to_file, args=(categ,))
#   jobs.append(_ps)
#   _ps.start()

# for j in jobs:
#   j.join()
#   print '%s.exitcode = %s' % (j.name, j.exitcode)



