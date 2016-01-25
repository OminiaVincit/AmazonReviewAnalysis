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

from settings import Settings

src_dir = Settings.PROCESSED_DIR
dst_dir = Settings.PROCESSED_DIR
categories = Settings.CATEGORIES
features = Settings.FEATURES

a, b = None, None
for categ in categories:
  for ft1 in features:
    for ft2 in features:
      if ft1 == ft2:
        continue
      a = np.load(os.path.join(src_dir, '%s_%s_features.npy' % (categ, ft1)))
      b = np.load(os.path.join(src_dir, '%s_%s_features.npy' % (categ, ft2)))
      tmp = np.sum(a[:,(-3):] - b[:,(-3):])
      if tmp != 0:
        print categ, ft1, ft2, tmp