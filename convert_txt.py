import pickle
import os
import numpy as np
from settings import Settings

src_dir = Settings.PROCESSED_DIR
dst_dir = Settings.PROCESSED_DIR
categories = Settings.CATEGORIES
features = Settings.FEATURES

for categ in categories:
  for ftype in features:
    print categ, ftype
    # Load features file
    a = np.load(os.path.join(src_dir, '%s_%s_features.npy' % (categ, ftype) ))
    np.savetxt(os.path.join(src_dir, '%s_%s_features.txt' % (categ, ftype) ), a)

