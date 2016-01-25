import numpy as np
from settings import Settings
import os

FEATURES = ['STR', 'tfidf', 'LIWC', 'INQUIRER', 'GALC', 'TOPICS_64']

# src_dir = Settings.PROCESSED_DIR
# dst_dir = Settings.PROCESSED_DIR

src_dir = r'/home/zoro/work/Dataset/'
dst_dir = r'/home/zoro/work/Dataset/'

# CATEGORIES = Settings.CATEGORIES
CATEGORIES = ['yelp', 'tripadvisor']

for categ in CATEGORIES:
    data = []
    for ftype in FEATURES[2:]:
        print categ, ftype
        arr = np.load(os.path.join(src_dir, '%s_%s_features.npy' % (categ, ftype)))
        data.append(arr[:, 0:(-3)].T)
        print arr.shape
    data.append(arr[:, (-3):].T)
    data = np.vstack(data)
    data = data.T
    np.save(os.path.join(dst_dir, '%s_JointSemantic_features' % categ), data)
    print categ, data.shape

