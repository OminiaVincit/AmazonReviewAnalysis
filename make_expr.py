import pickle
import os
import numpy as np
from settings import Settings

src_dir = Settings.PROCESSED_DIR
dst_dir = Settings.PROCESSED_DIR
categories = Settings.CATEGORIES

num_test = 50
train_rate = 0.8

for categ in categories[:2]:
  # Load features file
  features = np.load(os.path.join(src_dir, categ + '_STR_features.npy'))
  N = features.shape[0]
  N_train = int(N * train_rate)
  part = {}
  for i in range(num_test):
    part[i] = {}
    perm = np.random.permutation(N)
    part[i]['train'] = perm[:N_train]
    part[i]['test'] = perm[N_train:]

  # Dumpt to pickle file
  exp_file = '%s_partition.pickle' % categ
  with open(os.path.join(dst_dir, exp_file), 'wb') as handle:
    pickle.dump(part, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print categ, 'Dumped partition file'
