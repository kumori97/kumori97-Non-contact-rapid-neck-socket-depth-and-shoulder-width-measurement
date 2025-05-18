import os
from glob import glob
import pickle
import random
from os.path import join, split, exists

npz_files = glob('./assets/CAPE_Dataset_Sampled_10000/**/*_naked.obj', recursive=True)

npz_files = [split(file)[0] for file in npz_files]


import random
# random.shuffle(npz_files)

train_ratio = 0.7
test_ratio = 0.3
val_ratio = 0.0

train_index = int(len(npz_files) * train_ratio)
test_index = train_index + int(len(npz_files) * test_ratio)
train_files = npz_files[:train_index]
test_files = npz_files[train_index:test_index]
val_files = npz_files[test_index:]

#
# with open('train_files.txt', 'w') as f:
#     for file in train_files:
#         f.write(file + '\n')
# with open('test_files.txt', 'w') as f:
#     for file in test_files:
#         f.write(file + '\n')
# with open('val_files.txt', 'w') as f:
#     for file in val_files:
#         f.write(file + '\n')

# Save the file paths to a pickle file
with open('./assets/CAPE_Dataset_Sampled_10000.pkl', 'wb') as f:
    pickle.dump({'train': train_files, 'test': test_files, 'val': val_files,'all':npz_files}, f)








