import os
from glob import glob
import shutil
import tensorflow as tf
import numpy as np

np.random.seed(123)
tf.random.set_seed(123)

base_data = './data_base_clean/data'
new_data = './dcai_gcb_00/dcai_gcb_00'
if os.path.exists(new_data):
    shutil.rmtree(new_data)
    shutil.copytree(base_data, new_data)
    shutil.rmtree(new_data + '/ignore')

n_val = 100

numerals = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"]

val_dir = new_data + '/val'
if os.path.exists(val_dir):
    shutil.rmtree(val_dir)

for i, num in enumerate(numerals):
    num_dir = new_data + '/train/{}/'.format(num)
    os.makedirs(num_dir.replace('/train/', '/val/'))

    num_pattern = new_data + '/train/{}/*.png'.format(num)
    fps = glob(num_pattern)
    choices = np.random.choice(np.arange(len(fps)), size=n_val, replace=False)
    for choice in choices:
        source = fps[choice]
        dest = source.replace('/train/', '/val/')
        shutil.move(source, dest)
