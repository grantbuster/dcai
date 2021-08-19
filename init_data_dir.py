import os
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
