"""
Backfill the data dir with supplemental vae images
"""
import os
import shutil
from glob import glob
import pandas as pd
import numpy as np

np.random.seed(123)

N = 1000
data_dir = './dcai_gcb_00/dcai_gcb_00'
train_dir = data_dir + '/train'
val_dir = data_dir + '/val'
numerals = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"]

for n in numerals:
    vae_df = pd.read_csv('./vae_labels_{}.csv'.format(n), index_col=0)
    vae_df = vae_df[vae_df.good.astype(bool)]

    n_train_dir = train_dir + '/{}'.format(n)
    n_val_dir = val_dir + '/{}'.format(n)

    n_t_files = len(glob(n_train_dir + '/*.png'))
    n_v_files = len(glob(n_val_dir + '/*.png'))
    n_tot_files = n_t_files + n_v_files
    n_vae_files = np.minimum(N - n_tot_files, len(vae_df))

    vae_df = vae_df.sample(n=n_vae_files, replace=False)
    assert len(vae_df) == n_vae_files
    assert (vae_df['label'] == 'good').all()
    for fp in vae_df['fp']:
        dest = os.path.join(n_train_dir, os.path.basename(fp))
        shutil.copy(fp, dest)

    n_t_files = len(glob(n_train_dir + '/*.png'))
    n_v_files = len(glob(n_val_dir + '/*.png'))
    n_tot_files = n_t_files + n_v_files
    print('{} has {} files'.format(n, n_tot_files))


