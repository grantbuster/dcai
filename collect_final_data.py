import numpy as np
import os
import shutil
from glob import glob

data_dir = './data/data/'
numerals = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"]

def clean_num_dir(num_dir):
    for f in os.listdir(num_dir):
        fp = os.path.join(num_dir, f)
        if os.path.isdir(fp):
            shutil.rmtree(fp)


for n in numerals:
    val_dir = data_dir + '/val/{}/'.format(n)
    train_dir = data_dir + '/train/{}/'.format(n)
    vae_dir = data_dir + '/vae_out/{}/'.format(n)

    clean_num_dir(val_dir)
    clean_num_dir(train_dir)
    clean_num_dir(vae_dir)

    train_vae_files = glob(train_dir + '{}_*.png'.format(n))
    for fp in train_vae_files:
        os.remove(fp)

    train_vae_files = glob(train_dir + '{}_*.png'.format(n))
    assert not any(train_vae_files)
    train_files = glob(train_dir + '*.png')
    val_files = glob(val_dir + '*.png')
    vae_files = glob(vae_dir + '*.png')
    assert len(vae_files) == 5000

    n_tot_files = len(train_files) + len(val_files)
    n_vae_files = 995 - n_tot_files

    choices = np.random.choice(np.arange(n_vae_files), n_vae_files,
                               replace=False)

    for i in choices:
        fp_source = vae_files[i]
        fp_dest = fp_source.replace('vae_out', 'train')
        shutil.copy(fp_source, fp_dest)

    train_files = glob(train_dir + '*.png')
    val_files = glob(val_dir + '*.png')
    n_tot_files = len(train_files) + len(val_files)
    assert n_tot_files <= 1000
    print(n, n_tot_files)
