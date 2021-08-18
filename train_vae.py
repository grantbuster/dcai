import os
import h5py
import pandas as pd
import numpy as np
from statistics import mode
from vae import Vae
import tensorflow as tf
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from PIL import Image, ImageEnhance


def train_vae(numeral, n_images, sharpness=10, eps_scale=1.0):
    with h5py.File('./image_data_{}.h5'.format(numeral), 'r') as f:
        images = f['images'][...].astype(np.float32) / 255
        images = np.expand_dims(images, 3)
        labels = f['labels'][...].astype(np.float32)

    images.shape, labels.shape

    vae = Vae(images, latent_dim=4, learning_rate=1e-3, standardize_data=False,
              kl_weight=1.5, validation_split=0.05)
    vae.train(n_epoch=100, n_batch=8)

    vae.save('./vae_{}.zip'.format(numeral))

    eps = tf.random.normal(shape=(n_images, vae.latent_dim)) * eps_scale
    out = vae.sample(eps=eps, standardize=False, numpy=True,
                     apply_sigmoid=True, to_image=True)

    out_dir = './data/data/vae_out/{}'.format(numeral)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for i in range(n_images):
        image = Image.fromarray(out[i])
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(sharpness)
        fn = "{}_{}.png".format(numeral, str(i).zfill(4))
        fp_out = os.path.join(out_dir, fn)
        image.save(fp_out)


if __name__ == '__main__':
    n_images = 1000
    numerals = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"]
    for numeral in numerals:
        print('Training VAE for "{}"'.format(numeral))
        train_vae(numeral, n_images)
