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


def run_vae(numeral, n_images, sharpness, eps_scale):
    with h5py.File('./image_data_{}.h5'.format(numeral), 'r') as f:
        images = f['images'][...].astype(np.float32) / 255
        images = np.expand_dims(images, 3)

    vae = Vae.load(images, './vae_{}.zip'.format(numeral))

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
    n_images = 5000
    sharpness = 10
    eps_scale = 0.75
    numerals = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"]
    for numeral in numerals:
        print('Training VAE for "{}"'.format(numeral))
        run_vae(numeral, n_images, sharpness, eps_scale)
