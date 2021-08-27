import os
import shutil
import pandas as pd
import numpy as np
from numpy import asarray
from glob import glob
import cv2
import PIL
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import zipfile

from train import train


NUMERALS = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"]


def init_data_dir(new_data_dir, base_data_dir='./data_baseline_clean/data_baseline_clean/'):
    if os.path.exists(new_data_dir):
        shutil.rmtree(new_data_dir)
    shutil.copytree(base_data_dir, new_data_dir)
    if os.path.exists(new_data_dir + '/ignore'):
        shutil.rmtree(new_data_dir + '/ignore')


def get_file_count(numeral, data_dir):
    num_pattern = data_dir + '/*/{}/*.png'.format(numeral)
    fps = glob(num_pattern)
    return len(fps)


def get_n_required(numeral, data_dir, target_counts):
    n_files = get_file_count(numeral, data_dir)
    n_required = target_counts[numeral] - n_files
    return n_required


def count_all_files(data_dir):
    num_count = {}
    n_tot = 0
    for n in NUMERALS:
        n_files = get_file_count(n, data_dir)
        num_count[n] = n_files
        n_tot += n_files
        print('Numeral "{}" has {} files'.format(n, n_files))
    print('Total of {} files'.format(n_tot))
    return num_count, n_tot


def process_image(fp_source, fp_dest,
                  horiz=False,
                  vert=False,
                  rotate=False,
                  erode=False,
                  dilate=False,
                  enhance=1,
                  contrast=1,
                  resize=(256, 256),
                  show=False):

    image = Image.open(fp_source)
    image_out = Image.open(fp_source)

    if len(asarray(image_out).shape) > 2:
        image_out = Image.fromarray(asarray(image_out)[:, :, 0])

    if rotate == 'random':
        rotate = np.random.normal(0, 12)
    if enhance == 'random':
        enhance = np.random.normal(1, 0.5)
    if contrast == 'random':
        contrast = np.maximum(0.3, np.random.normal(1, 0.5))
    if erode == 'random':
        erode = np.random.choice([0, 1, 2], 1)
    if dilate == 'random':
        dilate = np.random.choice([0, 1], 1)

    if horiz:
        arr = asarray(image_out)
        arr = arr[:, ::-1]  # horizontal sym
        image_out = Image.fromarray(arr)
    if vert:
        arr = asarray(image_out)
        arr = arr[::-1, :]  # vert sym
        image_out = Image.fromarray(arr)
    if rotate:
        image_out = image_out.rotate(rotate, PIL.Image.NEAREST,
                                     expand=False,
                                     fillcolor='white')
    if erode:
        kernel = np.ones((5, 5), np.uint8)
        arr_out = cv2.erode(asarray(image_out), kernel, iterations=int(erode))
        image_out = Image.fromarray(arr_out)
    if dilate:
        kernel = np.ones((5, 5), np.uint8)
        arr_out = cv2.dilate(asarray(image_out), kernel, iterations=1)
        image_out = Image.fromarray(arr_out)

    image_out = ImageEnhance.Sharpness(image_out)
    image_out = image_out.enhance(enhance)
    image_out = ImageEnhance.Contrast(image_out)
    image_out = image_out.enhance(contrast)

    image_out = image_out.resize(resize)
    image_out = asarray(image_out)
    image_out = np.round(image_out).astype(np.uint8)
    image_out = Image.fromarray(image_out)

    if show:
        fig = plt.figure(figsize=(15, 5))
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        ax1.imshow(image, cmap='binary_r')
        ax2.imshow(image.resize(resize), cmap='binary_r')
        ax3.imshow(image_out, cmap='binary_r')
        plt.show()
        plt.close()

    if fp_dest is not None:
        image_out.save(fp_dest)


def engineer_images_by_num(eng_kwargs, data_dir, target_counts, ifile=0):
    for num, kwargs in eng_kwargs.items():
        n_required = get_n_required(num, data_dir, target_counts)
        num_dir = data_dir + '/train/{}/*.png'.format(num)

        if n_required < 0:
            # remove images
            fps = [fp for fp in glob(num_dir)]
            ifps = np.random.choice(np.arange(len(fps)), -n_required,
                                    replace=False)
            for i in ifps:
                fp = fps[i]
                os.remove(fp)
        else:
            # engineer new images
            fps = [fp for fp in glob(num_dir) if '_eng_' not in fp]
            ifps = np.random.choice(np.arange(len(fps)), n_required,
                                    replace=True)
            for i in ifps:
                fp = fps[i]

                if 'target' in kwargs:
                    target = kwargs.pop('target')
                    base_dir = os.path.dirname(fp).replace(num, target)
                    fn = '{}_eng_{}.png'.format(target, str(ifile).zfill(5))
                    fp_out = os.path.join(base_dir, fn)
                else:
                    base_dir = os.path.dirname(fp)
                    fn = '{}_eng_{}.png'.format(num, str(ifile).zfill(5))
                    fp_out = os.path.join(base_dir, fn)

                process_image(fp, fp_out,
                    enhance='random',
                    contrast=1,
                    rotate='random',
                    erode='random',
                    dilate='random',
                    show=False,
                    **kwargs)
                ifile += 1


def engineer_bad_images(bad_images, n_eng_images, data_dir, ifile=0):
    print('Engineering bad images with target count of {}'
          .format(n_eng_images))
    previous_fps = glob(data_dir + '/train/*/*eng_bad_*.png')
    for fp in previous_fps:
        os.remove(fp)

    count_all_files(data_dir)

    ifps = np.random.choice(np.arange(len(bad_images)), n_eng_images,
                            replace=True)
    for i in ifps:
        fp = bad_images[i]
        base_dir = os.path.dirname(fp)
        fn = 'eng_bad_{}.png'.format(str(ifile).zfill(5))
        fp_out = os.path.join(base_dir, fn)
        fp_out = fp_out.replace('/val/', '/train/')

        process_image(fp, fp_out,
            enhance='random',
            contrast=1,
            rotate='random',
            erode='random',
            dilate='random',
            show=False,
            )
        ifile += 1

    count_all_files(data_dir)
    print('Engineered {} bad images.'.format(ifile))


def parse_bad_images(predictions_df):
    if predictions_df is None:
        return 0
    else:
        val_mask = predictions_df['dataset'] == 'val'
        bad_mask = predictions_df['truth'] != predictions_df['prediction']
        bad_images = predictions_df.loc[bad_mask & val_mask, 'fp'].values
        bad_images = ['./' + fp for fp in bad_images]
        return bad_images


def zipdir(source, destination):
    # ziph is zipfile handle
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as ziph:
        for root, _, files in os.walk(source):
            for file in files:
                ziph.write(os.path.join(root, file),
                           os.path.relpath(os.path.join(root, file),
                                           os.path.join(source, '..')))
