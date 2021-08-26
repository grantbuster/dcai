import os
import json
import pandas as pd

from train import train
from utilities import (init_data_dir, count_all_files, engineer_images_by_num,
                       parse_bad_images, engineer_bad_images, zipdir)


NUMERALS = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"]


if __name__ == '__main__':
    data_dir = './dcai_gcb_10/dcai_gcb_10'
    test_dir = './label_book/label_book'
    base_dir = './data_baseline_aug/data_baseline_aug'
    job_tag = os.path.basename(data_dir)
    target_count_tot = 9995
    epochs = 50
    n_iter = 50

    # initial target file count 8000 files (80% of max)
    target_counts = {'i': 240, 'ii': 560, 'iii': 560, 'iv': 1360,
                     'v': 640, 'vi': 800, 'vii': 1120, 'viii': 1200,
                     'ix': 1120, 'x': 400}

    # initial target file count - balanced w perceived challenge (7250 files)
    target_counts = {'i': 250, 'ii': 500, 'iii': 500, 'iv': 1000,
                     'v': 500, 'vi': 1000, 'vii': 1000, 'viii': 1000,
                     'ix': 1000, 'x': 500}

    # initial target file count - naive (8010 files)
    target_counts = {'i': 500, 'ii': 930, 'iii': 930, 'iv': 930,
                     'v': 500, 'vi': 930, 'vii': 930, 'viii': 930,
                     'ix': 930, 'x': 500}

    init_data_dir(data_dir, base_dir)
    _, file_count = count_all_files(data_dir)

    eng_kwargs = {
        'i': {'horiz': True, 'vert': True},
        'ii': {'horiz': True, 'vert': True},
        'iii': {'horiz': True, 'vert': True},
        'iv': {'horiz': False, 'vert': False, 'target': 'vi'},
        'v': {'horiz': True, 'vert': False},
        'vi': {'horiz': False, 'vert': False, 'target': 'iv'},
        'vii': {'horiz': False, 'vert': False},
        'viii': {'horiz': False, 'vert': False},
        'ix': {'horiz': False, 'vert': True},
        'x': {'horiz': True, 'vert': True},
    }

    engineer_images_by_num(eng_kwargs, data_dir, target_counts)
    _, file_count = count_all_files(data_dir)
    n_opt_files = target_count_tot - file_count

    predictions = None
    optm_df = pd.DataFrame()
    bad_images = []
    bad_image_record = {}
    for i in range(n_iter):
        if i > 0:
            i_bad_images = parse_bad_images(predictions)
            bad_images += i_bad_images
            bad_images = list(set(bad_images))
            bad_image_record[i] = bad_images
            engineer_bad_images(bad_images, n_opt_files, data_dir)
            zipdir(data_dir, './{}_{}.zip'.format(job_tag, i))

        predictions, val_acc, test_acc = train(data_dir, test_dir,
                                               epochs=epochs)

        optm_df.at[i, 'val_acc'] = val_acc
        optm_df.at[i, 'test_acc'] = test_acc
        optm_df.at[i, 'n_bad_images'] = len(bad_images)
        for num in NUMERALS:
            num_bad = [fp for fp in bad_images if '/{}/'.format(num) in fp]
            optm_df.at[i, '{}_bad_images'.format(num)] = len(num_bad)

        optm_df.to_csv('./optimization_record_{}.csv'.format(job_tag))
        with open('./bad_image_record_{}.json'.format(job_tag), 'w') as f:
            json.dump(bad_image_record, f, indent=2, sort_keys=True)
