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
    epochs = 100
    n_iter = 50

    # initial target file count - naive (7500 files)
    target_counts = {'i': 400, 'ii': 900, 'iii': 900, 'iv': 900,
                     'v': 400, 'vi': 900, 'vii': 900, 'viii': 900,
                     'ix': 900, 'x': 400}

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
