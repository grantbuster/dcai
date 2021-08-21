import os
import pandas as pd

from train import train
from utilities import (init_data_dir, count_all_files, engineer_images_by_num,
                       parse_bad_images, engineer_bad_images)


NUMERALS = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"]


if __name__ == '__main__':
    data_dir = './dcai_gcb_06/dcai_gcb_06'
    test_dir = './label_book/label_book'
    base_dir = './data_baseline_clean/data_baseline_clean'
    target_count_tot = 9995
    epochs = 10

    # initial target file count 8000 files (80% of max)
    target_counts = {'i': 240, 'ii': 560, 'iii': 560, 'iv': 1360,
                     'v': 640, 'vi': 800, 'vii': 1120, 'viii': 1200,
                     'ix': 1120, 'x': 400}

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
    for i in range(2):
        if i > 0:
            i_bad_images = parse_bad_images(predictions)
            bad_images += i_bad_images
            engineer_bad_images(bad_images, n_opt_files, data_dir)

        predictions, val_acc, test_acc = train(data_dir, test_dir,
                                               epochs=epochs)

        optm_df.at[i, 'val_acc'] = val_acc
        optm_df.at[i, 'test_acc'] = test_acc
        optm_df.at[i, 'n_bad_images'] = len(bad_images)
        optm_df.at[i, 'n_unique_bad_images'] = len(set(bad_images))

    optm_df.to_csv('./optmization_record_{}.csv'
                   .format(os.path.basename(data_dir)))
