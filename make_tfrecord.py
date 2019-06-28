import numpy as np
import os
import sys
import pandas as pd

import tensorflow as tf
import nibabel as nib

from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold

from tqdm import tqdm
from utils.pad import *
from utils import preprocess
from utils.tfrecord_utils import *
from utils.patch_ops import *


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Missing cmd line argument")
        sys.exit()

    tf.enable_eager_execution()
    os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'

    ######### DIRECTRY SETUP #########

    DATA_DIR = sys.argv[1]
    df = pd.read_csv(sys.argv[2])
    TF_RECORD_FILENAME = os.path.join(
            "data", "dataset_fold_{}_{}.tfrecord"
    )

    PATCH_DIMS = (128, 128)

    ######### GET DATA FILENAMES #######
    filenames = [x for x in os.listdir(DATA_DIR) if ".nii" in x]
    filenames.sort()

    pos_count = 0
    neg_count = 0

    X = []
    y = []

    ######### PAIR FILENAME WITH CLASS #########
    for x_file in filenames:
        for row in df.itertuples():
            if row[3] in x_file:
                # no finding is labeled 0
                if np.sum(row[6:]) == 0:
                    X.append(x_file)
                    y.append(0)
                    neg_count += 1
                # extraaxial hematoma is labeled 1
                elif row[6] == 1:
                    X.append(x_file)
                    y.append(1)
                    pos_count += 1

    X = np.array(X)
    y = np.array(y)

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    pos_idx = shuffle(pos_idx, random_state=4)
    neg_idx = shuffle(neg_idx, random_state=4)

    ######### TRAIN/TEST SPLIT #########
    LIMIT_TRAIN_SPLIT = int(0.8 * min(len(pos_idx), len(neg_idx)))
    print("Num train pos: {} train neg: {}\ntest pos: {} test neg: {}".format(
        len(pos_idx[:LIMIT_TRAIN_SPLIT]),
        len(neg_idx[:LIMIT_TRAIN_SPLIT]),
        len(pos_idx[LIMIT_TRAIN_SPLIT:]),
        len(neg_idx[LIMIT_TRAIN_SPLIT:]),
    ))

    train_idx = np.concatenate([
        pos_idx[:LIMIT_TRAIN_SPLIT],
        neg_idx[:LIMIT_TRAIN_SPLIT],
    ])

    test_idx = np.concatenate([
        pos_idx[LIMIT_TRAIN_SPLIT:],
        neg_idx[LIMIT_TRAIN_SPLIT:],
    ])

    X_test = X[test_idx]
    y_test = y[test_idx]

    # exclude test indices from full dataset
    X = X[train_idx]
    y = y[train_idx]

    ######### 5-FOLD TRAIN/VAL SPLIT #########
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=4)

    for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        # Train TFRecord
        with tf.python_io.TFRecordWriter(TF_RECORD_FILENAME.format(i, "train")) as writer:
            train_pos = 0
            train_neg = 0
            for train_i in tqdm(train_idx):
                x = nib.load(
                        os.path.join(DATA_DIR, X[train_i])
                    ).get_fdata()
                x[np.where(x <= 0)] = 0

                x_patches = get_nonoverlapping_patches(x, PATCH_DIMS)
                x_patches = x_patches.astype(np.float16)

                if y[train_i] == 0:
                    y_label = np.array([1, 0], dtype=np.int8)
                    train_neg += 1
                else:
                    y_label = np.array([0, 1], dtype=np.int8)
                    train_pos += 1

                tf_example = image_example(x_patches, y_label, len(x_patches))
                writer.write(tf_example.SerializeToString())
        print("Train pos: {} Train neg: {}".format(train_pos, train_neg))

        # Val TFRecord
        with tf.python_io.TFRecordWriter(TF_RECORD_FILENAME.format(i, "val")) as writer:
            val_pos = 0
            val_neg = 0
            for val_i in tqdm(val_idx):
                x = nib.load(
                        os.path.join(DATA_DIR, X[val_i])
                    ).get_fdata()
                x[np.where(x <= 0)] = 0

                x_patches = get_nonoverlapping_patches(x, PATCH_DIMS)
                x_patches = x_patches.astype(np.float16)

                if y[val_i] == 0:
                    y_label = np.array([1, 0], dtype=np.int8)
                    val_neg += 1
                else:
                    y_label = np.array([0, 1], dtype=np.int8)
                    val_pos += 1

                tf_example = image_example(x_patches, y_label, len(x_patches))
                writer.write(tf_example.SerializeToString())
        print("Val pos: {} Val neg: {}".format(val_pos, val_neg))

    # Test TFRecord
    with tf.python_io.TFRecordWriter(TF_RECORD_FILENAME.format("_", "test")) as writer:
        test_pos = 0
        test_neg = 0
        for x_test_name, y_test_label in tqdm(zip(X_test, y_test), total=len(X_test)):
            x = nib.load(
                    os.path.join(DATA_DIR, x_test_name)
                ).get_fdata()
            x[np.where(x <= 0)] = 0

            x_patches = get_nonoverlapping_patches(x, PATCH_DIMS)
            x_patches = x_patches.astype(np.float16)

            if y_test_label == 0:
                y_label = np.array([1, 0], dtype=np.int8)
                test_neg += 1
            else:
                y_label = np.array([0, 1], dtype=np.int8)
                test_pos += 1

            tf_example = image_example(x_patches, y_label, len(x_patches))
            writer.write(tf_example.SerializeToString())
    print("Test pos: {} Test neg: {}".format(test_pos, test_neg))

