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

    manual_check_file = sys.argv[1]
    healthy_lines = []
    with open(manual_check_file, 'r') as f:
        lines = f.readlines()
        for l in lines:
            if int(l.strip().split()[-1]) == 0:
                healthy_lines.append(l.strip().split())

    severe_file = sys.argv[2]
    with open(severe_file, 'r') as f:
        severe_lines = [l.strip() for l in f.readlines()]
    
    
    TF_RECORD_FILENAME = os.path.join(
            os.sep, "home", "remedis", "data", "dataset_fold_{}_{}.tfrecord"
    )

    TARGET_DIMS = (512, 512) 

    ######### GET DATA FILENAMES #######

    filenames = []
    scores = []
    omitted = []

    # severe lesions
    for l in tqdm(severe_lines):
        f = l.replace("_predicted_mask.nii.gz", ".nii.gz").replace("mask", "preprocessed")
        if not os.path.exists(f):
            omitted.append(f)
            continue
        filenames.append(f)
        scores.append(1)

    print(len(filenames), "severe")

    # healthy subjects
    for l in tqdm(healthy_lines):
        if int(l[-1]) == 0:
            f = l[0].replace(".png", ".nii.gz").replace("mask", "preprocessed")
            if not os.path.exists(f):
                omitted.append(f)
                continue
            filenames.append(f)
            scores.append(int(l[-1]))

    print(len(filenames), "healthy")

        
    pos_count = 0
    neg_count = 0

    X = []
    y = []

    ######### PAIR FILENAME WITH CLASS #########
    for x_file, y_score in zip(filenames, scores):
        # no finding is labeled 0
        if y_score == 0:
            X.append(x_file)
            y.append(0)
            neg_count += 1
        # Zihao scores have 3 as guaranteed hematoma
        # 1 is possibly hematoma
        elif y_score == 1:
            X.append(x_file)
            y.append(1)
            pos_count += 1

    print("Num positive: {} Num negative: {}".format(
        pos_count,
        neg_count,
        )
    )

    print("Omitted {}".format(len(omitted)))

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

    # shuffle indices for randomness
    train_idx = shuffle(train_idx, random_state=4)
    test_idx = shuffle(test_idx, random_state=4)

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
                try:
                    x = nib.load(X[train_i]).get_fdata()
                except EOFError:
                    print("Error loading {}, omitting".format(X[train_i]))
                    continue

                x[np.where(x <= 0)] = 0

                x_patches = get_slices(x, TARGET_DIMS)
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
                try:
                    x = nib.load(X[val_i]).get_fdata()
                except EOFError:
                    print("Error loading {}, omitting".format(X[val_i]))
                    continue
                x[np.where(x <= 0)] = 0

                x_patches = get_slices(x, TARGET_DIMS)
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
            try:
                x = nib.load(x_test_name).get_fdata()
            except EOFError:
                print("Error loading {}, omitting".format(x_test_name))
                continue

            x[np.where(x <= 0)] = 0

            x_patches = get_slices(x, TARGET_DIMS)
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
