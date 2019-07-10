import numpy as np
import os
import sys
import pandas as pd

import tensorflow as tf
import nibabel as nib

from sklearn.utils import shuffle
from sklearn.model_selection import KFold

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
    TF_RECORD_FILENAME = os.path.join(
            "data", "seg_dataset_fold_{}_{}.tfrecord"
    )

    PATCH_DIMS = (128, 128)


    ######### LOAD PATCHES INTO RAM #######

    filenames = [x for x in os.listdir(DATA_DIR) if ".nii" in x]
    ct_filenames = [x for x in filenames if "CT" in x]
    mask_filenames = [x for x in filenames if "mask" in x]
    ct_filenames.sort()
    mask_filenames.sort()

    ct_filenames = np.array(ct_filenames)
    mask_filenames = np.array(mask_filenames)

    ######### 5-FOLD TRAIN/VAL SPLIT #########

    # cross validate over filenames
    skf = KFold(n_splits=5, shuffle=True, random_state=4)

    for i, (train_idx, val_idx) in enumerate(skf.split(ct_filenames, mask_filenames)):
        # Train TFRecord
        with tf.python_io.TFRecordWriter(TF_RECORD_FILENAME.format(i, "train")) as writer:
            ct_patches = []
            mask_patches = []

            for train_i in tqdm(train_idx):
                x = nib.load(
                        os.path.join(DATA_DIR, ct_filenames[train_i])
                    ).get_fdata()
                y = nib.load(
                        os.path.join(DATA_DIR, mask_filenames[train_i])
                    ).get_fdata()

                x_patches, y_patches = get_patches(
                        invols=[x],
                        mask=y,
                        patchsize=PATCH_DIMS, 
                        maxpatch=10000,
                        num_channels=1,
                )

                # keep track of all patches
                for x_patch, y_patch in zip(x_patches, y_patches):
                    ct_patches.append(x_patch)
                    mask_patches.append(y_patch)

            # shuffle among subjects
            print("Shuffling...")
            ct_patches, mask_patches = shuffle(ct_patches, mask_patches)

            print("Writing...")
            # write TFRecord
            for x_patch, y_patch in\
                    tqdm(zip(ct_patches, mask_patches), total=len(ct_patches)):
                tf_example = image_seg_example(x_patch, y_patch)
                writer.write(tf_example.SerializeToString())


        # Val TFRecord
        with tf.python_io.TFRecordWriter(TF_RECORD_FILENAME.format(i, "val")) as writer:
            ct_patches = []
            mask_patches = []

            for val_i in tqdm(val_idx):
                x = nib.load(
                        os.path.join(DATA_DIR, ct_filenames[val_i])
                    ).get_fdata()
                y = nib.load(
                        os.path.join(DATA_DIR, mask_filenames[val_i])
                    ).get_fdata()

                x_patches, y_patches = get_patches(
                        invols=[x],
                        mask=y,
                        patchsize=PATCH_DIMS, 
                        maxpatch=10000,
                        num_channels=1,
                )

                # keep track of all patches
                for x_patch, y_patch in zip(x_patches, y_patches):
                    ct_patches.append(x_patch)
                    mask_patches.append(y_patch)

            # shuffle among subjects
            print("Shuffling...")
            ct_patches, mask_patches = shuffle(ct_patches, mask_patches)

            print("Writing...")

            # write TFRecord
            for x_patch, y_patch in\
                    tqdm(zip(ct_patches, mask_patches), total=len(ct_patches)):
                tf_example = image_seg_example(x_patch, y_patch)
                writer.write(tf_example.SerializeToString())

