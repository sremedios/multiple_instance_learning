import numpy as np
import os
import sys
import pandas as pd

import tensorflow as tf
import nibabel as nib

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
    TF_RECORD_FILENAME = os.path.join(DATA_DIR, "dataset.tfrecords")
    PATCH_DIMS = (64, 64)

    ######### GET DATA FILENAMES #########
    pos_dir = os.path.join(DATA_DIR, "positive")
    pos_filenames = os.listdir(pos_dir)
    pos_filenames = [x for x in pos_filenames if ".nii.gz" in x]
    pos_filenames.sort()

    neg_dir = os.path.join(DATA_DIR, "negative")
    neg_filenames = os.listdir(neg_dir)
    neg_filenames = [x for x in neg_filenames if ".nii.gz" in x]
    neg_filenames.sort()

    ##################### WRITE TF RECORD ######################
    with tf.python_io.TFRecordWriter(TF_RECORD_FILENAME) as writer:
        for filenames, file_dir in zip([pos_filenames, neg_filenames], [pos_dir, neg_dir]):
            for x_file in tqdm(filenames):
                x = nib.load(os.path.join(file_dir, x_file)).get_fdata()
                x[np.where(x <= 0)] = 0

                x_patches = get_nonoverlapping_patches(x, PATCH_DIMS)
                x_patches = x_patches.astype(np.float16)

                y_label = np.array([1], dtype=np.int8)

                tf_example = image_example(x_patches, y_label, len(x_patches))
                writer.write(tf_example.SerializeToString())
