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
<<<<<<< HEAD

    df = pd.read_csv(sys.argv[2])
    PATCH_DIMS = (64, 64)
=======
    TF_RECORD_FILENAME = sys.argv[2]
    PATCH_DIMS = (128, 128)
>>>>>>> 5adc217c78d2251e9bb5e2d68bab8d7e2eab4909

    ######### GET DATA FILENAMES #########
    sub_dirs = [os.path.join(x, "preprocessed") for x in os.listdir(DATA_DIR) 
                if os.path.isdir(os.path.join(DATA_DIR, x))]

    filenames_dict = {}

    for sub_dir in sub_dirs:
        filenames = os.listdir(os.path.join(DATA_DIR, sub_dir))
        filenames_dict[sub_dir] = [x for x in filenames if ".nii" in x]
        filenames_dict[sub_dir].sort()

    ##################### WRITE TF RECORD ######################
<<<<<<< HEAD
    for file_dir, filenames in filenames_dict.items():
        TF_RECORD_FILENAME = os.path.join(DATA_DIR, file_dir, "dataset.tfrecords")
        with tf.python_io.TFRecordWriter(TF_RECORD_FILENAME) as writer:
=======
    with tf.python_io.TFRecordWriter(TF_RECORD_FILENAME) as writer:
        for file_dir, filenames in filenames_dict.items():
            if "pos" in file_dir:
                y_label = np.array([1], dtype=np.int8)
            else:
                y_label = np.array([0], dtype=np.int8)

>>>>>>> 5adc217c78d2251e9bb5e2d68bab8d7e2eab4909
            for x_file in tqdm(filenames):
                for row in df.itertuples():
                    if row[3] in x_file:
                        x = nib.load(
                                os.path.join(DATA_DIR, file_dir, x_file)
                            ).get_fdata()
                        x[np.where(x <= 0)] = 0

                        x_patches = get_nonoverlapping_patches(x, PATCH_DIMS)
                        x_patches = x_patches.astype(np.float16)

<<<<<<< HEAD
                        y_label = np.array(row[5:], dtype=np.int8)
=======
>>>>>>> 5adc217c78d2251e9bb5e2d68bab8d7e2eab4909

                        tf_example = image_example(x_patches, y_label, len(x_patches))
                        writer.write(tf_example.SerializeToString())
