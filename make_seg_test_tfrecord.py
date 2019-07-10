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
            "data", "seg_dataset_test.tfrecord"
    )


    ######### LOAD PATCHES INTO RAM #######

    filenames = [x for x in os.listdir(DATA_DIR) if ".nii" in x]
    ct_filenames = [x for x in filenames if "CT" in x]
    mask_filenames = [x for x in filenames if "mask" in x]
    ct_filenames.sort()
    mask_filenames.sort()

    ct_filenames = np.array(ct_filenames)
    mask_filenames = np.array(mask_filenames)


    with tf.python_io.TFRecordWriter(TF_RECORD_FILENAME) as writer:

        for ct_filename, mask_filename in\
                tqdm(zip(ct_filenames, mask_filenames), total=len(ct_filenames)):
            x = nib.load(
                    os.path.join(DATA_DIR, ct_filename)
                ).get_fdata()
            y = nib.load(
                    os.path.join(DATA_DIR, mask_filename)
                ).get_fdata()

            tf_example = image_seg_example(x, y)
            writer.write(tf_example.SerializeToString())
