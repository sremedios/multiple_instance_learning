import numpy as np
import os
import sys
import pandas as pd

import tensorflow as tf
import nibabel as nib

from sklearn.utils import shuffle

from tqdm import tqdm
from utils.pad import *
from utils import preprocess
from utils.tfrecord_utils import *
from utils.patch_ops import *
from utils.class_mapping import *


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Missing cmd line argument")
        sys.exit()

    tf.enable_eager_execution()
    os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'

    ######### DIRECTRY SETUP #########

    DATA_DIR = sys.argv[1]
    df = pd.read_csv(sys.argv[2])
    TF_RECORD_FILENAME = sys.argv[3]

    PATCH_DIMS = (64, 64)

    class_mapping = read_mapping("class_mapping.pkl")

    ######### GET DATA FILENAMES #######
    filenames = [x for x in os.listdir(DATA_DIR) if ".nii" in x]
    filenames.sort()
    filenames = shuffle(filenames)

    ##################### WRITE TF RECORD ######################
    with tf.python_io.TFRecordWriter(TF_RECORD_FILENAME) as writer:
        for x_file in tqdm(filenames):
            for row in df.itertuples():
                if row[3] in x_file:
                    x = nib.load(
                            os.path.join(DATA_DIR, x_file)
                        ).get_fdata()
                    x[np.where(x <= 0)] = 0

                    x_patches = get_nonoverlapping_patches(x, PATCH_DIMS)
                    x_patches = x_patches.astype(np.float16)

                    y_label = np.array(row[6:], dtype=np.int8)
                    '''
                    y_label = tf.keras.utils.to_categorical(
                            class_mapping[tuple(y_label)],
                            len(class_mapping),
                            dtype='int8',
                    )
                    '''

                    tf_example = image_example(x_patches, y_label, len(x_patches))
                    writer.write(tf_example.SerializeToString())
