import matplotlib.pyplot as plt
import nibabel as nib
import matplotlib.cm as cm
from vis.visualization import visualize_cam, overlay
import numpy as np
import os
import json
import pandas as pd
import sys
import time
from tqdm import tqdm

import tensorflow as tf

from utils.tfrecord_utils import *
from models.gradcam_unet import *

def running_average(old_average, cur_val, n):
    return old_average * (n-1)/n + cur_val/n

def mil_prediction(pred):
    idx = tf.argmax(pred[:, 1], axis=0)
    return pred[idx], idx

os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if __name__ == "__main__":

    tf.enable_eager_execution()

    ########## HYPERPARAMETER SETUP ##########

    dataset_count = int(sys.argv[1])
    instance_size = (512, 512)
    num_classes = 2
    progbar_length = 40

    ########## MODEL SETUP ##########

    WEIGHT_PATH = os.path.join(
            "models", 
            "weights", 
            "class_resnet", 
            "dataset_{}".format(dataset_count), 
            "best_weights_fold_1.h5"
    )

    MODEL_PATH = os.path.join(
            "models", 
            "weights", 
            "class_resnet", 
            "class_resnet.json",
    )

    with open(MODEL_PATH, 'r') as json_data:
        model = tf.keras.models.model_from_json(json.load(json_data))

    # specify shape is necessary for gradCAM
    model.layers[0]._batch_input_shape = (None, *instance_size, 1)

    ########## DATASET SETUP ##########

    test_ds_filename = os.path.join(
            os.sep,
            'home',
            'remedis',
            'data',
            'dataset_fold___test.tfrecord',
    )

    test_dataset = tf.data.TFRecordDataset(test_ds_filename)\
            .map(lambda r: parse_bag(
                r,
                instance_size,
                num_classes,
                )
            )

    ########## GRAD CAM ##########

    num_pos = 0
    num_neg = 0

    for i, (x, y) in enumerate(test_dataset):

        if i > 2:
            sys.exit()

        if tf.argmax(y).numpy() == 0:
            num_pos += 1
            cur_label = 'pos'
        else:
            num_neg += 1
            cur_label = 'neg'

        logits = model(x)
        pred, _ = mil_prediction(tf.nn.softmax(logits))

        DST_DIR = os.path.join('grad_cams', 'subject_{}'.format(i))

        if not os.path.exists(DST_DIR):
            os.makedirs(DST_DIR)

        img_vol = x.numpy()

        for i in range(img_vol.shape[0]):
            img_slice = img_vol[i, ...]

            jet_heatmap = np.zeros((img_slice.shape[1], img_slice.shape[2], 3,))

            modifier = None
            layer_idx = -1
            
            grads = visualize_cam(
                model,
                layer_idx,
                filter_indices=1,
                seed_input=img_slice,
                penultimate_layer_idx=-3,
                backprop_modifier=modifier,
            )
            jet_heatmap += (cm.jet(grads) * 255)[:, :, :, 0]

            jet_heatmap = np.uint8(jet_heatmap / NUM_FOLDS)

            img = np.reshape(img, (*img.shape[1:3], 1, 1))
            img = np.concatenate([
                    img[:, :, :, 0],
                    img[:, :, :, 0],
                    img[:, :, :, 0],
                ],
                axis=-1,)
            img[np.where(img < 0)] = 0
            if img.max() != 0:
                img /= img.max()
            img = np.uint8(img * 255)

            output_filename = os.path.join(
                    DST_DIR, 
                    "{}_orig_{:02d}.jpg".format(cur_label, i))
            plt.imshow(img)
            plt.savefig(output_filename)

            output_filename = os.path.join(
                    DST_DIR, 
                    "{}_heatmap_{:02d}.jpg".format(cur_label, i))
            plt.imshow(jet_heatmap)
            plt.savefig(output_filename)
            

            output_filename = os.path.join(
                    DST_DIR, 
                    "{}_gradcam_{:02d}.jpg".format(cur_label, i))
            plt.imshow(img)
            plt.imshow(jet_heatmap, alpha=0.3)
            plt.savefig(output_filename)
