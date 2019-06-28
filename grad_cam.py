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

#import tensorflow as tf
import keras

#from utils.tfrecord_utils import *
from models.gradcam_unet import *

def running_average(old_average, cur_val, n):
    return old_average * (n-1)/n + cur_val/n

def mil_prediction(pred):
    idx = tf.argmax(pred[:, 1], axis=0)
    return pred[idx], idx

os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

if __name__ == "__main__":
    os.system("clear")

    #tf.enable_eager_execution()

    csv_file = os.path.join(
            os.sep, "ISFILE3", "USERS", "remediossw",
            "data", "tbi_weak_labels", "weak_labels.csv")

    df = pd.read_csv(csv_file)

    ########## HYPERPARAMETER SETUP ##########

    instance_size = (64, 64)
    num_classes = 2
    progbar_length = 40

    ########## DIRECTORY SETUP ##########

    NUM_FOLDS = 5
    DATA_DIR = "/ISFILE3/USERS/remediossw/data/tbi_weak_labels/preprocessed"
    filenames = os.listdir(DATA_DIR)
    #MODEL_PATH = os.path.join("models", "weights", "class_unet.json")
    #with open(MODEL_PATH, 'r') as json_data:
        #model = keras.models.model_from_json(json.load(json_data))

    ########## GRAD CAM ##########

    num_positive = 0
    num_negative = 0
    limit_positive = 10
    limit_negative = 10

    for filename in tqdm(filenames):
        for row in df.itertuples():
            if row[3] in filename:
                if np.sum(row[6:]) != 0:
                    if num_positive >= limit_positive:
                        continue
                    num_positive += 1
                    cur_label = "positive"
                else:
                    if num_negative >= limit_negative:
                        continue
                    num_negative += 1
                    cur_label = "negative"

                DST_DIR = os.path.join('grad_cams', filename[:filename.find('.nii')])
                if not os.path.exists(DST_DIR):
                    os.makedirs(DST_DIR)

                img_vol = nib.load(os.path.join(DATA_DIR, filename)).get_fdata()
                for i in range(img_vol.shape[2]):
                    img = img_vol[:, :, i]


                    img = np.reshape(img, (1,) + img.shape + (1,))

                    model = class_unet_2D(
                            img_shape=(img.shape[1:]),
                            num_channels=1,
                            num_classes=num_classes,
                            ds=2
                        )
                    WEIGHT_DIR = os.path.join("models", "weights")

                    MODEL_NAME = "class_unet" 
                    jet_heatmap = np.zeros((img.shape[1], img.shape[2], 3,))


                    for cur_fold in range(NUM_FOLDS):
                        WEIGHT_PATH = os.path.join(WEIGHT_DIR, 
                                "best_weights_fold_{}.h5".format(cur_fold))
                        model.load_weights(WEIGHT_PATH)

                        modifier = None
                        layer_idx = -1
                        
                        grads = visualize_cam(
                            model,
                            layer_idx,
                            filter_indices=1,
                            seed_input=img,
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
                    plt.imsave(
                            output_filename, 
                            img)

                    output_filename = os.path.join(
                            DST_DIR, 
                            "{}_heatmap_{:02d}.jpg".format(cur_label, i))
                    plt.imsave(
                            output_filename, 
                            jet_heatmap)
                    

                    output_filename = os.path.join(
                            DST_DIR, 
                            "{}_grad_cam_{:02d}.jpg".format(cur_label, i))
                    plt.imsave(
                            output_filename, 
                            overlay(jet_heatmap, img))
                    
