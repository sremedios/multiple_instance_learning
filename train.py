import pickle
import requests

import matplotlib.pyplot as plt

import json
import numpy as np
import os
from subprocess import Popen, PIPE
import sys
import time

import tensorflow as tf
import tensorflow.keras.backend as K

from utils.grad_ops import *
from utils import utils, patch_ops
from utils import preprocess
from utils.augmentations import *

from utils.tfrecord_utils import * 

from models.multi_gpu import ModelMGPU
from models.old_losses import *
from models.new_unet import *

os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


if __name__ == "__main__":

    results = utils.parse_args("train")

    ########## GPU SETUP ##########

    NUM_GPUS = 1

    if results.GPUID == None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    elif results.GPUID == -1:
        # find maximum number of available GPUs
        call = "nvidia-smi --list-gpus"
        pipe = Popen(call, shell=True, stdout=PIPE).stdout
        available_gpus = pipe.read().decode().splitlines()
        NUM_GPUS = len(available_gpus)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(results.GPUID)

    tf.enable_eager_execution()

    ########## HYPERPARAMETER SETUP ##########

    num_epochs = 1000000
    batch_size = results.batch_size
    vol_size = (256, 256, 32, 1)
    start_time = utils.now()
    learning_rate = 1e-5

    ########## DIRECTORY SETUP ##########

    WEIGHT_DIR = os.path.join("models", "weights", results.experiment_details)
    TB_LOG_DIR = os.path.join("models", "tensorboard", results.experiment_details)

    MODEL_NAME = results.experiment_details
    MODEL_PATH = os.path.join(WEIGHT_DIR, MODEL_NAME + ".json")

    HISTORY_PATH = os.path.join(WEIGHT_DIR, MODEL_NAME + "_history.json")

    # files and paths
    for d in [TB_LOG_DIR, WEIGHT_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)

    ######### MODEL AND CALLBACKS #########
    model = class_unet(num_channels=1,
                 ds=32,
                 lr=learning_rate,
                 verbose=1,)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['acc'])

    json_string = model.to_json()
    with open(MODEL_PATH, 'w') as f:
        json.dump(json_string, f)

    print(model.summary())

    loss_fn = continuous_dice_coef_loss
    monitor = "val_acc"

    # callbacks
    checkpoint_filename = "epoch_{epoch:04d}_"\
        + monitor + "_"\
        + "{" + monitor + ":.4f}.hdf5"
    checkpoint_filename = os.path.join(WEIGHT_DIR, checkpoint_filename)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_filename,
                                                    monitor='val_loss',
                                                    save_best_only=True,
                                                    save_weights_only=True,
                                                    mode='auto',
                                                    verbose=0,)

    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                          min_delta=1e-4,
                                          patience=50,
                                          verbose=1,
                                          mode='auto')

    tb = tf.keras.callbacks.TensorBoard(log_dir=TB_LOG_DIR)

    callbacks_list = [checkpoint, es, tb]

    ######### DATA IMPORT #########
    TRAIN_TF_RECORD_FILENAME = os.path.join(
        "data", "classification", "train", "dataset.tfrecords")
    VAL_TF_RECORD_FILENAME = os.path.join(
        "data", "classification", "val", "dataset.tfrecords")

    # data augmentations
    augmentations = [flip_dim1, flip_dim2, flip_dim3]

    train_dataset = tf.data.TFRecordDataset(TRAIN_TF_RECORD_FILENAME)\
        .repeat()\
        .map(lambda record: parse_classification_example(record, vol_size, num_labels=4))\
        .shuffle(buffer_size=100)

    val_dataset = tf.data.TFRecordDataset(VAL_TF_RECORD_FILENAME)\
        .repeat()\
        .map(lambda record: parse_classification_example(record, vol_size, num_labels=4))\
        .shuffle(buffer_size=100)

    for f in augmentations:
        train_dataset = train_dataset.map(
            lambda x, y: (f(x), y), num_parallel_calls=2)

    train_dataset = train_dataset.batch(batch_size=batch_size)
    val_dataset = val_dataset.batch(batch_size=batch_size)

    ######### TRAINING #########

    # steps per epoch is number of datapoints * num augmentations that could have been applied
    model.fit(train_dataset,
              validation_data=val_dataset,
              steps_per_epoch=106//batch_size*len(augmentations),
              validation_steps=1,
              epochs=10000000,
              callbacks=callbacks_list)
