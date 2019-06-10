import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time

import tensorflow as tf

from utils.tfrecord_utils import *
from models.new_unet import *

def running_average(old_average, cur_val, n):
    return old_average * (n-1)/n + cur_val/n

os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == "__main__":

    tf.enable_eager_execution()

    ########## HYPERPARAMETER SETUP ##########

    num_epochs = 1000000
    batch_size = 32
    instance_size = (64, 64)
    num_classes = 5
    start_time = utils.now()
    learning_rate = 1e-4

    ########## DIRECTORY SETUP ##########

    WEIGHT_DIR = os.path.join("models", "weights")

    MODEL_NAME = "class_unet" 
    MODEL_PATH = os.path.join(WEIGHT_DIR, MODEL_NAME + ".json")

    HISTORY_PATH = os.path.join(WEIGHT_DIR, MODEL_NAME + "_history.json")

    # files and paths
    for d in [WEIGHT_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)

    ######### MODEL AND CALLBACKS #########
    model = class_unet(num_channels=1,
                       ds=32,
                       lr=learning_rate,
                       verbose=1,)

    json_string = model.to_json()
    with open(MODEL_PATH, 'w') as f:
        json.dump(json_string, f)

    print(model.summary())

    ######### DATA IMPORT #########
    TRAIN_TF_RECORD_FILENAME = os.path.join(
        "data", "classification", "train", "dataset.tfrecords")
    VAL_TF_RECORD_FILENAME = os.path.join(
        "data", "classification", "val", "dataset.tfrecords")

    train_dataset = tf.data.TFRecordDataset(TRAIN_TF_RECORD_FILENAME)\
        .repeat()\
        .map(lambda record: parse_classification_example(
            record,
            vol_size,
            num_labels=num_classes))\
        .shuffle(buffer_size=100)

    val_dataset = tf.data.TFRecordDataset(VAL_TF_RECORD_FILENAME)\
        .repeat()\
        .map(lambda record: parse_classification_example(
            record,
            vol_size,
            num_labels=num_classes))\
        .shuffle(buffer_size=100)

    ######### TRAINING #########

    grads = [tf.zeros_like(l) for l in model.trainable_variables]

    for cur_epoch in range(N_EPOCHS):
        print("\nEpoch {}/{}".format(cur_epoch + 1, N_EPOCHS))

        for i, (x, y) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                repeated_y = np.repeat(y.numpy(), len(x), axis=0)

                logits = model(x, training=True)

                losses = tf.losses.sigmoid_cross_entropy(
                        multi_class_labels=tf.reshape(repeated_y, repeated_y.shape + (1,)),
                        logits=logits,
                        reduction=tf.losses.Reduction.NONE
                    )

                if tf.reduce_sum(y) == 0:
                    # in all-zero class situtation, take mean of entire bag
                    loss = tf.reduce_mean(losses, axis=0)
                    loss = tf.reshape(loss, (num_classes,1))
                    grad = tape.gradient(loss, trainable_variables)
                    # aggregate current element in batch
                    for k in range(len(grad)):
                        grads[k] = running_average(grads[k], grad[k], i + 1)
                else:
                    # otherwise, take top instance for each class
                    multiclass_grads = [tf.zeros_like(l) for l in model.trainable_variables]
                    top_polluted_indices = tf.argmax(logits, dimension=0).numpy()[0]
                    # average among top num_classes instances
                    for j, top_polluted_idx in enumerate(top_polluted_indices):
                        loss = tf.reduce_min(losses[top_polluted_idx], axis=0)
                        loss = tf.reshape(loss, (num_classes,1))
                        grad = tape.gradient(loss, trainable_variables)
                        for k in range(len(grad)):
                            multiclass_grads = running_average(multiclass_grads, grad[k], j + 1)
                    # aggregate current element in batch
                    for k in range(len(multiclass_grads)):
                        grads[k] = running_average(grads[k], multiclass_grads[k], i + 1)



            if i > 0 and i % batch_size == 0 or i == num_elements - 1:
                opt.apply_gradients(zip(grads, model.trainable_variables))
                grads = [tf.zeros_like(l) for l in model.trainable_variables]
        
        model.save_weights(os.path.join(WEIGHT_DIR, "mil_weights.tf"))

