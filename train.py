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

    N_EPOCHS = 100
    batch_size = 32
    instance_size = (64, 64)
    num_classes = 5
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
    model = class_unet_2D(num_channels=1,
                        num_classes=num_classes,
                       ds=32,
                       lr=learning_rate,
                       verbose=1,)

    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

    print(model.summary())

    ######### DATA IMPORT #########
    TRAIN_TF_RECORD_FILENAME = "train_dataset.tfrecord"
    VAL_TF_RECORD_FILENAME = "val_dataset.tfrecord"

    tmp = tf.data.TFRecordDataset(TRAIN_TF_RECORD_FILENAME)\
        .map(lambda record: parse_bag(
            record,
            instance_size,
            num_labels=num_classes))
    for num_elements, data in enumerate(tmp):
        continue

    print("Found {} bags in {}".format(num_elements, TRAIN_TF_RECORD_FILENAME))

    train_dataset = tf.data.TFRecordDataset(TRAIN_TF_RECORD_FILENAME)\
        .repeat()\
        .map(lambda record: parse_bag(
            record,
            instance_size,
            num_labels=num_classes))\
        .shuffle(buffer_size=100)

    val_dataset = tf.data.TFRecordDataset(VAL_TF_RECORD_FILENAME)\
        .repeat()\
        .map(lambda record: parse_bag(
            record,
            instance_size,
            num_labels=num_classes))\
        .shuffle(buffer_size=100)

    ######### TRAINING #########

    grads = [tf.zeros_like(l) for l in model.trainable_variables]

    for cur_epoch in range(N_EPOCHS):
        epoch_loss = 0
        epoch_acc = 0
        correct = 0

        print("\nEpoch {}/{}".format(cur_epoch + 1, N_EPOCHS))

        for i, (x, y) in enumerate(train_dataset):
            with tf.GradientTape(persistent=True) as tape:

                logits = model(x, training=True)
                if tf.reduce_sum(y) == 0:
                    loss= tf.losses.sigmoid_cross_entropy(
                            multi_class_labels=y,
                            logits=logits,
                            reduction=tf.losses.Reduction.MEAN
                        )
                    # in all-zero class situtation, take mean of entire bag
                    loss = tf.reshape(loss, (num_classes,1))
                    grad = tape.gradient(loss, model.trainable_variables)
                    # aggregate current element in batch
                    for k in range(len(grad)):
                        grads[k] = running_average(grads[k], grad[k], i + 1)

                else:
                    # otherwise, take top polluted instance for each class
                    multiclass_grads = [tf.zeros_like(l) for l in model.trainable_variables]
                    top_polluted_indices = tf.argmax(logits, dimension=0).numpy()
                    # average among top num_classes instances
                    for j, top_polluted_idx in enumerate(top_polluted_indices):
                        loss= tf.losses.sigmoid_cross_entropy(
                                multi_class_labels=y,
                                logits=logits[top_polluted_idx],
                                reduction=tf.losses.Reduction.NONE
                            )
                        grad = tape.gradient(loss, model.trainable_variables)
                        for k in range(len(grad)):
                            multiclass_grads[k] = running_average(multiclass_grads[k], grad[k], j + 1)
                    # aggregate current element in batch
                    for k in range(len(multiclass_grads)):
                        grads[k] = running_average(grads[k], multiclass_grads[k], i + 1)


            # Acc is based off max predictions
            # TODO: figure out how to handle prediction correctly for multiclass
            # For now, this is a garbage value
            pred = tf.reshape(tf.round(tf.reduce_max(logits)), y.shape)
            if pred.numpy() == y.numpy():
                correct += 1
            cur_acc = correct / (i + 1)
            epoch_loss = running_average(epoch_loss, loss, i + 1)
            epoch_acc = running_average(epoch_acc, cur_acc, i + 1)


            if i > 0 and i % batch_size == 0 or i == num_elements - 1:
                opt.apply_gradients(zip(grads, model.trainable_variables))
                sys.stdout.write("\r[{:{}<{}}] Loss: {:.4f} Acc: {:.2%} = {}/{}".format(
                    "=" * i,
                    "-",
                    progbar_length,
                    epoch_loss.numpy()[0],
                    epoch_acc,
                    correct,
                    i + 1
                ))
                sys.stdout.flush()
                # wipe tape and records
                grads = [tf.zeros_like(l) for l in model.trainable_variables]
                del tape

        
        model.save_weights(os.path.join(WEIGHT_DIR, "mil_weights.tf"))
