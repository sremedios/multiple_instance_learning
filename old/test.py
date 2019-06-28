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

def mil_prediction(pred):
    idx = tf.argmax(tf.reduce_sum(pred, axis=1), axis=0)
    return pred[idx], idx

os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == "__main__":

    tf.enable_eager_execution()

    ########## HYPERPARAMETER SETUP ##########

    instance_size = (64, 64)
    num_classes = 4
    progbar_length = 40

    ########## DIRECTORY SETUP ##########

    WEIGHT_DIR = os.path.join("models", "weights")

    MODEL_NAME = "class_unet" 
    WEIGHT_PATH = os.path.join(WEIGHT_DIR, "best_weights.h5")
    model = class_unet_2D(num_channels=1,
            num_classes=num_classes,
            ds=2,)
    model.load_weights(WEIGHT_PATH)

    print(model.summary())

    ######### DATA IMPORT #########
    TEST_TF_RECORD_FILENAME = "dataset_test.tfrecord"

    test_dataset = tf.data.TFRecordDataset(TEST_TF_RECORD_FILENAME)\
        .map(lambda record: parse_bag(
            record,
            instance_size,
            num_labels=num_classes))

    ######### TESTING #########

    test_correct = 0
    classwise_acc = [0 for _ in range(num_classes+1)]
    classwise_total = [0 for _ in range(num_classes+1)]
    debug = 0

    for i, (x, y) in enumerate(test_dataset):
        logits = model(x, training=True)

        # Acc is based off max predictions
        pred, _ = mil_prediction(tf.nn.sigmoid(logits))
        pred = tf.round(pred)

        if debug <=10 and np.sum(y.numpy()) != 0:
            print(pred.numpy(), '\n', y.numpy(), '\n')
            debug += 1

        if np.all(pred.numpy() == y.numpy()):
            test_correct += 1

        # first index is for the all-zero class
        if np.sum(y.numpy()) == 0:
            classwise_total[0] += 1
            if np.sum(pred.numpy()) == 0:
                classwise_acc[0] += 1
        else:
            for j in range(num_classes):
                classwise_total[j+1] += y.numpy()[j]
                classwise_acc[j+1] += pred.numpy()[j]

    cur_test_acc = test_correct / (i + 1)

    sys.stdout.write("{}/{} = {:.2%}\n".format(
        test_correct,
        i,
        cur_test_acc
    ))
    print("Classwise acc", classwise_acc)
    print("Classwise total", classwise_total)

