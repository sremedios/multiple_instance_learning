import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import json
import time
from tqdm import tqdm

import tensorflow as tf

from utils.tfrecord_utils import *
from models.new_unet import *

def running_average(old_average, cur_val, n):
    return old_average * (n-1)/n + cur_val/n

def mil_prediction(pred):
    idx = tf.argmax(pred[:, 1], axis=0)
    return pred[idx], idx

os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == "__main__":

    tf.enable_eager_execution()

    ########## HYPERPARAMETER SETUP ##########

    instance_size = (64, 64)
    num_classes = 2
    NUM_FOLDS = 3

    ########## DIRECTORY SETUP ##########

    WEIGHT_DIR = os.path.join("models", "weights")

    MODEL_NAME = "class_unet" 
    MODEL_PATH = os.path.join("models", "weights", "class_resnet.json")
    with open(MODEL_PATH) as json_data:
        model = tf.keras.models.model_from_json(json.load(json_data))
            

    ######### DATA IMPORT #########
    #TEST_TF_RECORD_FILENAME = "dataset_fold___test.tfrecord"
    TEST_TF_RECORD_FILENAME = "dataset_fold_4_val.tfrecord"

    test_dataset = tf.data.TFRecordDataset(TEST_TF_RECORD_FILENAME)\
        .map(lambda record: parse_bag(
            record,
            instance_size,
            num_labels=num_classes))

    for num_bags, data in enumerate(test_dataset):
        continue
    num_bags += 1

    ######### TESTING #########

    test_correct = 0
    classwise_acc = [0 for _ in range(num_classes)]
    classwise_total = [0 for _ in range(num_classes)]

    for i, (x, y) in tqdm(enumerate(test_dataset), total=num_bags):
        pred = tf.zeros((2,)) 
        for cur_fold in range(NUM_FOLDS):
            WEIGHT_PATH = os.path.join(WEIGHT_DIR, "best_weights_fold_{}.h5".format(cur_fold))
            model.load_weights(WEIGHT_PATH)

            logits = model(x)
            cur_pred, _ = mil_prediction(tf.nn.softmax(logits))
            pred += cur_pred

        pred /= NUM_FOLDS

        if tf.argmax(pred).numpy() == 0:
            pred = 0
        else:
            pred = 1

        if pred == tf.argmax(y).numpy():
            test_correct += 1

        if tf.argmax(y).numpy() == 0:
            classwise_total[0] += 1
        else:
            classwise_total[1] += 1

        if pred == 0:
            classwise_acc[0] += 1
        else:
            classwise_acc[1] += 1

    cur_test_acc = test_correct / (i + 1)

    sys.stdout.write("{}/{} = {:.2%}\n".format(
        test_correct,
        i+1,
        cur_test_acc
    ))
    print("Classwise pred", classwise_acc)
    print("Classwise total", classwise_total)

