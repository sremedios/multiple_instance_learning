import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve, confusion_matrix, average_precision_score
import os
import sys
from utils.tfrecord_utils import *
from tqdm import tqdm
import json

tf.enable_eager_execution()

def mil_prediction(pred, n=1):
    i = tf.argsort(pred[:, 1], axis=0)
    i = i[len(pred) - n : len(pred)]
    return (tf.gather(pred, i), i)

for dataset_count in [672, 500, 400, 300, 200, 100]:

    # load model
    weight_path = "/nfs/share5/remedis/projects/multiple_instance_learning/models/weights/class_resnet/dataset_{}/best_weights_fold_1.h5".format(dataset_count)
    model_path = "/nfs/share5/remedis/projects/multiple_instance_learning/models/weights/class_resnet/class_resnet.json"
    with open(model_path) as json_data:
        model = tf.keras.models.model_from_json(json.load(json_data))
    model.load_weights(weight_path)

    DST_DIR = os.path.join("figures", "figure_3_dataset_{}".format(dataset_count))
    if not os.path.exists(DST_DIR):
        os.makedirs(DST_DIR)
    dst_file = os.path.join(DST_DIR, "model_predictions_{}".format(dataset_count))

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
                (512, 512),
                2,
                )
            )

    num_pos = 0
    num_neg = 0

    y_true = []
    y_prob = []
    y_pred = []

    with open(dst_file, 'w') as f:
        f.write("y_true,y_pred,y_prob\n")

    print("Performing inference...")

    # forward pass
    for x, y in tqdm(test_dataset, total=4042):
        logits = model(x)
        pred, _ = mil_prediction(tf.nn.softmax(logits))
        
        y_true.append(tf.argmax(y).numpy())
        y_prob.append(pred.numpy()[0, 1])
        y_pred.append(tf.argmax(pred[0]).numpy())
        
        if tf.argmax(y).numpy() == 0:
            num_neg += 1
        else:
            num_pos += 1

    with open(dst_file, 'a') as f:
        for a, b, c in zip(y_true, y_pred, y_prob):
            f.write("{},{},{}\n".format(a,b,c))
