import os
import numpy as np
import tensorflow as tf
import sys

from utils.tfrecord_utils import *

tf.enable_eager_execution()

TF_RECORD_FILENAME = sys.argv[1]

instance_size = (64, 64)
num_labels = 5
train_dataset = tf.data.TFRecordDataset(TF_RECORD_FILENAME)\
    .map(lambda record: parse_bag(record, instance_size, num_labels=num_labels))

for x, y in train_dataset:
    print(x.shape, y.shape)
    print(y.numpy())
