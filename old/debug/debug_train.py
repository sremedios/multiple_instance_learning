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
    tf.logging.set_verbosity(tf.logging.ERROR)

    ########## HYPERPARAMETER SETUP ##########

    N_EPOCHS = 10000
    batch_size = 16
    minibatch_size = 16
    instance_size = (64, 64)
    num_classes = 4
    learning_rate = 1e-5
    progbar_length = 40

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
                       ds=2,)

    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

    print(model.summary())

    ######### DATA IMPORT #########
    TRAIN_TF_RECORD_FILENAME = "debug.tfrecord"
    VAL_TF_RECORD_FILENAME = "debug.tfrecord"

    tmp = tf.data.TFRecordDataset(TRAIN_TF_RECORD_FILENAME)\
        .map(lambda record: parse_bag(
            record,
            instance_size,
            num_labels=num_classes))
    for num_elements, data in enumerate(tmp):
        continue

    print("Found {} bags in {}".format(num_elements, TRAIN_TF_RECORD_FILENAME))

    train_dataset = tf.data.TFRecordDataset(TRAIN_TF_RECORD_FILENAME)\
        .map(lambda record: parse_bag(
            record,
            instance_size,
            num_labels=num_classes))\
        .shuffle(buffer_size=100)

    val_dataset = tf.data.TFRecordDataset(VAL_TF_RECORD_FILENAME)\
        .map(lambda record: parse_bag(
            record,
            instance_size,
            num_labels=num_classes))\

    ######### TRAINING #########

    best_val_loss = 100000
    convergence_epoch_counter = 0
    CONVERGENCE_EPOCH_LIMIT = 20
    epsilon = 1e-4

    grads = [tf.zeros_like(l) for l in model.trainable_variables]

    for cur_epoch in range(N_EPOCHS):
        epoch_train_loss = 0
        epoch_train_acc = 0
        train_correct = 0

        epoch_val_loss = 0
        epoch_val_acc = 0
        val_correct = 0

        print("\nEpoch {}/{}".format(cur_epoch + 1, N_EPOCHS))
        sys.stdout.write("\r[{:{}<{}}]".format(
            "=" * 0, '-', progbar_length
        ))

        loss_weights = tf.convert_to_tensor([[47/106, 10/106, 18/106, 2/106]])

        for i, (x, y) in enumerate(train_dataset):
            with tf.GradientTape(persistent=True) as tape:
                logits = model(x, training=True)
                pred, top_idx = mil_prediction(tf.nn.sigmoid(logits))

                if tf.reduce_sum(y).numpy() == 0:
                    repeated_y = np.repeat(
                            tf.reshape(y, (1,) + y.numpy().shape),
                            len(x),
                            axis=0
                    )
                    loss = tf.losses.sigmoid_cross_entropy(
                            multi_class_labels=repeated_y,
                            logits=logits,
                            weights=loss_weights,
                            reduction=tf.losses.Reduction.NONE
                        )
                    loss = tf.reduce_mean(loss, axis=0)

                else:
                    loss = tf.losses.sigmoid_cross_entropy(
                            multi_class_labels=y,
                            logits=logits[top_idx],
                            reduction=tf.losses.Reduction.NONE
                        )
                grad = tape.gradient(loss, model.trainable_variables)
                # aggregate current element in batch
                for k in range(len(grad)):
                    grads[k] = running_average(grads[k], grad[k], i + 1)

            pred = tf.round(pred)

            if np.all(pred.numpy() == y.numpy()):
                train_correct += 1
            cur_train_acc = train_correct / (i + 1)
            epoch_train_loss = running_average(epoch_train_loss, loss, i + 1)
            epoch_train_acc = running_average(epoch_train_acc, cur_train_acc, i + 1)


            if (i > 0 and i % batch_size == 0) or i == num_elements - 1:
                opt.apply_gradients(zip(grads, model.trainable_variables))
                sys.stdout.write("\r[{:{}<{}}] Loss: {:.4f} Acc: {:.2%}".format(
                    "=" * min(int(progbar_length * (i/num_elements)), progbar_length),
                    "-",
                    progbar_length,
                    epoch_train_loss.numpy()[0],
                    epoch_train_acc,
                ))
                sys.stdout.flush()
                # wipe tape and records
                grads = [tf.zeros_like(l) for l in model.trainable_variables]
                del tape

        
        for i, (x, y) in enumerate(val_dataset):
            logits = model(x, training=True)
            repeated_y = np.repeat(
                    tf.reshape(y, (1,) + y.numpy().shape),
                    len(x),
                    axis=0
            )

            # keep track of avg loss
            loss = tf.losses.sigmoid_cross_entropy(
                    multi_class_labels=repeated_y,
                    logits=logits,
                    reduction=tf.losses.Reduction.NONE
                )
            loss = tf.reduce_sum(tf.reduce_mean(loss, axis=0)).numpy()
            epoch_val_loss = running_average(epoch_val_loss, loss, i + 1)

            # Acc is based off max predictions

            pred, _ = mil_prediction(tf.nn.sigmoid(logits))
            pred = tf.round(pred)

            if np.all(pred.numpy() == y.numpy()):
                val_correct += 1
            cur_val_acc = val_correct / (i + 1)
            epoch_val_acc = running_average(epoch_val_acc, cur_val_acc, i + 1)


        if convergence_epoch_counter >= CONVERGENCE_EPOCH_LIMIT:
            print("\nNo improvement in {} epochs, model is converged.".format(
                CONVERGENCE_EPOCH_LIMIT
            ))
            sys.exit()

        if epoch_val_loss > best_val_loss and np.abs(epoch_val_loss - best_val_loss) > epsilon:
            convergence_epoch_counter += 1
        else:
            convergence_epoch_counter = 0

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            model.save_weights(os.path.join(WEIGHT_DIR, "best_weights.h5"))

        sys.stdout.write(" Val Loss: {:.4f} Val Acc: {:.2%}".format(
            epoch_val_loss,
            epoch_val_acc
        ))

