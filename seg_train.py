import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
import json

import tensorflow as tf

from utils.augmentations import *
from utils.tfrecord_utils import *
from models.new_unet import *
from models.losses import *

def running_average(old_average, cur_val, n):
    return old_average * (n-1)/n + cur_val/n

def mil_prediction(pred):
    # Most polluted is in column idx 1
    # So we take the max likely according to this column
    idx = tf.argmax(pred[:, 1], axis=0)
    return pred[idx], idx

os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

if __name__ == "__main__":

    tf.enable_eager_execution()
    tf.logging.set_verbosity(tf.logging.ERROR)

    ########## HYPERPARAMETER SETUP ##########

    N_EPOCHS = 10000
    batch_size = 4096
    minibatch_size = 16
    ds = 4
    instance_size = (128, 128)
    learning_rate = 1e-4
    progbar_length = 10
    CONVERGENCE_EPOCH_LIMIT = 50
    epsilon = 1e-4

    ########## DIRECTORY SETUP ##########

    WEIGHT_DIR = os.path.join("models", "weights")

    MODEL_NAME = "joint_unet" 
    MODEL_PATH = os.path.join(WEIGHT_DIR, MODEL_NAME + ".json")
    HISTORY_PATH = os.path.join(WEIGHT_DIR, MODEL_NAME + "_history.json")

    # files and paths
    for d in [WEIGHT_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)

    model = unet_2D(
        num_channels=1,
        ds=ds,
    )
    INIT_WEIGHT_PATH = os.path.join(WEIGHT_DIR, "init_weights.h5")
    model.save_weights(INIT_WEIGHT_PATH)
    json_string = model.to_json()
    with open(MODEL_PATH, 'w') as f:
        json.dump(json_string, f)

    print(model.summary(line_length=75))


    ######### FIVE FOLD CROSS VALIDATION #########

    TRAIN_SEG_TF_RECORD_FILENAME = os.path.join(
            "data", "seg_dataset_fold_{}_train.tfrecord"
    )
    VAL_SEG_TF_RECORD_FILENAME = os.path.join(
            "data", "seg_dataset_fold_{}_val.tfrecord"
    )

    for cur_fold in range(5):
        ######### MODEL AND CALLBACKS #########
        model.load_weights(INIT_WEIGHT_PATH)
        #opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

        ######### DATA IMPORT #########

        seg_tmp = tf.data.TFRecordDataset(
                TRAIN_SEG_TF_RECORD_FILENAME.format(cur_fold)
            )\
            .map(lambda record: parse_seg_bag(
                record,
                instance_size,
                ))

        print("Counting elements...")
        start_time = time.time()
        for num_seg_elements, data in enumerate(seg_tmp):
            continue
        num_seg_elements += 1

        print("Found {} bags in {}".format(
            num_seg_elements, TRAIN_SEG_TF_RECORD_FILENAME.format(cur_fold))
        )
        print("Using {} as epoch limit".format(
            num_seg_elements)
        )
        print("Took {:.2f}s to count all elements".format(
            time.time() - start_time)
        )

        #augmentations = [flip_dim1, flip_dim2, flip_dim3, rotate_2D]
        augmentations = []

        train_seg_dataset = tf.data.TFRecordDataset(
                TRAIN_SEG_TF_RECORD_FILENAME.format(cur_fold))\
            .map(lambda record: parse_seg_bag(
                record,
                instance_size,
                )
            )\
            .shuffle(buffer_size=100)

        val_seg_dataset = tf.data.TFRecordDataset(
                VAL_SEG_TF_RECORD_FILENAME.format(cur_fold))\
            .map(lambda record: parse_seg_bag(
                record,
                instance_size,
                )
            )

        for f in augmentations:
            train_seg_dataset = train_seg_dataset.map(
                    lambda x, y: 
                    tf.cond(tf.random_uniform([], 0, 1) > 0.75, 
                        lambda: (f(x), y),
                        lambda: (x, y)
                    ), num_parallel_calls=4,)
                    
        ######### TRAINING #########

        best_val_seg_loss = 100000
        convergence_epoch_counter = 0

        seg_grads = [tf.zeros_like(l) for l in model.trainable_variables]

        print()

        best_epoch = 1
        for cur_epoch in range(N_EPOCHS):
            epoch_train_seg_loss = 0
            train_correct = 0

            epoch_val_seg_loss = 0
            epoch_val_seg_acc = 0
            val_correct = 0

            sys.stdout.write("\rEpoch {}/{} [{:{}<{}}]".format(
                cur_epoch + 1, N_EPOCHS, 
                "=" * 0, '-', progbar_length
            ))

            for i, (seg_x, seg_y) in enumerate(train_seg_dataset):
                # Forward pass
                with tf.GradientTape(persistent=True) as tape:
                    seg_out = model(seg_x, training=True)

                    # segmentation gradient
                    seg_loss = dice_coef_loss(
                            seg_out, 
                            seg_y,
                        )
                    grad = tape.gradient(seg_loss, model.trainable_variables)
                    for k in range(len(grad)):
                        seg_grads[k] = running_average(seg_grads[k], grad[k], i + 1)

                epoch_train_seg_loss = running_average(
                        epoch_train_seg_loss,
                        seg_loss.numpy(),
                        i + 1
                )


                # apply gradients per batch
                if (i > 0 and i % batch_size == 0) or i == num_seg_elements - 1:
                    opt.apply_gradients(zip(seg_grads, model.trainable_variables))
                    sys.stdout.write("\rEpoch {}/{} [{:{}<{}}] Seg Loss: {:.4f}".format(
                        cur_epoch + 1, N_EPOCHS,
                        "=" * min(
                            int(progbar_length * (i/num_seg_elements)), 
                            progbar_length),
                        "-",
                        progbar_length,
                        epoch_train_seg_loss,
                    ))
                    sys.stdout.flush()
                    # wipe tape and records
                    grads = [tf.zeros_like(l) for l in model.trainable_variables]
                    del tape

            # validation metrics
            for i, (x, y) in enumerate(val_seg_dataset):
                seg_out = model(seg_x, training=True)

                # segmentation gradient
                seg_loss = dice_coef_loss(
                        seg_out, 
                        seg_y,
                    )

                epoch_val_seg_loss = running_average(
                        epoch_val_seg_loss,
                        seg_loss.numpy(),
                        i + 1
                )

            if convergence_epoch_counter >= CONVERGENCE_EPOCH_LIMIT:
                print("\nCurrent Fold: {}\
                        \nNo improvement in {} epochs, model is converged.\
                        \nModel achieved best val loss at epoch {}.\
                        \nTrain Loss: {:.4f}\
                        \nVal   Loss: {:.4f}".format(
                    cur_fold,
                    CONVERGENCE_EPOCH_LIMIT,
                    best_epoch,
                    epoch_train_seg_loss, 
                    best_val_seg_loss,
                ))
                break

            if epoch_val_seg_loss > best_val_seg_loss and\
                    np.abs(epoch_val_seg_loss - best_val_seg_loss) > epsilon:
                convergence_epoch_counter += 1
            else:
                convergence_epoch_counter = 0

            if epoch_val_seg_loss < best_val_seg_loss:
                best_epoch = cur_epoch + 1
                best_val_seg_loss = epoch_val_seg_loss
                best_val_seg_acc = epoch_val_seg_acc
                model.save_weights(os.path.join(
                    WEIGHT_DIR, "best_weights_fold_{}.h5".format(cur_fold))
                )

            sys.stdout.write(" Val Loss: {:.4f}".format(
                epoch_val_seg_loss,
            ))
