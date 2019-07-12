import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
import json

import tensorflow as tf

from utils.augmentations import *
from utils.tfrecord_utils import *
from models.joint_unet import *

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
    batch_size = 96
    minibatch_size = 16
    ds = 4
    instance_size = (128, 128)
    num_classes = 2
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

    model = joint_unet(
        shared_weights,
        num_classes=num_classes,
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

    TRAIN_CLASS_TF_RECORD_FILENAME = os.path.join(
            "data", "dataset_fold_{}_train.tfrecord"
    )
    VAL_CLASS_TF_RECORD_FILENAME = os.path.join(
            "data", "dataset_fold_{}_val.tfrecord"
    )

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

        class_tmp = tf.data.TFRecordDataset(
                TRAIN_CLASS_TF_RECORD_FILENAME.format(cur_fold)
            )\
            .map(lambda record: parse_bag(
                record,
                instance_size,
                num_labels=num_classes))
        for num_class_elements, data in enumerate(class_tmp):
            continue
        num_class_elements += 1

        seg_tmp = tf.data.TFRecordDataset(
                TRAIN_SEG_TF_RECORD_FILENAME.format(cur_fold)
            )\
            .map(lambda record: parse_bag(
                record,
                instance_size,
                num_labels=num_seges))
        for num_seg_elements, data in enumerate(seg_tmp):
            continue
        num_seg_elements += 1

        print("\nFound {} bags in {}".format(
            num_class_elements, TRAIN_CLASS_TF_RECORD_FILENAME.format(cur_fold))
        )
        print("Found {} bags in {}".format(
            num_seg_elements, TRAIN_SEG_TF_RECORD_FILENAME.format(cur_fold))
        )
        print("Using {} as epoch limit".format(
            num_seg_elements
        )

        augmentations = [flip_dim1, flip_dim2, flip_dim3, rotate_2D]

        train_class_dataset = tf.data.TFRecordDataset(
                TRAIN_CLASS_TF_RECORD_FILENAME.format(cur_fold))\
            .map(lambda record: parse_bag(
                record,
                instance_size,
                num_labels=num_classes))\
            .shuffle(buffer_size=100)

        val_class_dataset = tf.data.TFRecordDataset(
                VAL_CLASS_TF_RECORD_FILENAME.format(cur_fold))\
            .map(lambda record: parse_bag(
                record,
                instance_size,
                num_labels=num_classes))\

        train_seg_dataset = tf.data.TFRecordDataset(
                TRAIN_CLASS_TF_RECORD_FILENAME.format(cur_fold))\
            .map(lambda record: parse_bag(
                record,
                instance_size,
                num_labels=num_seges))\
            .shuffle(buffer_size=100)

        val_seg_dataset = tf.data.TFRecordDataset(
                VAL_CLASS_TF_RECORD_FILENAME.format(cur_fold))\
            .map(lambda record: parse_bag(
                record,
                instance_size,
                num_labels=num_seges))\

        for f in augmentations:
            train_class_dataset = train_class_dataset.map(
                    lambda x, y: 
                    tf.cond(tf.random_uniform([], 0, 1) > 0.75, 
                        lambda: (f(x), y),
                        lambda: (x, y)
                    ), num_parallel_calls=4,)

            train_seg_dataset = train_seg_dataset.map(
                    lambda x, y: 
                    tf.cond(tf.random_uniform([], 0, 1) > 0.75, 
                        lambda: (f(x), y),
                        lambda: (x, y)
                    ), num_parallel_calls=4,)

        train_joint_dataset = tf.data.Dataset.zip(
                (train_seg_dataset, train_class_dataset)
        )
        val_joint_dataset = tf.data.Dataset.zip(
                (val_seg_dataset, val_class_dataset)
        )
                    

        ######### TRAINING #########

        best_val_loss = 100000
        best_val_acc = 0
        convergence_epoch_counter = 0

        seg_grads = [tf.zeros_like(l) for l in model.trainable_variables]
        class_grads = [tf.zeros_like(l) for l in model.trainable_variables]

        print()

        best_epoch = 1
        for cur_epoch in range(N_EPOCHS):
            epoch_train_class_loss = 0
            epoch_train_seg_loss = 0
            epoch_train_class_acc = 0
            train_correct = 0

            epoch_val_loss = 0
            epoch_val_acc = 0
            val_correct = 0

            #print("\nEpoch {}/{}".format(cur_epoch + 1, N_EPOCHS))
            sys.stdout.write("\rEpoch {}/{} [{:{}<{}}]".format(
                cur_epoch + 1, N_EPOCHS, 
                "=" * 0, '-', progbar_length
            ))

            for i, ((seg_x, seg_y), (class_x, class_y)) in enumerate(train_dataset):
                # Forward pass
                with tf.GradientTape(persistent=True) as tape:
                    seg_out, class_logits = model((seg_x, class_x), training=True)

                    # MIL for class network
                    pred, top_idx = mil_prediction(tf.nn.softmax(logits))

                    class_loss = tf.losses.softmax_cross_entropy(
                            onehot_labels=y,
                            logits=logits[top_idx],
                            reduction=tf.losses.Reduction.NONE,
                        )
                    grad = tape.gradient(class_loss, model.trainable_variables)
                    for k in range(len(grad)):
                        class_grads[k] = running_average(class_grads[k], grad[k], i + 1)

                    # segmentation gradient
                    seg_loss = continuous_dice_coef_loss(
                            seg_out, 
                            seg_y,
                        )
                    grad = tape.gradient(seg_loss, model.trainable_variables)
                    for k in range(len(grad)):
                        seg_grads[k] = running_average(seg_grads[k], grad[k], i + 1)

                # bag-wise metrics 
                if tf.argmax(pred).numpy() == tf.argmax(y).numpy():
                    train_correct += 1
                epoch_train_class_loss = running_average(
                        epoch_train_class_loss, 
                        class_loss.numpy(), 
                        i + 1
                )
                epoch_train_seg_loss = running_average(
                        epoch_train_seg_loss,
                        seg_loss.numpy(),
                        i + 1
                )

                # epoch-wise accuracy
                # We can calculate this as we go for metric printing
                epoch_train_class_acc = train_correct / (i + 1)

                # apply gradients per batch
                if (i > 0 and i % batch_size == 0) or i == num_class_elements - 1:
                    opt.apply_gradients(zip(seg_grads, model.trainable_variables))
                    opt.apply_gradients(zip(class_grads, model.trainable_variables))
                    sys.stdout.write("\rEpoch {}/{} [{:{}<{}}] \
                            Class Loss: {:.4f} \
                            Seg Loss: {:.4f} \
                            Acc: {:.2%}".format(
                        cur_epoch + 1, N_EPOCHS,
                        "=" * min(
                            int(progbar_length * (i/num_class_elements)), 
                            progbar_length),
                        "-",
                        progbar_length,
                        epoch_train_class_loss,
                        epoch_train_seg_loss * -1,
                        epoch_train_class_acc,
                    ))
                    sys.stdout.flush()
                    # wipe tape and records
                    grads = [tf.zeros_like(l) for l in model.trainable_variables]
                    del tape


            # validation metrics
            for i, (x, y) in enumerate(val_dataset):
                logits = model(x)
                pred, top_idx = mil_prediction(tf.nn.softmax(logits))

                # keep track of avg loss
                loss = tf.losses.softmax_cross_entropy(
                        onehot_labels=y,
                        logits=logits[top_idx],
                        reduction=tf.losses.Reduction.NONE,
                    )

                # bag-wise loss
                epoch_val_loss = running_average(epoch_val_loss, loss.numpy(), i + 1)

                if tf.argmax(pred).numpy() == tf.argmax(y).numpy():
                    val_correct += 1
            # calculate accuracy after "val epoch" is done
            epoch_val_acc = val_correct / (i + 1)

            if convergence_epoch_counter >= CONVERGENCE_EPOCH_LIMIT:
                print("\nCurrent Fold: {}\
                        \nNo improvement in {} epochs, model is converged.\
                        \nModel achieved best val loss at epoch {}.\
                        \nTrain Loss: {:.4f} Train Acc: {:.2%}\
                        \nVal   Loss: {:.4f} Val   Acc: {:.2%}".format(
                    cur_fold,
                    CONVERGENCE_EPOCH_LIMIT,
                    best_epoch,
                    epoch_train_class_loss, epoch_train_class_acc,
                    best_val_loss, best_val_acc,
                ))
                break

            if epoch_val_loss > best_val_loss and np.abs(epoch_val_loss - best_val_loss) > epsilon:
                convergence_epoch_counter += 1
            else:
                convergence_epoch_counter = 0

            if epoch_val_loss < best_val_loss:
                best_epoch = cur_epoch + 1
                best_val_loss = epoch_val_loss
                best_val_acc = epoch_val_acc
                model.save_weights(os.path.join(
                    WEIGHT_DIR, "best_weights_fold_{}.h5".format(cur_fold))
                )

            sys.stdout.write(" Val Loss: {:.4f} Val Acc: {:.2%}".format(
                epoch_val_loss,
                epoch_val_acc
            ))

