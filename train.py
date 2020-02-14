import numpy as np
import os
import sys
import json

from pathlib import Path

import tensorflow as tf
from tqdm import tqdm

from utils.augmentations import *
from utils.tfrecord_utils import *
from utils.pad import *
from models.resnet import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def running_average(old_average, cur_val, n):
    return old_average * (n-1)/n + cur_val/n

def mil_prediction(pred, n=1):
    i = tf.argsort(pred[:, 1], axis=0)
    i = i[len(pred) - n : len(pred)]
    return (tf.gather(pred, i), i)

def show_progbar(cur_epoch, total_epochs, cur_step, num_instances, loss, acc, color_code):
    TEMPLATE = "\r{}Epoch {}/{} [{:{}<{}}] Loss: {:>3.4f} Acc: {:>3.2%}\033[0;0m"
    progbar_length = 20

    sys.stdout.write(TEMPLATE.format(
        color_code,
        cur_epoch,
        total_epochs,
        "=" * min(int(progbar_length*(cur_step/num_instances)), progbar_length),
        "-",
        progbar_length,
        loss,
        acc,
    ))
    sys.stdout.flush()

def step_bag_gradient(inputs, model):
    x, y = inputs

    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        pred, top_idx = mil_prediction(tf.nn.softmax(logits), n=1)
        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.reshape(tf.tile(y, [1]), (1, len(y))),
            logits=tf.gather(logits, top_idx),
        )
        loss = tf.reduce_mean(loss)

    grad = tape.gradient(loss, model.trainable_variables)

    return grad, loss, pred

def step_bag_val(inputs, model):
    x, y = inputs
        
    logits = model(x, training=True)
    pred, top_idx = mil_prediction(tf.nn.softmax(logits), n=1)
    loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.reshape(tf.tile(y, [1]), (1, len(y))),
        logits=tf.gather(logits, top_idx),
    )
    loss = tf.reduce_mean(loss)

    return loss, pred



if __name__ == "__main__":

    print(
            ("\n\n***WARNING***\n")
            ("Please remove these lines from the source code before running.\n")
            ("Please also change all filenames provided to suit your directory ")
            ("structure and all hyperparameters for your specific problem.\n")
            ("Thank you.\n\n")
        )
    sys.exit()


    if len(sys.argv) < 2:
        print(
                ("Missing cmd line arguments; please first run `make_tfrecord.py`\n")
                ("First argument: path to TFRecord directory")
             )
        sys.exit()

    ########## HYPERPARAMETER SETUP ##########

    N_EPOCHS = 10000
    BATCH_SIZE = 2**7
    BUFFER_SIZE = 2**2
    ds = 4
    instance_size = (512, 512)
    num_classes = 2
    learning_rate = 1e-4
    train_color_code = "\033[0;32m"
    val_color_code = "\033[0;36m"
    CONVERGENCE_EPOCH_LIMIT = 50
    epsilon = 1e-4

    ########## DIRECTORY SETUP ##########


    MODEL_NAME = "resnetish_ds_{}".format(ds)
    WEIGHT_DIR = Path("models/weights") / MODEL_NAME
    RESULTS_DIR = Path("results") / MODEL_NAME
    DATA_DIR = Path(sys.argv[1])
    NUM_INSTANCES_FILE = DATA_DIR / "count.txt"
    with open(NUM_INSTANCES_FILE, 'r') as f:
        lines = [l.strip() for l in f.readlines()]
    num_instances = {}
    for l in lines:
        cur_name, cur_fold, n = l.split()
        num_instances["{}_{}".format(cur_name, cur_fold)] = int(n)

    # files and paths
    for d in [WEIGHT_DIR, RESULTS_DIR]:
        if not d.exists():
            d.mkdir(parents=Path('.'))

    MODEL_PATH = WEIGHT_DIR / (MODEL_NAME + ".json")
    HISTORY_PATH = WEIGHT_DIR / (MODEL_NAME + "_history.json")

    # Actual instantiation happens for each fold
    model = resnet(num_classes=num_classes, ds=ds)
    
    INIT_WEIGHT_PATH = WEIGHT_DIR / "init_weights.h5"
    model.save_weights(str(INIT_WEIGHT_PATH))
    json_string = model.to_json()
    with open(str(MODEL_PATH), 'w') as f:
        json.dump(json_string, f)

    print(model.summary(line_length=75))

    ######### FIVE FOLD CROSS VALIDATION #########

    TRAIN_CURVE_FILENAME = RESULTS_DIR / "training_curve_fold_{}.csv"

    TRAIN_TF_RECORD_FILENAME = DATA_DIR / "dataset_fold_{}_train.tfrecord"
    VAL_TF_RECORD_FILENAME = DATA_DIR / "dataset_fold_{}_val.tfrecord"
    
    for cur_fold in range(5):

        with open(str(TRAIN_CURVE_FILENAME).format(cur_fold), 'w') as f:
            f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")

        ######### MODEL AND CALLBACKS #########
        model.load_weights(str(INIT_WEIGHT_PATH))
        opt = tf.optimizers.Adam(learning_rate=learning_rate)

        ######### DATA IMPORT #########
        train_dataset = tf.data.TFRecordDataset(
                str(TRAIN_TF_RECORD_FILENAME).format(cur_fold))\
            .map(lambda record: parse_into_volume(
                record,
                instance_size,
                num_labels=num_classes))\
            .take(num_training_samples)\
            .shuffle(BUFFER_SIZE)\


        val_dataset = tf.data.TFRecordDataset(
                str(VAL_TF_RECORD_FILENAME).format(cur_fold))\
            .map(lambda record: parse_into_volume(
                record,
                instance_size,
                num_labels=num_classes))

        augmentations = [flip_dim1, flip_dim2, rotate_2D]
        for f in augmentations:
            train_dataset = train_dataset.map(
                lambda x, y:
                tf.cond(tf.random_uniform([], 0, 1) > 0.9, # with 90% chance, call first `lambda`:
                    lambda: (f(x), y),  # apply augmentation `f`, don't touch `y`
                    lambda: (x, y),     # don't apply any aug
                ), num_parallel_calls=4,
            )

        # metrics
        train_accuracy = tf.keras.metrics.Accuracy(name='train_acc')
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        val_accuracy = tf.keras.metrics.Accuracy(name='val_acc')
        val_loss = tf.keras.metrics.Mean(name='val_loss')

        ######### TRAINING #########
        best_val_loss = 100000
        best_val_acc = 0
        convergence_epoch_counter = 0

        print()

        grads = [tf.zeros_like(l) for l in model.trainable_variables]

        train_n = num_instances["train_{}".format(cur_fold)]
        val_n = num_instances["val_{}".format(cur_fold)]

        best_epoch = 1
        for cur_epoch in range(N_EPOCHS):

            print("\n{}Training...\033[0;0m".format(train_color_code))
            for i, (x, y) in enumerate(train_dataset):
                grad, loss, pred = step_bag_gradient((x,y), model)
                for g in range(len(grads)):
                    grads[g] = running_average(grads[g], grad[g], i + 1)

                train_accuracy.update_state(
                    tf.argmax(tf.convert_to_tensor([y]), axis=1), 
                    tf.argmax(pred, axis=1),
                )
                train_loss.update_state(loss)

                if (i+1)%BATCH_SIZE==0 or (i+1)==train_n:
                    opt.apply_gradients(zip(grads, model.trainable_variables))

                    show_progbar(
                        cur_epoch + 1,
                        N_EPOCHS,
                        (i + 1),
                        train_n,
                        train_loss.result(),
                        train_accuracy.result(),
                        train_color_code,
                    )




            # validation metrics
            print("\n{}Validating...\033[0;0m".format(val_color_code))
            for i, (x, y) in enumerate(val_dataset):
                loss, pred = step_bag_val((x,y), model)

                val_accuracy.update_state(
                    tf.argmax(tf.convert_to_tensor([y]), axis=1), 
                    tf.argmax(pred, axis=1),
                )
                val_loss.update_state(loss)

                if (i+1)%BATCH_SIZE==0 or (i+1)==val_n:
                    show_progbar(
                        cur_epoch + 1,
                        N_EPOCHS,
                        (i + 1),
                        val_n,
                        val_loss.result(),
                        val_accuracy.result(),
                        val_color_code,
                    )

            

            with open(str(TRAIN_CURVE_FILENAME).format(cur_fold), 'a') as f:
                f.write("{},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(
                    cur_epoch + 1,
                    train_loss.result(),
                    train_accuracy.result(),
                    val_loss.result(),
                    val_accuracy.result(),
                ))


            if convergence_epoch_counter >= CONVERGENCE_EPOCH_LIMIT:
                print("\nCurrent Fold: {}\
                        \nNo improvement in {} epochs, model is converged.\
                        \nModel achieved best val loss at epoch {}.\
                        \nTrain Loss: {:.4f} Train Acc: {:.2%}\
                        \nVal   Loss: {:.4f} Val   Acc: {:.2%}".format(
                    cur_fold,
                    CONVERGENCE_EPOCH_LIMIT,
                    best_epoch,
                    train_loss.result(), 
                    train_accuracy.result(),
                    val_loss.result(), 
                    val_accuracy.result(),
                ))
                break

            if val_loss.result() > best_val_loss and\
                    np.abs(val_loss.result() - best_val_loss) > epsilon:
                convergence_epoch_counter += 1
            else:
                convergence_epoch_counter = 0

            if val_loss.result() < best_val_loss:
                best_epoch = cur_epoch + 1
                best_val_loss = val_loss.result() 
                best_val_acc = val_accuracy.result()
                model.save_weights(
                    str(WEIGHT_DIR / "best_weights_fold_{}.h5".format(cur_fold))
                )
