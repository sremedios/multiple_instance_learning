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
from models.resnet import *

def running_average(old_average, cur_val, n):
    return old_average * (n-1)/n + cur_val/n

def mil_prediction(pred, n=1):
    # generalized, returns top n 
    i = tf.argsort(pred[:, 1], axis=0)
    i = i[len(pred) - n : len(pred)]

    return (tf.gather(pred, i), i)

os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == "__main__":

    opts = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.98)
    conf = tf.compat.v1.ConfigProto(gpu_options=opts)
    tf.compat.v1.enable_eager_execution(config=conf)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    
    ########## HYPERPARAMETER SETUP ##########

    num_training_samples = int(sys.argv[1])

    N_EPOCHS = 10000
    batch_size = 128
    minibatch_size = 32
    ds = 4
    instance_size = (512, 512)
    num_classes = 2
    learning_rate = 1e-4
    progbar_length = 10
    CONVERGENCE_EPOCH_LIMIT = 50
    epsilon = 1e-4

    ########## DIRECTORY SETUP ##########

    #MODEL_NAME = "class_unet" 
    MODEL_NAME = "class_resnet" 
    WEIGHT_DIR = os.path.join(
            "models", 
            "weights", 
            MODEL_NAME, 
            "dataset_{}".format(num_training_samples)
    )

    RESULTS_DIR = os.path.join(
            "results",
            "dataset_{}".format(num_training_samples),
    )            

    # files and paths
    for d in [WEIGHT_DIR, RESULTS_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)

    MODEL_PATH = os.path.join(WEIGHT_DIR, MODEL_NAME + ".json")
    HISTORY_PATH = os.path.join(WEIGHT_DIR, MODEL_NAME + "_history.json")


    # Actual instantiation happens for each fold
    #model = class_unet_2D(
        #num_channels=1,
        #num_classes=num_classes,
        #ds=ds,
    #)
    model = resnet(num_classes=num_classes, ds=ds)
    INIT_WEIGHT_PATH = os.path.join(WEIGHT_DIR, "init_weights.h5")
    model.save_weights(INIT_WEIGHT_PATH)
    json_string = model.to_json()
    with open(MODEL_PATH, 'w') as f:
        json.dump(json_string, f)

    print(model.summary(line_length=75))


    ######### FIVE FOLD CROSS VALIDATION #########

    TRAIN_CURVE_FILENAME = os.path.join(
            RESULTS_DIR, "training_curve_fold_{}.csv"
    )

    TRAIN_TF_RECORD_FILENAME = os.path.join(
            os.sep, "home", "remedis", "data", "dataset_fold_{}_train.tfrecord"
    )
    VAL_TF_RECORD_FILENAME = os.path.join(
            os.sep, "home", "remedis", "data", "dataset_fold_{}_val.tfrecord"
    )

    for cur_fold in range(5):

        # Already did fold 1; skipping whenever it occurs
        if cur_fold == 1:
            continue
        with open(TRAIN_CURVE_FILENAME.format(cur_fold), 'w') as f:
            f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")

        ######### MODEL AND CALLBACKS #########
        model.load_weights(INIT_WEIGHT_PATH)
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

        ######### DATA IMPORT #########

        '''
        tmp = tf.data.TFRecordDataset(TRAIN_TF_RECORD_FILENAME.format(cur_fold))\
            .map(lambda record: parse_bag(
                record,
                instance_size,
                num_labels=num_classes))
        for num_elements, data in enumerate(tmp):
            continue

        num_elements += 1
        '''
        #num_elements = 672
        num_elements = num_training_samples

        print("\nFound {} bags in {}".format(
            num_elements, TRAIN_TF_RECORD_FILENAME.format(cur_fold))
        )

        augmentations = [flip_dim1, flip_dim2, rotate_2D]
        #augmentations = [rotate_2D, ]

        train_dataset = tf.data.TFRecordDataset(
                TRAIN_TF_RECORD_FILENAME.format(cur_fold))\
            .map(lambda record: parse_bag(
                record,
                instance_size,
                num_labels=num_classes))\
            .take(num_training_samples)\
            .shuffle(100)\

        val_dataset = tf.data.TFRecordDataset(
                VAL_TF_RECORD_FILENAME.format(cur_fold))\
            .map(lambda record: parse_bag(
                record,
                instance_size,
                num_labels=num_classes))\

        for f in augmentations:
            train_dataset = train_dataset.map(
                    lambda x, y: 
                    tf.cond(tf.random_uniform([], 0, 1) > 0.9, 
                        lambda: (f(x), y),
                        lambda: (x, y)
                    ), num_parallel_calls=4,)
                    

        ######### TRAINING #########

        best_val_loss = 100000
        best_val_acc = 0
        convergence_epoch_counter = 0

        grads = [tf.zeros_like(l) for l in model.trainable_variables]

        print()

        best_epoch = 1
        for cur_epoch in range(N_EPOCHS):
            epoch_train_loss = 0
            epoch_train_acc = 0
            train_correct = 0

            epoch_val_loss = 0
            epoch_val_acc = 0
            val_correct = 0

            sys.stdout.write("\rEpoch {}/{} [{:{}<{}}]".format(
                cur_epoch + 1, N_EPOCHS, 
                "=" * 0, '-', progbar_length
            ))


            for i, (x, y) in enumerate(train_dataset):
                N_INSTANCES = 1
                N_INSTANCES = min(N_INSTANCES, len(x))

                # Forward pass
                with tf.GradientTape(persistent=True) as tape:
                    # minibatches of forward pass for GPU memory constraints
                    logits = []
                    for minibatch in range(len(x)//minibatch_size + 1):
                        start = minibatch * minibatch_size
                        end = min(len(x), minibatch * minibatch_size + minibatch_size)
                        minibatch_logits = model(x[start:end], training=True)
                        for minibatch_logit in minibatch_logits:
                            logits.append(minibatch_logit)
                    logits = tf.convert_to_tensor(logits)

                    pred, top_idx = mil_prediction(tf.nn.softmax(logits), n=N_INSTANCES)

                    repeated_y = tf.reshape(
                            tf.tile(y, [N_INSTANCES]), 
                            (N_INSTANCES, len(y))
                    )

                    if repeated_y.shape != pred.shape:
                        print("\nShape mismatch:\
                                \nPred shape: {}\
                                \nRepeated Y shape: {}".format(
                                    pred.shape,
                                    repeated_y.shape
                                    ))
                        continue

                    loss = tf.losses.softmax_cross_entropy(
                            onehot_labels=repeated_y,
                            logits=tf.gather(logits, top_idx),
                            reduction=tf.losses.Reduction.NONE,
                    )
                    loss = tf.reduce_mean(loss)
                    grad = tape.gradient(loss, model.trainable_variables)
                    # aggregate current element in batch
                    for k in range(len(grad)):
                        grads[k] = running_average(grads[k], grad[k], i + 1)

                # bag-wise metrics 
                if tf.argmax(tf.reduce_mean(pred, axis=0)).numpy() == tf.argmax(y).numpy():
                    train_correct += 1
                epoch_train_loss = running_average(epoch_train_loss, loss.numpy(), i + 1)

                # epoch-wise accuracy
                # We can calculate this as we go for metric printing
                epoch_train_acc = train_correct / (i + 1)


                # apply gradients per batch
                if (i > 0 and i % batch_size == 0) or i == num_elements - 1:
                    opt.apply_gradients(zip(grads, model.trainable_variables))
                    sys.stdout.write("\rEpoch {}/{} [{:{}<{}}] Loss: {:.4f} Acc: {:.2%}"\
                            .format(
                        cur_epoch + 1, N_EPOCHS,
                        "=" * min(int(progbar_length*(i/num_elements)), progbar_length),
                        "-",
                        progbar_length,
                        epoch_train_loss,
                        epoch_train_acc,
                    ))
                    sys.stdout.flush()
                    # wipe tape and records
                    grads = [tf.zeros_like(l) for l in model.trainable_variables]
                    del tape


            # validation metrics
            for i, (x, y) in enumerate(val_dataset):
                N_INSTANCES = 1
                N_INSTANCES = min(N_INSTANCES, len(x))

                # minibatches of forward pass for GPU memory constraints
                logits = []
                for minibatch in range(len(x)//minibatch_size + 1):
                    start = minibatch * minibatch_size
                    end = min(len(x), minibatch * minibatch_size + minibatch_size)
                    minibatch_logits = model(x[start:end], training=False)
                    for minibatch_logit in minibatch_logits:
                        logits.append(minibatch_logit)
                logits = tf.convert_to_tensor(logits)

                pred, top_idx = mil_prediction(tf.nn.softmax(logits), n=N_INSTANCES)


                repeated_y = tf.reshape(
                        tf.tile(y, [N_INSTANCES]), 
                        (N_INSTANCES, len(y))
                )

                loss = tf.losses.softmax_cross_entropy(
                        onehot_labels=repeated_y,
                        logits=tf.gather(logits, top_idx),
                        reduction=tf.losses.Reduction.NONE,
                )
                loss = tf.reduce_mean(loss)
                    
                # bag-wise metrics 
                if tf.argmax(tf.reduce_mean(pred, axis=0)).numpy() == tf.argmax(y).numpy():
                    val_correct += 1
                epoch_val_loss = running_average(epoch_val_loss, loss.numpy(), i + 1)

                # epoch-wise accuracy
                # We can calculate this as we go for metric printing
                epoch_val_acc = val_correct / (i + 1)

            with open(TRAIN_CURVE_FILENAME.format(cur_fold), 'a') as f:
                f.write("{},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(
                    cur_epoch + 1,
                    epoch_train_loss,
                    epoch_train_acc,
                    epoch_val_loss,
                    epoch_val_acc,
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
                    epoch_train_loss, epoch_train_acc,
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

