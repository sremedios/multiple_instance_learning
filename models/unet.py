from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, concatenate,\
    GlobalAveragePooling2D, add, UpSampling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
import tensorflow as tf

from .losses import *

from .multi_gpu import ModelMGPU
import json


def unet(model_path,
         num_channels,
         loss="binary_crossentropy",
         ds=2,
         lr=1e-4,
         num_gpus=1,
         verbose=0,):
    inputs = Input((None, None, num_channels))

    conv1 = Conv2D(64//ds, 3, activation='relu', padding='same', )(inputs)
    conv1 = Conv2D(64//ds, 3, activation='relu', padding='same', )(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128//ds, 3, activation='relu', padding='same',)(pool1)
    conv2 = Conv2D(128//ds, 3, activation='relu', padding='same', )(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256//ds, 3, activation='relu', padding='same', )(pool2)
    conv3 = Conv2D(256//ds, 3, activation='relu', padding='same', )(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512//ds, 3, activation='relu', padding='same', )(pool3)
    conv4 = Conv2D(512//ds, 3, activation='relu', padding='same', )(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024//ds, 3, activation='relu', padding='same', )(pool4)
    conv5 = Conv2D(1024//ds, 3, activation='relu', padding='same', )(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512//ds, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512//ds, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv2D(512//ds, 3, activation='relu', padding='same')(conv6)

    up7 = Conv2D(256//ds, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256//ds, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv2D(256//ds, 3, activation='relu', padding='same')(conv7)

    up8 = Conv2D(128//ds, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128//ds, 3, activation='relu', padding='same')(merge8)
    conv8 = Conv2D(128//ds, 3, activation='relu', padding='same')(conv8)

    up9 = Conv2D(64//ds, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64//ds, 3, activation='relu', padding='same')(merge9)
    conv9 = Conv2D(64//ds, 3, activation='relu', padding='same')(conv9)

    conv9 = Conv2D(2, 3, activation='relu', padding='same', )(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    # dice as a human-readble metric
    model.compile(optimizer=Adam(lr=lr),
                  metrics=[dice_coef],
                  loss=loss)

    # save json before checking if multi-gpu
    json_string = model.to_json()
    with open(model_path, 'w') as f:
        json.dump(json_string, f)

    if verbose:
        print(model.summary())

    # recompile if multi-gpu model
    if num_gpus > 1:
        model = ModelMGPU(model, num_gpus)
        model.compile(optimizer=Adam(lr=lr),
                      metrics=[dice_coef],
                      loss=loss)

    return model
