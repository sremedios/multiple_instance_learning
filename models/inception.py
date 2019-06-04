from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, concatenate, GlobalAveragePooling2D, add
from keras.layers.merge import Concatenate
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
import tensorflow as tf

from .losses import *

from .multi_gpu import ModelMGPU
import json


def get_inception_layer(prev_layer, ds=2):
    a = Conv2D(64//ds, (1,1), strides=(1,1), activation='relu', padding='same')(prev_layer)
    
    b = Conv2D(96//ds, (1,1), strides=(1,1), activation='relu', padding='same')(prev_layer)
    b = Conv2D(128//ds, (3,3), strides=(1,1), activation='relu', padding='same')(b)
    
    c = Conv2D(16//ds, (1,1), strides=(1,1), activation='relu', padding='same')(prev_layer)
    c = Conv2D(32//ds, (5,5), strides=(1,1), activation='relu', padding='same')(c)

    d = AveragePooling2D((3,3), strides=(1,1), padding='same')(prev_layer)
    d = Conv2D(32//ds, (1,1), strides=(1,1), activation='relu', padding='same')(d)
    
    e = MaxPooling2D((3,3), strides=(1,1), padding='same')(prev_layer)
    e = Conv2D(32//ds, (1,1), strides=(1,1), activation='relu', padding='same')(e)

    out_layer = concatenate([a,b,c,d,e],axis=-1)

    return out_layer

def inception(model_path,
              num_channels=1,
              loss="binary_crossentropy",
              ds=2,
              lr=1e-4,
              num_gpus=1,
              verbose=0):

    inputs = Input((None, None, num_channels))

    x = Conv2D(64//ds, (9,9), strides=(1,1), activation='relu', padding='same')(inputs)
    x = Conv2D(64//ds, (7,7), strides=(1,1), activation='relu', padding='same')(x)

    x = get_inception_layer(x, ds)
    x = get_inception_layer(x, ds)
    x = get_inception_layer(x, ds)

    x = Conv2D(32//ds, (5,5), strides=(1,1), activation='relu', padding='same')(x)
    x = Conv2D(16//ds, (3,3), strides=(1,1), activation='relu', padding='same')(x)
    x = Conv2D(1, (1,1), strides=(1,1), activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=x)

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

def get_inception_layer_3D(prev_layer, ds=2):
    a = Conv2D(64//ds, (1,1), strides=(1,1), activation='relu', padding='same')(prev_layer)
    
    b = Conv2D(96//ds, (1,1), strides=(1,1), activation='relu', padding='same')(prev_layer)
    b = Conv2D(128//ds, (3,3), strides=(1,1), activation='relu', padding='same')(b)
    
    c = Conv2D(16//ds, (1,1), strides=(1,1), activation='relu', padding='same')(prev_layer)
    c = Conv2D(32//ds, (5,5), strides=(1,1), activation='relu', padding='same')(c)

    d = AveragePooling2D((3,3), strides=(1,1), padding='same')(prev_layer)
    d = Conv2D(32//ds, (1,1), strides=(1,1), activation='relu', padding='same')(d)
    
    e = MaxPooling2D((3,3), strides=(1,1), padding='same')(prev_layer)
    e = Conv2D(32//ds, (1,1), strides=(1,1), activation='relu', padding='same')(e)

    out_layer = concatenate([a,b,c,d,e],axis=-1)

    return out_layer

def inception(model_path,
              num_channels=1,
              loss="binary_crossentropy",
              ds=2,
              lr=1e-4,
              num_gpus=1,
              verbose=0):

    inputs = Input((None, None, num_channels))

    x = Conv2D(64//ds, (9,9), strides=(1,1), activation='relu', padding='same')(inputs)
    x = Conv2D(64//ds, (7,7), strides=(1,1), activation='relu', padding='same')(x)

    x = get_inception_layer(x, ds)
    x = get_inception_layer(x, ds)
    x = get_inception_layer(x, ds)

    x = Conv2D(32//ds, (5,5), strides=(1,1), activation='relu', padding='same')(x)
    x = Conv2D(16//ds, (3,3), strides=(1,1), activation='relu', padding='same')(x)
    x = Conv2D(1, (1,1), strides=(1,1), activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=x)

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
