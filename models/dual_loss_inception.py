from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, concatenate, GlobalAveragePooling2D, add
from keras.layers.merge import Concatenate
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.models import Model
from keras import backend as K
from .losses import *
import tensorflow as tf


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

def base_inception(ds=2):
    conv1 = Conv2D(64//ds, (3,3), strides=(1,1), activation='relu', padding='same')
    conv2 = Conv2D(64//ds, (3,3), strides=(1,1), activation='relu', padding='same')(conv1)

    incep1 = get_inception_layer(conv2, ds=ds)
    incep2 = get_inception_layer(incep1, ds=ds)
    incep3 = get_inception_layer(incep2, ds=ds)

    conv3 = Conv2D(32//ds, (5,5), strides=(1,1), activation='relu', padding='same')(incep3)
    conv4 = Conv2D(16//ds, (3,3), strides=(1,1), activation='relu', padding='same')(conv3)
    conv5 = Conv2D(1, (1,1), strides=(1,1), activation='sigmoid')(conv4)

    return conv5


def multi_inception(num_channels, healthy_loss, unhealthy_loss, ds=2, lr=1e-4):

    inception_root = base_inception(ds=ds)

    unhealthy_input = Input((None, None, num_channels), name='unhealthy_input')
    unhealthy_output = inception_root(unhealthy_input)
    unhealthy_output = Activation('linear', name='unhealthy_output')(unhealthy_output)

    healthy_input = Input((None, None, num_channels), name='healthy_input')
    healthy_output = inception_root(healthy_input)
    healthy_output = Activation('linear', name='healthy_output')(healthy_output)

    model = Model(inputs=[unhealthy_input, healthy_input], 
                  outputs=[unhealthy_output, healthy_output])

    # binary crossentropy with dice as a metric 
    model.compile(optimizer=Adam(lr=lr),
                  metrics=[continuous_dice_coef],
                  loss={'unhealthy_output':unhealthy_loss, 
                        'healthy_output':healthy_loss}, 
                  loss_weights={'unhealthy_output':1,
                                'healthy_output':0.75,})

    return model


def inception(num_channels, healthy_loss, unhealthy_loss, ds=2, lr=1e-4):

    # instantiate layers to share inputs
    conv1 = Conv2D(64//ds, (3,3), strides=(1,1), activation='relu', padding='same')
    conv2 = Conv2D(64//ds, (3,3), strides=(1,1), activation='relu', padding='same')

    inception1_a = Conv2D(64//ds, (1,1), strides=(1,1), activation='relu', padding='same')
    inception1_b = Conv2D(96//ds, (1,1), strides=(1,1), activation='relu', padding='same')
    inception1_b_2 = Conv2D(128//ds, (3,3), strides=(1,1), activation='relu', padding='same')
    inception1_c = Conv2D(16//ds, (1,1), strides=(1,1), activation='relu', padding='same')
    inception1_c_2 = Conv2D(32//ds, (5,5), strides=(1,1), activation='relu', padding='same')
    inception1_d = AveragePooling2D((3,3), strides=(1,1), padding='same')
    inception1_d_2 = Conv2D(32//ds, (1,1), strides=(1,1), activation='relu', padding='same')
    inception1_e = MaxPooling2D((3,3), strides=(1,1), padding='same')
    inception1_e_2 = Conv2D(32//ds, (1,1), strides=(1,1), activation='relu', padding='same')

    inception2_a = Conv2D(64//ds, (1,1), strides=(1,1), activation='relu', padding='same')
    inception2_b = Conv2D(96//ds, (1,1), strides=(1,1), activation='relu', padding='same')
    inception2_b_2 = Conv2D(128//ds, (3,3), strides=(1,1), activation='relu', padding='same')
    inception2_c = Conv2D(16//ds, (1,1), strides=(1,1), activation='relu', padding='same')
    inception2_c_2 = Conv2D(32//ds, (5,5), strides=(1,1), activation='relu', padding='same')
    inception2_d = AveragePooling2D((3,3), strides=(1,1), padding='same')
    inception2_d_2 = Conv2D(32//ds, (1,1), strides=(1,1), activation='relu', padding='same')
    inception2_e = MaxPooling2D((3,3), strides=(1,1), padding='same')
    inception2_e_2 = Conv2D(32//ds, (1,1), strides=(1,1), activation='relu', padding='same')
    
    inception3_a = Conv2D(64//ds, (1,1), strides=(1,1), activation='relu', padding='same')
    inception3_b = Conv2D(96//ds, (1,1), strides=(1,1), activation='relu', padding='same')
    inception3_b_2 = Conv2D(128//ds, (3,3), strides=(1,1), activation='relu', padding='same')
    inception3_c = Conv2D(16//ds, (1,1), strides=(1,1), activation='relu', padding='same')
    inception3_c_2 = Conv2D(32//ds, (5,5), strides=(1,1), activation='relu', padding='same')
    inception3_d = AveragePooling2D((3,3), strides=(1,1), padding='same')
    inception3_d_2 = Conv2D(32//ds, (1,1), strides=(1,1), activation='relu', padding='same')
    inception3_e = MaxPooling2D((3,3), strides=(1,1), padding='same')
    inception3_e_2 = Conv2D(32//ds, (1,1), strides=(1,1), activation='relu', padding='same')

    conv3 = Conv2D(32//ds, (5,5), strides=(1,1), activation='relu', padding='same')
    conv4 = Conv2D(16//ds, (3,3), strides=(1,1), activation='relu', padding='same')
    conv5 = Conv2D(1, (1,1), strides=(1,1), activation='sigmoid')

    # connect inputs and outputs
    unhealthy_input = Input((None, None, num_channels), name='unhealthy_input')

    unhealthy_conv_1_x = conv1(unhealthy_input) 
    unhealthy_conv_2_x = conv2(unhealthy_conv_1_x)

    unhealthy_block_1_a = inception1_a(unhealthy_conv_2_x)
    unhealthy_block_1_b = inception1_b(unhealthy_conv_2_x)
    unhealthy_block_1_b = inception1_b_2(unhealthy_block_1_b)
    unhealthy_block_1_c = inception1_c(unhealthy_conv_2_x)
    unhealthy_block_1_c = inception1_c_2(unhealthy_block_1_c)
    unhealthy_block_1_d = inception1_d(unhealthy_conv_2_x)
    unhealthy_block_1_d = inception1_d_2(unhealthy_block_1_d)
    unhealthy_block_1_e = inception1_e(unhealthy_conv_2_x)
    unhealthy_block_1_e = inception1_e_2(unhealthy_block_1_e)
    unhealthy_block_1_x = concatenate([unhealthy_block_1_a,
                                        unhealthy_block_1_b,
                                        unhealthy_block_1_c,
                                        unhealthy_block_1_d,
                                        unhealthy_block_1_e],axis=-1)

    unhealthy_block_2_a = inception2_a(unhealthy_block_1_x)
    unhealthy_block_2_b = inception2_b(unhealthy_block_1_x)
    unhealthy_block_2_b = inception2_b_2(unhealthy_block_2_b)
    unhealthy_block_2_c = inception2_c(unhealthy_block_1_x)
    unhealthy_block_2_c = inception2_c_2(unhealthy_block_2_c)
    unhealthy_block_2_d = inception2_d(unhealthy_block_1_x)
    unhealthy_block_2_d = inception2_d_2(unhealthy_block_2_d)
    unhealthy_block_2_e = inception2_e(unhealthy_block_1_x)
    unhealthy_block_2_e = inception2_e_2(unhealthy_block_2_e)
    unhealthy_block_2_x = concatenate([unhealthy_block_2_a,
                                        unhealthy_block_2_b,
                                        unhealthy_block_2_c,
                                        unhealthy_block_2_d,
                                        unhealthy_block_2_e],axis=-1)

    unhealthy_block_3_a = inception3_a(unhealthy_block_2_x)
    unhealthy_block_3_b = inception3_b(unhealthy_block_2_x)
    unhealthy_block_3_b = inception3_b_2(unhealthy_block_3_b)
    unhealthy_block_3_c = inception3_c(unhealthy_block_2_x)
    unhealthy_block_3_c = inception3_c_2(unhealthy_block_3_c)
    unhealthy_block_3_d = inception3_d(unhealthy_block_2_x)
    unhealthy_block_3_d = inception3_d_2(unhealthy_block_3_d)
    unhealthy_block_3_e = inception3_e(unhealthy_block_2_x)
    unhealthy_block_3_e = inception3_e_2(unhealthy_block_3_e)
    unhealthy_block_3_x = concatenate([unhealthy_block_3_a,
                                        unhealthy_block_3_b,
                                        unhealthy_block_3_c,
                                        unhealthy_block_3_d,
                                        unhealthy_block_3_e],axis=-1)

    unhealthy_conv_3_x = conv3(unhealthy_block_3_x)
    unhealthy_conv_4_x = conv4(unhealthy_conv_3_x)
    unhealthy_conv_5_x = conv5(unhealthy_conv_4_x)

    unhealthy_output = Activation('linear', name='unhealthy_output')(unhealthy_conv_5_x)

    healthy_input = Input((None, None, num_channels), name='healthy_input')
    healthy_conv_1_x = conv1(healthy_input) 
    healthy_conv_2_x = conv2(healthy_conv_1_x)

    healthy_block_1_a = inception1_a(healthy_conv_2_x)
    healthy_block_1_b = inception1_b(healthy_conv_2_x)
    healthy_block_1_b = inception1_b_2(healthy_block_1_b)
    healthy_block_1_c = inception1_c(healthy_conv_2_x)
    healthy_block_1_c = inception1_c_2(healthy_block_1_c)
    healthy_block_1_d = inception1_d(healthy_conv_2_x)
    healthy_block_1_d = inception1_d_2(healthy_block_1_d)
    healthy_block_1_e = inception1_e(healthy_conv_2_x)
    healthy_block_1_e = inception1_e_2(healthy_block_1_e)
    healthy_block_1_x = concatenate([healthy_block_1_a,
                                        healthy_block_1_b,
                                        healthy_block_1_c,
                                        healthy_block_1_d,
                                        healthy_block_1_e],axis=-1)

    healthy_block_2_a = inception2_a(healthy_block_1_x)
    healthy_block_2_b = inception2_b(healthy_block_1_x)
    healthy_block_2_b = inception2_b_2(healthy_block_2_b)
    healthy_block_2_c = inception2_c(healthy_block_1_x)
    healthy_block_2_c = inception2_c_2(healthy_block_2_c)
    healthy_block_2_d = inception2_d(healthy_block_1_x)
    healthy_block_2_d = inception2_d_2(healthy_block_2_d)
    healthy_block_2_e = inception2_e(healthy_block_1_x)
    healthy_block_2_e = inception2_e_2(healthy_block_2_e)
    healthy_block_2_x = concatenate([healthy_block_2_a,
                                        healthy_block_2_b,
                                        healthy_block_2_c,
                                        healthy_block_2_d,
                                        healthy_block_2_e],axis=-1)

    healthy_block_3_a = inception3_a(healthy_block_2_x)
    healthy_block_3_b = inception3_b(healthy_block_2_x)
    healthy_block_3_b = inception3_b_2(healthy_block_3_b)
    healthy_block_3_c = inception3_c(healthy_block_2_x)
    healthy_block_3_c = inception3_c_2(healthy_block_3_c)
    healthy_block_3_d = inception3_d(healthy_block_2_x)
    healthy_block_3_d = inception3_d_2(healthy_block_3_d)
    healthy_block_3_e = inception3_e(healthy_block_2_x)
    healthy_block_3_e = inception3_e_2(healthy_block_3_e)
    healthy_block_3_x = concatenate([healthy_block_3_a,
                                        healthy_block_3_b,
                                        healthy_block_3_c,
                                        healthy_block_3_d,
                                        healthy_block_3_e],axis=-1)

    healthy_conv_3_x = conv3(healthy_block_3_x)
    healthy_conv_4_x = conv4(healthy_conv_3_x)
    healthy_conv_5_x = conv5(healthy_conv_4_x)
    healthy_output = Activation('linear', name='healthy_output')(healthy_conv_5_x)
    
    model = Model(inputs=[unhealthy_input, healthy_input], 
                  outputs=[unhealthy_output, healthy_output])

    # binary crossentropy with dice as a metric 
    model.compile(optimizer=Adam(lr=lr),
                  metrics=[continuous_dice_coef],
                  loss={'unhealthy_output':unhealthy_loss, 
                        'healthy_output':healthy_loss}, 
                  loss_weights={'unhealthy_output':14/15,
                                'healthy_output':1/15,})

    return model
