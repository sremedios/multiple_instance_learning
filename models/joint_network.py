from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, AveragePooling3D, concatenate,\
    GlobalAveragePooling3D, GlobalMaxPooling3D, add, UpSampling3D, Dropout, Activation, Dense
from tensorflow.keras.models import Model


def init_shared_weights(ds):
    return [
        Conv3D(64//ds, 3, activation='relu',
               padding='same', name='level_1_conv_1'),
        Conv3D(64//ds, 3, activation='relu',
               padding='same', name='level_1_conv_2'),
        MaxPooling3D(pool_size=2),

        Conv3D(128//ds, 3, activation='relu',
               padding='same', name='level_2_conv_1'),
        Conv3D(128//ds, 3, activation='relu',
               padding='same', name='level_2_conv_2'),
        MaxPooling3D(pool_size=2),

        Conv3D(256//ds, 3, activation='relu',
               padding='same', name='level_3_conv_1'),
        Conv3D(256//ds, 3, activation='relu',
               padding='same', name='level_3_conv_2'),
        MaxPooling3D(pool_size=2),

        Conv3D(512//ds, 3, activation='relu',
               padding='same', name='level_4_conv_1'),
        Conv3D(512//ds, 3, activation='relu',
               padding='same', name='level_4_conv_2'),
        Dropout(0.5),
        MaxPooling3D(pool_size=2),

        Conv3D(1024//ds, 3, activation='relu',
               padding='same', name='level_5_conv_1'),
        Conv3D(1024//ds, 3, activation='relu',
               padding='same', name='level_5_conv_2'),
        Dropout(0.5),
    ]


def wire_shared_weights(shared_weights, prev_layer):
    layer_iterator = iter(shared_weights)

    conv1 = next(layer_iterator)(prev_layer)
    conv1 = next(layer_iterator)(conv1)
    pool1 = next(layer_iterator)(conv1)

    conv2 = next(layer_iterator)(pool1)
    conv2 = next(layer_iterator)(conv2)
    pool2 = next(layer_iterator)(conv2)

    conv3 = next(layer_iterator)(pool2)
    conv3 = next(layer_iterator)(conv3)
    pool3 = next(layer_iterator)(conv3)

    conv4 = next(layer_iterator)(pool3)
    conv4 = next(layer_iterator)(conv4)
    drop4 = next(layer_iterator)(conv4)
    pool4 = next(layer_iterator)(drop4)

    conv5 = next(layer_iterator)(pool4)
    conv5 = next(layer_iterator)(conv5)
    drop5 = next(layer_iterator)(conv5)

    return conv1, conv2, conv3, drop4, drop5


def joint_unet(shared_weights,
               num_channels,
               ds,
               lr=1e-4):

    seg_inputs = Input((None, None, None, num_channels), name='seg_inputs')

    ########## SEGMENTATION SIDE ##########

    conv1, conv2, conv3, drop4, drop5 = wire_shared_weights(shared_weights=shared_weights,
                                                            prev_layer=seg_inputs)

    up6 = Conv3D(512//ds, 2, activation='relu',
                 padding='same')(UpSampling3D(size=2)(drop5))
    merge6 = concatenate([drop4, up6], axis=-1)
    conv6 = Conv3D(512//ds, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv3D(512//ds, 3, activation='relu', padding='same')(conv6)

    up7 = Conv3D(256//ds, 2, activation='relu',
                 padding='same')(UpSampling3D(size=2)(conv6))
    merge7 = concatenate([conv3, up7], axis=-1)
    conv7 = Conv3D(256//ds, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv3D(256//ds, 3, activation='relu', padding='same')(conv7)

    up8 = Conv3D(128//ds, 2, activation='relu',
                 padding='same')(UpSampling3D(size=2)(conv7))
    merge8 = concatenate([conv2, up8], axis=-1)
    conv8 = Conv3D(128//ds, 3, activation='relu', padding='same')(merge8)
    conv8 = Conv3D(128//ds, 3, activation='relu', padding='same')(conv8)

    up9 = Conv3D(64//ds, 2, activation='relu',
                 padding='same')(UpSampling3D(size=2)(conv8))
    merge9 = concatenate([conv1, up9], axis=-1)
    conv9 = Conv3D(64//ds, 3, activation='relu', padding='same')(merge9)
    conv9 = Conv3D(64//ds, 3, activation='relu', padding='same')(conv9)

    conv9 = Conv3D(2, 3, activation='relu', padding='same', )(conv9)
    seg_outputs = Conv3D(1, 1, activation='sigmoid', name='seg_outputs')(conv9)

    ########## CLASSIFICATION SIDE ##########

    class_inputs = Input((None, None, None, num_channels), name='class_inputs')
    _, _, _, _, drop5 = wire_shared_weights(shared_weights=shared_weights,
                                            prev_layer=class_inputs)

    conv5 = Conv3D(1024//ds, 3, activation='relu',
                   padding='same', name='class_conv_1')(drop5)
    conv5 = Conv3D(1024//ds, 3, activation='relu',
                   padding='same', name='class_conv_2')(conv5)

    x = GlobalMaxPooling3D()(conv5)
    class_outputs = Dense(4, activation='sigmoid', name='class_outputs')(x)

    ########## JOINT WIRING ##########

    model = Model(inputs=[seg_inputs, class_inputs],
                  outputs=[seg_outputs, class_outputs])

    return model
