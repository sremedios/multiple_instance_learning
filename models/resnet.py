from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    AveragePooling2D,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    UpSampling2D,
    BatchNormalization,
    add,
    Dropout,
    Activation,
    Dense,
    concatenate,
)
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

def residual_block(prev_layer, repetitions, num_filters):
    block = prev_layer
    for i in range(repetitions):
        x = Conv2D(
                filters=num_filters,
                kernel_size=3, 
                kernel_regularizer=regularizers.l2(1e-2), 
                bias_regularizer=regularizers.l2(1e-2), 
                strides=1, 
                padding='same'
        )(block)
        #x = Activation('relu')(x)
        y = Activation('relu')(x)
        #y = BatchNormalization()(x)
        x = Conv2D(
                filters=num_filters,
                kernel_size=3, 
                kernel_regularizer=regularizers.l2(1e-2), 
                bias_regularizer=regularizers.l2(1e-2), 
                strides=1, 
                padding='same'
        )(x)
        x = add([x, y])
        x = Activation('relu')(x)
        #x = BatchNormalization()(x)
        block = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    return block 


def resnet(num_classes, num_channels=1, ds=2):

    inputs = Input(shape=(None, None, num_channels))

    x = Conv2D(
        filters=64//ds, 
        kernel_size=7, 
        kernel_regularizer=regularizers.l2(1e-2), 
        bias_regularizer=regularizers.l2(1e-2), 
        strides=2, 
        padding='same'
    )(inputs)
    block_0 = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    block_1 = residual_block(prev_layer=block_0, repetitions=3, num_filters=64//ds)
    block_2 = residual_block(prev_layer=block_1, repetitions=4, num_filters=128//ds)
    block_3 = residual_block(prev_layer=block_2, repetitions=6, num_filters=256//ds)
    block_4 = residual_block(prev_layer=block_3, repetitions=3, num_filters=512//ds)

    x = GlobalAveragePooling2D()(block_4)
    outputs = Dense(num_classes)(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model
