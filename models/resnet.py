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

def resnet(num_classes, num_channels=1, ds=2):

    inputs = Input(shape=(None, None, num_channels))

    x = Conv2D(64//ds, kernel_size=7, kernel_regularizer=regularizers.l2(1e-2), bias_regularizer=regularizers.l2(1e-2), strides=2, padding='same')(inputs)
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    block = x

    depth = 7
    for i in range(depth):
        x = Conv2D(int((64/ds)*2**i), kernel_size=3, kernel_regularizer=regularizers.l2(1e-2), bias_regularizer=regularizers.l2(1e-2), strides=2, padding='same')(block)
        x = BatchNormalization()(x)
        y = Activation('relu')(x)
        x = Conv2D(int((64/ds)*2**i), kernel_size=3, kernel_regularizer=regularizers.l2(1e-2), bias_regularizer=regularizers.l2(1e-2), strides=1, padding='same')(x)
        x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
        x = BatchNormalization()(y)
        x = add([x, y])
        if depth - i <= 3:
            x = Dropout(0.5)(x)
        block = Activation('relu')(x)

    x = GlobalMaxPooling2D()(block)
    #x = Dense(256//ds)(x)
    #x = Dropout(0.5)(x)
    #x = Dense(128//ds)(x)
    #x = Dropout(0.5)(x)
    outputs = Dense(num_classes)(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model
