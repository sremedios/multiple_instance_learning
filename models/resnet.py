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

def resnet(num_classes, num_channels=1, ds=2):

    inputs = Input(shape=(None, None, num_channels))

    x = Conv2D(64//ds, kernel_size=7, strides=2, padding='same')(inputs)
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    block = x

    for i in range(8):
        x = Conv2D(int((64/ds)*2**i), kernel_size=3, strides=2, padding='same')(block)
        x = BatchNormalization()(x)
        y = Activation('relu')(x)
        x = Conv2D(int((64/ds)*2**i), kernel_size=3, strides=1, padding='same')(x)
        x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
        x = BatchNormalization()(y)
        x = add([x, y])
        if i >= 5:
            x = Dropout(0.4)(x)
        block = Activation('relu')(x)

    x = GlobalMaxPooling2D()(block)
    outputs = Dense(num_classes)(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model
