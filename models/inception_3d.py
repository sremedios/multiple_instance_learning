from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, AveragePooling3D, concatenate, GlobalAveragePooling3D, add, Activation, BatchNormalization, UpSampling3D
from tensorflow.keras.models import Model

def get_inception_layer_3d(prev_layer, ds=2):
    a = Conv3D(64//ds, 1, strides=1, activation='relu',
               padding='same')(prev_layer)

    b = Conv3D(96//ds, 1, strides=1, activation='relu',
               padding='same')(prev_layer)
    b = Conv3D(128//ds, 3, strides=1, activation='relu', padding='same')(b)

    c = Conv3D(16//ds, 1, strides=1, activation='relu',
               padding='same')(prev_layer)
    c = Conv3D(32//ds, 5, strides=1, activation='relu', padding='same')(c)

    d = AveragePooling3D(3, strides=1, padding='same')(prev_layer)
    d = Conv3D(32//ds, 1, strides=1, activation='relu', padding='same')(d)

    e = MaxPooling3D(3, strides=1, padding='same')(prev_layer)
    e = Conv3D(32//ds, 1, strides=1, activation='relu', padding='same')(e)

    out_layer = concatenate([a, b, c, d, e], axis=-1)

    return out_layer


def inception_3d(num_channels=1, ds=2):

    inputs = Input((None, None, None, num_channels))

    x = Conv3D(64//ds, 3, strides=1, activation='relu', padding='same')(inputs)
    x = AveragePooling3D(2, strides=1, padding='same')(x)
    x = Conv3D(64//ds, 3, strides=1, activation='relu', padding='same')(x)
    x = AveragePooling3D(2, strides=1, padding='same')(x)

    x = get_inception_layer_3d(x, ds)
    x = AveragePooling3D(2, strides=1, padding='same')(x)
    x = get_inception_layer_3d(x, ds)
    x = AveragePooling3D(2, strides=1, padding='same')(x)

    x = UpSampling3D(size=2)(x)
    x = Conv3D(32//ds, 5, strides=1, activation='relu', padding='same')(x)
    x = UpSampling3D(size=2)(x)
    x = Conv3D(16//ds, 3, strides=1, activation='relu', padding='same')(x)
    x = UpSampling3D(size=2)(x)
    x = Conv3D(1, 1, strides=1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=x)

    return model
