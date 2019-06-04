from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, AveragePooling3D, concatenate,\
    GlobalAveragePooling3D, add, UpSampling3D, Dropout, Activation
from tensorflow.keras.models import Model

def tiny_net(num_channels):

    inputs = Input((None, None, None, num_channels))

    x = Conv3D(4, 9, activation='relu', padding='same', )(inputs)
    x = Conv3D(1, 1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=x)

    return model
