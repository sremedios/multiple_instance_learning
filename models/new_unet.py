from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, AveragePooling3D, concatenate,\
    GlobalAveragePooling3D, GlobalMaxPooling3D, add, UpSampling3D, Dropout, Activation, Dense
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D,\
    GlobalAveragePooling2D, GlobalMaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model

def tiny_net(num_channels,
             lr=1e-4):

    inputs = Input((None, None, None, num_channels))

    x = Conv3D(4, 3, activation='relu', padding='same', )(inputs)
    x = Conv3D(1, 1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=x)

    return model



def unet(num_channels,
         ds=2,
         lr=1e-4,
         verbose=0,):
    inputs = Input((None, None, None, num_channels))

    conv1 = Conv3D(64//ds, 3, activation='relu', padding='same', )(inputs)
    conv1 = Conv3D(64//ds, 3, activation='relu', padding='same', )(conv1)
    pool1 = MaxPooling3D(pool_size=2)(conv1)

    conv2 = Conv3D(128//ds, 3, activation='relu', padding='same',)(pool1)
    conv2 = Conv3D(128//ds, 3, activation='relu', padding='same', )(conv2)
    pool2 = MaxPooling3D(pool_size=2)(conv2)

    conv3 = Conv3D(256//ds, 3, activation='relu', padding='same', )(pool2)
    conv3 = Conv3D(256//ds, 3, activation='relu', padding='same', )(conv3)
    pool3 = MaxPooling3D(pool_size=2)(conv3)

    conv4 = Conv3D(512//ds, 3, activation='relu', padding='same', )(pool3)
    conv4 = Conv3D(512//ds, 3, activation='relu', padding='same', )(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling3D(pool_size=2)(drop4)

    conv5 = Conv3D(1024//ds, 3, activation='relu', padding='same', )(pool4)
    conv5 = Conv3D(1024//ds, 3, activation='relu', padding='same', )(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv3D(512//ds, 2, activation='relu', padding='same')(UpSampling3D(size=2)(drop5))
    merge6 = concatenate([drop4, up6], axis=-1)
    conv6 = Conv3D(512//ds, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv3D(512//ds, 3, activation='relu', padding='same')(conv6)

    up7 = Conv3D(256//ds, 2, activation='relu', padding='same')(UpSampling3D(size=2)(conv6))
    merge7 = concatenate([conv3, up7], axis=-1)
    conv7 = Conv3D(256//ds, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv3D(256//ds, 3, activation='relu', padding='same')(conv7)

    up8 = Conv3D(128//ds, 2, activation='relu', padding='same')(UpSampling3D(size=2)(conv7))
    merge8 = concatenate([conv2, up8], axis=-1)
    conv8 = Conv3D(128//ds, 3, activation='relu', padding='same')(merge8)
    conv8 = Conv3D(128//ds, 3, activation='relu', padding='same')(conv8)

    up9 = Conv3D(64//ds, 2, activation='relu', padding='same')(UpSampling3D(size=2)(conv8))
    merge9 = concatenate([conv1, up9], axis=-1)
    conv9 = Conv3D(64//ds, 3, activation='relu', padding='same')(merge9)
    conv9 = Conv3D(64//ds, 3, activation='relu', padding='same')(conv9)

    conv9 = Conv3D(2, 3, activation='relu', padding='same', )(conv9)
    conv10 = Conv3D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    return model

def class_unet(num_channels,
         ds=2,
         lr=1e-4,
         verbose=0,):
    inputs = Input((None, None, None, num_channels))
    #inputs = Input((256, 256, 32, num_channels))

    conv1 = Conv3D(64//ds, 3, activation='relu', padding='same', )(inputs)
    conv1 = Conv3D(64//ds, 3, activation='relu', padding='same', )(conv1)
    pool1 = MaxPooling3D(pool_size=2)(conv1)

    conv2 = Conv3D(128//ds, 3, activation='relu', padding='same',)(pool1)
    conv2 = Conv3D(128//ds, 3, activation='relu', padding='same', )(conv2)
    pool2 = MaxPooling3D(pool_size=2)(conv2)

    conv3 = Conv3D(256//ds, 3, activation='relu', padding='same', )(pool2)
    conv3 = Conv3D(256//ds, 3, activation='relu', padding='same', )(conv3)
    pool3 = MaxPooling3D(pool_size=2)(conv3)

    conv4 = Conv3D(512//ds, 3, activation='relu', padding='same', )(pool3)
    conv4 = Conv3D(512//ds, 3, activation='relu', padding='same', )(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling3D(pool_size=2)(drop4)

    conv5 = Conv3D(1024//ds, 3, activation='relu', padding='same', )(pool4)
    conv5 = Conv3D(1024//ds, 3, activation='relu', padding='same', )(conv5)

    x = GlobalMaxPooling3D()(conv5)
    x = Dense(4, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=x)

    return model

def class_unet_2D(num_channels,
         num_classes,
         ds=2,
         lr=1e-4,
         verbose=0,):
    inputs = Input((None, None, num_channels))

    conv1 = Conv2D(64//ds, 3, activation='relu', padding='same', )(inputs)
    conv1 = Conv2D(64//ds, 3, activation='relu', padding='same', )(conv1)
    pool1 = MaxPooling2D(pool_size=2)(conv1)

    conv2 = Conv2D(128//ds, 3, activation='relu', padding='same',)(pool1)
    conv2 = Conv2D(128//ds, 3, activation='relu', padding='same', )(conv2)
    pool2 = MaxPooling2D(pool_size=2)(conv2)

    conv3 = Conv2D(256//ds, 3, activation='relu', padding='same', )(pool2)
    conv3 = Conv2D(256//ds, 3, activation='relu', padding='same', )(conv3)
    pool3 = MaxPooling2D(pool_size=2)(conv3)

    conv4 = Conv2D(512//ds, 3, activation='relu', padding='same', )(pool3)
    conv4 = Conv2D(512//ds, 3, activation='relu', padding='same', )(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=2)(drop4)

    conv5 = Conv2D(1024//ds, 3, activation='relu', padding='same', )(pool4)
    conv5 = Conv2D(1024//ds, 3, activation='relu', padding='same', )(conv5)

    x = GlobalMaxPooling2D()(conv5)
    x = Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=x)

    return model
