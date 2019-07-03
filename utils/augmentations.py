import tensorflow as tf
import numpy as np

def flip_dim1(x):
    x = x[:, ::-1, :, :]
    return x

def flip_dim2(x):
    x = x[:, :, ::-1, :]
    return x

def flip_dim3(x):
    x = x[:, :, :, ::-1]
    return x

def rotate_2D(bag, max_angle=60):
    return tf.contrib.image.rotate(
        bag, 
        np.random.uniform(max_angle//6, max_angle)
    )
