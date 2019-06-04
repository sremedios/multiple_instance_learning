import tensorflow as tf
import numpy as np


def continuous_dice_coef_loss(model, x, y, smooth=1):

    input_shape = tf.shape(x)

    y_pred_f = tf.reshape(model(x, training=True), [-1, *input_shape[1:]])
    y_true_f = tf.reshape(y, [-1, *input_shape[1:]])

    prod = y_true_f * y_pred_f
    intersection = tf.reduce_sum(prod , axis=-1, keepdims=True)

    c = np.ones(shape=intersection.shape)
    for i, (y_t, y_p) in enumerate(zip(y_true_f, y_pred_f)):
        sample_intersection = tf.reduce_sum(tf.reshape(y_t, [-1]) * tf.reshape(y_p, [-1]))
        if tf.cond(tf.greater(sample_intersection, 0.), lambda: 1, lambda: 0) == 1:
            c[i] = tf.reduce_sum(y_t * y_p) /\
                tf.reduce_sum(y_t * tf.sign(y_p))

    continuous_union = c * tf.reduce_sum(y_true_f, axis=-1, keepdims=True) + tf.reduce_sum(y_pred_f, axis=-1, keepdims=True)

    cdc = (2. * intersection + smooth) / (continuous_union + smooth)
    return -cdc


def dice_coef(model, x, y, smooth=1):
    y_true_f = tf.reshape(model(x, training=True), [-1])
    y_pred_f = tf.round(tf.reshape(y, [-1]))
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    return (2. * intersection + smooth) / (union + smooth)


custom_losses = {
    'continuous_dice_coef_loss': continuous_dice_coef_loss,
    'dice_coef': dice_coef,
}
