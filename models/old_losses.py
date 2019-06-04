from keras.losses import binary_crossentropy
from keras import backend as K
from keras.backend.common import epsilon
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes 
from tensorflow.python.framework import tensor_shape 
from tensorflow.python.ops import nn_ops 
from tensorflow.python.ops import math_ops 


"""
def true_positive_rate(y_true, y_pred, smooth=1):
    ALPHA = 1.
    BETA = 1.
    bce = binary_crossentropy(y_true, y_pred)
    y_true_f = K.cast(K.flatten(y_true), dtype=tf.int32)
    y_pred_f = K.cast(K.round(K.flatten(y_pred)), dtype=tf.int32)
    c_matrix = tf.confusion_matrix(y_true_f, y_pred_f, num_classes=2)
    true_positive = c_matrix[1,1]
    false_positive = c_matrix[1,0]
    true_negative = c_matrix[0,0]
    false_negative = c_matrix[0,1]
    sensitivity = (smooth + true_positive) / (smooth + true_positive + false_negative)
    sensitivity = K.cast(sensitivity, dtype=tf.float32)
    # bce plus false positive rate plus false negative rate
    # this weights the bce by the FPR and FNR
    return ALPHA*bce + BETA*sensitivity

def false_positive_rate(y_true, y_pred, smooth=1):
    # two hyperparameters to scale the impact of BCE and FPR, respectively
    ALPHA = 1.
    BETA = 1.
    bce = binary_crossentropy(y_true, y_pred)
    y_true_f = K.cast(K.flatten(y_true), dtype=tf.int32)
    y_pred_f = K.cast(K.round(K.flatten(y_pred)), dtype=tf.int32)
    c_matrix = tf.confusion_matrix(y_true_f, y_pred_f, num_classes=2)
    true_positive = c_matrix[1,1]
    false_positive = c_matrix[1,0]
    true_negative = c_matrix[0,0]
    false_negative = c_matrix[0,1]
    specificity = (smooth + true_negative) / (smooth + true_negative + false_positive)
    specificity = K.cast(specificity, dtype=tf.float32)
    # bce plus false positive rate plus false negative rate
    # this weights the bce by the FPR and FNR
    return ALPHA*bce + BETA*(1-specificity)
"""

def weighted_dice_TPR(y_true, y_pred, smooth=1):
    dice = dice_coef_loss(y_true, y_pred)
    y_true_f = K.cast(K.flatten(y_true), dtype=tf.int32)
    y_pred_f = K.cast(K.round(K.flatten(y_pred)), dtype=tf.int32)
    c_matrix = tf.confusion_matrix(y_true_f, y_pred_f, num_classes=2)
    true_positive = c_matrix[1,1]
    false_positive = c_matrix[1,0]
    true_negative = c_matrix[0,0]
    false_negative = c_matrix[0,1]
    sensitivity = (smooth + true_positive) / (smooth + true_positive + false_negative)
    sensitivity = K.cast(sensitivity, dtype=tf.float32)
    return dice + sensitivity

def weighted_dice_FPR(y_true, y_pred, smooth=1):
    dice = dice_coef_loss(y_true, y_pred)
    y_true_f = K.cast(K.flatten(y_true), dtype=tf.int32)
    y_pred_f = K.cast(K.round(K.flatten(y_pred)), dtype=tf.int32)
    c_matrix = tf.confusion_matrix(y_true_f, y_pred_f, num_classes=2)
    true_positive = c_matrix[1,1]
    false_positive = c_matrix[1,0]
    true_negative = c_matrix[0,0]
    false_negative = c_matrix[0,1]
    specificity = (smooth + true_negative) / (smooth + true_negative + false_positive)
    specificity = K.cast(specificity, dtype=tf.float32)
    return dice + (1-specificity)

def weighted_bce(y_true, y_pred, smooth=1):
    bce = binary_crossentropy(y_true, y_pred)
    y_true_f = K.cast(K.flatten(y_true), dtype=tf.int32)
    y_pred_f = K.cast(K.round(K.flatten(y_pred)), dtype=tf.int32)
    c_matrix = tf.confusion_matrix(y_true_f, y_pred_f, num_classes=2)
    true_positive = c_matrix[1,1]
    false_positive = c_matrix[1,0]
    true_negative = c_matrix[0,0]
    false_negative = c_matrix[0,1]
    sensitivity = (smooth + true_positive) / (smooth + true_positive + false_negative)
    specificity = (smooth + true_negative) / (smooth + true_negative + false_positive)
    sensitivity = K.cast(sensitivity, dtype=tf.float32)
    specificity = K.cast(specificity, dtype=tf.float32)
    # bce plus false positive rate plus false negative rate
    # this weights the bce by the FPR and FNR
    return bce + (1-specificity) + (1-sensitivity)

def true_positive_rate(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    inverse_y_pred_f = 1 - y_pred_f
    inverse_y_true_f = 1 - y_true_f
    true_positives = K.sum(y_true_f * y_pred_f)
    false_positives = K.sum(y_pred_f) - true_positives
    true_negatives = K.sum(inverse_y_pred_f * inverse_y_true_f)
    false_negatives = K.sum(inverse_y_pred_f) - true_negatives
    tpr = true_positives / (true_positives + false_negatives)
    return tpr

def true_positive_rate_loss(y_true, y_pred):
    return -true_positive_rate(y_true, y_pred)

def false_positive_rate(y_true, y_pred, smooth=1):
    '''
    Returns a value in [0,1], where 1 is "all false positives" 
    and 0 is "all true negatives"
    '''
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    inverse_y_pred_f = 1 - y_pred_f
    inverse_y_true_f = 1 - y_true_f
    true_positives = K.sum(y_true_f * y_pred_f)
    false_positives = K.sum(y_pred_f) - true_positives
    true_negatives = K.sum(inverse_y_pred_f * inverse_y_true_f)
    false_negatives = K.sum(inverse_y_pred_f) - true_negatives
    fpr = (false_positives + smooth) / (false_positives + true_negatives + smooth)
    return fpr

def continuous_dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    # this is a workaround to allow a boolean check for continuous dice
    if tf.cond(tf.greater(intersection, 0.), lambda: 1, lambda:0) == 1: 
        c = K.sum(y_true_f * y_pred_f) / K.sum(y_true_f * K.sign(y_pred_f))
    else:
        c = 1
    continuous_union = c * K.sum(y_true_f) + K.sum(y_pred_f)
    return (2. * intersection + smooth) / (continuous_union + smooth)

def continuous_dice_coef_loss(y_true, y_pred):
    return (-continuous_dice_coef(y_true, y_pred))

def true_positive_continuous_dice_coef_loss(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    false_positives = K.sum(y_pred_f) - intersection 

    # this is a workaround to allow a boolean check for continuous dice
    if tf.cond(tf.greater(intersection, 0.), lambda: 1, lambda:0) == 1: 
        c = K.sum(y_true_f * y_pred_f) / K.sum(y_true_f * K.sign(y_pred_f))
    else:
        c = 1

    continuous_union = c * K.sum(y_true_f) + K.sum(y_pred_f)
    true_positive_augmented_union = continuous_union - false_positives 

    return (1- (2. * intersection + smooth) / (true_positive_augmented_union + smooth))

def false_positive_continuous_dice_coef_loss(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    # this is a workaround to allow a boolean check for continuous dice
    if tf.cond(tf.greater(intersection, 0.), lambda: 1, lambda:0) == 1: 
        c = K.sum(y_true_f * y_pred_f) / K.sum(y_true_f * K.sign(y_pred_f))
    else:
        c = 1

    continuous_union = c * K.sum(y_true_f) + K.sum(y_pred_f)
    false_positive_augmented_union = continuous_union - intersection 

    return -(2. * intersection + smooth) / (false_positive_augmented_union + smooth) + 1


def tpr_weighted_cdc_loss(y_true, y_pred):
    return -true_positive_rate(y_true, y_pred) + continuous_dice_coef_loss(y_true, y_pred)

def fpr_weighted_cdc_loss(y_true, y_pred):
    return false_positive_rate(y_true, y_pred) + continuous_dice_coef_loss(y_true, y_pred)

def tpr_weighted_bce_loss(y_true, y_pred):
    return -true_positive_rate(y_true, y_pred) + binary_crossentropy(y_true, y_pred)

def fpr_weighted_bce_loss(y_true, y_pred):
    return false_positive_rate(y_true, y_pred) + binary_crossentropy(y_true, y_pred)

def tpr_weighted_dice_loss(y_true, y_pred):
    return -true_positive_rate(y_true, y_pred) + dice_coef_loss(y_true, y_pred)

def fpr_weighted_dice_loss(y_true, y_pred):
    return false_positive_rate(y_true, y_pred) + dice_coef_loss(y_true, y_pred)


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.round(K.flatten(y_pred))
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f)
    return (2. * intersection + smooth) / (union + smooth)

def dice_coef_no_round(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f)
    return (2. * intersection + smooth) / (union + smooth)

def dice_coef_loss(y_true, y_pred):
    return (-dice_coef_no_round(y_true, y_pred) + 1)

def bce_of_true_positive(y_true, y_pred, from_logits=False, _sentinel=None, name=None):
    if not from_logits:
        _epsilon = tf.convert_to_tensor(epsilon(), y_pred.dtype.base_dtype)
        output = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
        output = tf.log(output / (1 - output))
    
    # alteration of sigmoid_crossentroy_with_logits
    nn_ops._ensure_xent_args("sigmoid_cross_entropy_with_logits", _sentinel,
            y_true, y_pred)

    with ops.name_scope(name, "logistic_loss_over_true_positives", [y_pred, y_true]) as name:
        logits = ops.convert_to_tensor(y_pred, name="logits")
        labels = ops.convert_to_tensor(y_true, name="labels")
        try:
            labels.get_shape().merge_with(logits.get_shape())
        except ValueError:
            raise ValueError("Logits and labels must have the same shape (%s vs %s)" %
                    (logits.get_shape(), labels.get_shape()))
        zeros = array_ops.zeros_like(logits, dtype=logits.dtype)
        cond = (logits >= zeros)
        relu_logits = array_ops.where(cond, logits, zeros)
        neg_abs_logits = array_ops.where(cond, -logits, logits)

        # here we calculate the mean to be in-line with Keras' binary crossentropy.
        return K.mean(math_ops.multiply(-labels,
                                        math_ops.log1p(math_ops.exp(neg_abs_logits)),
                                        name=name))

def dice_of_true_positive(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    # this differs here from normal dice in the second term of the union
    union = K.sum(y_true_f) + K.sum(intersection)
    return (2. * intersection + smooth) / (union + smooth)

def dice_of_true_positive_loss(y_true, y_pred):
    return (-dice_of_true_positive(y_true, y_pred) + 1)

def cdc_of_true_positive(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    # this is a workaround to allow a boolean check for continuous dice
    if tf.cond(tf.greater(intersection, 0.), lambda: 1, lambda:0) == 1: 
        c = K.sum(y_true_f * y_pred_f) / K.sum(y_true_f * K.sign(y_pred_f))
    else:
        c = 1
    # this differs here from normal dice in the second term of the union
    continuous_union = c * K.sum(y_true_f) + K.sum(intersection)
    return (2. * intersection + smooth) / (continuous_union + smooth)

def cdc_of_true_positive_loss(y_true, y_pred):
    return (-cdc_of_true_positive(y_true, y_pred) + 1)



custom_losses = {
        'true_positive_continuous_dice_coef_loss': true_positive_continuous_dice_coef_loss,
        'false_positive_continuous_dice_coef_loss': false_positive_continuous_dice_coef_loss,
        'true_positive_rate': true_positive_rate,
        'true_positive_rate_loss': true_positive_rate_loss,
        'false_positive_rate': false_positive_rate,
        'tpr_weighted_bce_loss': tpr_weighted_bce_loss,
        'fpr_weighted_bce_loss': fpr_weighted_bce_loss,
        'tpr_weighted_cdc_loss': tpr_weighted_cdc_loss,
        'fpr_weighted_cdc_loss': fpr_weighted_cdc_loss,
        'continuous_dice_coef_loss': continuous_dice_coef_loss,
        'continuous_dice_coef': continuous_dice_coef,
        'dice_coef_loss': dice_coef_loss,
        'dice_coef': dice_coef,
        'bce_of_true_positive': bce_of_true_positive,
        'dice_of_true_positive_loss': dice_of_true_positive_loss,
        'cdc_of_true_positive_loss': cdc_of_true_positive_loss,
        'tpr_weighted_dice_loss': tpr_weighted_dice_loss,
        'fpr_weighted_dice_loss': fpr_weighted_dice_loss,
        }
