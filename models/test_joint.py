from joint_network import *
import numpy as np
import tensorflow as tf
from old_losses import *

ds = 64

a_size = (32, 32, 32, 1)
b_size = (128, 128, 32, 1)

shared_weights = init_shared_weights(ds)
model = joint_unet(shared_weights, 1, ds, 1e-4)
model.compile(optimizer=tf.keras.optimizers.Adam(),
        loss={'seg_outputs': continuous_dice_coef_loss,
            'class_outputs': 'binary_crossentropy'},
        metrics={'seg_outputs': dice_coef,
            'class_outputs': 'accuracy'})


print(model.summary())

a_data = np.empty((10,) + a_size)
a_truth = np.empty((10,) + a_size)

b_data = np.empty((100,) + b_size)
b_truth = np.empty((100,)+ (4,))

a_dataset = tf.data.Dataset.from_tensor_slices((a_data, a_truth)).repeat()
b_dataset = tf.data.Dataset.from_tensor_slices((b_data, b_truth)).repeat()

joint_dataset = tf.data.Dataset.zip((a_dataset, b_dataset)).batch(2)

model.fit(joint_dataset, steps_per_epoch=10, epochs=2)
