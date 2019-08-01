import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from utils.tfrecord_utils import *
from itertools import chain
tf.enable_eager_execution()
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

FIGURE_1_PATH = os.path.join(
    os.sep, 
    'nfs', 
    'share5', 
    'remedis', 
    'projects', 
    'multiple_instance_learning', 
    'figures', 
    'figure_1.png',
)


f = [x for x in os.listdir('.') if "tfrecord" in x]
f.sort()

ds_0 = tf.data.TFRecordDataset(f[0]).map(lambda r: parse_bag(r, (512, 512), 2)).take(350)

pos_slices = []
neg_slices = []
for (x, y) in ds_0:
    if tf.argmax(y).numpy() == 0 and len(neg_slices) < 50:
        neg_slices.append(x[len(x)//2, :, :, 0].numpy().T)
    elif len(pos_slices) < 50:
        pos_slices.append(x[len(x)//2, :, :, 0].numpy().T)


manually_chosen_pos_slices = [3, 9, 8, 17, 40]
manually_chosen_neg_slices = [1, 9, 15, 17, 29]


figure_slices = [pos_slices[i] for i in manually_chosen_pos_slices] +     [neg_slices[i] for i in manually_chosen_neg_slices]

columns = 5
rows = 2

fig, axs = plt.subplots(rows, columns, sharex=True, sharey=True, figsize=(512*columns/128, 512*rows/128))
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace = 0)
plt.tight_layout(-2)

for i, img in enumerate(figure_slices):
    
    img[np.where(img < 0)] = 0
    img[np.where(img > 150)] = 150
    img =  img/img.max() * 255
    img[np.where(img > 255)] = 255
    img[np.where(img < 0)] = 0
    
    
    if i < columns:
        cur_row = 0
    else:
        cur_row = 1
    axs[cur_row, i%columns].axis('off')
    axs[cur_row, i%columns].imshow(img, cmap='Greys_r', interpolation='nearest')
    
plt.savefig(FIGURE_1_PATH, dpi=300)    
