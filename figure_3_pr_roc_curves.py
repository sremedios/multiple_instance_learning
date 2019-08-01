import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve, confusion_matrix, average_precision_score
import os
import sys
from utils.tfrecord_utils import *
from tqdm import tqdm
import json

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

tf.enable_eager_execution()

def mil_prediction(pred, n=1):
    i = tf.argsort(pred[:, 1], axis=0)
    i = i[len(pred) - n : len(pred)]
    return (tf.gather(pred, i), i)

dataset_count = int(sys.argv[1])


# load model
weight_path = "/nfs/share5/remedis/projects/multiple_instance_learning/models/weights/class_resnet/dataset_{}/best_weights_fold_1.h5".format(dataset_count)
model_path = "/nfs/share5/remedis/projects/multiple_instance_learning/models/weights/class_resnet/class_resnet.json"
with open(model_path) as json_data:
    model = tf.keras.models.model_from_json(json.load(json_data))
model.load_weights(weight_path)

DST_DIR = os.path.join("figures", "figure_3_dataset_{}".format(dataset_count))
if not os.path.exists(DST_DIR):
    os.makedirs(DST_DIR)


test_ds_filename = os.path.join(
    os.sep,
    'home',
    'remedis',
    'data',
    'dataset_fold___test.tfrecord',
)
test_dataset = tf.data.TFRecordDataset(test_ds_filename)\
        .map(lambda r: parse_bag(
            r,
            (512, 512),
            2,
            )
        )

num_pos = 0
num_neg = 0

y_true = []
y_prob = []
y_pred = []

print("Performing inference...")

# forward pass
for x, y in tqdm(test_dataset, total=4042):
    logits = model(x)
    pred, _ = mil_prediction(tf.nn.softmax(logits))
    
    y_true.append(tf.argmax(y).numpy())
    y_prob.append(pred.numpy()[0, 1])
    y_pred.append(tf.argmax(pred[0]).numpy())
    
    if tf.argmax(y).numpy() == 0:
        num_neg += 1
    else:
        num_pos += 1

print("Inference complete!  Generating figures...")


# PR Curve

precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
average_precision = average_precision_score(y_true, y_prob)
step_kwargs = {'step': 'post'}
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall Curve: AP={0:0.4f}'.format(average_precision))
plt.savefig(os.path.join(
    DST_DIR,
    'figure_3_dataset_{}_dataset_672_PR_Curve.png'.format(dataset_count),
))


# ROC Curve

fpr, tpr, thresholds = roc_curve(y_true, y_prob, pos_label=1)
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='red', lw=lw, label='ROC curve (area = {0:0.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.ylim([0.0, 1.05])
plt.xlim([-0.05, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve: Imbalanced Dataset. AUC={0:0.4f}'.format(roc_auc))
plt.savefig(os.path.join(
    DST_DIR,
    'figure_3_dataset_{}_dataset_672_ROC_Curve.png'.format(dataset_count),
))


# Confusion Matrix

class_names = ['Healthy', 'Large Hemorrhage']
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(10, 7))
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(
    xticks=np.arange(cm.shape[1]),
    yticks=np.arange(cm.shape[0]),
    xticklabels=class_names,
    yticklabels=class_names,
    title='Confusion Matrix',
    ylabel='True label',
    xlabel='Pred label',
)
plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

fmt = 'd'
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else 'black')
fig.tight_layout()
plt.savefig(os.path.join(
    DST_DIR,
    'figure_3_dataset_{}_Confusion_Matrix.png'.format(dataset_count),
))


# Normalized Confusion Matrix

class_names = ['Healthy', 'Large Hemorrhage']
cm = confusion_matrix(y_true, y_pred)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(10, 7))
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(
    xticks=np.arange(cm.shape[1]),
    yticks=np.arange(cm.shape[0]),
    xticklabels=class_names,
    yticklabels=class_names,
    title='Normalized Confusion Matrix',
    ylabel='True label',
    xlabel='Pred label',
)
plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

fmt = '.2f'
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else 'black')
fig.tight_layout()
plt.savefig(os.path.join(
    DST_DIR,
    'figure_3_dataset_{}_Normalized_Confusion_Matrix.png'.format(dataset_count),
))

