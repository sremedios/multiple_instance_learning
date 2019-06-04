import matplotlib as mpl
mpl.use("Agg")
import os
import matplotlib.pyplot as plt
import seaborn as sns

import random

LOGDIR = "logs"
plt.figure(figsize=(20,5))

filenames = [os.path.join(LOGDIR, x) for x in os.listdir(LOGDIR) if ".txt" in x]
filenames.sort()
TIME = 0
SITE = 1
TRAIN_LOSS = 2
VAL_LOSS = 3
CUR_PATIENCE = 4
BEST_LOSS = 5
CUR_EPOCH = 6

cur_ax = 0

colors = ["light blue", "orange", "dark red", "purply blue", "grass green", ]

for i in range(len(filenames)):
    epochs = []
    train_losses = []
    val_losses = []
    with open(filenames[i], 'r') as f:
        iter_lines = iter(f.readlines())
        next(iter_lines)
        for line in iter_lines:
            cur_data = line.split()

            epochs.append(int(cur_data[CUR_EPOCH]))
            train_losses.append(float(cur_data[TRAIN_LOSS]))
            val_losses.append(float(cur_data[VAL_LOSS]))

    if "full" in filenames[i]:
        session = "MSL Full"
    elif "other" in filenames[i]:
        session = "MSL 1/2 B"
    elif "half" in filenames[i]:
        session = "MSL 1/2 A"
    elif "nih" in filenames[i]:
        session = "SSL CNRM"
    elif "vu" in filenames[i]:
        session = "SSL VUMC"


    ax = sns.lineplot(x=epochs,
                      y=val_losses,
                      color=sns.xkcd_rgb[colors[i]],
                      label=session + " " + "Validation Loss")

    #ax.lines[cur_ax].set_linestyle("--")
    '''

    ax = sns.lineplot(x=epochs,
                      y=train_losses,
                      color=sns.xkcd_rgb[colors[i]],
                      label=session + " " + "Train Loss")
    cur_ax += 2 # used to manually make validation graph a dashed line
    '''

fontsize = 15

ax.set_xlabel("Epoch", fontsize=fontsize)
ax.set_ylabel("Loss", fontsize=fontsize)

plt.title("Model Validation Loss by Epoch", fontsize=fontsize)
plt.legend(prop={'size': fontsize})
plt.savefig("validation_loss_font_bigger.png")
#plt.show()
