'''
Parses history object into loss log file
'''
import json
import os
from utils import logger

histories = {"vu": "unet_model_2018-11-22_03:49:55_unet_vu_ssl_history.json", 
             "nih": "unet_model_2018-11-22_03:50:11_unet_nih_ssl_history.json"}


for THIS_COMPUTER, h in histories.items():
    with open(h) as f:
        data = json.load(f)

    LOGFILE = os.path.join(THIS_COMPUTER + "_ssl_training_log.txt")
    if not os.path.exists(LOGFILE):
        os.system("touch " + LOGFILE)

    for cur_epoch in range(len(data['loss'])):
        cur_train_loss = data['loss'][cur_epoch]
        cur_val_loss = data['val_loss'][cur_epoch]

        # unimportant for these log files
        cur_patience = 0
        best_loss = 0

        logger.write_log(LOGFILE,
                         THIS_COMPUTER,
                         cur_train_loss,
                         cur_val_loss,
                         cur_patience,
                         best_loss,
                         cur_epoch)

