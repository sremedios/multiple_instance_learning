from time import strftime
from urllib.request import urlopen

from .skullstrip import skullstrip
from .reorient import orient, reorient

from multiprocessing.pool import ThreadPool

import os
import argparse
import numpy as np
import nibabel as nib
from sklearn.utils import shuffle
from skimage import measure
from skimage import morphology
from subprocess import Popen, PIPE
from tqdm import tqdm
import random
import copy
import csv
import shutil


def preprocess(filename, src_dir, dst_dir, tmp_dir, skullstrip_script_path, verbose=0,
               remove_tmp_files=True):
    '''
    Preprocesses a single file.
    Can be used in parallel

    1. skullstrip
    2. reorient to RAI

    Params: TODO
    Returns: TODO, the directory location of the final processed image

    '''
    if os.path.isdir(os.path.join(src_dir, filename)):
        return

    if os.path.exists(os.path.join(dst_dir, filename)):
        return

    ########## Directory Setup ##########
    SKULLSTRIP_DIR = os.path.join(tmp_dir, "skullstripped")
    ORIENT_DIR = os.path.join(tmp_dir, "orient")

    DIR_LIST = [SKULLSTRIP_DIR, ORIENT_DIR]

    for d in DIR_LIST:
        if not os.path.exists(d):
            os.makedirs(d)


    # apply preprocessing
    if not "mask" in filename:
        skullstrip(filename, src_dir, SKULLSTRIP_DIR, skullstrip_script_path, verbose)
        orient(filename, SKULLSTRIP_DIR, ORIENT_DIR, verbose)
    else:
        orient(filename, src_dir, ORIENT_DIR, verbose)

    # move to dst_dir
    final_preprocess_dir = ORIENT_DIR 
    shutil.copy(os.path.join(final_preprocess_dir, filename),
                os.path.join(dst_dir, filename))

    # delete intermediate files
    if remove_tmp_files:
        for d in DIR_LIST:
            tmp_file = os.path.join(d, filename)
            if os.path.exists(tmp_file):
                os.remove(tmp_file)

def preprocess_dir(train_dir, preprocess_dir, skullstrip_script_path):
    '''
    Preprocesses a directory in parallel using preprocess(...)

    Params: TODO
    '''

    TMPDIR = os.path.join(preprocess_dir,
                          "tmp_intermediate_preprocessing_steps")
    if not os.path.exists(TMPDIR):
        os.makedirs(TMPDIR)

    print("*** PREPROCESSSING ***")

    filenames = os.listdir(train_dir)

    # process using free threads, server-friendly
    tp = ThreadPool(30)
    for f in tqdm(filenames):
        tp.apply_async(preprocess(filename=f,
                                  src_dir=train_dir,
                                  dst_dir=preprocess_dir,
                                  tmp_dir=TMPDIR,
                                  verbose=0,
                                  skullstrip_script_path=skullstrip_script_path,
                                  remove_tmp_files=True))

    tp.close()
    tp.join()


    # if the preprocessed data exists, delete its folder
    if os.path.exists(TMPDIR):
        shutil.rmtree(TMPDIR)
