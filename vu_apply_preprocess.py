'''
Apply preprocessing to VU data
'''

import os
import sys
from tqdm import tqdm
from utils.preprocess import *

manual_annotations_filename = sys.argv[1]

with open(manual_annotations_filename, 'r') as f:
    lines = [l.strip().split() for l in f.readlines()]

filenames = []
mask_filenames = []
nifti_dirs = []
mask_dirs = []
preprocess_dirs = []
tmp_dirs = []

print("Reading files...")
for l in tqdm(lines):
    ##### TEMPORARY; ONLY PREPROCESS IMAGES FOR INLIER TRAINING #####
    if l[1] != '3':
        continue
    '''
    if l[1] == '9' or l[1] == '2':
        continue
    '''

    mask_dir, filename = os.path.split(l[0])
    nifti_dir = mask_dir.replace('mask', 'nifti')
    filename = filename.replace('.png', '.nii.gz')
    mask_filename = filename.replace('.nii.gz', '_predicted_mask.nii.gz')

    preprocess_dir = nifti_dir.replace('nifti', 'preprocessed')

    tmp_dir = os.path.join(
        preprocess_dir,
        "tmp_intermediate_preprocessing_steps"
    )

    '''
    if os.path.exists(os.path.join(preprocess_dir, filename)):
        print("Already preprocessed {}, continuing".format(filename))
        continue
    if not os.path.exists(os.path.join(nifti_dir, filename)):
        print("Requested file {} does not exist; skipping".format(
            filename)
        )
        continue
    '''
    # MASK ONLY
    if os.path.exists(os.path.join(preprocess_dir, mask_filename)):
        print("Already preprocessed {}, continuing".format(mask_filename))
        continue
    if not os.path.exists(os.path.join(mask_dir, mask_filename)):
        print("Requested file {} does not exist; skipping".format(
            mask_filename)
        )
        continue

    for d in [preprocess_dir, tmp_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    filenames.append(filename)
    mask_filenames.append(mask_filename)
    nifti_dirs.append(nifti_dir)
    mask_dirs.append(mask_dir)
    preprocess_dirs.append(preprocess_dir)
    tmp_dirs.append(tmp_dir)


print("Processing files...")
skullstrip_script_path = os.path.join(
            os.sep,
            "nfs",
            "share5", 
            "remedis", 
            "projects", 
            "multiple_instance_learning", 
            "utils", 
            "CT_BET.sh"
        )
tp = ThreadPool(30)
'''
for filename, nifti_dir, preprocess_dir, tmp_dir in\
        tqdm(zip(filenames, nifti_dirs, preprocess_dirs, tmp_dirs), total=len(filenames)):
    preprocess(
        filename=filename,
        src_dir=nifti_dir,
        dst_dir=preprocess_dir,
        tmp_dir=tmp_dir,
        skullstrip_script_path=skullstrip_script_path,
        verbose=0,
        remove_tmp_files=True,
    )
'''
# PROCESS ONLY MASKS
for mask_filename, mask_dir, preprocess_dir, tmp_dir in\
        tqdm(zip(mask_filenames, mask_dirs, preprocess_dirs, tmp_dirs), total=len(filenames)):
    preprocess(
        filename=mask_filename,
        src_dir=mask_dir,
        dst_dir=preprocess_dir,
        tmp_dir=tmp_dir,
        skullstrip_script_path=skullstrip_script_path,
        verbose=0,
        remove_tmp_files=True,
    )
tp.close()
tp.join()
