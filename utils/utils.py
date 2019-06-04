from time import strftime
from datetime import datetime
from urllib.request import urlopen
from urllib.error import URLError

from .skullstrip import skullstrip
from .reorient import orient, reorient
from .pad import pad_image

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

def save_args_to_csv(args_obj, out_dir):
    '''
    Saves arguments to a csv for future reference

    args_obj: argparse object, collected after running parse_args
    out_dir: string, path where to save the csv file
    '''

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out_file = os.path.join(out_dir, "script_arguments.csv")
    with open(out_file, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(["Argument", "Value"])

        for arg in vars(args_obj):
            writer.writerow([arg, getattr(args_obj, arg)])


def preprocess(filename, src_dir, preprocess_root_dir, skullstrip_script_path, n4_script_path,
               verbose=0):
    '''
    Preprocesses an image:
    1. skullstrip
    2. N4 bias correction
    3. resample
    4. reorient to RAI

    Params: TODO
    Returns: TODO, the directory location of the final processed image

    '''
    ########## Directory Setup ##########
    SKULLSTRIP_DIR = os.path.join(preprocess_root_dir, "skullstripped")
    N4_DIR = os.path.join(preprocess_root_dir, "n4_bias_corrected")
    RESAMPLE_DIR = os.path.join(preprocess_root_dir, "resampled")
    RAI_DIR = os.path.join(preprocess_root_dir, "RAI")

    for d in [SKULLSTRIP_DIR, N4_DIR, RESAMPLE_DIR, RAI_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)

    if "CT" in filename:
        skullstrip(filename, src_dir, SKULLSTRIP_DIR,
                   skullstrip_script_path, verbose)
        #n4biascorrect(filename, SKULLSTRIP_DIR, N4_DIR, n4_script_path, verbose)
        #resample(filename, N4_DIR, RESAMPLE_DIR, verbose)
        #orient(filename, SKULLSTRIP_DIR, RAI_DIR, verbose)
    '''
    elif "mask" in filename or "multiatlas" in filename:
        #resample(filename, src_dir, RESAMPLE_DIR, verbose)
        #orient(filename, SKULLSTRIP_DIR, RAI_DIR, verbose)
    '''

    final_preprocess_dir = SKULLSTRIP_DIR

    return final_preprocess_dir


def parse_args(session):
    '''
    Parse command line arguments.

    Params:
        - session: string, one of "train", "validate", or "test"
    Returns:
        - parse_args: object, accessible representation of arguments
    '''
    parser = argparse.ArgumentParser(
        description="Arguments for Training and Testing")

    if session == "train":
        parser.add_argument('--datadir', required=True, action='store', dest='SRC_DIR',
                            help='Where the initial unprocessed data is. See readme for\
                                    further information')
        parser.add_argument('--plane', required=False, action='store', dest='plane',
                            default='axial', type=str,
                            help='Which plane to train the model on. Default is axial. \
                                    Other options are only "sagittal" or "coronal".')
        parser.add_argument('--psize', required=True, action='store', dest='patch_size',
                            help='Patch size, eg: 45x45. Patch sizes are separated by x\
                                    and in voxels')
        parser.add_argument('--batch_size', required=False, action='store', dest='batch_size',
                            default=256, type=int,
                            help='Batch size for training.')
        parser.add_argument('--loss', required=False, action='store', dest='loss',
                            default='cdc', type=str,
                            help='Loss for the model to optimize over. Options are: \
                            bce, dice_coef, tpr, cdc, tpw_cdc, bce_tp')
        parser.add_argument('--model', required=False, action='store', dest='model',
                            default=None,
                            help='If provided, picks up training from this model.')
        parser.add_argument('--experiment_details', required=False, action='store',
                            dest='experiment_details', default='experiment_details', type=str,
                            help='Description of experiment, used to create folder to save\
                                    weights.')
        parser.add_argument('--num_patches', required=False, action='store', dest='num_patches',
                            default=1500000, type=int,
                            help='Maximum allowed number of patches. Default is all possible.')
    elif session == "test":
        parser.add_argument('--infile', required=True, action='store', dest='INFILE',
                            help='Image to segment')
        parser.add_argument('--weights', required=True, action='store', dest='weights',
                            help='Learnt weights (.hdf5) file')
        parser.add_argument('--outdir', required=True, action='store', dest='segdir',
                            help='Directory in which to place segmentations')
    elif session == "validate":
        parser.add_argument('--datadir', required=True, action='store', dest='VAL_DIR',
                            help='Where the initial unprocessed data is')
        parser.add_argument('--weights', required=True, action='store', 
                            dest='weights',
                            help='Learnt weights on axial plane (.hdf5) file')
        parser.add_argument('--threshold', required=False, action='store', dest='threshold',
                            type=float, default=0.5,
                            help='Scalar in [0,1] to use as binarizing threshold.')
    elif session == "multiseg":
        parser.add_argument('--datadir', required=True, action='store', dest='DATA_DIR',
                            help='Where the initial unprocessed data is')
        parser.add_argument('--weights', required=True, action='store', 
                            dest='weights',
                            help='Learnt weights on axial plane (.hdf5) file')
    elif session == "calc_dice":
        parser.add_argument('--gt_dir', required=True, action='store', dest='GT_DATA_DIR',
                            help='Where the manual masks are')
        parser.add_argument('--indata', required=True, action='store', dest='IN_DATA',
                            help='Predicted data, either a file or directory')
    else:
        print("Invalid session. Must be one of \"train\", \"validate\", or \"test\"")
        sys.exit()

    parser.add_argument('--num_channels', required=False, type=int, action='store',
                        dest='num_channels', default=1,
                        help='Number of channels to include. First is CT, second is atlas,\
                                third is unskullstripped CT')
    parser.add_argument('--gpuid', required=False, action='store', type=int, dest='GPUID',
                        help='For a multi-GPU system, the trainng can be run on different GPUs.\
                        Use a GPU id (single number), eg: 1 or 2 to run on that particular GPU.\
                        0 indicates first GPU.  Optional argument. Default is the first GPU.\
                        -1 for all GPUs.')


    return parser.parse_args()


def now():
    '''
    Formats time for use in the log file
    Pulls time from internet in UTC to sync properly
    '''
    try:
        res = urlopen('http://just-the-time.appspot.com/')
        result = res.read().strip()
        result_str = result.decode('utf-8')
        result_str = result_str.split()
        result_str = '_'.join(result_str)
    except URLError as err:
        result_str = strftime("%Y-%m-%d_%H:%M:%S", datetime.now())

    return result_str


def write_log(log_file, host_id, acc, val_acc, loss):
    update_log_file = False
    new_log_file = False
    with open(log_file, 'r') as f:
        logfile_data = [x.split() for x in f.readlines()]
        if (len(logfile_data) >= 1 and logfile_data[-1][1] != host_id)\
                or len(logfile_data) == 0:
            update_log_file = True
        if len(logfile_data) == 0:
            new_log_file = True
    if update_log_file:
        with open(log_file, 'a') as f:
            if new_log_file:
                f.write("{:<30}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\n".format(
                    "timestamp",
                    "host_id",
                    "train_acc",
                    "val_acc",
                    "loss",))
            f.write("{:<30}\t{:<10}\t{:<10.4f}\t{:<10.4f}\t{:<10.4f}\n".format(
                    now(),
                    host_id,
                    acc,
                    val_acc,
                    loss,))


def remove_ext(filename):
    if ".nii" in filename:
        return filename[:filename.find(".nii")]
    else:
        return filename


def get_root_filename(filename):
    if "CT" in filename:
        return filename[:filename.find("_CT")]
    elif "mask" in filename:
        return filename[:filename.find("_mask")]
    else:
        return filename


def get_dice(img1, img2):
    '''
    Returns the dice score as a voxel-wise comparison of two nifti files.

    Params:
        - img1: ndarray, tensor of first .nii.gz file 
        - img2: ndarray, tensor of second .nii.gz file 
    Returns:
        - dice: float, the dice score between the two files
    '''

    img_data_1 = img1.astype(np.bool)
    img_data_2 = img2.astype(np.bool)

    if img_data_1.shape != img_data_2.shape:
        print("Pred shape", img_data_1.shape)
        print("GT shape", img_data_2.shape)
        raise ValueError("Shape mismatch between files")

    volume_dice = dice_metric(img_data_1.flatten(), img_data_2.flatten())
    slices_dice = []
    for slice_idx in range(img_data_1.shape[2]):
        slices_dice.append(dice_metric(img_data_1[:, :, slice_idx],
                                       img_data_2[:, :, slice_idx]))

    return volume_dice, slices_dice


def dice_metric(A, B):
    '''
    Dice calculation over two BOOLEAN numpy tensors
    '''
    union = A.sum() + B.sum()
    intersection = np.logical_and(A, B).sum()

    if union == 0:
        return 1.0

    return 2.0 * intersection / union


def write_stats(filename, nii_obj, nii_obj_gt, stats_file, threshold=0.5):
    '''
    Writes to csv probability volumes and thresholded volumes.

    Params:
        - filename: string, name of the subject/file which was segmented
        - nii_obj: nifti object, segmented CT
        - nii_obj_gt: nifti object, ground truth segmentation
        - stats_file: string, path and filename of .csv file to hold statistics
    '''
    SEVERE_HEMATOMA = 25000  # in mm^3

    # get ground truth severity
    img_data_gt = nii_obj_gt.get_data()

    # pad ground truth
    img_data_gt = pad_image(img_data_gt)

    zooms_gt = nii_obj_gt.header.get_zooms()
    scaling_factor_gt = zooms_gt[0] * zooms_gt[1] * zooms_gt[2]

    # get volumes
    probability_vol_gt = np.sum(img_data_gt)
    prob_thresh_vol_gt = np.sum(
        img_data_gt[np.where(img_data_gt >= threshold)])

    thresh_data_gt = img_data_gt.copy()
    thresh_data_gt[np.where(thresh_data_gt < threshold)] = 0
    thresh_data_gt[np.where(thresh_data_gt >= threshold)] = 1
    thresholded_vol_gt = np.sum(thresh_data_gt)

    thresholded_vol_mm_gt = scaling_factor_gt * thresholded_vol_gt

    # classify severity of largest hematoma in ground truth
    label_gt = measure.label(img_data_gt)
    props_gt = measure.regionprops(label_gt)
    if len(props_gt) > 0:
        areas = [x.area for x in props_gt]
        areas.sort()
        largest_contig_hematoma_vol_mm_gt = areas[-1] * scaling_factor_gt
    else:
        largest_contig_hematoma_vol_mm_gt = 0

    if largest_contig_hematoma_vol_mm_gt > SEVERE_HEMATOMA:
        severe_gt = 1
    else:
        severe_gt = 0

    ##### SEGMENTATION DATA #####

    # load object tensor for calculations
    img_data = nii_obj.get_data()[:, :, :]
    img_data = pad_image(img_data)

    zooms = nii_obj.header.get_zooms()
    scaling_factor = zooms[0] * zooms[1] * zooms[2]

    # get volumes
    probability_vol = np.sum(img_data)
    prob_thresh_vol = np.sum(img_data[np.where(img_data >= threshold)])

    thresh_data = img_data.copy()
    thresh_data[np.where(thresh_data < threshold)] = 0
    thresh_data[np.where(thresh_data >= threshold)] = 1
    thresholded_vol = np.sum(thresh_data)

    probability_vol_mm = scaling_factor * probability_vol
    prob_thresh_vol_mm = scaling_factor * prob_thresh_vol
    thresholded_vol_mm = scaling_factor * thresholded_vol

    # classify severity of hematoma in seg
    label = measure.label(thresh_data)
    props = measure.regionprops(label)

    if len(props) > 0:
        areas = [x.area for x in props]
        areas.sort()
        largest_contig_hematoma_vol_mm = areas[-1] * scaling_factor
    else:
        largest_contig_hematoma_vol_mm = 0

    # remove small lesions
    smallest_lesion_size = 27 
    thresh_data = morphology.remove_small_objects(label, smallest_lesion_size)
    # turn results back into binary values
    thresh_data[np.where(thresh_data < threshold)] = 0
    thresh_data[np.where(thresh_data >= threshold)] = 1

    ############## record results ############

    if largest_contig_hematoma_vol_mm > SEVERE_HEMATOMA:
        severe_pred = 1
    else:
        severe_pred = 0

    volume_dice, slices_dice = get_dice(thresh_data, thresh_data_gt)

    # write to file the two sums
    if not os.path.exists(stats_file):
        with open(stats_file, 'w') as csvfile:
            fieldnames = [
                "filename",
                "dice",
                "thresholded volume(mm)",
                "thresholded volume ground truth(mm)",
                "largest hematoma ground truth(mm)",
                "largest hematoma prediction(mm)",
                "severe hematoma ground truth",
                "severe hematoma pred",
                "vox dim 1(mm)",
                "vox dim 2(mm)",
                "vox dim 3(mm)",
                "probability vol(mm)",
                "probability volume(voxels)",
                "thresholded volume(voxels)",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    with open(stats_file, 'a') as csvfile:
        fieldnames = [
            "filename",
            "dice",
            "thresholded volume(mm)",
            "thresholded volume ground truth(mm)",
            "largest hematoma ground truth(mm)",
            "largest hematoma prediction(mm)",
            "severe hematoma ground truth",
            "severe hematoma pred",
            "vox dim 1(mm)",
            "vox dim 2(mm)",
            "vox dim 3(mm)",
            "probability vol(mm)",
            "probability volume(voxels)",
            "thresholded volume(voxels)",
        ]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writerow({
                        "filename": os.path.basename(filename),
                        "dice": volume_dice,
                        "thresholded volume(mm)": thresholded_vol_mm,
                        "thresholded volume ground truth(mm)": thresholded_vol_mm_gt,
                        "largest hematoma ground truth(mm)": largest_contig_hematoma_vol_mm_gt,
                        "largest hematoma prediction(mm)": largest_contig_hematoma_vol_mm,
                        "severe hematoma ground truth": severe_gt,
                        "severe hematoma pred": severe_pred,
                        "vox dim 1(mm)": zooms[0],
                        "vox dim 2(mm)": zooms[1],
                        "vox dim 3(mm)": zooms[2],
                        "probability vol(mm)": probability_vol_mm,
                        "probability volume(voxels)": probability_vol,
                        "thresholded volume(voxels)": thresholded_vol,
                        })

    return volume_dice, slices_dice, thresholded_vol_mm, thresholded_vol_mm_gt


def write_dice_scores(filename, volume_dice, slices_dice, results_dst):
    if not os.path.exists(results_dst):
        with open(results_dst, 'w') as csvfile:
            fieldnames = [
                "filename",
                "volume_dice",
                "slice_idx",
                "slices_dice",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    for idx, slice_dice in enumerate(slices_dice):
        with open(results_dst, 'a') as csvfile:
            fieldnames = [
                "filename",
                "volume_dice",
                "slice_idx",
                "slices_dice",
            ]

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writerow({
                "filename": os.path.basename(filename),
                "volume_dice": volume_dice,
                "slice_idx": idx,
                "slices_dice": slice_dice,
            })


def threshold(filename, src_dir, dst_dir, threshold=0.5):
    '''
    Saves the thresholded image to the destination directory.
    Calls write_stats() to save statistics to file

    Params:
        - filename: string, name of segmentation NIFTI file
        - src_dir: string, source directory where segmented NIFTI file exists
        - dst_dir: string, destination directory
        - threshold: float in [0,1], threshold at which to split between 0 and 1
    '''
    seg_filename = filename
    # load object tensor for calculations
    nii_obj = nib.load(os.path.join(src_dir, seg_filename))
    img_data = nii_obj.get_data()

    # threshold image and save thresholded image
    img_data[np.where(img_data < threshold)] = 0
    img_data[np.where(img_data >= threshold)] = 1

    # remove small segmentation areas
    label = measure.label(img_data)
    smallest_lesion_size = 27 
    img_data = morphology.remove_small_objects(label, smallest_lesion_size)
    # turn results back into binary values
    img_data[np.where(img_data < threshold)] = 0
    img_data[np.where(img_data >= threshold)] = 1

    thresh_obj = nib.Nifti1Image(
        img_data, affine=nii_obj.affine, header=nii_obj.header)
    nib.save(thresh_obj, os.path.join(
        dst_dir, get_root_filename(seg_filename)+"_thresh.nii.gz"))

