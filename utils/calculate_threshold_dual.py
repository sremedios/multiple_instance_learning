'''
Author: Samuel Remedios

Optimizing threshold for lesion memberships with semi-exhaustive search.
'''
import os
import numpy as np
import nibabel as nib
from utils import utils
from utils.apply_model import apply_model, apply_model_single_input
from keras.models import load_model
from keras import backend as K
from models.losses import *
import csv
from tqdm import tqdm


def get_dice(img1, img2):
    '''
    Returns the dice score as a voxel-wise comparison of two nifti files.

    Params:
        - img1: ndarray, tensor of first .nii.gz file 
        - img2: ndarray, tensor of second .nii.gz file 
    Returns:
        - dice: float, the dice score between the two files
    '''

    empty_score = 1.0

    img_data_1 = img1.astype(np.bool)
    img_data_2 = img2.astype(np.bool)

    if img_data_1.shape != img_data_2.shape:
        raise ValueError("Shape mismatch between files")

    img_sum = img_data_1.sum() + img_data_2.sum()
    if img_sum == 0:
        return empty_score

    intersection = np.logical_and(img_data_1, img_data_2)

    return 2. * intersection.sum() / img_sum


def calc_dice(filename, gt_filename, threshold):
    '''
    Calculates threshold with a semi-exhaustive search by finding DICE values
    using thresholds between [0.1, 0.9] and taking steps of size h.

    Params:
        - filename: string, name of segmentation NIFTI file
        - gt_filename: string, name of ground truth NIFTI file
        - suffix: string, _CNNLesionMembership.nii.gz as specified in Roy's seg script

    Returns:
        - threshold: float, threshold which returns the best DICE for this filename
    '''

    ##### GROUND TRUTH #####
    nii_obj_gt = nib.load(gt_filename)
    img_data_gt = nii_obj_gt.get_data()

    zooms_gt = nii_obj_gt.header.get_zooms()
    scaling_factor_gt = zooms_gt[0] * zooms_gt[1] * zooms_gt[2]

    ##### SEGMENTATION DATA #####
    nii_obj = nib.load(filename)
    img_data = nii_obj.get_data()

    zooms = nii_obj.header.get_zooms()
    scaling_factor = zooms[0] * zooms[1] * zooms[2]

    thresh_data_gt = img_data_gt.copy()
    thresh_data_gt[np.where(thresh_data_gt < threshold)] = 0
    thresh_data_gt[np.where(thresh_data_gt >= threshold)] = 1

    thresh_data = img_data.copy()
    thresh_data[np.where(thresh_data < threshold)] = 0
    thresh_data[np.where(thresh_data >= threshold)] = 1

    # dice calculation
    dice = get_dice(thresh_data, thresh_data_gt)

    # volume calculation
    thresholded_vol_mm_gt = scaling_factor_gt * np.sum(thresh_data_gt)
    thresholded_vol_mm = scaling_factor * np.sum(thresh_data)

    return dice, thresholded_vol_mm, thresholded_vol_mm_gt


if __name__ == "__main__":

    ######################## COMMAND LINE ARGUMENTS ########################
    results = utils.parse_args("validate")
    num_channels = results.num_channels
    model_filename = results.weights
    experiment_details = os.path.basename(model_filename)[:os.path.basename(model_filename)
                                                          .find('.hdf5')]
    DATA_DIR = results.VAL_DIR

    ######################## PATH CONSTANTS ########################
    # script constants
    SEG_SUFFIX = "_CNNLesionMembership.nii.gz"

    SEG_SCRIPT_PATH = os.path.join("TBISegmentation_for_CT_Test.py")

    ######################## PREPROCESS TESTING DATA ########################
    SKULLSTRIP_SCRIPT_PATH = os.path.join("utils", "CT_BET.sh")
    N4_SCRIPT_PATH = os.path.join("utils", "N4BiasFieldCorrection")
    PREPROCESSING_DIR = os.path.join(DATA_DIR, "preprocessing")
    SEG_ROOT_DIR = os.path.join(DATA_DIR, "segmentations")
    STATS_DIR = os.path.join("results")
    SEG_DIR = os.path.join(SEG_ROOT_DIR, experiment_details)
    REORIENT_DIR = os.path.join(SEG_DIR, "reoriented")

    for d in [PREPROCESSING_DIR, SEG_ROOT_DIR, STATS_DIR, SEG_DIR, REORIENT_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)
    # Stats file
    stat_filename = "result_" + experiment_details + ".csv"
    STATS_FILE = os.path.join(STATS_DIR, stat_filename)

    ######################## PREPROCESSING ########################
    filenames = [x for x in os.listdir(DATA_DIR)
                 if not os.path.isdir((os.path.join(DATA_DIR, x)))]
    filenames.sort()
    for filename in filenames:
        final_preprocess_dir = utils.preprocess(filename,
                                                DATA_DIR,
                                                PREPROCESSING_DIR,
                                                SKULLSTRIP_SCRIPT_PATH,
                                                N4_SCRIPT_PATH)

    ######################## LOAD MODEL ########################
    model = load_model(model_filename, custom_objects=custom_losses)

    ######################## SEGMENT FILES ########################
    filenames = [x for x in os.listdir(final_preprocess_dir)
                 if not os.path.isdir((os.path.join(DATA_DIR, x)))]
    masks = [x for x in filenames if "mask" in x]
    filenames = [x for x in filenames if "CT" in x]
    filenames.sort()
    masks.sort()
    if len(filenames) != len(masks):
        print("Error, file missing. #CT:{}, #masks:{}".format(
            len(filenames), len(masks)))
        import sys
        sys.exit()

    print("Using model:", model_filename)

    segmented_filenames = []
    for filename, mask in zip(filenames, masks):
        # filenames
        segmented_filename = os.path.join(SEG_DIR, filename)
        segmented_filenames.append(segmented_filename)

        # skip if segmentation complete
        if os.path.exists(segmented_filename):
            continue

        # load nifti file data
        nii_obj = nib.load(os.path.join(final_preprocess_dir, filename))
        nii_img = nii_obj.get_data()
        header = nii_obj.header
        affine = nii_obj.affine

        # reshape to account for implicit "1" channel
        nii_img = np.reshape(nii_img, nii_img.shape + (1,))

        print("Max: {} Min: {}".format(np.max(nii_img), np.min(nii_img)))

        # segment
        segmented_img = apply_model(nii_img, model)
        #segmented_img = apply_model_single_input(nii_img, model)
        # save resultant image
        segmented_nii_obj = nib.Nifti1Image(
            segmented_img, affine=affine, header=header)
        nib.save(segmented_nii_obj, segmented_filename)

    ######################## FIND BEST THRESHOLD ########################

    h = 0.025  # step size to check for in exhaustive search
    init = 0.5
    halt = 1
    patience = 10
    cur_iter = 0

    best_dice_opt_for_dice = 0
    best_corr_opt_for_dice = 0
    threshold_opt_for_dice = 0
    
    best_dice_opt_for_corr = 0
    best_corr_opt_for_corr = 0
    threshold_opt_for_corr = 0

    best_dice_opt_for_both = 0
    best_corr_opt_for_both = 0
    threshold_opt_for_both = 0

    cur_threshold = init

    masks.sort()
    segmented_filenames.sort()

    while cur_threshold < halt and cur_iter < patience:
        print("** Current Threshold: {:.3f} **".format(cur_threshold))

        avg_dice = 0
        vols = []
        vols_gt = []

        for segmented_filename, mask in tqdm(zip(segmented_filenames, masks),
                                             total=len(segmented_filenames)):
            ground_truth_filename = os.path.join(final_preprocess_dir, mask)

            #print("Comparing:\n\t{}\n\t{}".format(segmented_filename, ground_truth_filename))

            # optimize threshold
            cur_dice, cur_vol, cur_vol_gt = calc_dice(segmented_filename,
                                                      ground_truth_filename,
                                                      cur_threshold)
            avg_dice += cur_dice
            vols.append(cur_vol)
            vols_gt.append(cur_vol_gt)

        avg_dice /= len(segmented_filenames)
        vols = np.array(vols)
        vols_gt = np.array(vols_gt)
        corr = np.corrcoef(vols, vols_gt)[0, 1]

        if np.abs(corr) > best_corr_opt_for_corr:
            threshold_opt_for_corr = cur_threshold
            best_dice_opt_for_corr = avg_dice
            best_corr_opt_for_corr = np.abs(corr)

        if avg_dice > best_dice_opt_for_dice:
            threshold_opt_for_dice = cur_threshold
            best_dice_opt_for_dice = avg_dice
            best_corr_opt_for_dice = np.abs(corr)

        if avg_dice > best_dice_opt_for_both and np.abs(corr) > best_corr_opt_for_both:
            threshold_opt_for_both = cur_threshold
            best_dice_opt_for_both = avg_dice
            best_corr_opt_for_both = np.abs(corr)
        else:
            cur_iter += 1

        print("=== Optimizing for Dice ===")
        print("Best_threshold: {:.4f}\tBest Dice: {:.4f}\tBest Corr: {:.4f}\n".format(
            threshold_opt_for_dice, best_dice_opt_for_dice, best_corr_opt_for_dice))

        print("=== Optimizing for Correlation ===")
        print("Best_threshold: {:.4f}\tBest Dice: {:.4f}\tBest Corr: {:.4f}\n".format(
            threshold_opt_for_corr, best_dice_opt_for_corr, best_corr_opt_for_corr))

        print("=== Optimizing for Both ===")
        print("Best_threshold: {:.4f}\tBest Dice: {:.4f}\tBest Corr: {:.4f}\n".format(
            threshold_opt_for_both, best_dice_opt_for_both, best_corr_opt_for_both))

        cur_threshold += h

    print("Best threshold for Dice: {}".format(threshold_opt_for_dice))
    print("Best threshold for Correlation: {}".format(threshold_opt_for_corr))
    print("Best threshold for Both: {}".format(threshold_opt_for_both))
    threshold_path = os.path.join("results", "thresholds")
    if not os.path.exists(threshold_path):
        os.makedirs(threshold_path)

    '''
    with open(os.path.join(threshold_path, "best_threshold.txt"), 'w') as f:
        f.write("Best_threshold: {}".format(np.around(threshold, decimals=6)))
    '''

    K.clear_session()
