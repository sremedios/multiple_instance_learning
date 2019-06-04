import nibabel as nib
import numpy as np
import os
import sys
from PIL import Image
from .utils import remove_ext

def scale_ct_for_png(img_data):
    ''' 
    new_img_data = img_data + 1024
    new_img_data[np.where(new_img_data < 0)] = 0
    new_img_data = np.divide(new_img_data, 2000)
    new_img_data *= 255
    new_img_data[np.where(new_img_data > 255)] = 255
    new_img_data = new_img_data.astype(np.uint8).T
    ''' 
    new_img_data = img_data
    new_img_data[np.where(new_img_data < 0)] = 0
    #new_img_data = np.divide(new_img_data, np.max(new_img_data) * 0.4)
    new_img_data = np.divide(new_img_data, np.max(new_img_data))
    new_img_data *= 255
    new_img_data = new_img_data.astype(np.uint8).T
    
    return new_img_data

# TODO: handle saving binary images
def save_slice(filename, ct_img_data, pred_mask_img_data, gt_mask_img_data, slices_dice, result_dst):
    # also ensure to get screencaps of these specific slices
    specified_slices = [10, 12, 14, 16, 18]

    best_dice = 0
    worst_dice = 1

    best_slice_idx = 0
    worst_slice_idx = 0
    
    for idx, slice_dice in enumerate(slices_dice):
        if slice_dice != 1 and best_dice < slice_dice:
            best_dice = slice_dice
            best_slice_idx = idx

        if slice_dice != 0 and worst_dice > slice_dice:
            worst_dice = slice_dice
            worst_slice_idx = idx

    for img_data in [ct_img_data, pred_mask_img_data, gt_mask_img_data]:
        best_slice = img_data[:,:,best_slice_idx]
        best_slice = scale_ct_for_png(best_slice)
        best_im = Image.fromarray(best_slice).convert('LA')

        worst_slice = img_data[:,:,worst_slice_idx]
        worst_slice = scale_ct_for_png(worst_slice)
        worst_im = Image.fromarray(worst_slice).convert('LA')

        if img_data is ct_img_data:
            best_slice_filename = remove_ext(filename) + "_best_slice_" + str(best_slice_idx).zfill(2) + "_orig.png"
            worst_slice_filename = remove_ext(filename) + "_worst_slice_" + str(worst_slice_idx).zfill(2) + "_orig.png"
        elif img_data is pred_mask_img_data:
            best_slice_filename = remove_ext(filename) + "_best_slice_" + str(best_slice_idx).zfill(2) + "_pred.png"
            worst_slice_filename = remove_ext(filename) + "_worst_slice_" + str(worst_slice_idx).zfill(2) + "_pred.png"
        elif img_data is gt_mask_img_data:
            best_slice_filename = remove_ext(filename) + "_best_slice_" + str(best_slice_idx).zfill(2) + "_gt.png"
            worst_slice_filename = remove_ext(filename) + "_worst_slice_" + str(worst_slice_idx).zfill(2) + "_gt.png"

        best_im.save(os.path.join(result_dst, best_slice_filename))
        worst_im.save(os.path.join(result_dst, worst_slice_filename))

        for specified_idx in specified_slices:
            if specified_idx in [best_slice_idx, worst_slice_idx]:
                continue

            # only gather specified slices if possible
            if specified_idx >= img_data.shape[-1]:
                continue

            cur_slice = img_data[:,:,specified_idx]
            cur_slice = scale_ct_for_png(cur_slice)
            cur_im = Image.fromarray(cur_slice).convert('LA')

            if img_data is ct_img_data:
                cur_slice_filename = remove_ext(filename) + "_specified_slice_" + str(specified_idx).zfill(2) + "_orig.png"
            elif img_data is pred_mask_img_data:
                cur_slice_filename = remove_ext(filename) + "_specified_slice_" + str(specified_idx).zfill(2) + "_pred.png"
            elif img_data is gt_mask_img_data:
                cur_slice_filename = remove_ext(filename) + "_specified_slice_" + str(specified_idx).zfill(2) + "_gt.png"

            cur_im.save(os.path.join(result_dst, cur_slice_filename))
