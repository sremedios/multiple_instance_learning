import os
import numpy as np
import nibabel as nib
from sklearn.utils import shuffle
from tqdm import tqdm
from .pad import *
import random
import copy
from time import strftime, time

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys


def PadImage(vol, padsize):
    dim = vol.shape
    padsize = np.asarray(padsize, dtype=int)
    dim2 = dim+2*padsize
    temp = np.zeros(dim2, dtype=float)
    temp[padsize:dim[0]+padsize,
         padsize:dim[1]+padsize,
         padsize:dim[2]+padsize] = vol
    return temp


def get_intersection(a, b):
    '''
    Gets intersection of coordinate arrays a and b returned from np.nonzero()-style functions.
    Used to find valid centers for healthy patches.

    TODO: CURRENTLY ONLY WORKS ON 3D ARRAYS

    Params:
        - a: tuple of N np.arrays, where N is the dimension of the original image
        - b: tuple of N np.arrays, where N is the dimension of the original image
    Returns:
        - intersection: set of tuples of rank N, where N is the dimension of the original image
    '''
    # TODO: this is the slowest operation
    start_time = time()
    print(np.array(a).shape)
    print(np.array(b).shape)

    if len(a) == 3:
        # first change format to be a list of coordinates rather than one list per dimension
        a_reformat = [(x, y, z) for x, y, z in zip(a[0], a[1], a[2])]
        b_reformat = [(x, y, z) for x, y, z in zip(b[0], b[1], b[2])]
    elif len(a) == 2:
        a_reformat = [(x, y) for x, y in zip(a[0], a[1])]
        b_reformat = [(x, y) for x, y in zip(b[0], b[1])]
    else:  # TODO: proper error handling
        print("Shape mismatch")

    intersection = set(a_reformat) & set(b_reformat)
    print("Time taken to calculate intersection:",
          time() - start_time, "seconds")

    return intersection


def get_center_coords(ct, mask, ratio):
    '''
    Gets coordinates for center pixel of all patches.

    Params:
        - ct: 3D ndarray, image data from which to find healthy coordinates
        - mask: 3D ndarray, image data from which to find coordinates
        - ratio: float in [0,1].
                 If 0 or 1, skip intersection calculation for speed
    Returns:
        - healthy_coords: set of tuples of rank 3, coordinates of healthy voxels
        - lesion_coords: set of tuples of rank 3, coordinates of lesion voxels
    '''

    # These first two must be shuffled.
    if ratio == 1:
        # ct-valid patches
        ct_possible_centers = np.nonzero(ct)
        healthy_coords = [(x, y, z) for x, y, z in zip(ct_possible_centers[0],
                                                       ct_possible_centers[1],
                                                       ct_possible_centers[2])]
        healthy_coords = set(shuffle(healthy_coords, random_state=0))
        lesion_coords = {(0, 0, 0), (0, 0, 0), (0, 0, 0), }
    elif ratio == 0:
        healthy_coords = {(0, 0, 0), (0, 0, 0), (0, 0, 0), }
        # mask lesion patches
        lesion_coords = np.nonzero(mask)
        # cuurently only works for 3D input images
        lesion_coords = [(x, y, z) for x, y, z in zip(lesion_coords[0],
                                                      lesion_coords[1],
                                                      lesion_coords[2])]
        lesion_coords = set(shuffle(lesion_coords, random_state=0))
    else:
        # ct-valid patches
        ct_possible_centers = np.nonzero(ct)
        zeros_coords = np.where(mask == 0)
        healthy_coords = get_intersection(ct_possible_centers, zeros_coords)
        # mask lesion patches
        lesion_coords = np.nonzero(mask)
        # currently only works for 3D input images
        # This does not need to be shuffled since it will be shuffled later
        lesion_coords = set([(x, y, z) for x, y, z in zip(lesion_coords[0],
                                                          lesion_coords[1],
                                                          lesion_coords[2])])

    return healthy_coords, lesion_coords


def get_patches(invols, mask, patchsize, maxpatch, num_channels):
    rng = random.SystemRandom()

    mask = np.asarray(mask, dtype=np.float32)
    patch_size = np.asarray(patchsize, dtype=int)
    dsize = np.floor(patch_size/2).astype(dtype=int)

    # find indices of all lesions in mask volume
    mask_lesion_indices = np.nonzero(mask)
    mask_lesion_indices = np.asarray(mask_lesion_indices, dtype=int)
    total_lesion_patches = len(mask_lesion_indices[0])

    num_patches = np.minimum(maxpatch, total_lesion_patches)
    '''
    print("Number of patches used: {} out of {} (max: {})"
          .format(num_patches,
                  total_lesion_patches,
                  maxpatch))
    '''

    randidx = rng.sample(range(0, total_lesion_patches), num_patches)
    # here, 3 corresponds to each axis of the 3D volume
    shuffled_mask_lesion_indices = np.ndarray((3, num_patches))
    for i in range(0, num_patches):
        for j in range(0, 3):
            shuffled_mask_lesion_indices[j,
                                         i] = mask_lesion_indices[j, randidx[i]]
    shuffled_mask_lesion_indices = np.asarray(
        shuffled_mask_lesion_indices, dtype=int)

    # mask out all lesion indices to get all healthy indices
    tmp = copy.deepcopy(invols[0])
    tmp[tmp > 0] = 1
    tmp[tmp <= 0] = 0
    tmp = np.multiply(tmp, 1-mask)

    healthy_brain_indices = np.nonzero(tmp)
    healthy_brain_indices = np.asarray(healthy_brain_indices, dtype=int)
    num_healthy_indices = len(healthy_brain_indices[0])

    randidx0 = rng.sample(range(0, num_healthy_indices), num_patches)
    # here, 3 corresponds to each axis of the 3D volume
    shuffled_healthy_brain_indices = np.ndarray((3, num_patches))
    for i in range(0, num_patches):
        for j in range(0, 3):
            shuffled_healthy_brain_indices[j,
                                           i] = healthy_brain_indices[j, randidx0[i]]
    shuffled_healthy_brain_indices = np.asarray(
        shuffled_healthy_brain_indices, dtype=int)

    newidx = np.concatenate([shuffled_mask_lesion_indices,
                             shuffled_healthy_brain_indices], axis=1)

    CT_matsize = (2*num_patches, patchsize[0], patchsize[1], num_channels)
    Mask_matsize = (2*num_patches, patchsize[0], patchsize[1], 1)

    #CTPatches = np.ndarray(CT_matsize, dtype=np.float16)
    #MaskPatches = np.ndarray(Mask_matsize, dtype=np.float16)

    CTPatches = []
    MaskPatches = []

    for i in range(0, 2*num_patches):
        I = newidx[0, i]
        J = newidx[1, i]
        K = newidx[2, i]

        if I - dsize[0] < 0 or\
                I + dsize[0] > invols[0].shape[0] or\
                J - dsize[1] < 0 or\
                J + dsize[1] > invols[0].shape[1]:
                    continue

        if np.sum(
            invols[0][I - dsize[0]: I + dsize[0],
                      J - dsize[1]: J + dsize[1],
                      K]
                ) == 0:
            continue

        for c in range(num_channels):
            CTPatches.append(invols[c][I - dsize[0]: I + dsize[0],
                                              J - dsize[1]: J + dsize[1],
                                              K])
        MaskPatches.append(mask[I - dsize[0]: I + dsize[0],
                                       J - dsize[1]:J + dsize[1],
                                       K])

    #CTPatches = np.asarray(CTPatches, dtype=np.float16)
    #MaskPatches = np.asarray(MaskPatches, dtype=np.float16)

    return CTPatches, MaskPatches

def get_slices(img_vol, target_size):
    num_z_steps = img_vol.shape[2]

    slices = []
    for z in range(num_z_steps):
        sl = img_vol[:, :, z]
        sl = pad_crop_image_2D(sl, target_size)

        if np.sum(sl) > 0:
            slices.append(sl)

    slices = np.array(slices)
    slices = np.reshape(slices, slices.shape + (1,))

    return slices


def get_nonoverlapping_patches(img_vol, patch_size):
    '''
    Gets all 2D non-overlapping, non-zero patches of `patchsize` from `img`
    '''
    num_x_steps = img_vol.shape[0] // patch_size[0]
    num_y_steps = img_vol.shape[1] // patch_size[1]
    num_z_steps = img_vol.shape[2]

    patches = []
    for z in range(num_z_steps):
        for x in range(0, img_vol.shape[0] - patch_size[0] + 1, patch_size[0]):
            for y in range(0, img_vol.shape[1] - patch_size[1] + 1, patch_size[1]):
                patch = img_vol[
                    x:x+patch_size[0],
                    y:y+patch_size[1],
                    z
                ]

                if np.sum(patch) > 0:
                    patches.append(patch)

    patches = np.array(patches)
    if len(patches.shape) != 3:
        return np.zeros((1, *patch_size))
    patches = np.rollaxis(patches, 2, -1)
    patches = np.reshape(patches, patches.shape + (1,))


    return patches


def CreatePatchesForTraining(atlasdir, plane, patchsize, max_patch=150000, num_channels=1):
    '''

    Params:
        - TODO
        - healthy: bool, False if not going over the healthy dataset, true otherwise

    '''

    # get filenames
    ct_names = os.listdir(atlasdir)
    mask_names = os.listdir(atlasdir)

    ct_names = [x for x in ct_names if "CT" in x]
    mask_names = [x for x in mask_names if "mask" in x]

    ct_names.sort()
    mask_names.sort()

    numatlas = len(ct_names)

    patchsize = np.asarray(patchsize, dtype=int)
    padsize = np.max(patchsize + 1)  # / 2

    # calculate total number of voxels for all images to pre-allocate array
    f = 0
    for i in range(0, numatlas):
        maskname = mask_names[i]
        maskname = os.path.join(atlasdir, maskname)
        temp = nib.load(maskname)
        mask = temp.get_data()
        f = f + np.sum(mask)

    print("Total number of lesion patches =", f)
    total_num_patches = int(np.minimum(max_patch * numatlas, f))
    single_subject_num_patches = total_num_patches // numatlas
    print("Allowed total number of patches =", total_num_patches)

    # note here we double the size of the tensors to allow for healthy patches too
    doubled_num_patches = total_num_patches * 2
    if plane == "axial":
        CT_matsize = (doubled_num_patches,
                      patchsize[0], patchsize[1], num_channels)
        Mask_matsize = (doubled_num_patches, patchsize[0], patchsize[1], 1)
    elif plane == "sagittal":
        CT_matsize = (doubled_num_patches,
                      patchsize[0], 16, num_channels)
        Mask_matsize = (doubled_num_patches, patchsize[0], 16, 1)
    elif plane == "coronal":
        CT_matsize = (doubled_num_patches,
                      16, patchsize[1], num_channels)
        Mask_matsize = (doubled_num_patches, 16, patchsize[1], 1)

    CTPatches = np.zeros(CT_matsize, dtype=np.float16)
    MaskPatches = np.zeros(Mask_matsize, dtype=np.float16)

    indices = [x for x in range(doubled_num_patches)]
    indices = shuffle(indices, random_state=0)
    cur_idx = 0

    # interpret plane
    planar_codes = {"axial": (0, 1, 2),
                    "sagittal": (1, 2, 0),
                    "coronal": (2, 0, 1)}
    planar_code = planar_codes[plane]

    for i in tqdm(range(0, numatlas)):
        ctname = ct_names[i]
        ctname = os.path.join(atlasdir, ctname)

        temp = nib.load(ctname)
        ct = temp.get_data()
        ct = np.asarray(ct, dtype=np.float16)

        maskname = mask_names[i]
        maskname = os.path.join(atlasdir, maskname)
        temp = nib.load(maskname)
        mask = temp.get_data()
        mask = np.asarray(mask, dtype=np.float16)

        # here, need to ensure that the CT and mask tensors
        # are padded out to larger than the size of the requested
        # patches, to allow for patches to be gathered from edges
        ct = PadImage(ct, padsize)
        mask = PadImage(mask, padsize)

        ct = np.transpose(ct, axes=planar_code)
        mask = np.transpose(mask, axes=planar_code)

        invols = [ct]  # can handle multichannel here

        # adjusting patch size after transpose
        if planar_code != planar_codes["axial"]:
            if ct.shape[0] < ct.shape[1]:
                patchsize = (ct.shape[0]//4, patchsize[1])
            if ct.shape[1] < ct.shape[0]:
                patchsize = (patchsize[0], ct.shape[1]//4)
        patchsize = np.asarray(patchsize, dtype=int)

        CTPatchesA, MaskPatchesA = get_patches(invols,
                                               mask,
                                               patchsize,
                                               single_subject_num_patches,
                                               num_channels,)

        CTPatchesA = np.asarray(CTPatchesA, dtype=np.float16)
        MaskPatchesA = np.asarray(MaskPatchesA, dtype=np.float16)

        for ct_patch, mask_patch in zip(CTPatchesA, MaskPatchesA):
            CTPatches[indices[cur_idx], :, :, :] = ct_patch
            MaskPatches[indices[cur_idx], :, :, :] = mask_patch
            cur_idx += 1

    return (CTPatches, MaskPatches)
