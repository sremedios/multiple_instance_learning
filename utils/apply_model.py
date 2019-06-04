import numpy as np
from tqdm import tqdm

def apply_model(img_volume, model):
    '''
    Segments an image volume slice-by-slice.

    Creates image slice with zero-elements to feed into the auxiliary input.

    Params:
        - img_volume: 4D ndarray, image volume to segement
        - model: keras model object, model with which to segment
    Returns:
        - out_vol: 3D ndarray, segmented volume (the "1" channel is implicit)
    '''
    num_channels = img_volume.shape[-1]
    num_slices = img_volume.shape[2]

    dim3D = img_volume.shape[:-1]
    dim2D = np.array([1, dim3D[0], dim3D[1], num_channels], dtype=int)

    img_slice = np.zeros(dim2D, dtype=float)
    aux_slice = np.zeros(dim2D, dtype=float)
    out_vol = np.zeros(dim3D, dtype=float)

    for k in tqdm(range(num_slices)):
        for c in range(num_channels):
            img_slice[0, :, :, c] = img_volume[:, :, k, c]

        pred = model.predict([img_slice, aux_slice])
        
        # the [0] index at the start refers to the first of two outputs,
        # since this is a dual-output network
        # the [1] index is the auxiliary output
        out_vol[:, :, k] = pred[0][0, :, :, 0]

    return out_vol

def apply_model_single_input(img_volume, model):
    '''
    Segments an image volume slice-by-slice.


    Params:
        - img_volume: 4D ndarray, image volume to segement
        - model: keras model object, model with which to segment
    Returns:
        - out_vol: 3D ndarray, segmented volume (the "1" channel is implicit)
    '''
    num_channels = img_volume.shape[-1]
    num_slices = img_volume.shape[2]

    dim3D = img_volume.shape[:-1]
    dim2D = np.array([1, dim3D[0], dim3D[1], num_channels], dtype=int)

    img_slice = np.zeros(dim2D, dtype=float)
    out_vol = np.zeros(dim3D, dtype=float)

    for k in tqdm(range(num_slices)):
        for c in range(num_channels):
            img_slice[0, :, :, c] = img_volume[:, :, k, c]

        pred = model.predict(img_slice)
        
        out_vol[:, :, k] = pred[0, :, :, 0]

    return out_vol

def apply_triplanar_models(img_volume, axial_model, sagittal_model, coronal_model):

    planar_codes = {"axial": (0,1,2),
                    "sagittal": (1,2,0),
                    "coronal": (2,0,1)}

    undo_planar_codes = {"sagittal": (2,0,1),
                         "coronal": (1,2,0)}

    planes = ["axial", "sagittal", "coronal"]

    out_vols = []

    for plane, model in zip(planes, [axial_model, sagittal_model, coronal_model]):

        img_copy = img_volume.copy()

        if plane != "axial":

            # transpose to other view, requires omission of channel dimension
            img_copy = np.transpose(img_copy[:,:,:,0], axes=planar_codes[plane])
            # re-add channel dimension
            img_copy = np.reshape(img_copy, img_copy.shape + (1,))

        num_channels = img_copy.shape[-1]
        num_slices = img_copy.shape[2]

        dim3D = img_copy.shape[:-1]
        dim2D = np.array([1, dim3D[0], dim3D[1], num_channels], dtype=int)

        img_slice = np.zeros(dim2D, dtype=float)
        out_vol = np.zeros(dim3D, dtype=float)

        for k in tqdm(range(num_slices)):
            for c in range(num_channels):
                img_slice[0, :, :, c] = img_copy[:, :, k, c]

            pred = model.predict(img_slice)

            out_vol[:, :, k] = pred[0, :, :, 0]

        if plane != "axial":
            out_vol = np.transpose(out_vol, undo_planar_codes[plane])
        out_vols.append(out_vol)

    out_vols = np.array(out_vols)
    out_vols = np.average(out_vols, axis=-1)

    return out_vol
