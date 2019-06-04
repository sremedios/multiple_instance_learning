import numpy as np

def pad_image(img_data, target_dims=None):
    '''
    Pads the image to the nearest greater multiple of 16.
    This is due to the downsample/upsample count in the Unet.
    '''

    # pad to nearest greater multiple of 2**NUM_DOWNSAMPLES
    # if target_dims not provided
    if not target_dims:
        NUM_DOWNSAMPLES = 4
        scaling = 2**NUM_DOWNSAMPLES
        target_dims = [int(np.ceil(x/scaling)) * scaling for x in img_data.shape[:3]]

    target_dims = list(target_dims)

    # handle number of channels
    if len(img_data.shape) == 4:
        num_channels = img_data.shape[-1]
    else:
        num_channels = 1
    target_dims.append(num_channels)

    left_pad = round(float(target_dims[0] - img_data.shape[0]) / 2)
    right_pad = round(float(target_dims[0] - img_data.shape[0]) - left_pad)
    top_pad = round(float(target_dims[1] - img_data.shape[1]) / 2)
    bottom_pad = round(float(target_dims[1] - img_data.shape[1]) - top_pad)
    #front_pad = round(float(target_dims[2] - img_data.shape[2]) / 2)
    #back_pad = round(float(target_dims[2] - img_data.shape[2]) - front_pad)
    # skip padding along axial slices; this model is a slice-by-slice segmentation
    front_pad = 0
    back_pad = 0
    # enforce that the axial slice dimension is the same as input image
    # this means no padding for axial slices
    target_dims[2] = img_data.shape[2]

    pads = ((left_pad, right_pad),
            (top_pad, bottom_pad),
            (front_pad, back_pad))
    
    new_img = np.zeros((target_dims))


    if len(img_data.shape) == 4:
        for c in range(num_channels):
            new_img[:,:,:,c] = np.pad(img_data[:,:,:,c], pads, 'constant', constant_values=0)
    else:
        new_img[:,:,:,0] = np.pad(img_data[:,:,:], pads, 'constant', constant_values=0)
        new_img = new_img[:,:,:,0]

    return new_img
