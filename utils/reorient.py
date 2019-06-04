import os
import nibabel as nib 
from subprocess import Popen, PIPE

def orient(filename, src_dir, dst_dir, verbose=0):
    '''
    Orients image to RAI using 3dresample into data_dir/preprocessing/rai.

    Requires AFNI 3dresample

    Params:
        - filename: string, name of original CT image
        - src_dir: string, path to skullstripped dir
        - dst_dir: string, path to RAI oriented dir
        - verbose: int, 0 for silent, 1 for verbose
    '''
    target_orientation = "RAI"

    infile = os.path.join(src_dir, filename)
    outfile = os.path.join(dst_dir, filename)

    if os.path.exists(outfile):
        if verbose == 1:
            print("Already oriented", filename)
        return

    if verbose == 1:
        print("Orienting to " + target_orientation+"...")

    call = "3dresample -orient" + " " + target_orientation + " " +\
        "-inset" + " " + infile + " " +\
        "-prefix" + " " + outfile
    os.system(call)

    if verbose == 1:
        print("Orientation complete")


def reorient(filename, orig_dir, seg_dir, seg_orient="RAI"):
    '''
    Reorients a single image to its original orientation prior to being run
    through Roy's segmentation script (which uses RAI) using 3dresample.  
    First checks against original scan to see if reorientation is necessary, 
    then reorients.

    Reoriented image is saved to dst_dir/
    Requires AFNI 3dresample

    Params:
        - filename: string, name of NIFTI file to skullstrip 
        - src_dir: string, source directory
    '''
    orig_filepath = os.path.join(orig_dir, filename)
    call = "3dinfo -orient " + orig_filepath
    pipe = Popen(call, shell=True, stdout=PIPE).stdout
    three_digit_code = pipe.read()[:3].decode()
    pipe.close()

    dst_dir = os.path.join(seg_dir, "reoriented")
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    # if three_digit_code != seg_orient:
    # reorient lesion membership
    infile = os.path.join(seg_dir, filename)
    outfile = os.path.join(dst_dir, filename)
    call = "3dresample -orient " + three_digit_code\
        + " -inset " + infile + " -prefix " + outfile
    os.system(call)
