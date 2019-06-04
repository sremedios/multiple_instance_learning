import os

def skullstrip(filename, src_dir, dst_dir, script_path, verbose=0):
    '''
    Skullstrips a CT nifti image into data_dir/preprocessing/skullstripped/

    Params:
        - filename: string, name of file to skullstrip
        - src_dir: string, path to directory where the CT to be skullstripped is
        - dst_dir: string, path to directory where the skullstripped CT is saved
        - script_path: string, path to the file of the skullstrip script
        - verbose: int, 0 for silent, 1 for verbose
    '''

    infile = os.path.join(src_dir, filename)
    outfile = os.path.join(dst_dir, filename)

    if os.path.exists(outfile):
        if verbose == 1:
            print("Already skullstripped", filename)
        return

    if verbose == 1:
        print("Skullstripping", filename, "into" + " " + dst_dir)

    call = "sh" + " " + script_path + " " + infile + " " + outfile
    os.system(call)

    if verbose == 1:
        print("Skullstripping complete")
