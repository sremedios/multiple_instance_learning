#!/bin/sh
if [ $# -lt 2 ];then
	echo ct-bet.sh input_ct.nii.gz output_stripped_ct.nii.gz
	exit 1
fi

FSLOUTPUTTYPE=NIFTI_GZ
img=$1
outfile=$2
#img="Head_Image_1.nii.gz"
intensity=0.01
#outfile="Head_Image_1_SS_0.01"
tmpfile=`mktemp`

# Thresholding Image to 0-100
fslmaths "${img}" -thr 0.000000 -uthr 100.000000  "${outfile}" 
# Creating 0 - 100 mask to remask after filling
fslmaths "${outfile}"  -bin   "${tmpfile}"
fslmaths "${tmpfile}" -bin -fillh "${tmpfile}" 
# Presmoothing image
fslmaths "${outfile}"  -s 1 "${outfile}"; 
# Remasking Smoothed Image
fslmaths "${outfile}" -mas "${tmpfile}"  "${outfile}" 
# Running bet2
bet2 "${outfile}" "${outfile}" -f ${intensity} -v 
# Using fslfill to fill in any holes in mask 
fslmaths "${outfile}" -bin -fillh "${outfile}" 
# Using the filled mask to mask original image
fslmaths "${img}" -mas "${outfile}"  "${outfile}" 


######################
## If no pre-smoothing
######################

#outfile_nosmooth="Head_Image_1_SS_0.01_nopresmooth"
#fslmaths "$img" -thr 0.000000 -uthr 100.000000  "${outfile_nosmooth}" 
# Creating 0 - 100 mask to remask after filling
#fslmaths "${outfile_nosmooth}"  -bin   "${tmpfile}"; 
#fslmaths "${tmpfile}" -bin -fillh "${tmpfile}" 
# Running bet2
#bet2 "${outfile_nosmooth}" "${outfile_nosmooth}" -f ${intensity} -v 
# Using fslfill to fill in any holes in mask 
#fslmaths "${outfile_nosmooth}" -bin -fillh "${outfile_nosmooth}_Mask" 
# Using the filled mask to mask original image
#fslmaths "$img" -mas "${outfile_nosmooth}_Mask"  "${outfile_nosmooth}" 
