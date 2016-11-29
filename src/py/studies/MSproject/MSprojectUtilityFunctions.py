##
# \file MSprojectUtilityFunctions.py
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       November 2016
#


## Import libraries 
import SimpleITK as sitk
import argparse
import numpy as np
import os


## Import modules
import utilities.SimpleITKHelper as sitkh

LEFT_RIGHT_MIRRORED_STACKS = [
      "A0045632-B1463033-5yr"
    , "A0045632-B1463033-5yr-0-PD"  # ground-truth stack

    , "A0742401-B4903730-5yr"
    , "A0742401-B4903730-5yr-0-PD"  # ground-truth stack
    
    , "A0885540-B5650504-5yr"
    , "A0885540-B5650504-5yr-0-PD"  # ground-truth stack
    
    , "A2192428-B3117859-5yr"
    , "A2192428-B3117859-5yr-0-PD"  # ground-truth stack
    
    , "A2890283-B1234334-5yr"
    , "A2890283-B1234334-5yr-0-PD"  # ground-truth stack
    
    , "A2921508-B6991743-5yr"
    , "A2921508-B6991743-5yr-0-PD"  # ground-truth stack
    
    , "A3451463-B7126393-5yr"
    , "A3451463-B7126393-5yr-0-PD"  # ground-truth stack
    
    , "A3594681-B1352386-5yr"
    , "A3594681-B1352386-5yr-0-PD"  # ground-truth stack
    
    , "A3733074-B3575525-5yr"
    , "A3733074-B3575525-5yr-0-PD"  # ground-truth stack
    
    , "A4345560-B3508298-5yr"
    , "A4345560-B3508298-5yr-0-PD"  # ground-truth stack
    
    , "A4892602-B9035827-5yr"
    , "A4892602-B9035827-5yr-0-PD"  # ground-truth stack
    
    , "A5222915-B2849383-5yr"
    , "A5222915-B2849383-5yr-0-PD"  # ground-truth stack
    
    , "A5330058-B8071825-5yr"
    , "A5330058-B8071825-5yr-0-PD"  # ground-truth stack
    
    , "A5602960-B3234150-5yr"
    , "A5602960-B3234150-5yr-0-PD"  # ground-truth stack
    
    , "A6125346-B0853727-5yr"
    , "A6125346-B0853727-5yr-0-PD"  # ground-truth stack
    
    , 'A5884831-B6601288-5yr-0-PD'  # ground-truth stack

    , "A6137029-B3531778-5yr"
    , "A6137029-B3531778-5yr-0-PD"  # ground-truth stack
    
    , "A7333875-B7213936-5yr"
    , "A7333875-B7213936-5yr-0-PD"  # ground-truth stack
    
    ## no reference available for motion correction!
    # , "A7429556-B9898126-5yr"
    # , "A7429556-B9898126-5yr-0-PD"  # ground-truth stack
    
    , "A7434519-B3842837-5yr-0-PD"  # ground-truth stack
    
    , "A9493003-B3862085-5yr"
    , "A9493003-B3862085-5yr-0-PD"  # ground-truth stack
    
    , "A9832766-B0023954-5yr"       # interesting case, references do not match?!
    , "A9832766-B0023954-5yr-0-PD"  # ground-truth stack ## interesting case, references do not match?!
]


##
def get_parsed_input_line():

    parser = argparse.ArgumentParser(description="Run regularization parameter study by specifying several parameters")

    parser.add_argument('--dir_input_reference', type=str, help="", required=True)
    parser.add_argument('--dir_input_stack', type=str, help="", required=True)
    parser.add_argument('--dir_input_stacked_volumes', type=str, help="", default="")
    parser.add_argument('--dir_output_verbose', type=str, help="", required=True)
    parser.add_argument('--dir_output_motion_correction', type=str, help="", required=True)
    parser.add_argument('--dir_output_reconstruction_analysis', type=str, help="", default="")
    parser.add_argument('--filename_reference', type=str, help="", required=True)
    parser.add_argument('--filename_subject', type=str, help="", required=True)
    parser.add_argument('--increase_inplane_spacing', type=float, help="", default=0)
    parser.add_argument('--downsampling_factor', type=int, help="", default=0)
    parser.add_argument('--debug', type=int, help="", default=0)
    
    args = parser.parse_args()

    dir_input_reference = args.dir_input_reference
    dir_input_stack = args.dir_input_stack
    dir_input_stacked_volumes = args.dir_input_stacked_volumes
    dir_output_verbose = args.dir_output_verbose
    dir_output_motion_correction = args.dir_output_motion_correction
    dir_output_reconstruction_analysis = args.dir_output_reconstruction_analysis
    filename_reference = args.filename_reference
    filename_subject = args.filename_subject
    increase_inplane_spacing = args.increase_inplane_spacing
    downsampling_factor = args.downsampling_factor
    debug = args.debug
    
    ## Debug
    print("dir_input_reference = " + str(dir_input_reference))        
    print("dir_input_stack = " + str(dir_input_stack))    
    print("dir_output_verbose = " + str(dir_output_verbose))    
    print("dir_output_motion_correction = " + str(dir_output_motion_correction))    
    print("dir_input_stacked_volumes = " + str(dir_input_stacked_volumes))    
    print("dir_output_reconstruction_analysis = " + str(dir_output_reconstruction_analysis))    
    print("filename_reference = " + str(filename_reference))    
    print("filename_subject = " + str(filename_subject))        
    print("increase_inplane_spacing = " + str(increase_inplane_spacing))    
    print("downsampling_factor = " + str(downsampling_factor))        
    print("debug = " + str(debug))        

    return args

##
# Gets the updated affine transforms. At least one input parameter must be a
# list of transforms!
# \date       2016-11-02 23:38:58+0000
#
# \param      transforms_outer  Either as np.array containing list or
#                               sitk.Transform object
# \param      transforms_inner  Either as np.array containing list or
#                               sitk.Transform object
#
# \return     The updated affine transforms with elements transforms_outer[i]
# \f$
# \c irc\f$ transforms_inner[i]
#
def get_updated_affine_transforms(transforms_outer, transforms_inner):

    ## In case transforms_inner is not an array
    try:
        len(transforms_inner)
    except:
        transforms_inner = np.array([transforms_inner])

    ## In case transforms_outer is not an array
    try:
        len(transforms_outer)
    except:
        transforms_outer = np.array([transforms_outer])

    ## Make len(transforms_inner) = len(transforms_outer)
    if len(transforms_inner) is 1:
        transforms_inner = [transforms_inner[0]] * len(transforms_outer)
    
    if len(transforms_outer) is 1:
        transforms_outer = [transforms_outer[0]] * len(transforms_inner)

    ## Prepare output
    composite_transforms = [None] * len(transforms_inner)

    ## Apply transform
    for i in range(0, len(transforms_inner)):
        composite_transforms[i] = sitkh.get_composite_sitk_affine_transform(transforms_outer[i], transforms_inner[i])
        
    return composite_transforms



##
# Gets the left right mirrored stack if required.
# \date       2016-11-26 18:20:39+0000
#
# Input stack is given by the originally stacked slices extracted from the MR
# film
#
# \param      image_sitk        The image sitk
# \param      filename_subject  The filename subject
#
# \return     The left right mirrored stack if required.
#
def get_left_right_mirrored_stack_if_required(image_sitk, filename_subject):

    image_sitk = sitk.Image(image_sitk)

    ## Affine transform defining the mirroring transform
    transform_sitk = sitk.AffineTransform(3)

    ## Mirror stack
    if filename_subject in LEFT_RIGHT_MIRRORED_STACKS:
        print("Mirrored stack: Flip left and right")
        matrix = np.eye(3)
        matrix[0,0] = -1
        direction_matrix = np.array(image_sitk.GetDirection()).reshape(3,3)

        ## Update direction and corresponding image header
        image_sitk.SetDirection((matrix.dot(direction_matrix)).flatten())
        # image_sitk = update_left_right_image_header_information(image_sitk)

        ## Transform to apply (in case of zero origin) to original image
        transform_sitk.SetMatrix(matrix.flatten())


    # return image_sitk, transform_sitk
    return image_sitk

##
# Update header information after left-right swapping
# \date       2016-11-28 17:46:22+0000
#
# \param      image_sitk  Either sitk Image to the path to the file
#
# \return     sitk image with updated header
#
def update_left_right_image_header_information(image_sitk):

    print("Update image header for left-right flipping")
    dir_output_tmp = "/tmp/fslhd/"
    filename_tmp = "A"
    os.system("mkdir -p " + dir_output_tmp)
    os.system("rm -rf " + dir_output_tmp + "*")
    
    try:
        sitk.WriteImage(image_sitk, dir_output_tmp + filename_tmp + ".nii.gz")
    except:
        os.system("cp " + image_sitk + " " + dir_output_tmp + filename_tmp + ".nii.gz")

    ## Possibility A:
    cmd  = "fslswapdim "
    cmd += dir_output_tmp + filename_tmp + " LR PA IS "
    cmd += dir_output_tmp + filename_tmp + "_swapped"
    flag = os.system(cmd)

    ## Possibility B:
    if flag is not 0:
        cmd  = "fslswapdim "
        cmd += dir_output_tmp + filename_tmp + " RL PA IS "
        cmd += dir_output_tmp + filename_tmp + "_swapped"
        os.system(cmd)
    print("done")

    image_swapped_sitk = sitk.ReadImage(dir_output_tmp + filename_tmp + "_swapped" + ".nii.gz")

    return image_swapped_sitk

