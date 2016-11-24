##
# \file MSprojectUtilityFunctions.py
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       November 2016
#


## Import libraries 
import argparse
import numpy as np


## Import modules
import utilities.SimpleITKHelper as sitkh


##
def get_parsed_input_line():

    parser = argparse.ArgumentParser(description="Run regularization parameter study by specifying several parameters")

    parser.add_argument('--dir_input_reference', type=str, help="", required=True)
    parser.add_argument('--dir_input_stack', type=str, help="", required=True)
    parser.add_argument('--dir_output_verbose', type=str, help="", required=True)
    parser.add_argument('--dir_output_motion_correction', type=str, help="", required=True)
    parser.add_argument('--filename_reference', type=str, help="", required=True)
    parser.add_argument('--filename_subject', type=str, help="", required=True)
    parser.add_argument('--increase_inplane_spacing', type=float, help="", default=0)
    parser.add_argument('--downsampling_factor', type=int, help="", default=0)
    parser.add_argument('--debug', type=int, help="", default=0)
    parser.add_argument('--dir_input_stacked_volumes', type=str, help="", default="")

    args = parser.parse_args()

    dir_input_reference = args.dir_input_reference
    dir_input_stack = args.dir_input_stack
    dir_output_verbose = args.dir_output_verbose
    dir_output_motion_correction = args.dir_output_motion_correction
    filename_reference = args.filename_reference
    filename_subject = args.filename_subject
    increase_inplane_spacing = args.increase_inplane_spacing
    downsampling_factor = args.downsampling_factor
    debug = args.debug
    dir_input_stacked_volumes = args.dir_input_stacked_volumes
    
    ## Debug
    print("dir_input_reference = " + str(dir_input_reference))        
    print("dir_input_stack = " + str(dir_input_stack))    
    print("dir_output_verbose = " + str(dir_output_verbose))    
    print("dir_output_motion_correction = " + str(dir_output_motion_correction))    
    print("filename_reference = " + str(filename_reference))    
    print("filename_subject = " + str(filename_subject))        
    print("increase_inplane_spacing = " + str(increase_inplane_spacing))    
    print("downsampling_factor = " + str(downsampling_factor))        
    print("debug = " + str(debug))        
    print("dir_input_stacked_volumes = " + str(dir_input_stacked_volumes))        

    return dir_input_reference, dir_input_stack, dir_output_verbose, dir_output_motion_correction, filename_reference, filename_subject, increase_inplane_spacing, downsampling_factor, debug, dir_input_stacked_volumes

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

