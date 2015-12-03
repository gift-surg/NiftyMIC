#!/usr/bin/python

## \file main.py
#  \brief main-file incorporating all the other files 
# 
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date September 2015


## Import libraries 
import SimpleITK as sitk
import numpy as np

## Import modules from src-folder
import ReconstructionManager as rm
# import SliceStack


def read_input_data(image_type):
    
    if image_type in ["fetal_neck"]:
        ## Fetal Neck Images:
        dir_input = "../data/fetal_neck/"
        # filenames = [
        #     "20150115_161038s006a1001_crop",
        #     "20150115_161038s003a1001_crop",
        #     "20150115_161038s004a1001_crop",
        #     "20150115_161038s005a1001_crop",
        #     "20150115_161038s007a1001_crop",
        #     "20150115_161038s5005a1001_crop",
        #     "20150115_161038s5006a1001_crop",
        #     "20150115_161038s5007a1001_crop"
        #     ]

        # filenames = [str(i) for i in range(0, 8)]
        filenames = [str(i) for i in range(4, 8)]


    elif image_type in ["kidney"]:
        ## Kidney Images:
        # dir_input = "/Users/mebner/UCL/Data/Kidney\\ \\(3T,\\ Philips,\\ UCH,\\ 20150713\\)/Nifti/"
        # filenames = [
        #     "20150713_09583130x3mmlongSENSEs2801a1028",
        #     "20150713_09583130x3mmlongSENSEs2701a1027",
        #     "20150713_09583130x3mmlongSENSEs2601a1026"
        #     # "20150713_09583130x3mmlongSENSEs2501a1025",
        #     # "20150713_09583130x3mmlongSENSEs2401a1024",
        #     # "20150713_09583130x3mmlongSENSEs2301a1023"
        #     ]

        dir_input = "../data/kidney/"
        # filenames = [
        #     "SENSEs2801a1028",
        #     "SENSEs2701a1027",
        #     "SENSEs2601a1026",
        #     "SENSEs2501a1025",
        #     "SENSEs2401a1024",
        #     "SENSEs2301a1023"
        #     ]
        filenames = [str(i) for i in range(0, 6)]


    else:
        ## Fetal Neck Images:
        dir_input = "../data/placenta_in-plane_Guotai/"
        filenames = [
            "a13_15"
            ]

    return dir_input, filenames


""" ###########################################################################
Main Function
"""
if __name__ == '__main__':

    """
    Choose variables
    """
    ## Types of input images to process
    input_stack_types_available = ("fetal_neck", "kidney", "placenta")

    ## Directory to save obtained results
    dir_output = "../results/"

    ## Choose input stacks and reference stack therein
    input_stacks_type = input_stack_types_available[0]
    reference_stack_id = 0

    """
    Run reconstruction
    """
    ## Prepare output directory
    reconstruction_manager = rm.ReconstructionManager(dir_output)

    ## Read input data
    dir_input, filenames = read_input_data(input_stacks_type)
    reconstruction_manager.read_input_data(dir_input, filenames)

    ## Compute first estimate of HR volume
    reconstruction_manager.compute_first_estimate_of_HR_volume()

    ## In-plane rigid registration
    # reconstruction_manager.run_in_plane_rigid_registration()

    ## Write results
    # reconstruction_manager.write_resampled_stacks_after_2D_in_plane_registration()
    # reconstruction_manager.write_results()

    """
    Playground
    """
    stacks = reconstruction_manager.get_stacks()
    slices = stacks[0].get_slices()
    # print stacks[0.sitk
    # print slices[-1].sitk
