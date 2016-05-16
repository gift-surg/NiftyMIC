#!/usr/bin/python

## \file main.py
#  \brief main-file incorporating all the other files 
# 
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date September 2015

## for IPython: reload all changed modules every time before executing
## However, added in file ~/.ipython/profile_default/ipython_config.py
# %load_ext autoreload
# %autoreload 2

## Import libraries 
import pdb # set "pdb.set_trace()" to break into the debugger from a running program
import itk
import SimpleITK as sitk
import numpy as np

## Import modules from src-folder
import ReconstructionManager as rm
import StackManager as sm
import HierarchicalSliceAlignment as hsa
import SimpleITKHelper as sitkh
import Stack as st
import Slice as sl


## Change viewer for sitk.Show command
#%env SITK_SHOW_COMMAND /Applications/ITK-SNAP.app/Contents/MacOS/ITK-SNAP

def read_input_data(image_type):
    
    if image_type in ["fetal_neck_mass_brain"]:
        ## Fetal Neck Images:
        dir_input = "../data/fetal_neck_mass_brain/"
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

        filenames = [str(i) for i in range(0, 8)]
        filenames.remove("6")

        # filenames = [str(i) for i in range(0, 3)]

    elif image_type in ["fetal_neck_mass_subject"]:
        dir_input = "../data/fetal_neck_mass_subject/"

        # filenames = [str(i) for i in range(0, 8)]
        filenames = [str(i) for i in range(0, 3)]

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
        filenames = [str(i) for i in range(0, 3)]

    elif image_type in ["StructuralData_Pig"]:
        dir_input = "../data/StructuralData_Pig/"
        filenames = [
            "T22D3mm05x05hresCLEARs601a1006",
            "T22D3mm05x05hresCLEARs701a1007",
            "T22D3mm05x05hresCLEARs901a1009"
            ]

        # filenames = filenames[0:1]

    else:
        ## Fetal Neck Images:
        dir_input = "../data/placenta/"
        filenames = [
            "a23_04",
            "a23_05"
            ]

    return dir_input, filenames


""" ###########################################################################
Main Function
"""
if __name__ == '__main__':

    np.set_printoptions(precision=3)

    ## Dummy to load itk for the first time (which takes 15 to 20 secs!)
    image_type = itk.Image[itk.D, 3]

    """
    Choose variables
    """
    ## Types of input images to process
    input_stack_types_available = ("fetal_neck_mass_brain", "fetal_neck_mass_subject", "StructuralData_Pig", "kidney", "placenta")

    ## Directory to save obtained results
    dir_output = "../results/"

    ## Choose input stacks and reference stack therein
    input_stacks_type = input_stack_types_available[0]
    reference_stack_id = 0

    print("Stacks chosen: %s" %input_stacks_type)
    dir_input, filenames = read_input_data(input_stacks_type)

    """
    Run reconstruction
    """
    ## Prepare output directory
    reconstruction_manager = rm.ReconstructionManager(dir_output, reference_stack_id, recon_name=input_stacks_type)

    ## Read input stack data (including data preprocessing)
    reconstruction_manager.read_input_stacks_from_filenames(dir_input, filenames, suffix_mask="_mask")
    # reconstruction_manager.read_input_stacks_from_filenames(dir_input, filenames,suffix_mask="_mask_trachea_rectangular")

    ## Read input stack data as bundle of slices (without data preprocessing)
    # reconstruction_manager.read_input_stacks_from_slice_filenames("../results/slices/", filenames, suffix_mask="_mask")
    # reconstruction_manager.set_HR_volume(st.Stack.from_filename(dir_output, "recon_fetal_neck_mass_brain_0_SRR_TK0"))

    ## Compute first estimate of HR volume
    reconstruction_manager.set_on_registration_of_stacks_before_estimating_initial_volume()
    reconstruction_manager.compute_first_estimate_of_HR_volume_from_stacks(display_info=1)
    
    ## Run hierarchical slice alignment strategy
    # reconstruction_manager.run_hierarchical_alignment_of_slices(interleave=2, display_info=1)

    ## Run two step reconstruction alignment approach
    reconstruction_manager.run_two_step_reconstruction_alignment_approach(iterations=3, display_info=1)

    ## Write results
    reconstruction_manager.write_results()

    HR_volume = reconstruction_manager.get_HR_volume()
    HR_volume.show()

    """
    Playground
    """
    stacks = reconstruction_manager.get_stacks()
    stack = stacks[0]
    slices = stack.get_slices()


    # hierarchical_slice_alignment = hsa.HierarchicalSliceAlignment(sm.StackManager.from_stacks(stacks), stacks[0].get_isotropically_resampled_stack_from_slices())
    # hierarchical_slice_alignment.run_hierarchical_alignment()
