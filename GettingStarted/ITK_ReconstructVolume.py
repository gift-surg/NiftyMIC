## \file ITK_ReconstructVolume.py
#  \brief  Translate algorithms which were tested in Optimization.py into
#       something which performs volume reconstructions from slices
#       given the ITK/SimpleITK framework
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date March 2016

import SimpleITK as sitk

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import leastsq
from scipy import ndimage
import unittest
import matplotlib.pyplot as plt
import sys
import time

sys.path.append("../src")

import itk
import SimpleITKHelper as sitkh
import InverseProblemSolver as ips
import Stack as st
# import Slice as sl
import ReconstructionManager as rm
import FirstEstimateOfHRVolume as efhrv



"""
Main Function
"""
if __name__ == '__main__':

    np.set_printoptions(precision=3)

    input_stack_types_available = ("pig", "fetalbrain" , "fetaltrachea")
    
    input_stack_type = input_stack_types_available[1]
    write_results = 1

    ## Settings for optimizer
    iter_max = 100       # maximum iterations
    alpha = 0.1        # regularization parameter
    reg_type = "TK1"    # regularization type; "TK0" or "TK1"
    DTD_comp_type = "Laplace" # "Laplace" or "FiniteDifference"
    # DTD_comp_type = "FiniteDifference" # "Laplace" or "FiniteDifference"
    
    ## Data of structural pig
    if input_stack_type in ["pig"]:
        dir_input = "../data/StructuralData_Pig/"
        filenames = [
            "T22D3mm05x05hresCLEARs601a1006",
            "T22D3mm05x05hresCLEARs701a1007",
            "T22D3mm05x05hresCLEARs901a1009"
            ]
        dir_ref = dir_input
        filename_HR_volume = "3DBrainViewT2SHCCLEARs1301a1013"
        filename_out = "pig"

    ## data of fetal brain
    elif input_stack_type in ["fetalbrain"]:
        # dir_input = "../data/fetal_neck/"
        # filenames = [
        #     "0",
        #     "1",
        #     "2"
        #     ]
        # dir_ref = "data/"
        # filename_HR_volume = "FetalBrain_reconstruction_4stacks"
        # filename_out = "fetalbrain"
        dir_input = "../data/fetal_neck/"
        filenames = [
            "0",
            "1",
            "2"
            ]
        dir_ref = "/Users/mebner/UCL/UCL/Other Toolkits/IRTK_BKainz/fetal_brain/brain_0_1_2_target0_inplaneres/"
        filename_HR_volume = "3TReconstruction"
        filename_out = "fetalbrain"

    ## data of fetal brain (but registered to HR volume)
    elif input_stack_type in ["fetaltrachea"]:
        dir_input = "VolumetricReconstructions/fetal_trachea/data/"
        filenames = [
            "croppedTemplate"
            ,"cropped1"
            ,"cropped2"
            ,"cropped3"
            ,"cropped4"
            ]
        dir_ref = dir_input
        filename_HR_volume = "3TReconstruction"
        filename_out = "fetaltrachea"

    ## Output folder
    dir_output = "VolumetricReconstructions/"

    ## Prepare output directory
    reconstruction_manager = rm.ReconstructionManager(dir_output)

    ## Read input data
    reconstruction_manager.read_input_stacks(dir_input, filenames)

    ## Compute first estimate of HR volume (averaged volume)
    reconstruction_manager.set_off_in_plane_rigid_registration_before_estimating_initial_volume()
    reconstruction_manager.set_on_registration_of_stacks_before_estimating_initial_volume()
    reconstruction_manager.compute_first_estimate_of_HR_volume_from_stacks()    

    HR_volume = reconstruction_manager.get_HR_volume()

    ## Copy initial HR volume for comparison later on
    HR_init_sitk = sitk.Image(HR_volume.sitk)

    # HR_volume.show()

    ## HR volume reconstruction obtained from Kainz toolkit
    HR_volume_ref = st.Stack.from_nifti(dir_ref,filename_HR_volume)

    ## Write initial and reference volume before starting reconstruction algorithm
    if write_results:
        sitk.WriteImage(HR_volume.sitk, dir_output+filename_out+"_init.nii.gz")
        sitk.WriteImage(HR_volume_ref.sitk, dir_output+filename_out+"_ref.nii.gz")

    ## Initialize optimizer with current state of motion estimation + guess of HR volume
    MyOptimizer = ips.InverseProblemSolver(reconstruction_manager.get_stacks(), HR_volume)

    ## Set regularization parameter and maximum number of iterations
    MyOptimizer.set_alpha( alpha )
    MyOptimizer.set_iter_max( iter_max )
    MyOptimizer.set_regularization_type( reg_type )
    MyOptimizer.set_DTD_computation_type( DTD_comp_type)

    ## Perform reconstruction
    print("\n--- Run reconstruction algorithm ---")
    MyOptimizer.run_reconstruction()

    ## Get reconstruction result
    recon = MyOptimizer.get_HR_volume()

    sitkh.show_sitk_image(HR_init_sitk, overlay_sitk=recon.sitk, title="HR_init+recon")

    ## Write reconstruction result
    if write_results:
        if reg_type in ["TK0"]:
            filename_out += "_" + reg_type+ "recon_alpha" + str(alpha) + "_iter" + str(iter_max)
        else:
            if DTD_comp_type in ["Laplace"]:
                filename_out += "_" + reg_type+ "recon_Laplace_alpha" + str(alpha) + "_iter" + str(iter_max)
            else:
                filename_out += "_" + reg_type+ "recon_DTD_alpha" + str(alpha) + "_iter" + str(iter_max)

        sitk.WriteImage(recon.sitk, dir_output+filename_out+".nii.gz")

    # stacks = reconstruction_manager.get_stacks()
    # N_stacks = len(stacks)
    # stack = stacks[1]
    # slices = stack.get_slices()
    # sitkh.show_sitk_image(HR_volume.sitk)


    # stacks[1].get_slice(0).write(directory=dir_output, filename="slice")
    # HR_volume.write(directory=dir_output, filename="HR_volume")




