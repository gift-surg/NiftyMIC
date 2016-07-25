#!/usr/bin/python

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
import datetime

sys.path.append("../../src/")
sys.path.append("../../src/base/")
sys.path.append("../../src/preprocessing/")
sys.path.append("../../src/reconstruction/")
sys.path.append("../../src/registration/")
sys.path.append("../../src/validation/")

import itk
import SimpleITKHelper as sitkh
import InverseProblemSolver as ips
import StackManager as sm
import ScatteredDataApproximation as sda
import Stack as st
import DataPreprocessing as dp
import ReconstructionManager as rm
import FirstEstimateOfHRVolume as efhrv
import RegularizationParameterEstimation as rpe

def read_input_data(image_type):
    if image_type in ["fetal_neck_mass_brain"]:
        ## Fetal Neck Images:
        dir_input = "../../data/fetal_neck_mass_brain/"
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

        filenames = [str(i) for i in range(0, 3)]

    elif image_type in ["fetal_neck_mass_subject"]:
        dir_input = "../../data/fetal_neck_mass_subject/"

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

        dir_input = "../../data/kidney/"
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
        dir_input = "../../data/StructuralData_Pig/"
        filenames = [
            "T22D3mm05x05hresCLEARs601a1006",
            "T22D3mm05x05hresCLEARs701a1007",
            "T22D3mm05x05hresCLEARs901a1009"
            ]

        # filenames = filenames[0:1]

    else:
        ## Fetal Neck Images:
        dir_input = "../../data/placenta/"
        filenames = [
            "a23_04",
            "a23_05"
            ]

    return dir_input, filenames


"""
Main Function
"""
if __name__ == '__main__':

    np.set_printoptions(precision=3)

    """
    Choose Input
    """
    ## Types of input images to process
    input_stack_types_available = ("fetal_neck_mass_brain", "fetal_neck_mass_subject", "StructuralData_Pig", "kidney", "placenta")

    ## Directory to save obtained results
    dir_output = "results/"

    ## Choose input stacks and reference stack therein
    input_stacks_type = input_stack_types_available[0]
    reference_stack_id = 0

    print("Stacks chosen: %s" %input_stacks_type)
    dir_input_data, filenames_data = read_input_data(input_stacks_type)

    """
    Choose Reconstruction
    """
    dilation_radius = 0
    extra_frame_target = 5
    
    SDA_sigma = 1
    
    SRR_approach = "TK1"
    SRR_alpha = 0.03
    SRR_iter_max = 10
    
    SRR_alpha_cut = 3 
    SRR_tolerance = 1e-5
    SRR_DTD_computation_type = "FiniteDifference"


    ## Choose input stacks and reference stack therein
    target_stack_number = 0
    mask_template_number = 1

    ## Directory to save obtained results
    now = datetime.datetime.now()
    time_stamp =  str(now.year) + str(now.month).zfill(2) + str(now.day).zfill(2)
    time_stamp += "_" + str(now.hour).zfill(2) + str(now.minute).zfill(2)

    dir_output_root = "results/" 
    dir_output_root += time_stamp + "/"

    ## Data Preprocessing from data on HDD
    data_preprocessing = dp.DataPreprocessing.from_filenames(dir_input_data, filenames_data, suffix_mask="_mask")
    data_preprocessing.set_dilation_radius(dilation_radius)
    data_preprocessing.use_N4BiasFieldCorrector(False)
    data_preprocessing.run_preprocessing(boundary=0, mask_template_number=mask_template_number)

    stacks = data_preprocessing.get_preprocessed_stacks()

    ## Get initial value for reconstruction
    stack_manager = sm.StackManager.from_stacks(stacks)

    HR_volume_ref_frame = stacks[target_stack_number].get_isotropically_resampled_stack(extra_frame=extra_frame_target)

    SDA = sda.ScatteredDataApproximation(stack_manager, HR_volume_ref_frame)
    SDA.set_sigma(SDA_sigma)
    SDA.run_reconstruction()

    HR_volume_init = SDA.get_HR_volume()
    # HR_volume_init.show()


    ## Super-Resolution Reconstruction
    HR_volume = st.Stack.from_stack(HR_volume_init)
    SRR = ips.InverseProblemSolver(stacks, HR_volume)
    SRR.set_regularization_type(SRR_approach)
    SRR.set_alpha_cut(SRR_alpha_cut)
    SRR.set_tolerance(SRR_tolerance)
    SRR.set_alpha(SRR_alpha)
    SRR.set_iter_max(SRR_iter_max)
    SRR.set_DTD_computation_type(SRR_DTD_computation_type)
    SRR.run_reconstruction()

    HR_volume = SRR.get_HR_volume()
    HR_volume.show()


