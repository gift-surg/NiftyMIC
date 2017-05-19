#!/usr/bin/python

## \file reconstructVolume.py
#  \brief  Reconstruction of fetal brain.
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date March 2016


## Import libraries 
import SimpleITK as sitk
import itk
import os 
import sys

sys.path.insert(1, os.path.abspath(os.path.join(os.environ['VOLUMETRIC_RECONSTRUCTION_DIR'], 'src', 'py')))

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import leastsq
from scipy import ndimage
import unittest
import matplotlib.pyplot as plt
import time

## Import modules
import base.Stack as st
import utilities.PythonHelper as ph
import utilities.SimpleITKHelper as sitkh
import utilities.StackMaskMorphologicalOperations as stmorph
import preprocessing.DataPreprocessing as dp
import registration.SegmentationPropagation as segprop

# from definitions import dir_test


"""
Main Function
"""
if __name__ == '__main__':

    dir_input = "/Users/mebner/UCL/UCL/Software/VolumetricReconstruction/studies/FetalBrain/input_data/"

    segmentation_propagator = segprop.SegmentationPropagation(
        # registration_method=regniftyreg.NiftyReg(use_verbose=False),
        # registration_method=regsitk.RegistrationSimpleITK(use_verbose=False),
        registration_method=regitk.RegistrationITK(use_verbose=False),
        dilation_radius=3,
        dilation_kernel="Ball",
        )

    data_preprocessing = dp.DataPreprocessing.from_directory(
        dir_input=dir_input, 
        suffix_mask="_mask",
        segmentation_propagator=segmentation_propagator,
        )
    data_preprocessing.run_preprocessing()
    stacks = data_preprocessing.get_preprocessed_stacks()

    stacks[1].show(1)
    # stack_mask_morpher = stmorph.StackMaskMorphologicalOperations.from_sitk_mask(
    #     mask_sitk=stack.sitk_mask,
    #     dilation_radius=3,
    #     dilation_kernel="Ball",
    #     use_dilation_in_plane_only=True,
    #     )

    # stack_mask_morpher.run_dilation()

    # # tmp = stack_mask_morpher.get_processed_stack()
    # mask_sitk = stack_mask_morpher.get_processed_mask_sitk()
    # sitkh.show_sitk_image(stack.sitk, segmentation=mask_sitk, label="proc")
    # stack.show(1)
    # # tmp.show(1, label="proc")