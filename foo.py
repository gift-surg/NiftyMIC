#!/usr/bin/python

## \file reconstructVolume.py
#  \brief  Reconstruction of fetal brain.
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date March 2016

## Import libraries 
import SimpleITK as sitk
import itk

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import leastsq
from scipy import ndimage
import unittest
import matplotlib.pyplot as plt
import sys
import time

## Add directories to import modules
dir_src_root = "src/py/"
sys.path.append(dir_src_root)

## Import modules
import utilities.PythonHelper as ph
import utilities.SimpleITKHelper as sitkh
import utilities.StackManager as sm
import preprocessing.DataPreprocessing as dp
import studies.MSproject.MSprojectUtilityFunctions as msutilitiy
import base.Stack as st
import reconstruction.ScatteredDataApproximation as sda
import utilities.FilenameParser as fp
import reconstruction.solver.TikhonovSolver as tk
import reconstruction.solver.NonnegativeTikhonovSolver as nntk
import reconstruction.solver.TVL2Solver as tvl2
import reconstruction.regularization_parameter_estimator.AnalysisRegularizationParameterEstimator as arpe

"""
Main Function
"""
if __name__ == '__main__':

    dir_input_data = "/Users/mebner/Downloads/20170210-angles/nifti/"   
    filename_parser = fp.FilenameParser()
    filenames = filename_parser.get_filenames_which_match_pattern_in_directory(directory=dir_input_data, patterns="T1W")

    N_stacks = len(filenames)
    stacks = [None] * N_stacks

    for i in range(N_stacks):
        stacks[i] = st.Stack.from_filename(dir_input_data, filenames[i])

    template = stacks[0].get_isotropically_resampled_stack(extra_frame=100, spacing_new_scalar=3)

    stacks_visualization = [None]*N_stacks
    stacks_visualization[0] = template
    for i in range(1,N_stacks):
        stacks_visualization[i] = stacks[i]

    sitkh.show_stacks(stacks_visualization, show_comparison_file=1)



    ph.exit()

    template = st.Stack.from_filename(dir_input_data,filenames_data[0],"_mask")
    stack = st.Stack.from_filename(dir_input_data,filenames_data[1])

    params_all = np.loadtxt("/tmp/RegistrationITK/RegistrationITK_transform_s601a1006_s701a1007.txt")
    transform_sitk = sitk.Euler3DTransform()
    transform_sitk.SetParameters(params_all[3:])

    transform_sitk_inv = sitk.Euler3DTransform(transform_sitk.GetInverse())

    stack_transformed = st.Stack.from_stack(stack, stack.get_filename()+"_transformed")
    stack_transformed.update_motion_correction(transform_sitk_inv)

    foo = stack_transformed.get_resampled_stack(resampling_grid=template.sitk)

    sitkh.show_stacks([foo,template])

    # foo.show()
    # stack_transformed.show()
