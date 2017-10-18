## \file TestIntensityCorrection.py

##
#  \brief  Class containing unit tests for module IntensityCorrection
# 
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date October 2016


## Import libraries 
import SimpleITK as sitk
import itk
import numpy as np
import unittest
import sys
from scipy.ndimage import imread

import pysitk.simple_itk_helper as sitkh

## Import modules
import niftymic.base.Stack as st
import niftymic.preprocessing.IntensityCorrection as ic

from niftymic.definitions import DIR_TEST

## Concept of unit testing for python used in here is based on
#  http://pythontesting.net/framework/unittest/unittest-introduction/
#  Retrieved: Aug 6, 2015
class TestIntensityCorrection(unittest.TestCase):

    ## Specify input data
    dir_test_data = DIR_TEST

    accuracy = 6
    use_verbose = False

    def setUp(self):        
        pass


    def test_linear_intensity_correction(self):

        ## Create stack of Lena slices
        shape_z = 15

        ## Original stack
        nda_2D = imread(self.dir_test_data + "2D_Lena_256.png", flatten=True)
        nda_3D = np.tile(nda_2D, (shape_z,1,1)).astype('double')
        stack_sitk = sitk.GetImageFromArray(nda_3D)
        stack = st.Stack.from_sitk_image(stack_sitk, "Lena")

        ## 1) Create linearly corrupted intensity stack
        nda_3D_corruped = np.zeros_like(nda_3D)
        for i in range(0, shape_z):
            nda_3D_corruped[i,:,:] = nda_3D[i,:,:]/(i+1.)
        stack_corrupted_sitk = sitk.GetImageFromArray(nda_3D_corruped)
        stack_corrupted = st.Stack.from_sitk_image(stack_corrupted_sitk, "stack_corrupted")
        # stack_corrupted.show_slices()
        # sitkh.show_stacks([stack, stack_corrupted])

        ## Ground truth-parameter:
        ic_values = np.zeros((shape_z,1))
        for i in range(0, shape_z):
            ic_values[i,:] = (i+1.)

        intensity_correction = ic.IntensityCorrection(stack=stack_corrupted, reference=stack,  use_individual_slice_correction=True, use_verbose=self.use_verbose)
        intensity_correction.run_linear_intensity_correction()
        ic_values_est = intensity_correction.get_intensity_correction_coefficients()

        nda_diff = ic_values - ic_values_est
        self.assertEqual(np.round(
                np.linalg.norm(nda_diff)
            , decimals = self.accuracy), 0)


    def test_affine_intensity_correction(self):

        ## Create stack of Lena slices
        shape_z = 15

        ## Original stack
        nda_2D = imread(self.dir_test_data + "2D_Lena_256.png", flatten=True)
        nda_3D = np.tile(nda_2D, (shape_z,1,1)).astype('double')
        stack_sitk = sitk.GetImageFromArray(nda_3D)
        stack = st.Stack.from_sitk_image(stack_sitk, "Lena")

        ## 1) Create linearly corrupted intensity stack
        nda_3D_corruped = np.zeros_like(nda_3D)
        for i in range(0, shape_z):
            nda_3D_corruped[i,:,:] = (nda_3D[i,:,:]-10*i)/(i+1.)
        stack_corrupted_sitk = sitk.GetImageFromArray(nda_3D_corruped)
        stack_corrupted = st.Stack.from_sitk_image(stack_corrupted_sitk, "stack_corrupted")
        # stack_corrupted.show_slices()
        # sitkh.show_stacks([stack, stack_corrupted])

        ## Ground truth-parameter:
        ic_values = np.zeros((shape_z,2))
        for i in range(0, shape_z):
            ic_values[i,:] = (i+1,10*i)

        intensity_correction = ic.IntensityCorrection(stack=stack_corrupted, reference=stack,  use_individual_slice_correction=True, use_verbose=self.use_verbose)
        intensity_correction.run_affine_intensity_correction()
        ic_values_est = intensity_correction.get_intensity_correction_coefficients()

        nda_diff = ic_values - ic_values_est
        self.assertEqual(np.round(
                np.linalg.norm(nda_diff)
            , decimals = self.accuracy), 0)




