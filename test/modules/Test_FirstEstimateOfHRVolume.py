## \file Test_FirstEstimateOfHRVolume.py
#  \brief  Class containing unit tests for module FirstEstimateOfHRVolume
# 
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date December 2015


# Import libraries 
import SimpleITK as sitk
import numpy as np
import unittest
import sys
sys.path.append("../src/")

## Import modules from src-folder
import FirstEstimateOfHRVolume as efhrv
import StackManager as sm
import Slice as sl
import Stack as st
import SimpleITKHelper as sitkh


## Concept of unit testing for python used in here is based on
#  http://pythontesting.net/framework/unittest/unittest-introduction/
#  Retrieved: Aug 6, 2015
class Test_FirstEstimateOfHRVolume(unittest.TestCase):

    ## Specify input data
    dir_input = "data/"

    accuracy = 7

    def setUp(self):
        pass

    ## Test whether private function _get_zero_framed_stack within FirstEstimateOfHRVolume
    #  works correctly
    def test_01_get_zero_framed_stack(self):

        ## Read stack
        stack = st.Stack.from_filename(self.dir_input, filename="fetal_brain_0", suffix_mask="_mask")

        ## Isotropically resample stack (simulate HR volume for FirstEstimateOfHRVolume)
        #  Idea: Don't "merge" more stacks but only use one isotropic stack where ground
        #       truth is available then
        stack_resampled = stack.get_isotropically_resampled_stack(interpolator="Linear")
        # stack_resampled.show()

        ## Create FirstEstimateOfHRVolume in order to access _get_zero_framed_stack function
        stack_manager = sm.StackManager.from_stacks([stack])
        first_estimate_of_HR_volume = efhrv.FirstEstimateOfHRVolume(stack_manager, "bla", 0)
        stack_resampled_framed = first_estimate_of_HR_volume._get_zero_framed_stack(stack_resampled, 5)
        # stack_resampled_framed.show()

        ## Resample back to stack_resampled
        stack_resampled_frame_resampled_sitk = sitk.Resample(stack_resampled_framed.sitk, stack_resampled.sitk, sitk.Euler3DTransform(), sitk.sitkLinear, 0.0, stack_resampled_framed.sitk.GetPixelIDValue())

        ## Check alignment
        nda_diff = sitk.GetArrayFromImage(stack_resampled_frame_resampled_sitk - stack_resampled.sitk)
        self.assertEqual(np.round(
                np.linalg.norm(nda_diff)
                , decimals = self.accuracy), 0)

