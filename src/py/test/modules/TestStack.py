## \file TestStack.py
#  \brief  Class containing unit tests for module Stack
# 
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date December 2015


## Import libraries 
import SimpleITK as sitk
import numpy as np
import unittest
import sys

## Add directories to import modules
dir_src_root = "../src/"
sys.path.append( dir_src_root )

# sys.path.append("../src/")
# sys.path.append("data/")

## Import modules
import base.Stack as st

## Concept of unit testing for python used in here is based on
#  http://pythontesting.net/framework/unittest/unittest-introduction/
#  Retrieved: Aug 6, 2015
class TestStack(unittest.TestCase):

    ## Specify input data
    dir_test_data = "../../../test-data/"

    accuracy = 7

    def setUp(self):
        pass

    def test_get_resampled_stack_from_slices(self):

        filename = "stack0"

        stack = st.Stack.from_filename(self.dir_test_data, filename, "_mask")

        nda_stack = sitk.GetArrayFromImage(stack.sitk)
        nda_stack_mask = sitk.GetArrayFromImage(stack.sitk_mask)

        ## Resample stack based on slices
        stack_resampled_from_slice = stack.get_resampled_stack_from_slices()

        ## Check alignment of image
        nda_stack_resampled = sitk.GetArrayFromImage(stack_resampled_from_slice.sitk)
        self.assertEqual(np.round(
            np.linalg.norm(nda_stack - nda_stack_resampled)
            , decimals = self.accuracy), 0)

        ## Check alignment of image mask
        nda_stack_resampled_mask = sitk.GetArrayFromImage(stack_resampled_from_slice.sitk_mask)
        self.assertEqual(np.round(
            np.linalg.norm(nda_stack_mask - nda_stack_resampled_mask)
            , decimals = self.accuracy), 0)
        
