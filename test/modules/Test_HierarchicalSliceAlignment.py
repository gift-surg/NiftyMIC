## \file Test_HierarchicalSliceAlignment.py
#  \brief  Class containing unit tests for module HierarchicalSliceAlignment
# 
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date April 2016


## Import libraries 
import SimpleITK as sitk
import numpy as np
import unittest
import sys
sys.path.append("../src/")
sys.path.append("data/")

## Import modules from src-folder
import Stack as st
import StackManager as sm
import HierarchicalSliceAlignment as hsa

## Concept of unit testing for python used in here is based on
#  http://pythontesting.net/framework/unittest/unittest-introduction/
#  Retrieved: Aug 6, 2015
class Stack(unittest.TestCase):

    ## Specify input data
    dir_input = "data/"

    accuracy = 7

    def setUp(self):
        pass

    def test_get_resampled_stack_from_slices(self):

        filenames = ["placenta_0", "placenta_1"]
        suffix_mask = "_mask"
        N_stacks = len(filenames)
        stacks = [None]*N_stacks

        for i in range(0, N_stacks):
            stacks[i] = st.Stack.from_filename(self.dir_input, filenames[i], suffix_mask)

        stack_manager = sm.StackManager.from_stack(stacks)

        hierarchical_slice_alignment = hsa.HierarchicalSliceAlignment(stack_manager)



        ## Check alignment of image
        nda_stack_resampled = sitk.GetArrayFromImage(stack_resampled_from_slice.sitk)
        self.assertEqual(np.round(
            np.linalg.norm(nda_stack - nda_stack_resampled)
            , decimals = self.accuracy), 0)

        
