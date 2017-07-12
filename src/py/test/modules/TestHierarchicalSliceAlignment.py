# \file TestHierarchicalSliceAlignment.py
#  \brief  Class containing unit tests for module HierarchicalSliceAlignment
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date April 2016


# Import libraries
import SimpleITK as sitk
import numpy as np
import unittest
import os
import sys

# Import modules from src-folder
import base.Stack as st
import base.DataReader as dr
import utilities.StackManager as sm
import registration.HierarchicalSliceAlignment as hsa

from definitions import dir_test

# Concept of unit testing for python used in here is based on
#  http://pythontesting.net/framework/unittest/unittest-introduction/
#  Retrieved: Aug 6, 2015


class Stack(unittest.TestCase):

    # Specify input data
    dir_test_data = dir_test

    accuracy = 7

    def setUp(self):
        pass

    def test_get_resampled_stack_from_slices(self):

        filenames = ["placenta_0.nii.gz", "placenta_1.nii.gz"]
        suffix_mask = "_mask"
        N_stacks = len(filenames)
        stacks = [None]*N_stacks

        path_filenames_list = [os.path.join(
            self.dir_test_data, f) for f in filenames]

        data_reader = dr.MultipleImagesReader(" ".join(path_filenames_list))
        data_reader.read_data()
        stacks = data_reader.get_stacks()

        stack_manager = sm.StackManager.from_stacks(stacks)

        hierarchical_slice_alignment = hsa.HierarchicalSliceAlignment(
            stack_manager)

        # Check alignment of image
        nda_stack_resampled = sitk.GetArrayFromImage(
            stack_resampled_from_slice.sitk)
        self.assertEqual(np.round(
            np.linalg.norm(nda_stack - nda_stack_resampled), decimals=self.accuracy), 0)
