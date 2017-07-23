# \file TestStack.py
#  \brief  Class containing unit tests for module Stack
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date December 2015


# Import libraries
import SimpleITK as sitk
import numpy as np
import unittest
import sys
import os

# Import modules
import base.Stack as st
import utilities.Exceptions as Exceptions

from definitions import DIR_TEST


# Concept of unit testing for python used in here is based on
#  http://pythontesting.net/framework/unittest/unittest-introduction/
#  Retrieved: Aug 6, 2015
class TestStack(unittest.TestCase):

    # Specify input data
    dir_test_data = DIR_TEST
    dir_test_data_io = os.path.join(DIR_TEST, "IO")

    accuracy = 7

    def setUp(self):
        pass

    def test_get_resampled_stack_from_slices(self):

        filename = "stack0"

        stack = st.Stack.from_filename(
            os.path.join(self.dir_test_data, filename + ".nii.gz"),
            os.path.join(self.dir_test_data, filename + "_mask.nii.gz")
        )

        nda_stack = sitk.GetArrayFromImage(stack.sitk)
        nda_stack_mask = sitk.GetArrayFromImage(stack.sitk_mask)

        # Resample stack based on slices
        stack_resampled_from_slice = stack.get_resampled_stack_from_slices()

        # Check alignment of image
        nda_stack_resampled = sitk.GetArrayFromImage(
            stack_resampled_from_slice.sitk)
        self.assertEqual(np.round(
            np.linalg.norm(nda_stack - nda_stack_resampled), decimals=self.accuracy), 0)

        # Check alignment of image mask
        nda_stack_resampled_mask = sitk.GetArrayFromImage(
            stack_resampled_from_slice.sitk_mask)
        self.assertEqual(np.round(
            np.linalg.norm(nda_stack_mask - nda_stack_resampled_mask), decimals=self.accuracy), 0)

    # Being handled (or should be in class DataReader)
    # def test_io_image_data_ambiguous(self):

    #     # fetal_brain_0.nii and fetal_brain_0.nii.gz exist (ambiguous)
    #     filename = "fetal_brain_0"
    #     self.assertRaises(Exceptions.FilenameAmbiguous, lambda:
    #                       st.Stack.from_filename(
    #                           os.path.join(self.dir_test_data, "IO",
    #                                        filename + ".nii.gz"))
    #                       )

    # def test_io_image_mask_data_ambiguous(self):
    #     # fetal_brain_1_mask.nii and fetal_brain_1_mask.nii.gz exist
    #     # (ambiguous)
    #     filename = "fetal_brain_1"
    #     self.assertRaises(Exceptions.FilenameAmbiguous, lambda:
    #                       st.Stack.from_filename(
    #                           os.path.join(self.dir_test_data, "IO",
    #                                        filename + ".nii.gz"))
    #                       )

    def test_io_image_not_existent(self):
        # Neither fetal_brain_2.nii.gz nor fetal_brain_2.nii exists
        filename = "fetal_brain_2"
        self.assertRaises(Exceptions.FileNotExistent, lambda:
                          st.Stack.from_filename(
                              os.path.join(self.dir_test_data_io,
                                           filename + ".nii.gz"))
                          )
        self.assertRaises(Exceptions.FileNotExistent, lambda:
                          st.Stack.from_filename(
                              os.path.join(self.dir_test_data_io,
                                           filename + ".nii"))
                          )

    def test_io_image_and_mask_1(self):
        # Read *.nii + *_mask.nii

        filename = "fetal_brain_3"
        stack = st.Stack.from_filename(
            os.path.join(self.dir_test_data_io, filename + ".nii"),
            os.path.join(self.dir_test_data_io, filename + "_mask.nii")
        )

        # If everything was correctly read the mask will have zeros and ones
        # in mask
        nda_mask = sitk.GetArrayFromImage(stack.sitk_mask)
        self.assertEqual(nda_mask.prod(), 0)

    def test_io_image_and_mask_2(self):
        # Read *.nii + *_mask.nii.gz

        filename = "fetal_brain_4"
        stack = st.Stack.from_filename(
            os.path.join(self.dir_test_data_io, filename + ".nii"),
            os.path.join(self.dir_test_data_io, filename + "_mask.nii.gz")
        )

        # If everything was correctly read the mask will have zero and ones
        nda_mask = sitk.GetArrayFromImage(stack.sitk_mask)
        self.assertEqual(nda_mask.prod(), 0)

    def test_io_image_and_mask_3(self):
        # Read *.nii.gz + *_mask.nii

        filename = "fetal_brain_5"
        stack = st.Stack.from_filename(
            os.path.join(self.dir_test_data_io, filename + ".nii.gz"),
            os.path.join(self.dir_test_data_io, filename + "_mask.nii")
        )

        # If everything was correctly read the mask will have zero and ones
        nda_mask = sitk.GetArrayFromImage(stack.sitk_mask)
        self.assertEqual(nda_mask.prod(), 0)

    def test_io_image_and_mask_4(self):
        # Read *.nii.gz + *_mask.nii.gz

        filename = "fetal_brain_6"
        stack = st.Stack.from_filename(
            os.path.join(self.dir_test_data_io, filename + ".nii.gz"),
            os.path.join(self.dir_test_data_io, filename + "_mask.nii.gz")
        )

        # If everything was correctly read the mask will have zero and ones
        nda_mask = sitk.GetArrayFromImage(stack.sitk_mask)
        self.assertEqual(nda_mask.prod(), 0)
