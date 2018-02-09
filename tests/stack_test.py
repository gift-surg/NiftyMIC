# \file TestStack.py
#  \brief  Class containing unit tests for module Stack
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date December 2015


import SimpleITK as sitk
import numpy as np
import unittest
import random
import os

import niftymic.base.stack as st
import niftymic.base.data_reader as dr
import niftymic.base.exceptions as exceptions
import niftymic.validation.motion_simulator as ms

from niftymic.definitions import DIR_TEST, DIR_TMP


class StackTest(unittest.TestCase):

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

    def test_io_image_not_existent(self):
        # Neither fetal_brain_2.nii.gz nor fetal_brain_2.nii exists
        filename = "fetal_brain_2"
        self.assertRaises(exceptions.FileNotExistent, lambda:
                          st.Stack.from_filename(
                              os.path.join(self.dir_test_data_io,
                                           filename + ".nii.gz"))
                          )
        self.assertRaises(exceptions.FileNotExistent, lambda:
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

    def test_delete_slices(self):
        filename = "stack0"
        stack = st.Stack.from_filename(
            os.path.join(self.dir_test_data, filename + ".nii.gz"),
            os.path.join(self.dir_test_data, filename + "_mask.nii.gz")
        )

        # Throw error in case slice at non-existent index shall be deleted
        self.assertRaises(
            ValueError,
            lambda: stack.delete_slice(stack.get_number_of_slices()))
        self.assertRaises(
            ValueError,
            lambda: stack.delete_slice(-stack.get_number_of_slices()))

        # Delete randomly slice indices
        for i in range(stack.get_number_of_slices()):
            # print ("----")
            slice_numbers = [s.get_slice_number() for s in stack.get_slices()]
            # print slice_numbers
            indices = np.arange(len(slice_numbers))
            random.shuffle(indices)
            index = indices[0]
            # print index
            stack.delete_slice(index)
            # print stack.get_number_of_slices()

        # No slice left at the end of the loop
        self.assertEqual(stack.get_number_of_slices(), 0)

        # No slice left for deletion
        self.assertRaises(RuntimeError, lambda: stack.delete_slice(-1))

    def test_update_write_transform(self):

        motion_simulator = ms.RandomRigidMotionSimulator(
            dimension=3,
            angle_max_deg=20,
            translation_max=30)

        filenames = ["fetal_brain_%d" % d for d in range(3)]
        stacks = [st.Stack.from_filename(
            os.path.join(self.dir_test_data, "%s.nii.gz" % f))
            for f in filenames]

        # Generate random motions for all slices of each stack
        motions_sitk = {f: {} for f in filenames}
        for i, stack in enumerate(stacks):
            motion_simulator.simulate_motion(
                seed=i, simulations=stack.get_number_of_slices())
            motions_sitk[stack.get_filename()] = \
                motion_simulator.get_transforms_sitk()

        # Apply random motion to all slices of all stacks
        dir_output = os.path.join(DIR_TMP, "test_update_write_transform")
        for i, stack in enumerate(stacks):
            for j, slice in enumerate(stack.get_slices()):
                slice.update_motion_correction(
                    motions_sitk[stack.get_filename()][j])

            # Write stacks to directory
            stack.write(dir_output, write_slices=True, write_transforms=True)

        # Read written stacks/slices/transformations
        data_reader = dr.ImageSlicesDirectoryReader(
            dir_output)
        data_reader.read_data()
        stacks_2 = data_reader.get_data()

        data_reader = dr.SliceTransformationDirectoryReader(
            dir_output)
        data_reader.read_data()
        transformations_dic = data_reader.get_data()

        filenames_2 = [s.get_filename() for s in stacks_2]
        for i, stack in enumerate(stacks):
            stack_2 = stacks_2[filenames_2.index(stack.get_filename())]
            slices = stack.get_slices()
            slices_2 = stack_2.get_slices()

            # test number of slices match
            self.assertEqual(len(slices), len(slices_2))

            # Test whether header of written slice coincides with transformed
            # slice
            for j in range(stack.get_number_of_slices()):

                # Check Spacing
                self.assertAlmostEqual(
                    np.max(np.abs(np.array(slices[j].sitk.GetSpacing()) -
                                  np.array(slices_2[j].sitk.GetSpacing()))),
                    0, places=10)
                # Check Origin
                self.assertAlmostEqual(
                    np.max(np.abs(np.array(slices[j].sitk.GetOrigin()) -
                                  np.array(slices_2[j].sitk.GetOrigin()))),
                    0, places=4)
                # Check Direction
                self.assertAlmostEqual(
                    np.max(np.abs(np.array(slices[j].sitk.GetDirection()) -
                                  np.array(slices_2[j].sitk.GetDirection()))),
                    0, places=4)

            # Test whether parameters of written slice transforms match
            params = np.array(
                motions_sitk[stack.get_filename()][j].GetParameters())
            params_2 = np.array(
                transformations_dic[stack.get_filename()][j].GetParameters())
            self.assertAlmostEqual(
                np.max(np.abs(params - params_2)), 0, places=16)
