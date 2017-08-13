# \file TestRegistrationCppITK.py
#  \brief  Class containing unit tests for module RegistrationCppITK
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date Nov 2016


# Import libraries
import unittest
import SimpleITK as sitk
import numpy as np
import sys
import os

import pythonhelper.SimpleITKHelper as sitkh

# Import modules
import volumetricreconstruction.base.Stack as st
import volumetricreconstruction.registration.Registration as myreg
import volumetricreconstruction.registration.RegistrationSimpleITK as regsitk
import volumetricreconstruction.registration.RegistrationCppITK as regitk

from volumetricreconstruction.definitions import DIR_TEST

# Concept of unit testing for python used in here is based on
#  http://pythontesting.net/framework/unittest/unittest-introduction/
#  Retrieved: Aug 6, 2015


class TestRegistrationCppITK(unittest.TestCase):

    # Specify input data
    dir_test_data = DIR_TEST

    accuracy = 1

    def setUp(self):
        # print(os.listdir(self.dir_test_data))
        pass

    def test_inplane3Dsimilarity_registration(self):

        # Define parameters to corrupt image
        scale = 0.9
        angleX, angleY, angleZ = (0.1, -0.1, 0.05)
        translation = (1, -5, 4)

        # Read stack and reference
        filename_ref = "FetalBrain_reconstruction_3stacks_myAlg"
        # filename_ref = "FetalBrain_reconstruction_4stacks"
        filename = "fetal_brain_1"

        stack = st.Stack.from_filename(
            os.path.join(self.dir_test_data, filename + ".nii.gz"),
            os.path.join(self.dir_test_data, filename + "_mask.nii.gz"),
        )
        reference = st.Stack.from_filename(
            os.path.join(self.dir_test_data, filename_ref + ".nii.gz"))

        # Motion corrupt stack
        motion_sitk = sitk.Euler3DTransform()
        motion_sitk.SetRotation(angleX, angleY, angleZ)
        motion_sitk.SetTranslation(translation)
        stack_corrupted_sitk = sitkh.get_transformed_sitk_image(
            stack.sitk, motion_sitk)
        stack_corrupted_sitk_mask = sitkh.get_transformed_sitk_image(
            stack.sitk_mask, motion_sitk)

        # Corrupt spacing
        spacing = np.array(stack.sitk.GetSpacing())
        spacing[0:-1] /= scale
        stack_corrupted_sitk.SetSpacing(spacing)
        stack_corrupted_sitk_mask.SetSpacing(spacing)

        # Created corrupted stack
        stack_corrupted = st.Stack.from_sitk_image(
            stack_corrupted_sitk, stack.get_filename()+"_corrupted", stack_corrupted_sitk_mask)

        # Perform in-plane 3D similarity registration
        registration = regitk.RegistrationCppITK(
            fixed=stack_corrupted, moving=reference)
        registration.set_registration_type("InplaneSimilarity")
        registration.set_interpolator("Linear")
        registration.set_metric("MeanSquares")
        registration.set_scales_estimator("PhysicalShift")
        registration.use_fixed_mask(True)
        # registration.use_verbose(True)
        registration.run_registration()

        # Get uniform in-plane scaling factor
        scale_est = registration.get_parameters()[6]

        self.assertEqual(np.round(
            abs(scale-scale_est), decimals=self.accuracy), 0)

        # Get uniformly, in-plane scaled, rigidly aligned stack of slices
        stack_inplaneSimilar = registration.get_stack_with_similarity_inplane_transformed_slices(
            stack)

        # Get all affine transforms describing the corrections
        transform_update = registration.get_registration_transform_sitk()

        # Compare whether the corrected stack can be obtained via the transform
        # too
        mask_transformed_resampled_sitk = sitk.Resample(
            stack_corrupted.sitk_mask, stack_inplaneSimilar.sitk_mask, transform_update.GetInverse(), sitk.sitkNearestNeighbor)
        # sitkh.show_sitk_image(mask_transformed_resampled_sitk-stack_inplaneSimilar.sitk_mask)
        nda_diff = sitk.GetArrayFromImage(
            mask_transformed_resampled_sitk-stack_inplaneSimilar.sitk_mask)

        self.assertEqual(np.round(
            np.linalg.norm(nda_diff), decimals=self.accuracy), 0)
