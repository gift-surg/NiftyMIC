# \file TestNiftyReg.py
#  \brief  Class containing unit tests for module Stack
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date May 2016


# Import libraries
import SimpleITK as sitk
import numpy as np
import unittest
import sys
import os

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh

import niftymic.registration.niftyreg as nreg
import niftymic.base.stack as st

from niftymic.definitions import DIR_TEST


class NiftyRegTest(unittest.TestCase):

    # Specify input data
    dir_test_data = DIR_TEST

    accuracy = 7

    def setUp(self):
        pass

    def test_affine_transform_reg_aladin(self):

        # Read data
        filename_fixed = "stack1_rotated_angle_z_is_pi_over_10.nii.gz"
        filename_moving = "FetalBrain_reconstruction_3stacks_myAlg.nii.gz"

        diff_ref = os.path.join(
            DIR_TEST,  "stack1_rotated_angle_z_is_pi_over_10_nreg_diff.nii.gz")
        moving = st.Stack.from_filename(
            os.path.join(self.dir_test_data, filename_moving),
        )
        fixed = st.Stack.from_filename(
            os.path.join(self.dir_test_data, filename_fixed)
        )

        # Set up NiftyReg
        nifty_reg = nreg.RegAladin()
        nifty_reg.set_fixed(fixed)
        nifty_reg.set_moving(moving)
        nifty_reg.set_registration_type("Rigid")
        nifty_reg.use_verbose(False)

        # Register via NiftyReg
        nifty_reg.run()

        # Get associated results
        affine_transform_sitk = nifty_reg.get_registration_transform_sitk()
        moving_warped = nifty_reg.get_warped_moving()

        # Get SimpleITK result with "similar" interpolator (NiftyReg does not
        # state what interpolator is used but it seems to be BSpline)
        moving_warped_sitk = sitk.Resample(
            moving.sitk, fixed.sitk, affine_transform_sitk, sitk.sitkBSpline, 0.0, moving.sitk.GetPixelIDValue())

        diff_res_sitk = moving_warped.sitk - moving_warped_sitk
        sitkh.write_nifti_image_sitk(diff_res_sitk, diff_ref)
        diff_ref_sitk = sitk.ReadImage(diff_ref)

        res_diff_nda = sitk.GetArrayFromImage(diff_res_sitk - diff_ref_sitk)

        self.assertAlmostEqual(
            np.linalg.norm(res_diff_nda), 0, places=self.accuracy)
