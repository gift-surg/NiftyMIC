# \file TestBrainStripping.py
#  \brief  Class containing unit tests for module BrainStripping
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date December 2015


import os
import unittest
import numpy as np
import SimpleITK as sitk

import pysitk.simple_itk_helper as sitkh

import niftymic.utilities.brain_stripping as bs
from niftymic.definitions import DIR_TEST


class BrainStrippingTest(unittest.TestCase):

    # Specify input data
    dir_test_data = DIR_TEST

    accuracy = 7

    def setUp(self):
        self.precision = 7
        self.dir_data = os.path.join(
            DIR_TEST, "case-studies", "fetal-brain", "input-data")
        self.filename = "axial"

    def test_01_input_output(self):

        brain_stripping = bs.BrainStripping.from_filename(
            self.dir_data, self.filename)
        brain_stripping.compute_brain_image(0)
        brain_stripping.compute_brain_mask(0)
        brain_stripping.compute_skull_image(0)
        brain_stripping.run()

        with self.assertRaises(ValueError) as ve:
            brain_stripping.get_brain_image_sitk()
        self.assertEqual(
            "Brain was not asked for. Do not set option '-n' and run again.",
            str(ve.exception))

        with self.assertRaises(ValueError) as ve:
            brain_stripping.get_brain_mask_sitk()
        self.assertEqual(
            "Brain mask was not asked for. Set option '-m' and run again.",
            str(ve.exception))

        with self.assertRaises(ValueError) as ve:
            brain_stripping.get_skull_mask_sitk()
        self.assertEqual(
            "Skull mask was not asked for. Set option '-s' and run again.",
            str(ve.exception))

    def test_02_brain_mask(self):
        path_to_reference = os.path.join(
            self.dir_data, "brain_stripping", "axial_seg.nii.gz")

        brain_stripping = bs.BrainStripping.from_filename(
            self.dir_data, self.filename)
        brain_stripping.compute_brain_image(0)
        brain_stripping.compute_brain_mask(1)
        brain_stripping.compute_skull_image(0)
        # brain_stripping.set_bet_options("-f 0.3")

        brain_stripping.run()
        original_sitk = brain_stripping.get_input_image_sitk()
        res_sitk = brain_stripping.get_brain_mask_sitk()

        ref_sitk = sitkh.read_nifti_image_sitk(path_to_reference)

        diff_sitk = res_sitk - ref_sitk
        error = np.linalg.norm(sitk.GetArrayFromImage(diff_sitk))
        self.assertAlmostEqual(error, 0, places=self.precision)
