##
# \file case_study_fetal_brain_test.py
#  \brief  Unit tests based on fetal brain case study
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date November 2017


import os
import unittest
import numpy as np
import re
import SimpleITK as sitk

import pysitk.python_helper as ph

from niftymic.definitions import DIR_TMP, DIR_TEST


class CaseStudyFetalBrainTest(unittest.TestCase):

    def setUp(self):
        self.precision = 7
        self.dir_data = os.path.join(DIR_TEST, "case-studies", "fetal-brain")
        self.dir_output = os.path.join(DIR_TMP, "case-studies", "fetal-brain")

    def test_reconstruct_volume_from_slices(self):
        dir_root = os.path.join(
            self.dir_data, "reconstruct_volume_from_slices")
        dir_input = os.path.join(dir_root, "input-data")
        dir_reference = os.path.join(dir_root, "result-comparison")
        filename_reference = "SRR_stacks3_TK1_lsmr_alpha0p02_itermax10.nii.gz"
        path_to_reference = os.path.join(dir_reference, filename_reference)

        cmd_args = []
        cmd_args.append("--dir-input %s" % dir_input)
        cmd_args.append("--dir-output %s" % self.dir_output)
        cmd_args.append("--reconstruction-space %s" % path_to_reference)

        cmd = "niftymic_reconstruct_volume_from_slices %s" % (
            " ").join(cmd_args)
        self.assertEqual(ph.execute_command(cmd), 0)

        # Check whether identical reconstruction has been created
        path_to_reconstruction = os.path.join(
            self.dir_output, filename_reference)
        reconstruction_sitk = sitk.ReadImage(path_to_reconstruction)
        reference_sitk = sitk.ReadImage(path_to_reference)

        difference_sitk = reconstruction_sitk - reference_sitk
        error = np.linalg.norm(sitk.GetArrayFromImage(difference_sitk))

        self.assertAlmostEqual(error, 0, places=self.precision)

    def test_reconstruct_volume(self):
        dir_root = os.path.join(self.dir_data, "reconstruct_volume")
        dir_input = os.path.join(dir_root, "input-data")
        dir_reference = os.path.join(dir_root, "result-comparison")
        filename_reference = "SRR_stacks3_TK1_lsmr_alpha0p02_itermax3.nii.gz"
        path_to_reference = os.path.join(dir_reference, filename_reference)

        two_step_cycles = 1
        iter_max_first = 3
        iter_max = 3

        cmd_args = []
        cmd_args.append("--dir-input %s" % dir_input)
        cmd_args.append("--dir-output %s" % self.dir_output)
        cmd_args.append("--two-step-cycles %s" % two_step_cycles)
        cmd_args.append("--iter-max-first %s" % iter_max_first)
        cmd_args.append("--iter-max %s" % iter_max)

        cmd = "niftymic_reconstruct_volume %s" % (
            " ").join(cmd_args)
        self.assertEqual(ph.execute_command(cmd), 0)

        # Check whether identical reconstruction has been created
        path_to_reconstruction = os.path.join(
            self.dir_output, filename_reference)
        reconstruction_sitk = sitk.ReadImage(path_to_reconstruction)
        reference_sitk = sitk.ReadImage(path_to_reference)

        difference_sitk = reconstruction_sitk - reference_sitk
        error = np.linalg.norm(sitk.GetArrayFromImage(difference_sitk))

        self.assertAlmostEqual(error, 0, places=self.precision)

        # Obtained reconstructions could be tested too
