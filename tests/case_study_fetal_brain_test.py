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
        self.reference = "reference-result.nii.gz"

    def test_volumetric_reconstruction(self):

        dir_input = os.path.join(self.dir_data, "SRR", "motion_correction")
        path_to_reference = os.path.join(self.dir_data,  self.reference)

        cmd_args = []
        cmd_args.append("--dir-input %s" % dir_input)
        cmd_args.append("--dir-output %s" % self.dir_output)
        cmd_args.append("--reconstruction-space %s" % path_to_reference)

        cmd = "niftymic_reconstruct_volume_from_slices %s" % (
            " ").join(cmd_args)
        self.assertEqual(ph.execute_command(cmd), 0)

        pattern = "[a-zA-Z0-9_]+(stacks)[a-zA-Z0-9_]+(.nii.gz)"
        p = re.compile(pattern)
        path_to_reconstruction = [
            os.path.join(
                self.dir_output, p.match(f).group(0))
            for f in os.listdir(self.dir_output)
            if p.match(f)][0]

        reconstruction_sitk = sitk.ReadImage(path_to_reconstruction)
        reference_sitk = sitk.ReadImage(path_to_reference)

        difference_sitk = reconstruction_sitk - reference_sitk
        error = np.linalg.norm(sitk.GetArrayFromImage(difference_sitk))

        self.assertAlmostEqual(error, 0, places=self.precision)
