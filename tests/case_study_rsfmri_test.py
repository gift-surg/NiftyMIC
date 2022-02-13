##
# \file case_study_fetal_brain_test.py
#  \brief  Unit tests based on fetal brain case study
#
#  \author Michael Ebner (michael.ebner@kcl.ac.uk)
#  \date July 2019


import re
import os
import unittest
import numpy as np
import SimpleITK as sitk

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh

from niftymic.definitions import DIR_TMP, DIR_TEST, REGEX_FILENAMES

import niftymic.application.rsfmri_estimate_motion as rsfmri_estimate_motion
import niftymic.application.rsfmri_reconstruct_volume_from_slices as rsfmri_reconstruct_volume_from_slices


class CaseStudyRestingStateFMRITest(unittest.TestCase):

    def setUp(self):
        self.precision = 7
        self.dir_data = os.path.join(DIR_TEST, "case-studies", "rsfmri")
        self.filename = os.path.join(
            self.dir_data, "data", "1000AB97_bold_3componentsonly.nii.gz")
        self.dir_output = os.path.join(DIR_TMP, "case-studies", "rsfmri")
        self.suffix_mask = "_mask"

    def test_estimate_motion(self):
        filename = "SRR_reference.nii.gz"
        output = os.path.join(self.dir_output, filename)
        dir_reference = os.path.join(self.dir_data, "estimate_motion")
        dir_reference_mc = os.path.join(dir_reference, "motion_correction")
        path_to_reference = os.path.join(dir_reference, filename)
        path_to_reference_mask = ph.append_to_filename(
            os.path.join(dir_reference, filename), self.suffix_mask)

        two_step_cycles = 1
        iter_max = 5

        exe = os.path.abspath(rsfmri_estimate_motion.__file__)
        cmd_args = ["python %s" % exe]
        cmd_args.append("--filename %s" % self.filename)
        cmd_args.append("--filename-mask %s" % ph.append_to_filename(
            self.filename, self.suffix_mask))
        cmd_args.append("--dir-output %s" % self.dir_output)
        cmd_args.append("--two-step-cycles %s" % two_step_cycles)
        cmd_args.append("--iter-max %d" % iter_max)
        cmd = (" ").join(cmd_args)
        self.assertEqual(ph.execute_command(cmd), 0)

        # Check SRR volume
        res_sitk = sitkh.read_nifti_image_sitk(output)
        ref_sitk = sitkh.read_nifti_image_sitk(path_to_reference)
        diff_sitk = res_sitk - ref_sitk
        error = np.linalg.norm(sitk.GetArrayFromImage(diff_sitk))
        self.assertAlmostEqual(error, 0, places=self.precision)

        # Check SRR mask volume
        res_sitk = sitkh.read_nifti_image_sitk(
            ph.append_to_filename(output, self.suffix_mask))
        ref_sitk = sitkh.read_nifti_image_sitk(path_to_reference_mask)
        diff_sitk = res_sitk - ref_sitk
        error = np.linalg.norm(sitk.GetArrayFromImage(diff_sitk))
        self.assertAlmostEqual(error, 0, places=self.precision)

        # Check transforms
        pattern = REGEX_FILENAMES + "[.]tfm"
        p = re.compile(pattern)
        dir_res_mc = os.path.join(self.dir_output, "motion_correction")
        trafos_res = sorted(
            [os.path.join(dir_res_mc, t)
             for t in os.listdir(dir_res_mc) if p.match(t)])
        trafos_ref = sorted(
            [os.path.join(dir_reference_mc, t)
             for t in os.listdir(dir_reference_mc) if p.match(t)])
        self.assertEqual(len(trafos_res), len(trafos_ref))
        for i in range(len(trafos_ref)):
            nda_res = sitkh.read_transform_sitk(trafos_res[i]).GetParameters()
            nda_ref = sitkh.read_transform_sitk(trafos_ref[i]).GetParameters()
            nda_diff = np.linalg.norm(np.array(nda_res) - nda_ref)
            self.assertAlmostEqual(nda_diff, 0, places=self.precision)

    def test_reconstruct_volume_from_slices(self):
        filename = "bold_s2v.nii.gz"
        output = os.path.join(self.dir_output, filename)
        dir_reference = os.path.join(
            self.dir_data, "reconstruct_volume_from_slices")
        dir_input_mc = os.path.join(
            self.dir_data, "reconstruct_volume_from_slices", "motion_correction")
        path_to_reference = os.path.join(dir_reference, filename)

        iter_max = 3
        alpha = 0.05
        beta = -1

        cmd_args = []
        exe = os.path.abspath(rsfmri_reconstruct_volume_from_slices.__file__)
        cmd_args = ["python %s" % exe]
        cmd_args.append("--filename %s" % self.filename)
        cmd_args.append("--filename-mask %s" % ph.append_to_filename(
            self.filename, self.suffix_mask))
        cmd_args.append("--dir-input-mc %s" % dir_input_mc)
        cmd_args.append("--output %s" % output)
        cmd_args.append("--iter-max %d" % iter_max)
        cmd_args.append("--alpha %f" % alpha)
        cmd_args.append("--beta %f" % beta)
        cmd = (" ").join(cmd_args)
        self.assertEqual(ph.execute_command(cmd), 0)

        # Check whether identical reconstruction has been created
        reconstruction_sitk = sitkh.read_sitk_vector_image(output)
        reference_sitk = sitkh.read_sitk_vector_image(path_to_reference)

        difference_sitk = reconstruction_sitk - reference_sitk
        error = np.linalg.norm(sitk.GetArrayFromImage(difference_sitk))

        self.assertAlmostEqual(error, 0, places=self.precision)

    def test_reconstruct_volume_from_slices_temporal_reg(self):
        filename = "bold_s2v_alpha0p05_beta0p5.nii.gz"
        output = os.path.join(self.dir_output, filename)
        dir_reference = os.path.join(
            self.dir_data, "reconstruct_volume_from_slices")
        dir_input_mc = os.path.join(
            self.dir_data, "reconstruct_volume_from_slices", "motion_correction")
        path_to_reference = os.path.join(dir_reference, filename)

        iter_max = 3
        alpha = 0.05
        beta = 0.5

        cmd_args = []
        exe = os.path.abspath(rsfmri_reconstruct_volume_from_slices.__file__)
        cmd_args = ["python %s" % exe]
        cmd_args.append("--filename %s" % self.filename)
        cmd_args.append("--filename-mask %s" % ph.append_to_filename(
            self.filename, self.suffix_mask))
        cmd_args.append("--dir-input-mc %s" % dir_input_mc)
        cmd_args.append("--output %s" % output)
        cmd_args.append("--iter-max %d" % iter_max)
        cmd_args.append("--alpha %f" % alpha)
        cmd_args.append("--beta %f" % beta)
        # cmd_args.append("--reconstruction-type TVL2")
        cmd = (" ").join(cmd_args)
        self.assertEqual(ph.execute_command(cmd), 0)

        # Check whether identical reconstruction has been created
        reconstruction_sitk = sitkh.read_sitk_vector_image(output)
        reference_sitk = sitkh.read_sitk_vector_image(path_to_reference)

        difference_sitk = reconstruction_sitk - reference_sitk
        error = np.linalg.norm(sitk.GetArrayFromImage(difference_sitk))

        self.assertAlmostEqual(error, 0, places=self.precision)
