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
import pysitk.simple_itk_helper as sitkh

from niftymic.definitions import DIR_TMP, DIR_TEST, REGEX_FILENAMES, DIR_TEMPLATES


class CaseStudyFetalBrainTest(unittest.TestCase):

    def setUp(self):
        self.precision = 7
        self.dir_data = os.path.join(DIR_TEST, "case-studies", "fetal-brain")
        self.filenames = [
            os.path.join(self.dir_data,
                         "input-data",
                         "%s.nii.gz" % f)
            for f in ["axial", "coronal", "sagittal"]]
        self.dir_output = os.path.join(DIR_TMP, "case-studies", "fetal-brain")
        self.suffix_mask = "_mask"

    def test_reconstruct_volume(self):
        filename = "SRR_stacks3_TK1_lsmr_alpha0p02_itermax5.nii.gz"
        output = os.path.join(self.dir_output, filename)
        dir_reference = os.path.join(self.dir_data, "reconstruct_volume")
        dir_reference_mc = os.path.join(dir_reference, "motion_correction")
        path_to_reference = os.path.join(dir_reference, filename)
        path_to_reference_mask = ph.append_to_filename(
            os.path.join(dir_reference, filename), self.suffix_mask)

        two_step_cycles = 1
        iter_max = 5
        threshold = 0.84
        alpha = 0.02
        sigma = 0.6
        intensity_correction = 1
        isotropic_resolution = 1.02
        v2v_method = "FLIRT"

        cmd_args = []
        cmd_args.append("--filenames %s" % " ".join(self.filenames))
        cmd_args.append("--output %s" % output)
        cmd_args.append("--suffix-mask %s" % self.suffix_mask)
        cmd_args.append("--two-step-cycles %s" % two_step_cycles)
        cmd_args.append("--iter-max %d" % iter_max)
        cmd_args.append("--threshold-first %f" % threshold)
        cmd_args.append("--sigma %f" % sigma)
        cmd_args.append("--threshold %f" % threshold)
        cmd_args.append("--intensity-correction %d" % intensity_correction)
        cmd_args.append("--isotropic-resolution %s" % isotropic_resolution)
        cmd_args.append("--alpha %f" % alpha)
        cmd_args.append("--v2v-method %s" % v2v_method)
        # cmd_args.append("--verbose 1")

        cmd = "niftymic_reconstruct_volume %s" % (
            " ").join(cmd_args)
        self.assertEqual(ph.execute_command(cmd), 0)

        # Check SRR volume
        res_sitk = sitkh.read_nifti_image_sitk(output)
        ref_sitk = sitkh.read_nifti_image_sitk(path_to_reference)

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
        filename = "SRR_stacks3_TK1_lsmr_alpha0p02_itermax5.nii.gz"
        output = os.path.join(self.dir_output, filename)
        dir_reference = os.path.join(
            self.dir_data, "reconstruct_volume_from_slices")
        dir_input_mc = os.path.join(
            self.dir_data, "reconstruct_volume_from_slices", "motion_correction")
        path_to_reference = os.path.join(dir_reference, filename)

        iter_max = 5
        alpha = 0.02
        intensity_correction = 1

        cmd_args = []
        cmd_args.append("--filenames %s" % " ".join(self.filenames))
        cmd_args.append("--dir-input-mc %s" % dir_input_mc)
        cmd_args.append("--output %s" % output)
        cmd_args.append("--iter-max %d" % iter_max)
        cmd_args.append("--intensity-correction %d" % intensity_correction)
        cmd_args.append("--alpha %f" % alpha)
        cmd_args.append("--reconstruction-space %s" % path_to_reference)

        cmd = "niftymic_reconstruct_volume_from_slices %s" % (
            " ").join(cmd_args)
        self.assertEqual(ph.execute_command(cmd), 0)

        # Check whether identical reconstruction has been created
        reconstruction_sitk = sitkh.read_nifti_image_sitk(output)
        reference_sitk = sitkh.read_nifti_image_sitk(path_to_reference)

        difference_sitk = reconstruction_sitk - reference_sitk
        error = np.linalg.norm(sitk.GetArrayFromImage(difference_sitk))

        self.assertAlmostEqual(error, 0, places=self.precision)

    def test_register_image(self):
        filename = "registration_transform_sitk.txt"

        path_to_recon = os.path.join(
            self.dir_data, "register_image",
            "SRR_stacks3_TK1_lsmr_alpha0p02_itermax5.nii.gz")
        dir_input_mc = os.path.join(
            self.dir_data, "register_image", "motion_correction")
        gestational_age = 28

        path_to_transform_res = os.path.join(self.dir_output, filename)
        path_to_transform_ref = os.path.join(
            self.dir_data, "register_image", filename)
        dir_res_mc = os.path.join(
            self.dir_data, "register_image", "motion_correction")

        template = os.path.join(
            DIR_TEMPLATES,
            "STA%d.nii.gz" % gestational_age)
        template_mask = os.path.join(
            DIR_TEMPLATES,
            "STA%d_mask.nii.gz" % gestational_age)

        cmd_args = ["niftymic_register_image"]
        cmd_args.append("--fixed %s" % template)
        cmd_args.append("--moving %s" % path_to_recon)
        cmd_args.append("--fixed-mask %s" % template_mask)
        cmd_args.append("--moving-mask %s" %
                        ph.append_to_filename(path_to_recon, self.suffix_mask))
        cmd_args.append("--dir-input-mc %s" % dir_input_mc)
        cmd_args.append("--output %s" % path_to_transform_res)
        cmd_args.append("--use-flirt 1")
        cmd_args.append("--use-regaladin 1")
        cmd_args.append("--test-ap-flip 1")
        # cmd_args.append("--verbose 1")
        self.assertEqual(ph.execute_command(" ".join(cmd_args)), 0)

        # Check registration transform
        res_sitk = sitkh.read_transform_sitk(path_to_transform_res)
        ref_sitk = sitkh.read_transform_sitk(path_to_transform_ref)

        res_nda = res_sitk.GetParameters()
        ref_nda = ref_sitk.GetParameters()
        diff_nda = np.array(res_nda) - ref_nda

        self.assertAlmostEqual(
            np.linalg.norm(diff_nda), 0, places=self.precision)

        # Check individual slice transforms
        pattern = REGEX_FILENAMES + "[.]tfm"
        p = re.compile(pattern)
        dir_res_mc = os.path.join(self.dir_output, "motion_correction")
        trafos_res = sorted(
            [os.path.join(dir_res_mc, t)
             for t in os.listdir(dir_res_mc) if p.match(t)])
        trafos_ref = sorted(
            [os.path.join(dir_res_mc, t)
             for t in os.listdir(dir_res_mc) if p.match(t)])
        self.assertEqual(len(trafos_res), len(trafos_ref))
        for i in range(len(trafos_ref)):
            nda_res = sitkh.read_transform_sitk(trafos_res[i]).GetParameters()
            nda_ref = sitkh.read_transform_sitk(trafos_ref[i]).GetParameters()
            nda_diff = np.linalg.norm(np.array(nda_res) - nda_ref)
            self.assertAlmostEqual(nda_diff, 0, places=self.precision)
