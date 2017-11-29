##
# \file linear_operators_test.py
#  \brief  unit tests of linear operators
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

import niftymic.base.stack as st
import niftymic.base.data_reader as dr
import niftymic.reconstruction.linear_operators as lin_op
import niftymic.validation.simulate_stacks_from_reconstruction as \
    simulate_stacks_from_reconstruction
from niftymic.definitions import DIR_TMP, DIR_TEST

class LinearOperatorsTest(unittest.TestCase):

    def setUp(self):
        self.precision = 7
        self.dir_output = os.path.join(DIR_TMP, "reconstruction")

        self.dir_data = os.path.join(DIR_TEST, "reconstruction")
        self.filenames = [
            "IC_N4ITK_HASTE_exam_3.5mm_800ms_3",
            # "IC_N4ITK_HASTE_exam_3.5mm_800ms_4",
            # "IC_N4ITK_HASTE_exam_3.5mm_800ms_5",
            # "IC_N4ITK_HASTE_exam_3.5mm_800ms_6",
            # "IC_N4ITK_HASTE_exam_3.5mm_800ms_7",
        ]
        self.filename_recon = "SRR_stacks5_alpha0p01"

        self.suffix_mask = "_brain"
        self.paths_to_filenames = [
            os.path.join(self.dir_data, "motion_correction", f + ".nii.gz")
            for f in self.filenames]

        self.path_to_recon = os.path.join(
            self.dir_data, self.filename_recon + ".nii.gz")
        self.path_to_recon_mask = os.path.join(
            self.dir_data, self.filename_recon + self.suffix_mask + ".nii.gz")

    ##
    # Test forward simulation of stack and associated propagation of
    # (potentially existing) mask
    # \date       2017-11-28 22:37:54+0000
    #
    def test_forward_operator_stack(self):

        data_reader = dr.MultipleImagesReader(
            self.paths_to_filenames, suffix_mask=self.suffix_mask)
        data_reader.read_data()
        stacks = data_reader.get_data()
        stack = stacks[0]

        reconstruction = st.Stack.from_filename(
            self.path_to_recon, self.path_to_recon_mask)

        linear_operators = lin_op.LinearOperators()
        simulated_stack = linear_operators.A(
            reconstruction, stack, interpolator_mask="Linear")
        simulated_stack.set_filename(stack.get_filename() + "_sim")

        # sitkh.show_stacks([stack, simulated_stack])
        # simulated_stack.show(1)
        # reconstruction.show(1)
        # stack.show(1)

        filename_reference = "IC_N4ITK_HASTE_exam_3.5mm_800ms_3_simulated"
        reference_simulated_stack = st.Stack.from_filename(
            os.path.join(
                self.dir_data,
                "result-comparison",
                filename_reference + ".nii.gz"),
            os.path.join(
                self.dir_data,
                "result-comparison",
                filename_reference + self.suffix_mask + ".nii.gz")
        )

        # Error simulated stack
        difference_sitk = simulated_stack.sitk - \
            reference_simulated_stack.sitk
        error = np.linalg.norm(
            sitk.GetArrayFromImage(difference_sitk))
        self.assertAlmostEqual(error, 0, places=self.precision)

        # Error propagated mask
        difference_sitk = simulated_stack.sitk_mask - \
            reference_simulated_stack.sitk_mask
        error = np.linalg.norm(
            sitk.GetArrayFromImage(difference_sitk))
        self.assertAlmostEqual(error, 0, places=self.precision)

    ##
    # Test script to simulate stacks from slices
    # \date       2017-11-28 23:13:02+0000
    #
    def test_simulate_stacks_from_slices(self):

        cmd_args = []
        cmd_args.append("--dir-input %s" %
                        os.path.join(self.dir_data, "motion_correction"))
        cmd_args.append("--reconstruction %s" % self.path_to_recon)
        cmd_args.append("--reconstruction-mask %s" % self.path_to_recon_mask)
        cmd_args.append("--copy-data 1")
        cmd_args.append("--suffix-mask _brain")
        # cmd_args.append("--verbose 1")
        cmd_args.append("--dir-output %s" % self.dir_output)

        exe = os.path.abspath(simulate_stacks_from_reconstruction.__file__)
        cmd = "python %s %s" % (exe, (" ").join(cmd_args))
        self.assertEqual(ph.execute_command(cmd), 0)
