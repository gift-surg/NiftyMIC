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

        self.dir_data = os.path.join(DIR_TEST, "case-studies", "fetal-brain")
        self.filename = "axial"
        self.suffix_mask = "_mask"
        self.path_to_file = os.path.join(
            self.dir_data, "input-data", "%s.nii.gz" % self.filename)
        self.filename_recon = "SRR_stacks3_TK1_lsmr_alpha0p02_itermax5.nii.gz"
        self.path_to_recon = os.path.join(
            self.dir_data, "recon_projections", self.filename_recon)
        self.path_to_recon_mask = ph.append_to_filename(
            self.path_to_recon, self.suffix_mask)

    ##
    # Test forward simulation of stack and associated propagation of
    # (potentially existing) mask
    # \date       2017-11-28 22:37:54+0000
    #
    def test_forward_operator_stack(self):

        stack = st.Stack.from_filename(self.path_to_file)
        reconstruction = st.Stack.from_filename(
            self.path_to_recon, self.path_to_recon_mask)

        linear_operators = lin_op.LinearOperators()
        simulated_stack = linear_operators.A(
            reconstruction, stack, interpolator_mask="Linear")
        simulated_stack.set_filename(stack.get_filename() + "_sim")

        # sitkh.show_stacks(
        #     [stack, simulated_stack], segmentation=simulated_stack)

        filename_reference = os.path.join(
            self.dir_data,
            "recon_projections",
            "stack",
            "%s_sim.nii.gz" % self.filename)
        filename_reference_mask = os.path.join(
            self.dir_data,
            "recon_projections",
            "stack",
            "%s_sim%s.nii.gz" % (self.filename, self.suffix_mask))

        reference_simulated_stack = st.Stack.from_filename(
            filename_reference, filename_reference_mask)

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
        cmd_args.append("--filenames %s" % self.path_to_file)
        cmd_args.append("--dir-input-mc %s" %
                        os.path.join(
                            self.dir_data,
                            "recon_projections",
                            "motion_correction"))
        cmd_args.append("--reconstruction %s" % self.path_to_recon)
        cmd_args.append("--reconstruction-mask %s" % self.path_to_recon_mask)
        cmd_args.append("--copy-data 1")
        cmd_args.append("--suffix-mask %s" % self.suffix_mask)
        cmd_args.append("--dir-output %s" % self.dir_output)

        exe = os.path.abspath(simulate_stacks_from_reconstruction.__file__)
        cmd = "python %s %s" % (exe, (" ").join(cmd_args))
        self.assertEqual(ph.execute_command(cmd), 0)

        path_orig = os.path.join(self.dir_output, "%s.nii.gz" % self.filename)
        path_sim = os.path.join(
            self.dir_output, "Simulated_%s.nii.gz" % self.filename)

        path_orig_ref = os.path.join(self.dir_data,
                                     "recon_projections",
                                     "slices",
                                     "%s.nii.gz" % self.filename)
        path_sim_ref = os.path.join(self.dir_data,
                                    "recon_projections",
                                    "slices",
                                    "Simulated_%s.nii.gz" % self.filename)

        for res, ref in zip(
                [path_orig, path_sim], [path_orig_ref, path_sim_ref]):
            res_sitk = sitk.ReadImage(res)
            ref_sitk = sitk.ReadImage(ref)

            nda_diff = np.nan_to_num(
                sitk.GetArrayFromImage(res_sitk - ref_sitk))
            self.assertAlmostEqual(np.linalg.norm(
                nda_diff), 0, places=self.precision)
