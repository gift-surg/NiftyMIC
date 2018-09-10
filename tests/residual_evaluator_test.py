##
# \file residual_evaluator_test.py
#  \brief  Test ResidualEvaluator class
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date November 2017


import os
import unittest
import numpy as np
import re
import SimpleITK as sitk

import pysitk.python_helper as ph

import niftymic.base.stack as st
import niftymic.validation.residual_evaluator as res_ev
import niftymic.base.exceptions as exceptions
from niftymic.definitions import DIR_TMP, DIR_TEST


class ResidualEvaluatorTest(unittest.TestCase):

    def setUp(self):
        self.precision = 7
        self.dir_tmp = os.path.join(DIR_TMP, "residual_evaluator")

    def test_compute_write_read_slice_similarities(self):

        paths_to_stacks = [
            os.path.join(
                DIR_TEST, "fetal_brain_%d.nii.gz" % d) for d in range(0, 3)
        ]
        path_to_reference = os.path.join(
            DIR_TEST, "FetalBrain_reconstruction_3stacks_myAlg.nii.gz")

        stacks = [
            st.Stack.from_filename(p, ph.append_to_filename(p, "_mask"))
            for p in paths_to_stacks
        ]
        reference = st.Stack.from_filename(
            path_to_reference, extract_slices=False)

        residual_evaluator = res_ev.ResidualEvaluator(stacks, reference)
        residual_evaluator.compute_slice_projections()
        residual_evaluator.evaluate_slice_similarities()
        residual_evaluator.write_slice_similarities(self.dir_tmp)
        slice_similarities = residual_evaluator.get_slice_similarities()

        residual_evaluator1 = res_ev.ResidualEvaluator()
        residual_evaluator1.read_slice_similarities(self.dir_tmp)
        slice_similarities1 = residual_evaluator1.get_slice_similarities()

        for stack_name in slice_similarities.keys():
            for m in slice_similarities[stack_name].keys():
                rho_res = slice_similarities[stack_name][m]
                rho_res1 = slice_similarities1[stack_name][m]
                error = np.linalg.norm(rho_res - rho_res1)
                self.assertAlmostEqual(error, 0, places=self.precision)

    def test_slice_projections_not_created(self):
        paths_to_stacks = [
            os.path.join(
                DIR_TEST, "fetal_brain_%d.nii.gz" % d) for d in range(0, 1)
        ]
        path_to_reference = os.path.join(
            DIR_TEST, "FetalBrain_reconstruction_3stacks_myAlg.nii.gz")

        stacks = [
            st.Stack.from_filename(p, ph.append_to_filename(p, "_mask"))
            for p in paths_to_stacks
        ]
        reference = st.Stack.from_filename(
            path_to_reference, extract_slices=False)

        residual_evaluator = res_ev.ResidualEvaluator(stacks, reference)
        self.assertRaises(exceptions.ObjectNotCreated, lambda:
            residual_evaluator.evaluate_slice_similarities())