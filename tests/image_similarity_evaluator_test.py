##
# \file image_similarity_evaluator_test.py
#  \brief  Test ImageSimilarityEvaluator class
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date February 2018


import os
import unittest
import numpy as np
import re
import SimpleITK as sitk

import pysitk.python_helper as ph

import niftymic.base.stack as st
import niftymic.validation.image_similarity_evaluator as ise
import niftymic.base.exceptions as exceptions
from niftymic.definitions import DIR_TMP, DIR_TEST


class ImageSimilarityEvaluatorTest(unittest.TestCase):

    def setUp(self):
        self.precision = 7

    def test_compute_write_read_similarities(self):

        paths_to_stacks = [
            os.path.join(
                DIR_TEST, "fetal_brain_%d.nii.gz" % d) for d in range(0, 3)
        ]
        path_to_reference = os.path.join(
            DIR_TEST, "FetalBrain_reconstruction_3stacks_myAlg.nii.gz")

        reference = st.Stack.from_filename(
            path_to_reference, extract_slices=False)

        stacks = [
            st.Stack.from_filename(p, ph.append_to_filename(p, "_mask"))
            for p in paths_to_stacks
        ]
        stacks = [s.get_resampled_stack(reference.sitk) for s in stacks]

        residual_evaluator = ise.ImageSimilarityEvaluator(stacks, reference)
        residual_evaluator.compute_similarities()
        residual_evaluator.write_similarities(DIR_TMP)
        similarities = residual_evaluator.get_similarities()

        similarities1 = ise.ImageSimilarityEvaluator()
        similarities1.read_similarities(DIR_TMP)
        similarities1 = similarities1.get_similarities()

        for m in residual_evaluator.get_measures():
            rho_res = similarities[m]
            rho_res1 = similarities1[m]
            error = np.linalg.norm(rho_res - rho_res1)
            self.assertAlmostEqual(error, 0, places=self.precision)

    def test_results_not_created(self):
        residual_evaluator = ise.ImageSimilarityEvaluator()

        # Directory does not exist
        self.assertRaises(
            IOError, lambda:
            residual_evaluator.read_similarities(
                os.path.join(DIR_TMP, "whatevertestasdfsfasdasf")))

        # Directory does exist but does not contain similarity result files
        self.assertRaises(IOError, lambda:
                          residual_evaluator.read_similarities(DIR_TEST))
