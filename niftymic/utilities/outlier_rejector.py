##
# \file outlier_rejector.py
# \brief      Class to identify and reject outliers.
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Jan 2019
#

import os
import scipy
import pymc3
import theano
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh
import niftymic.validation.residual_evaluator as re


##
# Class to identify and reject outliers
# \date       2019-01-28 19:24:52+0100
#
class OutlierRejector(object):

    def __init__(self,
                 stacks,
                 reference,
                 threshold,
                 use_slice_masks=False,
                 use_reference_mask=True,
                 measure="NCC",
                 verbose=True,
                 ):

        self._stacks = stacks
        self._reference = reference
        self._threshold = threshold
        self._measure = measure
        self._use_slice_masks = use_slice_masks
        self._use_reference_mask = use_reference_mask
        self._verbose = verbose

    def get_stacks(self):
        return self._stacks

    def run(self):
        residual_evaluator = re.ResidualEvaluator(
            stacks=self._stacks,
            reference=self._reference,
            use_slice_masks=self._use_slice_masks,
            use_reference_mask=self._use_reference_mask,
            verbose=False,
            measures=[self._measure],
        )
        residual_evaluator.compute_slice_projections()
        residual_evaluator.evaluate_slice_similarities()
        slice_sim = residual_evaluator.get_slice_similarities()
        # residual_evaluator.show_slice_similarities(
        #     threshold=self._threshold,
        #     measures=[self._measure],
        #     directory="/tmp/spina/figs%s" % self._print_prefix[0:7],
        # )

        remove_stacks = []
        for i, stack in enumerate(self._stacks):
            nda_sim = np.nan_to_num(
                slice_sim[stack.get_filename()][self._measure])
            indices = np.where(nda_sim < self._threshold)[0]
            slices = stack.get_slices()

            # only those indices that match the available slice numbers
            rejections = [
                j for j in [s.get_slice_number() for s in slices]
                if j in indices
            ]

            for slice in slices:
                if slice.get_slice_number() in rejections:
                    stack.delete_slice(slice)

            if self._verbose:
                ph.print_info("Stack %d/%d: Slice rejections %d/%d %s (%s)" % (
                    i + 1,
                    len(self._stacks),
                    len(stack.get_deleted_slice_numbers()),
                    stack.sitk.GetSize()[-1],
                    stack.get_deleted_slice_numbers(),
                    stack.get_filename(),
                ))
                if len(rejections) > 0:
                    res_values = nda_sim[rejections]
                    print("    Latest rejections: %s (%s < %g): %s" % (
                        rejections, self._measure, self._threshold, res_values)
                    )

            # Log stack where all slices were rejected
            if stack.get_number_of_slices() == 0:
                remove_stacks.append(stack)

        # Remove stacks where all slices where rejected
        for stack in remove_stacks:
            self._stacks.remove(stack)
            if self._verbose:
                ph.print_info("Stack '%s' removed entirely." %
                              stack.get_filename())
