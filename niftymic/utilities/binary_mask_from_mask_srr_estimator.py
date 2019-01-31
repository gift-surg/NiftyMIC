##
# \file binary_mask_from_mask_srr_estimator.py
# \brief      Class to estimate binary mask from mask SRR stack
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       January 2019
#


import os
import re
import numpy as np
import SimpleITK as sitk

import niftymic.base.stack as st
import pysitk.simple_itk_helper as sitkh


##
# Class to estimate binary mask from mask SRR stack
# \date       2019-01-15 16:35:36+0000
#
class BinaryMaskFromMaskSRREstimator(object):

    def __init__(self,
                 srr_mask_sitk,
                 suffix="_mask",
                 sigma=2,
                 lower=0.5,
                 upper=100,
                 ):

        if not isinstance(srr_mask_sitk, sitk.Image):
            raise ValueError("Input must be of type sitk.Image")

        self._srr_mask_sitk = srr_mask_sitk
        self._suffix = suffix
        self._sigma = sigma
        self._lower = lower
        self._upper = upper

        self._mask_sitk = None
        self._mask = None

    def get_mask_sitk(self):
        return sitk.Image(self._mask_sitk)

    def run(self):
        mask_sitk = self._srr_mask_sitk

        # Smooth mask
        mask_sitk = sitk.SmoothingRecursiveGaussian(mask_sitk, self._sigma)

        # Binarize images given thresholds
        mask_sitk = sitk.BinaryThreshold(
            mask_sitk, lowerThreshold=self._lower, upperThreshold=self._upper)

        self._mask_sitk = mask_sitk
