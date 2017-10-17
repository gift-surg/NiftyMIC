##
# \file N4BiasFieldCorrection.py
# \brief      N4 bias field correction according to the file
#             runN4BiasFieldCorrectionImageFilter.cpp in src/cpp/source
#
# SimpleITK 1.0 does contain N4 Bias field corrector. Thus, file here has
# become obsolete
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       May 2017
#


# Import libraries
import os
import sys
import itk
import SimpleITK as sitk
import numpy as np

import pysitk.SimpleITKHelper as sitkh
import pysitk.PythonHelper as ph

import niftymic.base.Stack as st


##
# Class implementing the segmentation propagation from one image to another
# \date       2017-05-10 23:48:08+0100
#
class N4BiasFieldCorrection(object):

    def __init__(self,
                 stack=None,
                 use_mask=True,
                 convergence_threshold=1e-6,
                 spline_order=3,
                 wiener_filter_noise=0.11,
                 bias_field_fwhm=0.15,
                 prefix_corrected="",
                 ):

        self._stack = stack
        self._use_mask = use_mask
        self._convergence_threshold = convergence_threshold
        self._spline_order = spline_order
        self._wiener_filter_noise = wiener_filter_noise
        self._bias_field_fwhm = bias_field_fwhm
        self._prefix_corrected = prefix_corrected

        self._stack_corrected = None
        self._computational_time = ph.get_zero_time()

    def set_stack(self, stack):
        self._stack = stack

    def get_bias_field_corrected_stack(self):
        return st.Stack.from_stack(self._stack_corrected)

    def get_computational_time(self):
        return self._computational_time

    def run_bias_field_correction(self):

        time_start = ph.start_timing()

        bias_field_corrector = sitk.N4BiasFieldCorrectionImageFilter()

        bias_field_corrector.SetBiasFieldFullWidthAtHalfMaximum(
            self._bias_field_fwhm)
        bias_field_corrector.SetConvergenceThreshold(
            self._convergence_threshold)
        bias_field_corrector.SetSplineOrder(self._spline_order)
        bias_field_corrector.SetWienerFilterNoise(self._wiener_filter_noise)

        if self._use_mask:
            image_sitk = bias_field_corrector.Execute(
                self._stack.sitk, self._stack.sitk_mask)
        else:
            image_sitk = bias_field_corrector.Execute(self._stack.sitk)

        # Reading of image might lead to slight differences
        stack_corrected_sitk_mask = sitk.Resample(
            self._stack.sitk_mask,
            image_sitk,
            sitk.Euler3DTransform(),
            sitk.sitkNearestNeighbor,
            0,
            self._stack.sitk_mask.GetPixelIDValue())

        self._stack_corrected = st.Stack.from_sitk_image(
            image_sitk=image_sitk,
            image_sitk_mask=stack_corrected_sitk_mask,
            filename=self._prefix_corrected+self._stack.get_filename(),
        )

        # Get computational time
        self._computational_time = ph.stop_timing(time_start)

        # Debug
        # sitkh.show_stacks([self._stack, self._stack_corrected], label=["orig", "corr"])
