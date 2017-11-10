##
# \file slice_acquisition.py
# \brief      Based on a given volume, this class aims to simulate the slice
#             acquisition.
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       May 2016
#


# Import libraries
import itk
import SimpleITK as sitk
import numpy as np
import os
import sys
from abc import ABCMeta, abstractmethod

# Import modules from src-folder
import pysitk.simple_itk_helper as sitkh

import niftymic.base.stack as st
import niftymic.base.psf as psf


# Class simulating the slice acquisition
class SliceAcqusition(object):
    __metaclass__ = ABCMeta

    def __init__(self,
                 reference,
                 interpolator,
                 alpha_cut,
                 pixel_type=itk.D):
        self._reference = reference
        self._interpolator = interpolator
        self._alpha_cut = alpha_cut
        self._pixel_type = pixel_type
        self._image_type = itk.Image[pixel_type, reference.sitk.GetDimension()]

        self._output = None

    def run(self):
        self._run()

    @abstractmethod
    def _run(self):
        pass

    def _get_interpolator(self, stack_slice):
        if self._interpolator == "OrientedGaussian":
            # Get oriented PSF covariance matrix
            cov = psf.PSF().get_covariance_matrix_in_reconstruction_space(
                stack_slice, self._reference)

            # Specify oriented Gaussian interpolator
            interpolator_itk = itk.OrientedGaussianInterpolateImageFunction[
                self._image_type, self._pixel_type].New()
            interpolator_itk.SetCovariance(cov.flatten())
            interpolator_itk.SetAlpha(self._alpha_cut)

        else:
            interpolator_itk = eval(
                "itk.%sInterpolateImageFunction"
                "[self._image_type, self._pixel_type].New()" %
                self._interpolator)

        return interpolator_itk

    def get_output(self):
        return st.Stack.from_stack(self._output)


class StaticSliceAcquisition(SliceAcqusition):

    def __init__(self,
                 stack_slice,
                 reference,
                 interpolator="Linear",
                 alpha_cut=3):

        SliceAcqusition.__init__(self,
                                 reference=reference,
                                 interpolator=interpolator,
                                 alpha_cut=alpha_cut,
                                 )
        self._stack_slice = stack_slice

    def set_stack_slice(self, stack_slice):
        self._stack_slice = stack_slice

    def _run(self):
        resampler_itk = itk.ResampleImageFilter[
            self._image_type, self._image_type].New()
        resampler_itk.SetOutputParametersFromImage(self._stack_slice.itk)
        resampler_itk.SetInterpolator(
            self._get_interpolator(self._stack_slice))
        resampler_itk.SetInput(self._reference.itk)
        resampler_itk.UpdateLargestPossibleRegion()
        resampler_itk.Update()

        output_itk = resampler_itk.GetOutput()
        output_itk.DisconnectPipeline()
        output_sitk = sitkh.get_sitk_from_itk_image(output_itk)

        self._output = st.Stack.from_sitk_image(
            image_sitk=output_sitk,
            image_sitk_mask=self._stack_slice.sitk_mask,
            filename=self._stack_slice.get_filename()
        )
