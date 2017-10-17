# \file RegistrationWrapITK.py
# \brief      Class to use registration method based on SimpleITK
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Aug 2017


# Import libraries
import os
import numpy as np
import itk
import SimpleITK as sitk

# Used to parse variable arguments to SimpleITK object, see
# http://stackoverflow.com/questions/20263839/python-convert-a-string-to-arguments-list:
from ast import literal_eval

import pythonhelper.PythonHelper as ph
import registrationtools.WrapItkRegistration

import volumetricreconstruction.base.PSF as psf
import volumetricreconstruction.base.Stack as st
from volumetricreconstruction.registration.RegistrationSimpleITK \
    import RegistrationSimpleITK


##
# Class to use registration method FLIRT
# \date       2017-08-09 11:22:33+0100
#
class RegistrationWrapITK(RegistrationSimpleITK):

    def __init__(
        self,
        fixed=None,
        moving=None,
        use_fixed_mask=False,
        use_moving_mask=False,
        registration_type="Rigid",
        interpolator="Linear",
        metric="Correlation",
        metric_params=None,
        # optimizer="ConjugateGradientLineSearch",
        # optimizer_params={
        #     "learningRate": 1,
        #     "numberOfIterations": 100,
        # },
        optimizer="RegularStepGradientDescent",
        optimizer_params={
            "MinimumStepLength": 1e-6,
            "NumberOfIterations": 200,
            "GradientMagnitudeTolerance": 1e-6,
            "LearningRate": 1,
            # "RelaxationFactor": 0.5,
        },
        scales_estimator="PhysicalShift",
        initializer_type=None,
        use_oriented_psf=False,
        use_multiresolution_framework=False,
        shrink_factors=[4, 2, 1],
        smoothing_sigmas=[2, 1, 0],
        use_verbose=False,
        alpha_cut=3,
    ):

        RegistrationSimpleITK.__init__(
            self,
            fixed=fixed,
            moving=moving,
            use_fixed_mask=use_fixed_mask,
            use_moving_mask=use_moving_mask,
            registration_type=registration_type,
            interpolator=interpolator,
            metric=metric,
            metric_params=metric_params,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            scales_estimator=scales_estimator,
            initializer_type=initializer_type,
            use_oriented_psf=use_oriented_psf,
            use_multiresolution_framework=use_multiresolution_framework,
            shrink_factors=shrink_factors,
            smoothing_sigmas=smoothing_sigmas,
            use_verbose=use_verbose,
        )

        self._alpha_cut = alpha_cut
        self._pixel_type = itk.D

    def _run_registration(self):

        dimension = self._fixed.sitk.GetDimension()

        if self._use_fixed_mask:
            fixed_itk_mask = self._fixed.itk_mask
        else:
            fixed_itk_mask = None

        if self._use_moving_mask:
            moving_itk_mask = self._moving.itk_mask
        else:
            moving_itk_mask = None

        # Blur moving image with oriented Gaussian prior to the registration
        if self._use_oriented_psf:

            image_type = itk.Image[self._pixel_type, dimension]

            # Get oriented Gaussian covariance matrix
            cov_HR_coord = psf.PSF(
            ).get_gaussian_PSF_covariance_matrix_reconstruction_coordinates(
                self._fixed, self._moving)
            itk_gaussian_interpolator = itk.OrientedGaussianInterpolateImageFunction[
                image_type, self._pixel_type].New()
            itk_gaussian_interpolator.SetCovariance(cov_HR_coord.flatten())
            itk_gaussian_interpolator.SetAlpha(self._alpha_cut)

        else:
            itk_gaussian_interpolator = None

        self._registration_method = \
            registrationtools.WrapItkRegistration.WrapItkRegistration(
                dimension=dimension,
                fixed_itk=self._fixed.itk,
                moving_itk=self._moving.itk,
                fixed_itk_mask=fixed_itk_mask,
                moving_itk_mask=moving_itk_mask,
                registration_type=self._registration_type,
                interpolator=self._interpolator,
                metric=self._metric,
                # metric_params=self._metric_params,
                optimizer=self._optimizer,
                optimizer_params=self._optimizer_params,
                initializer_type=self._initializer_type,
                use_multiresolution_framework=self._use_multiresolution_framework,
                # optimizer_scales=self._scales_estimator,
                shrink_factors=self._shrink_factors,
                smoothing_sigmas=self._smoothing_sigmas,
                verbose=self._use_verbose,
                itk_oriented_gaussian_interpolate_image_filter=itk_gaussian_interpolator,
            )

        self._registration_method.run()

        self._registration_transform_sitk = \
            self._registration_method.get_registration_transform_sitk()
