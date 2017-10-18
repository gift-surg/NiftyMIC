# \file RegistrationSimpleITK.py
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

import pysitk.simple_itk_helper as sitkh
import pysitk.python_helper as ph
import simplereg.simple_itk_registration

import niftymic.base.PSF as psf
import niftymic.base.Stack as st
from niftymic.registration.RegistrationMethod \
    import AffineRegistrationMethod


##
# Class to use registration method FLIRT
# \date       2017-08-09 11:22:33+0100
#
class RegistrationSimpleITK(AffineRegistrationMethod):

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
            "minStep": 1e-6,
            "numberOfIterations": 200,
            "gradientMagnitudeTolerance": 1e-6,
            "learningRate": 1,
        },
        scales_estimator="PhysicalShift",
        initializer_type=None,
        use_oriented_psf=False,
        use_multiresolution_framework=False,
        shrink_factors=[2, 1],
        smoothing_sigmas=[1, 0],
        use_verbose=False,
    ):

        AffineRegistrationMethod.__init__(self,
                                          fixed=fixed,
                                          moving=moving,
                                          use_fixed_mask=use_fixed_mask,
                                          use_moving_mask=use_moving_mask,
                                          use_verbose=use_verbose,
                                          registration_type=registration_type,
                                          )

        self._REGISTRATION_TYPES = ["Rigid", "Similarity", "Affine"]
        self._INITIALIZER_TYPES = [None, "MOMENTS", "GEOMETRY",
                                   "SelfGEOMETRY", "SelfMOMENTS"]
        self._SCALES_ESTIMATORS = ["IndexShift", "PhysicalShift", "Jacobian"]

        self._interpolator = interpolator
        self._metric = metric
        self._metric_params = metric_params

        self._optimizer = optimizer
        self._optimizer_params = optimizer_params

        self._scales_estimator = scales_estimator

        self._initializer_type = initializer_type

        self._use_oriented_psf = use_oriented_psf

        self._use_multiresolution_framework = use_multiresolution_framework
        self._shrink_factors = shrink_factors
        self._smoothing_sigmas = smoothing_sigmas

    # Use multiresolution framework
    #  \param[in] flag boolean
    def use_multiresolution_framework(self, flag):
        self._use_multiresolution_framework = flag

    # Decide whether oriented PSF shall be applied, i.e. blur moving image
    #  with (axis aligned) Gaussian kernel given by the relative position of
    #  the coordinate systems of fixed and moving
    #  \param[in] flag boolean
    def use_oriented_psf(self, flag):
        self._use_oriented_psf = flag

    # Set type of centered transform initializer
    #  \param[in] initializer_type
    def set_initializer_type(self, initializer_type):
        if initializer_type not in self._INITIALIZER_TYPES:
            raise ValueError("Possible initializer types: " +
                             str(self._INITIALIZER_TYPES))
        else:
            self._initializer_type = initializer_type

    # Get type of centered transform initializer
    def get_initializer_type(self):
        return self._initializer_type

    # Set interpolator
    #  \param[in] interpolator_type
    def set_interpolator(self, interpolator_type):
        self._interpolator = interpolator_type

    # Get interpolator
    #  \return interpolator as string
    def get_interpolator(self):
        return self._interpolator

    def set_metric(self, metric):
        self._metric = metric

    def set_metric_params(self, metric_params):
        self._metric_params = metric_params

    def set_optimizer(self, optimizer):
        self._optimizer = optimizer

    def set_optimizer_params(self, optimizer_params):
        self._optimizer_params = optimizer_params

    # Set optimizer scales
    #  \param[in] scales
    def set_scales_estimator(self, scales_estimator):
        if scales_estimator not in self._SCALES_ESTIMATORS:
            raise ValueError("Possible optimizer scales: " +
                             str(self._SCALES_ESTIMATORS))
        else:
            self._scales_estimator = scales_estimator

    def _run_registration(self):

        if self._use_fixed_mask:
            fixed_sitk_mask = self._fixed.sitk_mask
        else:
            fixed_sitk_mask = None

        if self._use_moving_mask:
            moving_sitk_mask = self._moving.sitk_mask
        else:
            moving_sitk_mask = None

        # Blur moving image with oriented Gaussian prior to the registration
        if self._use_oriented_psf:

            # Get oriented Gaussian covariance matrix
            cov_HR_coord = psf.PSF(
            ).get_gaussian_PSF_covariance_matrix_reconstruction_coordinates(
                self._fixed, self._moving)

            # Create recursive YVV Gaussianfilter
            image_type = itk.Image[itk.D, self._fixed.sitk.GetDimension()]
            gaussian_yvv = itk.SmoothingRecursiveYvvGaussianImageFilter[
                image_type, image_type].New()

            # Feed Gaussian filter with axis aligned covariance matrix
            sigma_axis_aligned = np.sqrt(np.diagonal(cov_HR_coord))
            print("Oriented PSF blurring with (axis aligned) sigma = " +
                  str(sigma_axis_aligned))
            print("\t(Based on computed covariance matrix = ")
            for i in range(0, 3):
                print("\t\t" + str(cov_HR_coord[i, :]))
            print("\twith square root of diagonal " +
                  str(np.diagonal(cov_HR_coord)) + ")")

            gaussian_yvv.SetInput(self._moving.itk)
            gaussian_yvv.SetSigmaArray(sigma_axis_aligned)
            gaussian_yvv.Update()
            moving_itk = gaussian_yvv.GetOutput()
            moving_itk.DisconnectPipeline()
            moving_sitk = sitkh.get_sitk_from_itk_image(moving_itk)

        else:
            moving_sitk = self._moving.sitk

        self._registration_method = \
            simplereg.simple_itk_registration.SimpleItkRegistration(
                fixed_sitk=self._fixed.sitk,
                moving_sitk=moving_sitk,
                fixed_sitk_mask=fixed_sitk_mask,
                moving_sitk_mask=moving_sitk_mask,
                registration_type=self._registration_type,
                interpolator=self._interpolator,
                metric=self._metric,
                metric_params=self._metric_params,
                optimizer=self._optimizer,
                optimizer_params=self._optimizer_params,
                initializer_type=self._initializer_type,
                use_multiresolution_framework=self._use_multiresolution_framework,
                optimizer_scales=self._scales_estimator,
                shrink_factors=self._shrink_factors,
                smoothing_sigmas=self._smoothing_sigmas,
                verbose=self._use_verbose,
            )

        self._registration_method.run()

        self._registration_transform_sitk = \
            self._registration_method.get_registration_transform_sitk()

    def _get_warped_moving_sitk(self):
        warped_moving_sitk = sitk.Resample(
            self._moving.sitk,
            self._fixed.sitk,
            self.get_registration_transform_sitk(),
            eval("sitk.sitk%s" % (self._interpolator)),
            0.,
            self._moving.sitk.GetPixelIDValue()
        )
        return warped_moving_sitk
