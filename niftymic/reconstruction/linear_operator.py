##
# \file linear_operator.py
# \brief      Implementation of linear operations associated with the physical
#             slice acquisition model.
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date October 2017


# Import libraries
import sys
import itk
import SimpleITK as sitk
import numpy as np

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh

import niftymic.base.psf as psf


class LinearOperator(object):

    def __init__(self,
                 deconvolution_mode="full_3D",
                 predefined_covariance=None,
                 alpha_cut=3,
                 image_type=itk.Image.D3,
                 default_pixel_type=0.0):

        self._deconvolution_mode = deconvolution_mode

        # In case only diagonal entries are given, create diagonal matrix
        if predefined_covariance is not None:
            if predefined_covariance.size is 3:
                self._predefined_covariance = np.diag(
                    np.array(predefined_covariance))
            else:
                self._predefined_covariance = np.array(predefined_covariance)

        self._psf = psf.PSF()

        # Allocate and initialize Oriented Gaussian Interpolate Image Filter
        self._filter_oriented_gaussian = \
            itk.OrientedGaussianInterpolateImageFilter[
                image_type, image_type].New()
        self._filter_oriented_gaussian.SetDefaultPixelValue(default_pixel_type)
        self._filter_oriented_gaussian.SetAlpha(alpha_cut)

        # Allocate and initialize Adjoint Oriented Gaussian Interpolate Image
        # Filter
        self._filter_adjoint_oriented_gaussian = \
            itk.AdjointOrientedGaussianInterpolateImageFilter[
                image_type, image_type].New()
        self._filter_adjoint_oriented_gaussian.SetDefaultPixelValue(
            default_pixel_type)
        self._filter_adjoint_oriented_gaussian.SetAlpha(alpha_cut)

        # Allocate and initialize masking image filter
        self._masking = itk.MultiplyImageFilter[
            image_type, image_type, image_type].New()

        self._get_covariance = {
            "full_3D": self._get_covariance_full_3d,
            "only_in_plane": self._get_covariance_only_in_plane,
            "predefined_covariance": self._get_covariance_predefined,
        }

    ##
    # Perform forward operation on reconstruction image, i.e.
    # \f$y = D B x =: A(x)
    # \f$ with
    # \f$ D\f$ and
    # \f$ B
    # \f$ being the downsampling and blurring operators, respectively.
    # \date       2017-10-31 23:36:18+0000
    #
    # \param      self                The object
    # \param      reconstruction_itk  Reconstruction image as itk.Image object
    # \param      slice_itk           Slice image as itk.Image object. Required
    #                                 to define output space and orientation
    #                                 for PSF.
    #
    # \return     Image A(x) as itk.Image object in slice_itk image space
    #
    def A_itk(self, reconstruction_itk, slice_itk):

        # Get covariance describing PSF orientation of slice in reconstruction
        # space
        cov = self._get_covariance[self._deconvolution_mode](
            reconstruction_itk, slice_itk)

        reconstruction_itk.Update()
        self._filter_oriented_gaussian.SetCovariance(cov.flatten())
        self._filter_oriented_gaussian.SetInput(reconstruction_itk)
        self._filter_oriented_gaussian.SetOutputParametersFromImage(slice_itk)
        self._filter_oriented_gaussian.UpdateLargestPossibleRegion()
        self._filter_oriented_gaussian.Update()

        A_itk_reconstruction = self._filter_oriented_gaussian.GetOutput()
        A_itk_reconstruction.DisconnectPipeline()

        return A_itk_reconstruction

    ##
    # Perform backward operation on slice image, i.e.
    # \f$z = B^* D^* y =: A^*(y)
    # \f$ with
    # \f$ D^*
    # \f$ and
    # \f$ B^*
    # \f$ being the adjoint downsampling and blurring operators, respectively.
    # \date       2017-10-31 23:44:41+0000
    #
    # \param      self                The object
    # \param      slice_itk           Slice image as itk.Image object
    # \param      reconstruction_itk  Reconstruction image as itk.Image object.
    #                                 Required to define output space and
    #                                 orientation for PSF
    #
    # \return     Image A^*(y) as itk.Image object in reconstruction_itk image
    #             space
    #
    def A_adj_itk(self, slice_itk, reconstruction_itk):

        # Get covariance describing PSF orientation of slice in reconstruction
        # space
        cov = self._get_covariance[self._deconvolution_mode](
            reconstruction_itk, slice_itk)

        reconstruction_itk.Update()
        self._filter_adjoint_oriented_gaussian.SetCovariance(cov.flatten())
        self._filter_adjoint_oriented_gaussian.SetInput(slice_itk)
        self._filter_adjoint_oriented_gaussian.SetOutputParametersFromImage(
            reconstruction_itk)
        self._filter_adjoint_oriented_gaussian.UpdateLargestPossibleRegion()
        self._filter_adjoint_oriented_gaussian.Update()

        A_adj_itk_slice = self._filter_adjoint_oriented_gaussian.GetOutput()
        A_adj_itk_slice.DisconnectPipeline()

        return A_adj_itk_slice

    ##
    # Perform masking operation on itk.Image object
    # \date       2017-10-31 23:59:00+0000
    #
    # \param      self            The object
    # \param      image_itk       Image as itk.Image object
    # \param      image_itk_mask  Image mask as itk.Image object
    #
    # \return     Masked image as itk.Image object
    #
    def M_itk(self, image_itk, image_itk_mask):

        self._masking.SetInput1(image_itk_mask)
        self._masking.SetInput2(image_itk)
        self._masking.UpdateLargestPossibleRegion()
        self._masking.Update()

        Mk_slice_itk = self._masking.GetOutput()
        Mk_slice_itk.DisconnectPipeline()

        return Mk_slice_itk

    def _get_covariance_full_3d(self,
                                reconstruction_itk,
                                slice_itk):

        reconstruction_direction_sitk = sitkh.get_sitk_from_itk_direction(
            reconstruction_itk.GetDirection())
        slice_direction_sitk = sitkh.get_sitk_from_itk_direction(
            slice_itk.GetDirection())
        slice_spacing = np.array(slice_itk.GetSpacing())

        cov = self._psf.get_covariance_matrix_in_reconstruction_space(
            reconstruction_direction_sitk=reconstruction_direction_sitk,
            slice_direction_sitk=slice_direction_sitk,
            slice_spacing=slice_spacing)

        return cov

    def _get_covariance_only_in_plane(self,
                                      reconstruction_itk,
                                      slice_itk):
        reconstruction_direction_sitk = sitkh.get_sitk_from_itk_direction(
            reconstruction_itk.GetDirection())
        slice_direction_sitk = sitkh.get_sitk_from_itk_direction(
            slice_itk.GetDirection())
        slice_spacing = np.array(slice_itk.GetSpacing())

        # Get spacing of slice and set it very small so that the corresponding
        # covariance is negligibly small in through-plane direction. Hence,
        # only in-plane deconvolution is approximated
        slice_spacing[2] = 1e-6

        cov = self._psf.get_covariance_matrix_in_reconstruction_space(
            reconstruction_direction_sitk=reconstruction_direction_sitk,
            slice_direction_sitk=slice_direction_sitk,
            slice_spacing=slice_spacing)

        return cov

    def _get_covariance_predefined(self,
                                   reconstruction_itk,
                                   slice_itk):
        reconstruction_direction_sitk = sitkh.get_sitk_from_itk_direction(
            reconstruction_itk.GetDirection())
        slice_direction_sitk = sitkh.get_sitk_from_itk_direction(
            slice_itk.GetDirection())

        cov = \
            self._psf.get_predefined_covariance_matrix_in_reconstruction_space(
                reconstruction_direction_sitk=reconstruction_direction_sitk,
                slice_direction_sitk=slice_direction_sitk,
                cov=self._predefined_covariance)

        return cov
