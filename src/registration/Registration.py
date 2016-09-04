#!/usr/bin/python

## \file Registration.py
#  \brief 
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date Aug 2016


## Import libraries
import sys
import itk
import SimpleITK as sitk
import numpy as np
from scipy.optimize import least_squares
import time

## Add directories to import modules
DIR_SRC_ROOT = "../../src/"
sys.path.append(DIR_SRC_ROOT + "base/")

## Import modules
import SimpleITKHelper as sitkh
import PSF as psf
import Slice as sl
import time
from datetime import timedelta

## Pixel type of used 3D ITK image
PIXEL_TYPE = itk.D

## ITK image type
IMAGE_TYPE = itk.Image[PIXEL_TYPE, 3]


class Registration(object):

    ##-------------------------------------------------------------------------
    # \brief      Constructor
    # \date       2016-08-02 15:49:34+0100
    #
    # \param      self       The object
    # \param      slice      The slice
    # \param      HR_volume  The hr volume
    # \param      alpha_cut  The alpha cut
    #
    def __init__(self, slice, HR_volume, alpha_cut=3):

        self._slice = slice
        self._HR_volume = HR_volume

        self._slice_affine_transform_sitk = sitkh.get_sitk_affine_transform_from_sitk_image(slice.sitk)
        self._slice_spacing = self._slice.sitk.GetSpacing()
        self._slice_size = self._slice.sitk.GetSize()

        ## Used for PSF modelling
        self._alpha_cut = alpha_cut
        self._psf = psf.PSF()

        ## Allocate and initialize Oriented Gaussian Interpolate Image Filter
        self._filter_oriented_Gaussian_interpolator = itk.OrientedGaussianInterpolateImageFunction[IMAGE_TYPE, PIXEL_TYPE].New()
        self._filter_oriented_Gaussian_interpolator.SetAlpha(self._alpha_cut)

        self._filter_oriented_Gaussian = itk.ResampleImageFilter[IMAGE_TYPE, IMAGE_TYPE].New()
        self._filter_oriented_Gaussian.SetInput(self._HR_volume.itk)
        self._filter_oriented_Gaussian.SetInterpolator(self._filter_oriented_Gaussian_interpolator)
        self._filter_oriented_Gaussian.SetDefaultPixelValue(0.0)

        ## Allocate and initialize Adjoint Oriented Gaussian Interpolate Image Filter
        self._filter_adjoint_oriented_Gaussian = itk.AdjointOrientedGaussianInterpolateImageFilter[IMAGE_TYPE, IMAGE_TYPE].New()
        self._filter_adjoint_oriented_Gaussian.SetDefaultPixelValue(0.0)
        self._filter_adjoint_oriented_Gaussian.SetAlpha(self._alpha_cut)
        self._filter_adjoint_oriented_Gaussian.SetOutputParametersFromImage(self._HR_volume.itk)

        ## Create PyBuffer object for conversion between NumPy arrays and ITK images
        self._itk2np = itk.PyBuffer[IMAGE_TYPE]

        self._parameters = [None]*6
    

    ##-------------------------------------------------------------------------
    # \brief      Gets the parameters estimated by registration algorithm.
    # \date       2016-08-03 00:10:45+0100
    #
    # \param      self  The object
    #
    # \return     The parameters.
    #
    def get_parameters(self):
        return self._parameters


    ##-------------------------------------------------------------------------
    # \date       2016-07-29 12:30:30+0100
    # \brief      Print statistics associated to performed reconstruction
    #
    # \param      self  The object
    #
    def print_statistics(self):
        # print("\nStatistics for performed registration:" %(self._reg_type))
        # if self._elapsed_time_sec < 0:
        #     raise ValueError("Error: Elapsed time has not been measured. Run 'run_reconstruction' first.")
        # else:
        print("\tElapsed time = %s" %(timedelta(seconds=self._elapsed_time_sec)))
        # print("\tell^2-residual sum_k ||M_k(A_k x - y_k||_2^2 = %.3e" %(self._residual_ell2))
        # print("\tprior residual = %.3e" %(self._residual_prior))


    ##-------------------------------------------------------------------------
    # \brief      Run registration for given slice
    # \date       2016-08-03 00:11:51+0100
    # \post       self._paramters is updated
    #
    # \param      self  The object
    #
    def run_registration(self):
        
        time_start = time.time()

        fun = lambda x: self._get_residual_data_fit(x)
        x0 = np.zeros(6)

        ## Non-linear least-squares method: but does not go ahead
        res = least_squares(fun=fun, x0=x0, method='trf', verbose=1) 
        # res = least_squares(fun=fun, x0=x0, method='lm', loss='linear', tr_solver='exact', verbose=1) 
        self._parameters = res.x
        
        ## Set elapsed time
        time_end = time.time()
        self._elapsed_time_sec = time_end-time_start


    ##-------------------------------------------------------------------------
    # \brief      Compute residual y_k - A_k(theta)x based on parameters
    #             (theta). 
    # \date       2016-08-03 00:12:43+0100
    #
    # \param      self        The object
    # \param      parameters  The parameters
    #
    # \return     The residual data fit as N_k-array
    #
    def _get_residual_data_fit(self, parameters):

        ## Create registration transform based on parameters
        transform_sitk = sitk.Euler3DTransform()
        transform_sitk.SetParameters(parameters)

        ## Get composite affine transform: reg_trafo \circ slice_space
        composite_transform_sitk = sitkh.get_composite_sitk_affine_transform(transform_sitk, self._slice_affine_transform_sitk)

        ## Extract direction and origin of transformed slice space
        direction_sitk = sitkh.get_sitk_image_direction_from_sitk_affine_transform(composite_transform_sitk, self._slice_spacing)
        origin_sitk = sitkh.get_sitk_image_origin_from_sitk_affine_transform(composite_transform_sitk)

        ## Set output to transformed slice space
        self._filter_oriented_Gaussian.SetOutputOrigin(origin_sitk)
        self._filter_oriented_Gaussian.SetOutputSpacing(self._slice_spacing)
        self._filter_oriented_Gaussian.SetOutputDirection(sitkh.get_itk_direction_form_sitk_direction(direction_sitk))
        self._filter_oriented_Gaussian.SetSize(self._slice_size)
        self._filter_oriented_Gaussian.UpdateLargestPossibleRegion()

        ## Set oriented PSF based on transformed slice space
        Cov_HR_coord = self._psf.get_gaussian_PSF_covariance_matrix_HR_volume_coordinates_from_direction_and_spacing(direction_sitk, self._slice_spacing, self._HR_volume)
        self._filter_oriented_Gaussian_interpolator.SetCovariance(Cov_HR_coord.flatten())

        ## Create simulated slice from volume
        self._filter_oriented_Gaussian.Update()
        Ak_vol_itk = self._filter_oriented_Gaussian.GetOutput();
        Ak_vol_itk.DisconnectPipeline()

        ## Compute residual y_k - A_k(theta)x and return as 1D array
        nda_slice = sitk.GetArrayFromImage(self._slice.sitk)
        nda_Ak_vol = self._itk2np.GetArrayFromImage(Ak_vol_itk)

        return (nda_slice - nda_Ak_vol).flatten()

    
    ##-------------------------------------------------------------------------
    # \brief      Compute squared ell^2 norm of residual, i.e. compute residual
    #             of data fit \f$ \Vert \vec{y}_k - A_k(\theta)\vec{x} \f$
    # \date       2016-08-03 00:43:29+0100
    #
    # \param      self        The object
    # \param      parameters  The parameters containing 6 DOF for rigid trafo
    #
    # \return     ell^2-residual of data fit.
    #
    def _get_residual_ell2(self, parameters):

        ## Get y_k - A_k(theta)x as 1D array
        residual_data_fit = self._get_residual_data_fit(parameters)

        return np.sum(residual_data_fit**2)





