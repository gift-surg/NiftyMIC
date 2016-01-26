#!/usr/bin/python

## \file ITK_BlurringOperator.py
#  \brief Figure out how to apply the blurring operator with different orientations on the HR volume
#
#  \author: Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date: January 2016

## Import libraries
import SimpleITK as sitk
import itk
# import nibabel as nib
import numpy as np
import unittest
import math                     # for error function (erf)
from scipy.stats import norm    # for normal distribution

import os                       # used to execute terminal commands in python
import sys
sys.path.append("../src/")

## Import modules from src-folder
import SimpleITKHelper as sitkh

"""
Functions
"""
## Compute the covariance matrix modelling the PSF in-plane and through-plane of a stack
#  The PSF is modelled as Gaussian with
#       FWHM = 1.2*in-plane-resolution (in-plane)
#       FWHM = slice thickness (through-plane)
#  \param[in] stack_sitk stack defining the PSF
#  \return Matrix representing the PSF modelled as Gaussian
def get_Sigma_PSF(stack_sitk):
    spacing = np.array(stack_sitk.GetSpacing())

    ## Compute Gaussian to approximate in-plane PSF:
    sigma_x2 = (1.2*spacing[0])**2/(8*np.log(2))
    sigma_y2 = (1.2*spacing[1])**2/(8*np.log(2))

    ## Compute Gaussian to approximate through-plane PSF:
    sigma_z2 = spacing[2]**2/(8*np.log(2))

    return np.diag([sigma_x2, sigma_y2, sigma_z2])


## Compute rotated covariance matrix which expresses the PSF of the stack
#  in the coordinates of the HR_volume
#  \param[in] HR_volume_sitk current isotropic HR volume
#  \param[in] stack_sitk stack which is aimed to be simulated according to the slice acquisition model
#  \param[in] Sigma_PSF Covariance matrix modelling the Gaussian PSF of an acquired slice
#  \return Covariance matrix U*Sigma_diag*U' where U represents the
#          orthogonal trafo between stack and HR_volume
def get_rotated_covariance_matrix(HR_volume_sitk, stack_sitk, Sigma_PSF):
    dim = stack_sitk.GetDimension()

    direction_matrix_HR_volume = np.array(HR_volume_sitk.GetDirection()).reshape(dim,dim)
    direction_matrix_stack = np.array(stack_sitk.GetDirection()).reshape(dim,dim)

    U = direction_matrix_HR_volume.transpose().dot(direction_matrix_stack)

    return U.dot(Sigma_PSF).dot(U.transpose())


## Compute covariance matrix ready to use for subsequent interpolation step
#  within the resampling process
#  \param[in] HR_volume_sitk current isotropic HR volume
#  \param[in] stack_sitk stack which is aimed to be simulated according to the slice acquisition model
#  \param[in] Sigma_PSF Covariance matrix modelling the Gaussian PSF of an acquired slice
#  \return Covariance matrix S*U*Sigma_diag*U'*S where U represents the
#          orthogonal trafo between stack and HR_volume and S the scaling matrix
def get_scaled_inverse_rotated_covariance_matrix(HR_volume_sitk, stack_sitk, Sigma_PSF):
    spacing = np.array(HR_volume_sitk.GetSpacing())
    S = np.diag(spacing)
    Sigma_inv = np.linalg.inv(get_rotated_covariance_matrix(HR_volume_sitk, stack_sitk, Sigma_PSF))

    return S.dot(Sigma_inv).dot(S)


## Compute blurred point with respect to oriented PSF
#  \return value proportional to PSF blurred point
def compute_PSF_blurred_point(position, center, Sigma):
    return np.exp(-0.5* np.sum( (position-center)*Sigma.dot(position-center), 0))


def get_interpolator(interpolator_type, HR_volume_sitk=None, stack_sitk=None):

    ## Nearest neighbour
    if interpolator_type is 'NearestNeighbour':
        interpolator_type = itk.NearestNeighborInterpolateImageFunction.ID3D #Input image type: Float 3D
        interpolator = interpolator_type.New()
    
    ## Linear
    elif interpolator_type is 'Linear':
        interpolator_type = itk.LinearInterpolateImageFunction[input_image_type, input_pixel_type]
        interpolator = interpolator_type.New()

    ## Gaussian
    elif interpolator_type is 'Gaussian':

        Sigma_PSF = get_Sigma_PSF(stack_sitk)
        Sigma_aligned = get_rotated_covariance_matrix(HR_volume_sitk, stack_sitk, Sigma_PSF)
        Sigma_diag = Sigma_aligned.diagonal()

        print("Sigma_PSF = \n%s" %Sigma_PSF)
        print("Sigma_aligned = \n%s" %Sigma_aligned)
        print("Sigma_diag = \n%s" %Sigma_diag)

        interpolator_type = itk.GaussianInterpolateImageFunction.ID3D #Input image type: Float 3D
        interpolator = interpolator_type.New()
        interpolator.SetAlpha(alpha)
        interpolator.SetSigma(Sigma_diag)

        # print("alpha = %s" %interpolator.GetAlpha())
        # print("Sigma = %s" %interpolator.GetSigma())

    return interpolator

"""
Unit Test Class
"""

class TestUM(unittest.TestCase):

    accuracy = 7
    dir_input = "data/"


    def setUp(self):
        pass

    def test_01_check_get_Sigma_PSF(self):
        filename_stack = "FetalBrain_stack2_registered"

        stack_sitk = sitk.ReadImage(self.dir_input + filename_stack + ".nii.gz", sitk.sitkFloat32)
        spacing = np.array(stack_sitk.GetSpacing())
        
        Sigma = get_Sigma_PSF(stack_sitk)
        FWHM = np.array([1.2*spacing[0], 1.2*spacing[1], spacing[2]])

        for i in range(0, len(Sigma)):
            M = norm.pdf(0, loc=0, scale=np.sqrt(Sigma[i,i]))
            M_half_left = norm.pdf(-FWHM[i]/2., loc=0, scale=np.sqrt(Sigma[i,i]))
            M_half_right = norm.pdf(FWHM[i]/2., loc=0, scale=np.sqrt(Sigma[i,i]))

            ## Check results      
            self.assertEqual(np.around(
                abs( M - (M_half_left + M_half_right) )
                , decimals = self.accuracy), 0 )




"""
Main Function
"""
if __name__ == '__main__':

    class Object(object):
        pass

    dir_input = "data/"
    dir_output = "results/"
    filename_HR_volume = "FetalBrain_reconstruction_4stacks"
    filename_stack = "FetalBrain_stack2_registered"
    filename_slice = "FetalBrain_stack2_registered_midslice"
    # filename = "CTL_0_baseline_deleted_0.5"

    ## Define types of input and output pixels and state dimension of images
    input_pixel_type = itk.D
    output_pixel_type = input_pixel_type

    input_dimension = 3
    output_dimension = input_dimension

    ## Define type of input and output image
    input_image_type = itk.Image[input_pixel_type, input_dimension]
    output_image_type = itk.Image[output_pixel_type, output_dimension]

    ## Instantiate types of reader and writer
    reader_type = itk.ImageFileReader[input_image_type]
    writer_type = itk.ImageFileWriter[output_image_type]
    image_IO_type = itk.NiftiImageIO

    ## Create reader and writer
    reader_HR_volume = reader_type.New()
    reader_stack = reader_type.New()
    reader_slice = reader_type.New()
    writer = writer_type.New()

    ## Set image IO type to nifti
    image_IO = image_IO_type.New()
    reader_HR_volume.SetImageIO(image_IO)
    reader_stack.SetImageIO(image_IO)
    reader_slice.SetImageIO(image_IO)

    ## Read images
    reader_HR_volume.SetFileName(dir_input + filename_HR_volume + ".nii.gz")
    reader_HR_volume.Update()
    
    reader_stack.SetFileName(dir_input + filename_stack + ".nii.gz")
    reader_stack.Update()

    reader_slice.SetFileName(dir_input + filename_slice + ".nii.gz")
    reader_slice.Update()

    ## Get image
    HR_volume_itk = reader_HR_volume.GetOutput()
    stack_itk = reader_stack.GetOutput()
    slice_itk = reader_slice.GetOutput()

    HR_volume_sitk = sitk.ReadImage(dir_input + filename_HR_volume + ".nii.gz", sitk.sitkFloat32)
    stack_sitk = sitk.ReadImage(dir_input + filename_stack + ".nii.gz", sitk.sitkFloat32)
    slice_sitk = sitk.ReadImage(dir_input + filename_slice + ".nii.gz", sitk.sitkFloat32)

    """
    'Real' start:
    """
    ## Resample Image Filter
    filter_type = itk.ResampleImageFilter[input_image_type, output_image_type]
    filter = filter_type.New()

    ## Set input image
    filter.SetInput(HR_volume_itk)
    filter.SetOutputParametersFromImage(stack_itk)
    # filter.Update()

    ## Choose interpolator
    s_interpolators = ['NearestNeighbour', 'Linear', 'Gaussian']
    interpolator_type = s_interpolators[2]
    interpolator = get_interpolator(
        'Gaussian',
        HR_volume_sitk=HR_volume_sitk,
        stack_sitk=stack_sitk)

    ## Specify interpolator for filter
    filter.SetInterpolator(interpolator)
    filter.Update()

    ## Set covariance matrix
    # sigma_x2 = 1
    # sigma_y2 = 2
    # sigma_z2 = 3
    # Sigma = np.diag([sigma_x2,sigma_y2,sigma_z2])

    ## Write warped image
    # writer.SetFileName(dir_output + "test.nii.gz")
    # writer.SetInput(filter.GetOutput())
    # writer.Update()


    """
    Unit tests:
    """
    print("\nUnit tests:\n--------------")
    unittest.main()