## \file PSF.py
#  \brief Compute point spread function (PSF) based on orientation of slice to volume
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date April 2016


## Import libraries
import os                       # used to execute terminal commands in python
import sys
import itk
import SimpleITK as sitk
import numpy as np

## Import modules from src-folder
import SimpleITKHelper as sitkh


class PSF:

    # def __init__(self):


    ## Compute rotated covariance matrix which expresses the PSF of the slice
    #  in the coordinates of the HR volume
    #  \param[in] slice Slice object which is aimed to be simulated according to the slice acquisition model
    #  \param[in] HR_volume Stack object containing the HR volume
    #  \return Covariance matrix U*Sigma_diag*U' where U represents the
    #          orthogonal trafo between slice and HR_volume
    def get_gaussian_PSF_covariance_matrix_HR_volume_coordinates(self, slice, HR_volume):

        spacing_slice = np.array(slice.sitk.GetSpacing())
        direction_slice = np.array(slice.sitk.GetDirection())

        return self.get_gaussian_PSF_covariance_matrix_HR_volume_coordinates_from_direction_and_spacing(direction_slice, spacing_slice, HR_volume)


    ## Compute rotated covariance marix which expresses the PSF of the slice,
    #  given by its directon and spacing, in the coordinates of the HR volume
    #  \param[in] direction_slice information obtained by GetDirection() from slice
    #  \param[in] spacing_slice voxel dimension of slice as np.array
    #  \return Covariance matrix U*Sigma_diag*U' where U represents the
    #          orthogonal trafo between slice and HR_volume
    def get_gaussian_PSF_covariance_matrix_HR_volume_coordinates_from_direction_and_spacing(self, direction_slice, spacing_slice, HR_volume):

        ## Compute rotation matrix to express the PSF in the coordinate system of the HR volume
        dim = HR_volume.sitk.GetDimension()
        direction_matrix_HR_volume = np.array(HR_volume.sitk.GetDirection()).reshape(dim,dim)
        direction_matrix_slice = np.array(direction_slice).reshape(dim,dim)

        U = direction_matrix_HR_volume.transpose().dot(direction_matrix_slice)
        # print("U = \n%s\ndet(U) = %s" % (U,np.linalg.det(U)))

        ## Get axis algined PSF
        cov = self._get_gaussian_PSF_covariance_matrix_from_spacing(spacing_slice)

        ## Return Gaussian blurring variance covariance matrix of slice in HR volume coordinates 
        return U.dot(cov).dot(U.transpose())


    ## Compute the covariance matrix modelling the PSF in-plane and 
    #  through-plane of a slice. Hence, associated PSF is axis aligned.
    #  The PSF is modelled as Gaussian with
    #       FWHM = 1.2*in-plane-resolution (in-plane)
    #       FWHM = slice thickness (through-plane)
    #  \param[in] slice Slice instance defining the PSF
    #  \return Covariance matrix representing the PSF modelled as Gaussian
    def _get_gaussian_PSF_covariance_matrix(self, slice):
        spacing = np.array(slice.sitk.GetSpacing())

        return self._get_gaussian_PSF_covariance_matrix_from_spacing(spacing)


    ## Compute (axis aligned) covariance matrix from spacing
    #  \param[in] spacing 3D array containing in-plane and through-plane dimensions
    #  \return get (axis aligned) covariance matrix as 3x3 np.array
    def _get_gaussian_PSF_covariance_matrix_from_spacing(self, spacing):
        
        ## Compute Gaussian to approximate in-plane PSF:
        sigma_x2 = (1.2*spacing[0])**2/(8*np.log(2))
        sigma_y2 = (1.2*spacing[1])**2/(8*np.log(2))

        ## Compute Gaussian to approximate through-plane PSF:
        sigma_z2 = spacing[2]**2/(8*np.log(2))

        return np.diag([sigma_x2, sigma_y2, sigma_z2])


