# \file PSF.py
#  \brief Compute point spread function (PSF) based on orientation of slice to volume
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date April 2016


# Import libraries
import os
import sys
import itk
import SimpleITK as sitk
import numpy as np

# Import modules from src-folder
import pysitk.simple_itk_helper as sitkh


class PSF:

    # Compute rotated covariance matrix which expresses the PSF of the slice
    #  in the coordinates of the HR volume
    #  \param[in] slice Slice object which is aimed to be simulated according to the slice acquisition model
    #  \param[in] reconstruction Stack object containing the HR volume
    #  \return Covariance matrix U*Sigma_diag*U' where U represents the
    #          orthogonal trafo between slice and reconstruction
    def get_gaussian_PSF_covariance_matrix_reconstruction_coordinates(self, slice, reconstruction):

        spacing_slice = np.array(slice.sitk.GetSpacing())
        direction_slice = np.array(slice.sitk.GetDirection())

        return self.get_gaussian_PSF_covariance_matrix_reconstruction_coordinates_from_direction_and_spacing(direction_slice, spacing_slice, reconstruction)

    ##
    #       Gets the Gaussian PSF covariance matrix HR volume coordinates from covariance.
    # \date       2016-10-14 16:02:26+0100
    #
    # \param      self       The object
    # \param      slice      The slice
    # \param      reconstruction  The HR volume
    # \param      cov        Slice axis-aligned covariances diag(sigma_x^2, sigma_y^2, sigma_z^2) as numpy 3x3 array
    #
    # \return     The Gaussian PSF covariance matrix HR volume coordinates from covariance.
    #
    def get_gaussian_PSF_covariance_matrix_reconstruction_coordinates_from_covariances(self, slice, reconstruction, cov):

        # Compute rotation matrix to express the PSF in the coordinate system
        # of the HR volume
        U = self._get_relative_rotation_matrix(
            slice.sitk.GetDirection(), reconstruction.sitk.GetDirection())

        # Return Gaussian blurring variance covariance matrix of slice in HR
        # volume coordinates
        return U.dot(cov).dot(U.transpose())

    # Compute rotated covariance marix which expresses the PSF of the slice,
    #  given by its directon and spacing, in the coordinates of the HR volume
    #  \param[in] direction_slice information obtained by GetDirection() from slice
    #  \param[in] spacing_slice voxel dimension of slice as np.array
    #  \return Covariance matrix U*Sigma_diag*U' where U represents the
    #          orthogonal trafo between slice and reconstruction
    def get_gaussian_PSF_covariance_matrix_reconstruction_coordinates_from_direction_and_spacing(self, direction_slice_sitk, spacing_slice, reconstruction):

        # Compute rotation matrix to express the PSF in the coordinate system
        # of the HR volume
        U = self._get_relative_rotation_matrix(
            direction_slice_sitk, reconstruction.sitk.GetDirection())

        # Get axis algined PSF
        cov = self.get_gaussian_PSF_covariance_matrix_from_spacing(
            spacing_slice)

        # Return Gaussian blurring variance covariance matrix of slice in HR
        # volume coordinates
        return U.dot(cov).dot(U.transpose())

    # Compute (axis aligned) covariance matrix from spacing
    #  The PSF is modelled as Gaussian with
    #       FWHM = 1.2*in-plane-resolution (in-plane)
    #       FWHM = slice thickness (through-plane)
    #  \param[in] spacing 3D array containing in-plane and through-plane dimensions
    #  \return (axis aligned) covariance matrix representing PSF modelled Gaussian as 3x3 np.array
    def get_gaussian_PSF_covariance_matrix_from_spacing(self, spacing):

        # Compute Gaussian to approximate in-plane PSF:
        sigma_x2 = (1.2*spacing[0])**2/(8*np.log(2))
        sigma_y2 = (1.2*spacing[1])**2/(8*np.log(2))

        # Compute Gaussian to approximate through-plane PSF:
        sigma_z2 = spacing[2]**2/(8*np.log(2))

        return np.diag([sigma_x2, sigma_y2, sigma_z2])

    ##
    #       Gets the relative rotation matrix to express slice-axis
    #             aligned covariance matrix in coordinates of HR volume
    # \date       2016-10-14 16:37:57+0100
    #
    # \param      self                      The object
    # \param      direction_slice_sitk      The direction slice sitk
    # \param      direction_reconstruction_sitk  The direction hr volume sitk
    #
    # \return     The relative rotation matrix as 3x3 numpy array
    #
    def _get_relative_rotation_matrix(self, direction_slice_sitk, direction_reconstruction_sitk):

        # Compute rotation matrix to express the PSF in the coordinate system
        # of the HR volume
        dim = np.sqrt(len(direction_slice_sitk)).astype('int')
        direction_matrix_reconstruction = np.array(
            direction_reconstruction_sitk).reshape(dim, dim)
        direction_matrix_slice = np.array(
            direction_slice_sitk).reshape(dim, dim)

        U = direction_matrix_reconstruction.transpose().dot(direction_matrix_slice)
        # print("U = \n%s\ndet(U) = %s" % (U,np.linalg.det(U)))

        return U

    def get_covariance_matrix_in_reconstruction_space(
            self,
            reconstruction_direction_sitk,
            slice_direction_sitk,
            slice_spacing):

        # Compute rotation matrix to express the PSF in the coordinate system
        # of the reconstruction space
        U = self._get_relative_rotation_matrix(
            slice_direction_sitk, reconstruction_direction_sitk)

        # Get axis-aligned PSF
        cov = self.get_gaussian_PSF_covariance_matrix_from_spacing(
            slice_spacing)

        # Return Gaussian blurring variance covariance matrix of slice in
        # reconstruction space coordinates
        return U.dot(cov).dot(U.transpose())

    ##
    # Gets the predefined covariance matrix in reconstruction space.
    # \date       2017-10-31 23:27:44+0000
    #
    # \param      self                           The object
    # \param      reconstruction_direction_sitk  The reconstruction direction sitk
    # \param      slice_direction_sitk           The slice direction sitk
    # \param      cov                            Axis-aligned covariance matrix
    #                                            describing the PSF
    #
    # \return     The predefined covariance matrix in reconstruction space.
    #
    def get_predefined_covariance_matrix_in_reconstruction_space(
            self,
            reconstruction_direction_sitk,
            slice_direction_sitk,
            cov):

        # Compute rotation matrix to express the PSF in the coordinate system
        # of the reconstruction space
        U = self._get_relative_rotation_matrix(
            slice_direction_sitk, reconstruction_direction_sitk)

        # Return Gaussian blurring variance covariance matrix of slice in
        # reconstruction space coordinates
        return U.dot(cov).dot(U.transpose())
