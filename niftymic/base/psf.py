##
# \file psf.py
# \brief      Compute the Gaussian point spread function (PSF) associated with
#             a slice acquisition in the coordinates of the reconstruction
#             space
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       April 2016
#


# Import libraries
import numpy as np


class PSF:

    ##
    # Compute rotated covariance matrix which expresses the PSF of the slice in
    # the coordinates of the HR volume
    # \date       2017-11-01 16:16:20+0000
    #
    # \param      self            The object
    # \param      slice           Slice object which is aimed to be simulated
    #                             according to the slice acquisition model
    # \param      reconstruction  Stack object containing the HR volume
    #
    # \return     Covariance matrix U*Sigma_diag*U' where U represents the
    #             orthogonal trafo between slice and reconstruction
    #
    def get_covariance_matrix_in_reconstruction_space(
            self,
            slice,
            reconstruction):

        cov = self.get_covariance_matrix_in_reconstruction_space_sitk(
            reconstruction.sitk.GetDirection(),
            slice.sitk.GetDirection(),
            slice.sitk.GetSpacing())

        return cov

    ##
    # Gets the axis-aligned covariance matrix describing the PSF in
    # reconstruction space coordinates.
    # \date       2017-11-01 16:21:31+0000
    #
    # \param      self                           The object
    # \param      reconstruction_direction_sitk  Image header (sitk) direction
    #                                            of reconstruction space
    # \param      slice_direction_sitk           Image header (sitk) direction
    #                                            of slice space
    # \param      slice_spacing                  Spacing of slice space
    #
    # \return     Axis-aligned covariance matrix describing the PSF.
    #
    def get_covariance_matrix_in_reconstruction_space_sitk(
            self,
            reconstruction_direction_sitk,
            slice_direction_sitk,
            slice_spacing):

        # Compute rotation matrix to express the PSF in the coordinate system
        # of the reconstruction space
        U = self._get_relative_rotation_matrix(
            slice_direction_sitk, reconstruction_direction_sitk)

        # Get axis-aligned PSF
        cov = self.get_gaussian_psf_covariance_matrix_from_spacing(
            slice_spacing)

        # Return Gaussian blurring variance covariance matrix of slice in
        # reconstruction space coordinates
        return U.dot(cov).dot(U.transpose())

    ##
    # Gets the predefined covariance matrix in reconstruction space.
    # \date       2017-10-31 23:27:44+0000
    #
    # \param      self                           The object
    # \param      reconstruction_direction_sitk  Image header (sitk) direction
    #                                            of reconstruction space
    # \param      slice_direction_sitk           Image header (sitk) direction
    #                                            of slice space
    # \param      cov                            Axis-aligned covariance matrix
    #                                            describing the PSF
    #
    # \return     The predefined covariance matrix in reconstruction space
    #             coordinates.
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

    ##
    # Compute (axis aligned) covariance matrix from spacing The PSF is modelled
    # as Gaussian with
    #   FWHM = 1.2*in-plane-resolution (in-plane)
    #   FWHM = slice thickness (through-plane)
    # \date       2017-11-01 16:16:36+0000
    #
    # \param      spacing  3D array containing in-plane and through-plane
    #                      dimensions
    #
    # \return     (axis aligned) covariance matrix representing PSF modelled
    #             Gaussian as 3x3 np.array
    #
    @staticmethod
    def get_gaussian_psf_covariance_matrix_from_spacing(spacing):

        # Compute Gaussian to approximate in-plane PSF:
        sigma_x2 = (1.2*spacing[0])**2/(8*np.log(2))
        sigma_y2 = (1.2*spacing[1])**2/(8*np.log(2))

        # Compute Gaussian to approximate through-plane PSF:
        sigma_z2 = spacing[2]**2/(8*np.log(2))

        return np.diag([sigma_x2, sigma_y2, sigma_z2])

    ##
    # Gets the relative rotation matrix to express slice-axis aligned
    # covariance matrix in coordinates of HR volume
    # \date       2016-10-14 16:37:57+0100
    #
    # \param      slice_direction_sitk           Image header (sitk) direction
    #                                            of slice space
    # \param      reconstruction_direction_sitk  Image header (sitk) direction
    #                                            of reconstruction space
    #
    # \return     The relative rotation matrix as 3x3 numpy array
    #
    @staticmethod
    def _get_relative_rotation_matrix(slice_direction_sitk,
                                      reconstruction_direction_sitk):

        # Compute rotation matrix to express the PSF in the coordinate system
        # of the HR volume
        dim = np.sqrt(len(slice_direction_sitk)).astype('int')
        direction_matrix_reconstruction = np.array(
            reconstruction_direction_sitk).reshape(dim, dim)
        direction_matrix_slice = np.array(
            slice_direction_sitk).reshape(dim, dim)

        U = direction_matrix_reconstruction.transpose().dot(
            direction_matrix_slice)

        return U
