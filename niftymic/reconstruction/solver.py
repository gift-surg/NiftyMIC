##
# \file solver.py
# \brief      Base class to solve the SRR problem y_k = D_k B_k W_k x = A_k x
#             for all slices k = 1, ..., K.
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       July 2016
#


# Import libraries
from abc import ABCMeta, abstractmethod
import sys
import itk
import SimpleITK as sitk
import numpy as np

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh

import niftymic.reconstruction.linear_operators as lin_op

# Allowed data loss functions
DATA_LOSS = ['linear', 'soft_l1', 'huber', 'cauchy', 'arctan']


##
# This class contains the common functions/attributes of the solvers
# \date       2017-11-01 01:04:31+0000
#
class Solver(object):
    __metaclass__ = ABCMeta

    ##
    # Constructor
    # \date          2016-08-01 22:53:37+0100
    #
    # \param         self                   The object
    # \param         stacks                 list of Stack objects containing
    #                                       all stacks used for the
    #                                       reconstruction
    # \param[in,out] reconstruction         Stack object containing the current
    #                                       estimate of the reconstruction
    #                                       (used as initial value + space
    #                                       definition)
    # \param         alpha_cut              Cut-off distance for Gaussian
    #                                       blurring filter
    # \param         alpha                  regularization parameter, scalar
    # \param         iter_max               number of maximum iterations,
    #                                       scalar
    # \param         minimizer              The minimizer
    # \param         deconvolution_mode     Either "full_3D" or
    #                                       "only_in_plane". Indicates whether
    #                                       full 3D or only in-plane
    #                                       deconvolution is considered
    # \param         data_loss              The data loss
    # \param         huber_gamma            The huber gamma
    # \param         predefined_covariance  Either only diagonal entries
    #                                       (sigma_x2, sigma_y2, sigma_z2) or
    #                                       as full 3x3 numpy array
    # \param         verbose                The verbose
    #
    def __init__(self,
                 stacks,
                 reconstruction,
                 alpha_cut,
                 alpha,
                 iter_max,
                 minimizer,
                 x_scale,
                 data_loss,
                 data_loss_scale,
                 huber_gamma,
                 deconvolution_mode,
                 predefined_covariance,
                 verbose,
                 image_type=itk.Image.D3,
                 use_masks=True,
                 ):

        # Initialize variables
        self._stacks = stacks
        self._reconstruction = reconstruction

        # Cut-off distance for Gaussian blurring filter
        self._alpha_cut = alpha_cut

        self._deconvolution_mode = deconvolution_mode
        self._predefined_covariance = predefined_covariance
        self._linear_operators = lin_op.LinearOperators(
            deconvolution_mode=self._deconvolution_mode,
            predefined_covariance=self._predefined_covariance,
            alpha_cut=self._alpha_cut,
            image_type=image_type
        )

        # Settings for solver
        self._alpha = alpha
        self._iter_max = iter_max

        self._use_masks = use_masks

        self._minimizer = minimizer
        self._data_loss = data_loss
        self._data_loss_scale = data_loss_scale

        if x_scale == "max":
            self._x_scale = sitk.GetArrayFromImage(
                reconstruction.sitk).max()

            # Avoid zero in case zero-image is given
            if self._x_scale == 0:
                self._x_scale = 1
        else:
            self._x_scale = x_scale

        self._huber_gamma = huber_gamma

        self._verbose = verbose

        # Allocate variables containing information about statistics of
        # reconstruction
        self._computational_time = None
        self._residual_ell2 = None
        self._residual_prior = None

        # Create PyBuffer object for conversion between NumPy arrays and ITK
        # images
        self._itk2np = itk.PyBuffer[image_type]

        # -----------------------------Set helpers-----------------------------
        self._N_stacks = len(self._stacks)

        # Compute total amount of pixels for all slices
        self._N_total_slice_voxels = 0
        for i in range(0, self._N_stacks):
            N_stack_voxels = np.array(self._stacks[i].sitk.GetSize()).prod()
            self._N_total_slice_voxels += N_stack_voxels

        # Extract information ready to use for itk image conversion operations
        self._reconstruction_shape = sitk.GetArrayFromImage(
            self._reconstruction.sitk).shape

        # Compute total amount of voxels of x:
        self._N_voxels_recon = np.array(
            self._reconstruction.sitk.GetSize()).prod()

    def set_stacks(self, stacks):
        self._stacks = stacks

        # Update helpers
        self._N_stacks = len(self._stacks)

        # Compute total amount of pixels for all slices
        self._N_total_slice_voxels = 0
        for i in range(0, self._N_stacks):
            N_stack_voxels = np.array(self._stacks[i].sitk.GetSize()).prod()
            self._N_total_slice_voxels += N_stack_voxels

    def set_reconstruction(self, reconstruction):
        self._reconstruction = reconstruction

        # Extract information ready to use for itk image conversion operations
        self._reconstruction_shape = sitk.GetArrayFromImage(
            self._reconstruction.sitk).shape

        # Compute total amount of voxels of x:
        self._N_voxels_recon = np.array(
            self._reconstruction.sitk.GetSize()).prod()

    #
    # Set regularization parameter for Tikhonov regularization
    # \date       2017-07-25 15:15:54+0100
    #
    # \param      self   The object
    # \param      alpha  regularization parameter, scalar
    #
    # \return     { description_of_the_return_value }
    #
    def set_alpha(self, alpha):
        self._alpha = alpha

    # Get value of chosen regularization parameter for Tikhonov regularization
    #  \return regularization parameter, scalar
    def get_alpha(self):
        return self._alpha

    ##
    # Sets the maximum number of iterations for Tikhonov solver.
    # \date       2016-08-01 16:35:09+0100
    #
    # \param      self      The object
    # \param      iter_max  number of maximum iterations, scalar
    #
    # \return     { description_of_the_return_value }
    #
    def set_iter_max(self, iter_max):
        self._iter_max = iter_max

    # Get chosen value of maximum number of iterations for minimizer for Tikhonov regularization
    #  \return maximum number of iterations set for minimizer, scalar
    def get_iter_max(self):
        return self._iter_max

    ##
    #       Sets the minimizer.
    # \date       2016-11-05 23:40:31+0000
    #
    # \param      self       The object
    # \param      minimizer  The minimizer
    #
    def set_minimizer(self, minimizer):
        self._minimizer = minimizer

    def get_minimizer(self):
        return self._minimizer

    ##
    # Sets the data loss rho in 1/2 ||rho(Ax-b)||^2
    # \date       2017-05-15 11:30:25+0100
    #
    # \param      self       The object
    # \param      data_loss  string
    #
    def set_data_loss(self, data_loss):
        if data_loss not in DATA_LOSS:
            raise ValueError("Loss function must be in " + str(DATA_LOSS))
        self._data_loss = data_loss

    def set_huber_gamma(self, huber_gamma):
        self._huber_gamma = huber_gamma

    def get_huber_gamma(self):
        return self._huber_gamma

    def set_verbose(self, verbose):
        self._verbose = verbose

    def get_verbose(self):
        return self._verbose

    def run(self):

        # Run solver specific reconstruction
        self._run()

    # Get current estimate of reconstruction
    #  \return current estimate of reconstruction, instance of Stack
    def get_reconstruction(self):
        return self._reconstruction

    # Get cut-off distance
    #  \return scalar value
    def get_alpha_cut(self):
        return self._alpha_cut

    # Get computational time for reconstruction
    #  \return computational time in seconds
    def get_computational_time(self):
        return self._computational_time

    ##
    # Get function call A = lambda x: A(x) with A: R^n -> R^m
    # \date       2017-07-25 16:02:47+0100
    #
    # \param      self  The object
    #
    # \return     Function call mapping from and to 1D numpy array.
    #
    def get_A(self):
        return lambda x: self._MA(x)

    ##
    # Gets function call A^* = lambda y: A^*(y) with A: R^m -> R^n
    # \date       2017-07-25 16:18:52+0100
    #
    # \param      self  The object
    #
    # \return     Function call mapping from and to 1D numpy array.
    #
    def get_A_adj(self):
        return lambda x: self._A_adj_M(x)

    ##
    # Gets the right hand-side vector b \in R^m
    # \date       2017-07-25 16:19:30+0100
    #
    # \param      self  The object
    #
    # \return     1D numpy array
    #
    def get_b(self):
        return self._get_M_y()

    ##
    # Gets the initial value given by the flattened reconstruction numpy data
    # array in R^n.
    # \date       2017-07-25 16:20:00+0100
    #
    # \param      self  The object
    #
    # \return     1D numpy array
    #
    def get_x0(self):
        return sitk.GetArrayFromImage(self._reconstruction.sitk).flatten()

    def get_x_scale(self):
        return self._x_scale

    ##
    #       Gets the setting specific filename indicating the information
    #             used for the reconstruction step
    # \date       2016-11-17 15:41:58+0000
    #
    # \param      self    The object
    # \param      prefix  The prefix as string
    #
    # \return     The setting specific filename as string.
    #
    @abstractmethod
    def get_setting_specific_filename(self, prefix=""):
        pass

    @abstractmethod
    def _run(self):
        pass

    @abstractmethod
    def get_solver(self):
        pass

    ##
    #       Gets the predefined covariance.
    # \date       2016-10-14 16:52:10+0100
    #
    # \param      self  The object
    #
    # \return     The predefined covariance as 3x3 numpy array
    #
    def get_predefined_covariance(self):
        return self._predefined_covariance

    ##
    # Evaluate
    # \f$ M \vec{y}
    # \f$
    # \f$ = \begin{pmatrix} M_1 \vec{y}_1 \\ M_2 \vec{y}_2 \\ \vdots \\ M_K
    # \vec{y}_K \end{pmatrix} \vec{x}\f$
    # \date       2017-11-01 00:17:28+0000
    #
    # \param      self  The object
    #
    # \return     My, i.e. all masked slices stacked to 1D array
    #
    def _get_M_y(self):

        # Allocate memory
        My = np.zeros(self._N_total_slice_voxels)

        # Define index for first voxel of first slice within array
        i_min = 0

        for i, stack in enumerate(self._stacks):
            slices = stack.get_slices()

            # Get number of voxels of each slice in current stack
            N_slice_voxels = np.array(slices[0].sitk.GetSize()).prod()

            for j, slice_j in enumerate(slices):

                # Define index for last voxel to specify current slice
                # (exlusive)
                i_max = i_min + N_slice_voxels

                # Apply M_k y_k
                if self._use_masks:
                    slice_itk = self._linear_operators.M_itk(
                        slice_j.itk, slice_j.itk_mask)
                else:
                    slice_itk = slice_j.itk
                slice_nda_vec = self._itk2np.GetArrayFromImage(
                    slice_itk).flatten()

                # Fill respective elements
                My[i_min:i_max] = slice_nda_vec

                # Define index for first voxel to specify subsequent slice
                # (inclusive)
                i_min = i_max

        return My

    ##
    # Operation M_k A_k x
    # \date       2017-07-25 15:15:53+0100
    #
    # \param      self       The object
    # \param      reconstruction_itk  reconstruction image as itk.Image object
    # \param      slice_k    Slice object which defines operator M_k and A_k
    #
    # \return     { description_of_the_return_value }
    #
    def _Mk_Ak(self, reconstruction_itk, slice_k):

        # Get slice spacing relevant for Gaussian blurring estimate
        in_plane_res = slice_k.get_inplane_resolution()
        slice_thickness = slice_k.get_slice_thickness()
        slice_spacing = np.array([in_plane_res, in_plane_res, slice_thickness])

        # Compute A_k x
        Ak_reconstruction_itk = self._linear_operators.A_itk(
            reconstruction_itk, slice_k.itk, slice_spacing)

        if not self._use_masks:
            return Ak_reconstruction_itk

        # Compute M_k A_k x
        Ak_reconstruction_itk = self._linear_operators.M_itk(
            Ak_reconstruction_itk, slice_k.itk_mask)

        return Ak_reconstruction_itk

    ##
    # Operation A_k^* M_k y_k
    # \date       2017-07-25 15:15:53+0100
    #
    # \param      self       The object
    # \param      slice_itk  LR image as itk.Image object
    # \param      slice_k    Slice object which defines operator A_k^*
    #
    # \return     image in reconstruction space as itk.Image object after
    #             performed backward operation
    #
    def _Ak_adj_Mk(self, slice_itk, slice_k):

        # Compute M_k y_k
        if self._use_masks:
            Mk_slice_itk = self._linear_operators.M_itk(
                slice_itk, slice_k.itk_mask)
        else:
            Mk_slice_itk = slice_itk

        # Get slice spacing relevant for Gaussian blurring estimate
        in_plane_res = slice_k.get_inplane_resolution()
        slice_thickness = slice_k.get_slice_thickness()
        slice_spacing = np.array([in_plane_res, in_plane_res, slice_thickness])

        # Compute A_k^* M_k y_k
        Mk_slice_itk = self._linear_operators.A_adj_itk(
            Mk_slice_itk, self._reconstruction.itk, slice_spacing)

        return Mk_slice_itk

    #
    # Evaluate
    # \f$ MA \vec{x}
    # \f$
    # \f$ = \b egin{pmatrix} M_1 A_1 \\ M_2 A_2 \\ \vdots \\ M_K A_K \em
    # nd{pmatrix} \vec{x}
    # \f$
    # \date       2017-07-25 15:15:53+0100
    #
    # \param      self           The object
    # \param      reconstruction_nda_vec  reconstruction data as 1D array
    #
    # \return     evaluated MAx as part of augmented linear operator as 1D
    #             array
    #
    def _MA(self, reconstruction_nda_vec):

        # Convert reconstruction data array back to itk.Image object
        x_itk = self._get_itk_image_from_array_vec(
            reconstruction_nda_vec, self._reconstruction.itk)

        # Allocate memory
        MA_x = np.zeros(self._N_total_slice_voxels)

        # Define index for first voxel of first slice within array
        i_min = 0

        for i, stack in enumerate(self._stacks):
            slices = stack.get_slices()

            # Get number of voxels of each slice in current stack
            N_slice_voxels = np.array(slices[0].sitk.GetSize()).prod()

            for j, slice_j in enumerate(slices):

                # Define index for last voxel to specify current slice
                # (exclusive)
                i_max = i_min + N_slice_voxels

                # Compute M_k A_k y_k
                slice_itk = self._Mk_Ak(x_itk, slice_j)
                slice_nda = self._itk2np.GetArrayFromImage(slice_itk)

                # Fill corresponding elements
                MA_x[i_min:i_max] = slice_nda.flatten()

                # Define index for first voxel to specify subsequent slice
                # (inclusive)
                i_min = i_max

        return MA_x

    ##
    # Evaluate
    # \f$ A^* M \vec{y} = \begin{bmatrix} A_1^* M_1 && A_2^* M_2 && \c dots &&
    # A_K^* M_K \end{bmatrix} \vec{y}
    # \f$
    # \date       2017-07-18 22:21:53+0100
    #
    # \param      self                    The object
    # \param      stacked_slices_nda_vec  stacked slice data as 1D array
    #
    # \return     evaluated A'My as part of augmented adjoint linear operator
    #             as 1D array
    #
    def _A_adj_M(self, stacked_slices_nda_vec):

        # Allocate memory
        A_adj_M_y = np.zeros(self._N_voxels_recon)

        # Define index for first voxel of first slice within array
        i_min = 0

        for i, stack in enumerate(self._stacks):
            slices = stack.get_slices()

            # Get number of voxels of each slice in current stack
            N_slice_voxels = np.array(slices[0].sitk.GetSize()).prod()

            for j, slice_j in enumerate(slices):

                # Define index for last voxel to specify current slice
                # (exlusive)
                i_max = i_min + N_slice_voxels

                # Extract 1D corresponding to current slice and convert it to
                # itk.Object
                slice_itk = self._get_itk_image_from_array_vec(
                    stacked_slices_nda_vec[i_min:i_max], slice_j.itk)

                # Apply A_k' M_k on current slice
                Ak_adj_Mk_slice_itk = self._Ak_adj_Mk(slice_itk, slice_j)
                Ak_adj_Mk_slice_nda_vec = self._itk2np.GetArrayFromImage(
                    Ak_adj_Mk_slice_itk).flatten()

                # Add contribution
                A_adj_M_y += Ak_adj_Mk_slice_nda_vec

                # Define index for first voxel to specify subsequent slice
                # (inclusive)
                i_min = i_max

        return A_adj_M_y

    #
    # Convert numpy data array (vector format) back to itk.Image object
    # \date       2017-07-25 15:15:53+0100
    #
    # \param      self           The object
    # \param      nda_vec        reconstruction data as 1D array
    # \param      image_itk_ref  The image itk reference
    #
    # \return     reconstruction with intensities according to reconstruction_nda_vec as
    #             itk.Image object
    #
    def _get_itk_image_from_array_vec(self, nda_vec, image_itk_ref):

        shape_nda = np.array(
            image_itk_ref.GetLargestPossibleRegion().GetSize())[::-1]

        image_itk = self._itk2np.GetImageFromArray(nda_vec.reshape(shape_nda))
        image_itk.SetOrigin(image_itk_ref.GetOrigin())
        image_itk.SetSpacing(image_itk_ref.GetSpacing())
        image_itk.SetDirection(image_itk_ref.GetDirection())

        return image_itk
