##
# \file registration.py
# \brief      { item_description }
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Aug 2016
#


# Import libraries
import sys
import itk
import SimpleITK as sitk
import numpy as np
import scipy.optimize
import time
from datetime import timedelta

import pysitk.simple_itk_helper as sitkh
import pysitk.python_helper as ph
from nsol.loss_functions import LossFunctions as lf

# Import modules
import niftymic.base.psf as psf
import niftymic.base.slice as sl
import niftymic.base.stack as st
from niftymic.registration.registration_method \
    import AffineRegistrationMethod


DIMENSION = 3

# Pixel type of used 3D ITK image
PIXEL_TYPE = itk.D

# ITK image type
IMAGE_TYPE = itk.Image[PIXEL_TYPE, DIMENSION]
IMAGE_TYPE_CV33 = itk.Image.CVD33
IMAGE_TYPE_CV183 = itk.Image.CVD183

# Allowed data loss functions
DATA_LOSS = ['linear', 'soft_l1', 'huber', 'cauchy', 'arctan']

##
#       Class for registration in 3D based on least-squares optimization
# \date       2016-09-21 12:27:13+0100
#


class Registration(AffineRegistrationMethod):

    ##
    # { constructor_description }
    # \date       2016-08-02 15:49:34+0100
    #
    # \param[in]  self            The object
    # \param[in]  fixed           Fixed image as Stack or Slice object (S2VReg:
    #                             Slice)
    # \param[in]  moving          Moving image as Stack or Slice object
    #                             (S2VReg: HR Volume)
    # \param      use_fixed_mask  The use fixed mask
    # \param[in]  alpha_cut       The alpha cut
    # \param[in]  use_verbose     The verbose
    #
    def __init__(self,
                 fixed=None,
                 moving=None,
                 use_fixed_mask=False,
                 use_moving_mask=False,
                 registration_type="Rigid",
                 use_verbose=False,
                 alpha_cut=3,
                 data_loss="linear",
                 minimizer="least_squares",
                 initializer_type=None,
                 x_scale=1.,
                 ):

        AffineRegistrationMethod.__init__(self,
                                          fixed=fixed,
                                          moving=moving,
                                          use_fixed_mask=use_fixed_mask,
                                          use_moving_mask=use_moving_mask,
                                          use_verbose=use_verbose,
                                          registration_type=registration_type,
                                          )

        self._REGISTRATION_TYPES = ["Rigid"]

        self._minimizer = minimizer
        self._data_loss = data_loss
        self._alpha_cut = alpha_cut
        self._x_scale = x_scale
        self._initializer_type = initializer_type

        # Used for PSF modelling
        self._psf = psf.PSF()

        # Create PyBuffer object for conversion between NumPy arrays and ITK
        # images
        self._itk2np = itk.PyBuffer[IMAGE_TYPE]
        self._itk2np_CVD33 = itk.PyBuffer[IMAGE_TYPE_CV33]
        self._itk2np_CVD183 = itk.PyBuffer[IMAGE_TYPE_CV183]

        # Create transform instances
        self._rigid_transform_itk = itk.Euler3DTransform[PIXEL_TYPE].New()
        self._parameters_itk = self._rigid_transform_itk.GetParameters()
        self._fixed_parameters_itk = \
            self._rigid_transform_itk.GetFixedParameters()
        self._dof_transform = self._rigid_transform_itk.GetNumberOfParameters()
        self._parameters = None
        self._fixed_parameters = None

    ##
    # Initialize registration class
    # \date       2016-09-21 12:58:57+0100
    #
    # Idea: Have same structure as other registration classes and add moving
    # and fixed images later on via set routines. Several filters need to be
    # initialized according to the updates
    #
    # \param      self  The object
    #
    def _initialize_class(self):

        # Properties of fixed
        self._fixed_affine_transform_sitk = sitkh.get_sitk_affine_transform_from_sitk_image(
            self._fixed.sitk)
        self._fixed_spacing = self._fixed.sitk.GetSpacing()
        self._fixed_size = self._fixed.sitk.GetSize()
        self._fixed_voxels = np.array(self._fixed_size).prod()

        # Allocate and initialize Oriented Gaussian Interpolate Image Filter
        self._filter_oriented_Gaussian = itk.OrientedGaussianInterpolateImageFilter[
            IMAGE_TYPE, IMAGE_TYPE].New()
        self._filter_oriented_Gaussian.SetDefaultPixelValue(0.0)
        self._filter_oriented_Gaussian.SetAlpha(self._alpha_cut)
        self._filter_oriented_Gaussian.SetInput(self._moving.itk)
        self._filter_oriented_Gaussian.SetUseJacobian(False)
        self._filter_oriented_Gaussian.SetUseImageDirection(True)

        # ## Allocate and initialize Adjoint Oriented Gaussian Interpolate Image Filter
        # self._filter_adjoint_oriented_Gaussian = itk.AdjointOrientedGaussianInterpolateImageFilter[IMAGE_TYPE, IMAGE_TYPE].New()
        # self._filter_adjoint_oriented_Gaussian.SetDefaultPixelValue(0.0)
        # self._filter_adjoint_oriented_Gaussian.SetAlpha(self._alpha_cut)
        # self._filter_adjoint_oriented_Gaussian.SetOutputParametersFromImage(self._moving.itk)

        # Allocate and initialize Gradient Euler3DTransform Image Filter
        self._filter_gradient_transform = itk.GradientEuler3DTransformImageFilter[
            IMAGE_TYPE, PIXEL_TYPE, PIXEL_TYPE].New()
        self._filter_gradient_transform.SetInput(self._fixed.itk)
        self._filter_gradient_transform.SetTransform(self._rigid_transform_itk)

        if self._use_fixed_mask:
            # self._nda_mask =
            # sitk.GetArrayFromImage(self._fixed.sitk_mask).reshape(-1,1)
            self._nda_mask = sitk.GetArrayFromImage(
                self._fixed.sitk_mask).flatten()

        initial_transform = sitk.Euler3DTransform()
        if self._initializer_type is not None:
            if self._initializer_type in ["MOMENTS", "GEOMETRY"]:
                moving_sitk = self._moving.sitk
                initializer_type = self._initializer_type
            elif self._initializer_type == "SelfMOMENTS":
                moving_sitk = self._fixed.sitk
                initializer_type = "MOMENTS"
            elif self._initializer_type == "SelfGEOMETRY":
                moving_sitk = self._fixed.sitk
                initializer_type = "GEOMETRY"
            else:
                raise ValueError("Initializer type '%s' unknown"
                                 % (self._initializer_type))
            initial_transform = sitk.CenteredTransformInitializer(
                self._fixed.sitk,
                moving_sitk,
                initial_transform,
                eval("sitk.CenteredTransformInitializerFilter.%s" % (
                    initializer_type))
            )
        self._parameters = initial_transform.GetParameters()
        self._fixed_parameters = initial_transform.GetFixedParameters()

        for i in range(0, self._dof_transform):
            self._parameters_itk.SetElement(i, self._parameters[i])
        for i in range(0, len(self._fixed_parameters)):
            self._fixed_parameters_itk.SetElement(i, self._fixed_parameters[i])
        self._rigid_transform_itk.SetParameters(
            self._parameters_itk)
        self._rigid_transform_itk.SetFixedParameters(
            self._fixed_parameters_itk)

    #
    # Set cut-off distance
    # \date       2017-07-25 16:38:48+0100
    #
    # \param      self       The object
    # \param[in]  alpha_cut  scalar value
    #
    # \return     { description_of_the_return_value }
    #
    def set_alpha_cut(self, alpha_cut):
        self._alpha_cut = alpha_cut

    # Get cut-off distance
    #  \return scalar value
    def get_alpha_cut(self):
        return self._alpha_cut

    ##
    #       Gets the parameters estimated by registration algorithm.
    # \date       2016-08-03 00:10:45+0100
    #
    # \param[in]  self  The object
    #
    # \return     The parameters.
    #
    def get_parameters(self):
        return self._parameters

    ##
    #       Run registration
    # \date       2016-08-03 00:11:51+0100
    # \post       self._paramters is updated
    #
    # \param[in]  self     The object
    #
    def _run_registration(self):

        if self._data_loss not in DATA_LOSS:
            raise ValueError("data_loss must be in " + str(DATA_LOSS))

        if self._use_verbose:
            ph.print_subtitle("Registration")
            ph.print_info("Minimizer: %s" % (self._minimizer))
            ph.print_info("Data loss: %s" % (self._data_loss))

        ##
        self._initialize_class()

        nda_fixed = sitk.GetArrayFromImage(self._fixed.sitk)

        # Define variables for least-squares optimization
        residual = lambda x: self._get_residual_data_fit(
            parameters=x,
            fixed_parameters=self._fixed_parameters,
            nda_fixed=nda_fixed)
        jacobian_residual = lambda x: self._get_jacobian_residual_data_fit(
            parameters=x,
            fixed_parameters=self._fixed_parameters)
        x0 = np.array(self._parameters)

        if self._minimizer == "least_squares":

            if self._data_loss == "linear":
                method = "lm"
            else:
                method = "trf"
            if self._use_verbose:
                ph.print_info("least_squares method: %s" % (method))

            # Non-linear least-squares method:
            self._parameters = scipy.optimize.least_squares(
                fun=residual,
                jac=jacobian_residual,
                x0=x0,
                x_scale=self._x_scale,
                # method=method,
                loss=self._data_loss,
                verbose=2*self._use_verbose,
            ).x

        else:
            cost = lambda x: lf.get_ell2_cost_from_residual(
                f=residual(x),
                loss=self._data_loss)
            grad_cost = lambda x: lf.get_gradient_ell2_cost_from_residual(
                f=residual(x),
                jac_f=jacobian_residual(x),
                loss=self._data_loss)

            self._parameters = scipy.optimize.minimize(
                method=self._minimizer,
                fun=cost,
                jac=grad_cost,
                x0=x0,
                options={
                    # 'maxiter': self._iter_max,
                    'disp': self._use_verbose},
            ).x

        # Build new transform
        transform_sitk = sitk.Euler3DTransform()
        transform_sitk.SetParameters(self._parameters)
        transform_sitk.SetFixedParameters(self._fixed_parameters)

        # Get transform where center (fixed_params) is set to zero
        self._registration_transform_sitk = \
            sitkh.get_composite_sitk_affine_transform(
                transform_sitk, sitk.Euler3DTransform())

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

    ##
    #       Compute residual y_k - A_k(theta)x based on parameters
    #             theta.
    # \date       2016-08-03 00:12:43+0100
    #
    # \param[in]  self        The object
    # \param[in]  parameters  The parameters
    #
    # \return     The residual data fit as Nk-array, i.e. the number of voxels
    #             of slice y_k.
    #
    def _get_residual_data_fit(self, parameters, fixed_parameters, nda_fixed):

        Ak_vol_itk = self._execute_oriented_gaussian_interpolate_image_filter(
            parameters=parameters,
            fixed_parameters=fixed_parameters,
            use_jacobian=False)

        nda_Ak_vol = self._itk2np.GetArrayFromImage(Ak_vol_itk)
        residual = (nda_fixed - nda_Ak_vol).flatten()

        if self._use_fixed_mask:
            residual = residual*self._nda_mask

        return residual

    ##
    #       Gets the jacobian residual data fit.
    # \date       2016-09-08 12:11:13+0100
    #
    # \param[in]  self                         The object
    # \param[in]  parameters                   The parameters
    #
    # \return     The Jacobian residual data fit as (Nk x 6)-array
    #
    def _get_jacobian_residual_data_fit(self, parameters, fixed_parameters):

        # Allocate array for Jacobian of residual
        jacobian_nda = np.zeros((self._fixed_voxels, self._dof_transform))

        # Get Jacobian of filtered image w.r.t. spatial coordinates
        jacobian_spatial_Ak_vol_itk = self._execute_oriented_gaussian_interpolate_image_filter(
            parameters=parameters,
            fixed_parameters=fixed_parameters,
            use_jacobian=True)

        # Get array of Jacobian of forward operator w.r.t. spatial coordinates
        nda_gradient_filter = self._itk2np_CVD33.GetArrayFromImage(
            jacobian_spatial_Ak_vol_itk)

        # Reshape to (self._fixed_voxels x DIMENSION)-array
        nda_gradient_filter_vec = nda_gradient_filter.reshape(-1, DIMENSION)

        # Get array of Jacobian of transform w.r.t. parameters
        for i in range(0, self._dof_transform):
            self._parameters_itk.SetElement(i, parameters[i])
        self._rigid_transform_itk.SetParameters(self._parameters_itk)

        # for i in range(0, len(fixed_parameters)):
        #     self._fixed_parameters_itk.SetElement(i, fixed_parameters[i])
        # self._rigid_transform_itk.SetFixedParameters(
        # self._fixed_parameters_itk)
        # for i in range(self._fixed_parameters_itk.GetNumberOfElements()):
        # print self._rigid_transform_itk.GetFixedParameters().GetElement(i)

        self._filter_gradient_transform.Update()

        # Get Jacobian of transform w.r.t. parameters at locations specified by
        # fixed image
        gradient_transform_itk = self._filter_gradient_transform.GetOutput()
        gradient_transform_itk.DisconnectPipeline()

        # Get data array and reshape to self._fixed_voxels x DIMENSION x DOF
        nda_gradient_transform = self._itk2np_CVD183.GetArrayFromImage(
            gradient_transform_itk).reshape(self._fixed_voxels, DIMENSION,
                                            self._dof_transform)

        # Vectorized: Compute Jacobian of residual w.r.t. to parameters
        # (see test_vectorization_of_dImage_times_dT)
        # Each multiplication is (1xDIMENSION) * (DIMENSIONxDOF) = 1xDOF
        jacobian_nda = np.sum(nda_gradient_filter_vec[:, :, np.newaxis] *
                              nda_gradient_transform,
                              axis=1)

        if self._use_fixed_mask:
            # Multiply each DOF column of Jacobian with mask pointwise.
            # Dimensions for multiplication are (fixed_voxels x DOF) *
            # (fixed_voxels,1)
            # https://mail.scipy.org/pipermail/numpy-discussion/2007-March/026506.html
            jacobian_nda = jacobian_nda*self._nda_mask[:, np.newaxis]

        return -jacobian_nda

    ##
    # Execute Oriented Gaussian Interpolate Image Filter
    # \date       2016-09-20 18:32:12+0100
    #
    # \param[in]  self          The object
    # \param[in]  parameters    The parameters
    # \param[in]  use_jacobian  Boolean to indicate whether Jacobian or the
    #                           filtered image itself is returned
    #
    # \return     Either return filtered image or Jacobian w.r.t to spatial
    #             coordinates of filtered image
    #
    def _execute_oriented_gaussian_interpolate_image_filter(self,
                                                            parameters,
                                                            fixed_parameters,
                                                            use_jacobian):

        # Create registration transform based on parameters
        transform_sitk = sitk.Euler3DTransform()
        transform_sitk.SetParameters(parameters)
        transform_sitk.SetFixedParameters(fixed_parameters)

        # Get composite affine transform: reg_trafo \circ fixed_space
        composite_transform_sitk = sitkh.get_composite_sitk_affine_transform(
            transform_sitk, self._fixed_affine_transform_sitk)

        # Extract direction and origin of transformed fixed space
        direction_sitk = sitkh.get_sitk_image_direction_from_sitk_affine_transform(
            composite_transform_sitk, self._fixed_spacing)
        origin_sitk = sitkh.get_sitk_image_origin_from_sitk_affine_transform(
            composite_transform_sitk)

        # Set output to transformed fixed space
        self._filter_oriented_Gaussian.SetOutputOrigin(origin_sitk)
        self._filter_oriented_Gaussian.SetOutputSpacing(self._fixed_spacing)
        self._filter_oriented_Gaussian.SetOutputDirection(
            sitkh.get_itk_from_sitk_direction(direction_sitk))
        self._filter_oriented_Gaussian.SetSize(self._fixed_size)
        self._filter_oriented_Gaussian.SetUseJacobian(use_jacobian)
        self._filter_oriented_Gaussian.UpdateLargestPossibleRegion()

        # Set oriented PSF based on transformed fixed space
        # transformed_fixed = st.Stack.from_sitk_image(
        #     sitkh.get_transformed_sitk_image(self._fixed.sitk, transform_sitk))
        # Cov_HR_coord = self._psf.get_covariance_matrix_in_reconstruction_space(transformed_fixed, self._moving)
        # print Cov_HR_coord
        # Cov_HR_coord = self._psf.get_gaussian_PSF_covariance_matrix_reconstruction_coordinates_from_direction_and_spacing(
        #     direction_sitk, self._fixed_spacing, self._moving)
        Cov_HR_coord = np.eye(3)
        self._filter_oriented_Gaussian.SetCovariance(Cov_HR_coord.flatten())

        # Compute simulated fixed from volume and its Jacobian w.r.t. to
        # spatial coordinates
        self._filter_oriented_Gaussian.Update()

        # 1) Compute A_k(theta)x as itk.Image
        if not use_jacobian:
            Ak_vol_itk = self._filter_oriented_Gaussian.GetOutput()
            Ak_vol_itk.DisconnectPipeline()

            return Ak_vol_itk

        # 2) Compute Jacobian w.r.t. to spatial coordinates
        else:
            jacobian_spatial_Ak_vol_itk = self._filter_oriented_Gaussian.GetJacobian()
            jacobian_spatial_Ak_vol_itk.DisconnectPipeline()

            return jacobian_spatial_Ak_vol_itk
