##
# \file intra_stack_registration.py
# \brief      Intra-stack registration steps where slices are only transformed
#             2D in-plane.
#
# Class has been mainly developed for the CIS30FU project.
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Nov 2016
#


# Import libraries
import SimpleITK as sitk
import itk
import numpy as np

import niftymic.base.slice as sl
import niftymic.base.stack as st
import niftymic.utilities.intensity_correction as ic
# Import modules
import pysitk.simple_itk_helper as sitkh
from niftymic.registration.stack_registration_base import StackRegistrationBase


class IntraStackRegistration(StackRegistrationBase):

    ##
    # { constructor_description }
    # \date       2017-07-14 14:24:00+0100
    #
    # \param      self                                           The object
    # \param      stack                                          The stack to
    #                                                            be aligned as
    #                                                            Stack object
    # \param      reference                                      The reference
    #                                                            used for
    #                                                            alignment as
    #                                                            Stack object
    # \param      use_stack_mask                                 Use stack mask
    #                                                            for
    #                                                            registration,
    #                                                            bool
    # \param      use_reference_mask                             Use reference
    #                                                            mask for
    #                                                            registration,
    #                                                            bool
    # \param      use_verbose                                    Verbose
    #                                                            output, bool
    # \param      transform_initializer_type                     The transform
    #                                                            initializer
    #                                                            type, e.g.
    #                                                            "identity",
    #                                                            "moments" or
    #                                                            "geometry"
    # \param      interpolator                                   The interpolator
    # \param      alpha_neighbour                                Weight >= 0
    #                                                            for neighbour
    #                                                            term
    # \param      alpha_reference                                Weight >= 0
    #                                                            for reference
    #                                                            term
    # \param      alpha_parameter                                Weight >= 0
    #                                                            for prior term
    # \param      transform_type                                 The transform
    #                                                            type, "rigid",
    #                                                            "similarity",
    #                                                            "affine"
    # \param      optimizer                                      Either
    #                                                            "least_squares"
    #                                                            to use
    #                                                            scipy.optimize.least_squares
    #                                                            or any method
    #                                                            used in
    #                                                            "scipy.optimize.minimize",
    #                                                            e.g.
    #                                                            "L-BFGS-B".
    # \param      optimizer_iter_max                             Maximum number
    #                                                            of
    #                                                            iterations/function
    #                                                            evaluations
    # \param      optimizer_loss                                 Loss function,
    #                                                            e.g. "linear",
    #                                                            "soft_l1" or
    #                                                            "huber".
    # \param      optimizer_method                               The optimizer
    #                                                            method used
    #                                                            for
    #                                                            "least_squares"
    #                                                            algorithm.
    #                                                            E.g. "trf"
    # \param      use_parameter_normalization                    Use parameter
    #                                                            normalization
    #                                                            for optimizer,
    #                                                            bool
    # \param      intensity_correction_initializer_type          The intensity
    #                                                            correction
    #                                                            initializer
    #                                                            type; None,
    #                                                            "linear" or
    #                                                            "affine"
    # \param      intensity_correction_type_slice_neighbour_fit  The intensity
    #                                                            correction
    #                                                            type used for
    #                                                            slice
    #                                                            neighbour
    #                                                            term, None,
    #                                                            "linear" or
    #                                                            "affine"
    # \param      prior_intensity_correction_coefficients        Prior used for
    #                                                            intensity
    #                                                            correction
    #                                                            coefficients
    # \param      prior_scale                                    Prior used for
    #                                                            scaling; only
    #                                                            valid for
    #                                                            "similarity"
    # \param      image_transform_reference_fit_term             The image
    #                                                            transform
    #                                                            reference fit
    #                                                            term; Either
    #                                                            "identity",
    #                                                            "gradient_magnitude",
    #                                                            "partial_derivative"
    #
    def __init__(self,
                 stack=None,
                 reference=None,
                 use_stack_mask=False,
                 use_reference_mask=False,
                 use_verbose=False,
                 transform_initializer_type="identity",
                 interpolator="Linear",
                 alpha_neighbour=1,
                 alpha_reference=1,
                 alpha_parameter=0,
                 transform_type="rigid",
                 optimizer="least_squares",
                 optimizer_iter_max=20,
                 optimizer_loss="soft_l1",
                 optimizer_method="trf",
                 use_parameter_normalization=False,
                 intensity_correction_initializer_type=None,
                 intensity_correction_type_slice_neighbour_fit=None,
                 prior_intensity_correction_coefficients=np.array([1, 0]),
                 prior_scale=1.0,
                 image_transform_reference_fit_term="identity",
                 ):

        # Run constructor of superclass
        StackRegistrationBase.__init__(
            self,
            stack=stack,
            reference=reference,
            use_stack_mask=use_stack_mask,
            use_reference_mask=use_reference_mask,
            use_verbose=use_verbose,
            transform_initializer_type=transform_initializer_type,
            use_parameter_normalization=use_parameter_normalization,
            optimizer=optimizer,
            optimizer_iter_max=optimizer_iter_max,
            optimizer_loss=optimizer_loss,
            optimizer_method=optimizer_method,
            interpolator=interpolator,
            alpha_neighbour=alpha_neighbour,
            alpha_reference=alpha_reference,
            alpha_parameter=alpha_parameter,
        )

        # Chosen transform type
        self._transform_type = transform_type

        # Dictionaries to create new transform depending on the chosen
        # transform type
        self._new_transform_sitk = {
            "rigid": self._new_rigid_transform_sitk,
            "similarity": self._new_similarity_transform_sitk,
            "affine": self._new_affine_transform_sitk
        }
        self._new_transform_itk = {
            "rigid": self._new_rigid_transform_itk,
            "similarity": self._new_similarity_transform_itk,
            "affine": self._new_affine_transform_itk
        }

        # Chosen intensity correction type
        self._intensity_correction_type_slice_neighbour_fit = \
            intensity_correction_type_slice_neighbour_fit
        self._intensity_correction_type_reference_fit = \
            intensity_correction_type_slice_neighbour_fit

        # Define image type for reference cost
        self._image_transform_reference_fit_term = image_transform_reference_fit_term

        # Dictionary to apply requested intensity correction
        self._apply_intensity_correction = {
            None:  self._apply_intensity_correction_None,
            "linear":  self._apply_intensity_correction_linear,
            "affine":  self._apply_intensity_correction_affine,
        }

        self._add_gradient_with_respect_to_intensity_correction_parameters = {
            None: self._add_gradient_with_respect_to_intensity_correction_parameters_None,
            "linear": self._add_gradient_with_respect_to_intensity_correction_parameters_linear,
            "affine": self._add_gradient_with_respect_to_intensity_correction_parameters_affine,
        }

        # Specifies how the initial values for the intensity correction shall
        # be computed
        self._intensity_correction_initializer_type = intensity_correction_initializer_type

        # Dictionary to get initial values for intensity correction
        self._get_initial_intensity_correction_parameters = {
            None: self._get_initial_intensity_correction_parameters_None,
            "linear": self._get_initial_intensity_correction_parameters_linear,
            "affine": self._get_initial_intensity_correction_parameters_affine
        }

        # Scale prior
        self._prior_scale = prior_scale

        # Intensity correction coefficient priors
        self._prior_intensity_correction_coefficients = prior_intensity_correction_coefficients

        ##
        self._get_residual_intensity_coefficients = {
            None: self._get_residual_intensity_coefficients_None,
            "linear": self._get_residual_intensity_coefficients_linear,
            "affine": self._get_residual_intensity_coefficients_affine
        }
        self._get_jacobian_residual_intensity_coefficients = {
            None: self._get_jacobian_residual_intensity_coefficients_None,
            "linear": self._get_jacobian_residual_intensity_coefficients_linear,
            "affine": self._get_jacobian_residual_intensity_coefficients_affine
        }

        # Dictionary, to update the the slices according to the obtained
        # registration
        self._apply_motion_correction_and_compute_slice_transforms = {
            "rigid":  self._apply_rigid_motion_correction_and_compute_slice_transforms,
            "similarity":  self._apply_similarity_motion_correction_and_compute_slice_transforms,
            "affine":  self._apply_affine_motion_correction_and_compute_slice_transforms
        }

        # Gradient Magnitude Filter
        self._gradient_magnitude_filter_sitk = sitk.GradientMagnitudeImageFilter()
        self._gradient_magnitude_filter_sitk.SetUseImageSpacing(True)

        # Gradient Image Filter
        self._gradient_image_filter_sitk = sitk.GradientImageFilter()
        self._gradient_image_filter_sitk.SetUseImageDirection(True)
        self._gradient_image_filter_sitk.SetUseImageSpacing(True)

        self._apply_image_transform = {
            "identity":   self._apply_image_transform_identity,
            "dx":   self._apply_image_transform_dx,
            "dy":   self._apply_image_transform_dy,
            "gradient_magnitude":   self._apply_image_transform_gradient_magnitude
        }

        # Costs
        self._final_cost = 0
        self._residual_paramters_ell2 = 0
        self._residual_reference_fit_ell2 = 0
        self._residual_slice_neighbours_ell2 = 0

        self._use_stack_mask_reference_fit_term = self._use_stack_mask
        self._use_stack_mask_neighbour_fit_term = self._use_stack_mask

    ##
    # Sets the transform type.
    # \date       2016-11-10 01:53:58+0000
    #
    # \param      self            The object
    # \param      transform_type  The transform type
    #
    def set_transform_type(self, transform_type):
        if transform_type not in self._new_transform_sitk.keys():
            raise ValueError("Transform type " + transform_type +
                             " not possible.\nAllowed values: " +
                             str(self._new_transform_sitk.keys()))
        self._transform_type = transform_type

    def get_transform_type(self):
        return self._transform_type

    ##
    # Set the intensity correction type
    # \date       2016-11-10 01:58:39+0000
    #
    # \param      self  The object
    # \param      flag  The flag
    #
    def set_intensity_correction_type_slice_neighbour_fit(self, intensity_correction_type_slice_neighbour_fit):
        if intensity_correction_type_slice_neighbour_fit \
                not in self._apply_intensity_correction.keys():
            raise ValueError("Intensity correction type " +
                             intensity_correction_type_slice_neighbour_fit +
                             " not possible.\nAllowed values: " + str(self._apply_intensity_correction.keys()))
        self._intensity_correction_type_slice_neighbour_fit = \
            intensity_correction_type_slice_neighbour_fit

    def get_intensity_correction_type_slice_neighbour_fit(self):
        return self._intensity_correction_type_slice_neighbour_fit

    ##
    # Set the intensity correction type for neighbour fit
    # \date       2016-11-10 01:58:39+0000
    #
    # \param      self  The object
    # \param      flag  The flag
    #
    def set_intensity_correction_type_reference_fit(self, intensity_correction_type_slice_reference_fit):
        if intensity_correction_type_slice_reference_fit \
                not in self._apply_intensity_correction.keys():
            raise ValueError("Intensity correction type " +
                             intensity_correction_type_slice_reference_fit +
                             " not possible.\nAllowed values: " +
                             str(self._apply_intensity_correction.keys()))
        self._intensity_correction_type_reference_fit = \
            intensity_correction_type_slice_reference_fit

    def get_intensity_correction_type_reference_fit(self):
        return self._intensity_correction_type_reference_fit

    ##
    # Sets the intensity correction initializer type. It specifies how the
    # initial values for the intensity correction shall be computed.
    # \date       2016-11-21 19:36:26+0000
    #
    # \param      self                                   The object
    # \param      intensity_correction_initializer_type  The intensity
    #                                                    correction initializer
    #                                                    type
    #
    def set_intensity_correction_initializer_type(self, intensity_correction_initializer_type):
        if intensity_correction_initializer_type \
                not in self._apply_intensity_correction.keys():
            raise ValueError("Intensity correction initializer type " +
                             intensity_correction_initializer_type +
                             " not possible.\nAllowed values: " +
                             str(self._apply_intensity_correction.keys()))
        self._intensity_correction_initializer_type = \
            intensity_correction_initializer_type

    def get_intensity_correction_initializer_type(self):
        return self._intensity_correction_initializer_type

    ##
    # Sets the estimated scale.
    # \date       2016-11-21 15:45:14+0000
    #
    # \param      self             The object
    # \param      prior_scale  The estimated scale
    #
    def set_prior_scale(self, prior_scale):
        self._prior_scale = prior_scale
        # self._parameters_prior_transform["similarity"][0] = prior_scale

    def set_prior_intensity_coefficients(self, coefficients):
        coefficients = np.array(coefficients)

        if coefficients.size is 1:
            self._prior_intensity_correction_coefficients[0] = coefficients
        elif coefficients.size is 2:
            self._prior_intensity_correction_coefficients = coefficients
        else:
            raise ValueError("Coefficients must be of length 1 or 2")

    ##
    # Set image type used to compute the reference cost
    # \date       2016-11-30 14:14:36+0000
    #
    # \param      self                       The object
    # \param      image_transform_reference_fit_term  The image type reference cost
    #
    def set_image_transform_reference_fit_term(self, image_transform_reference_fit_term):
        if image_transform_reference_fit_term \
                not in ["identity", "gradient_magnitude", "partial_derivative"]:
            raise ValueError("Registration image type" +
                             image_transform_reference_fit_term +
                             " for the reference residuals is not possible.")
        self._image_transform_reference_fit_term = \
            image_transform_reference_fit_term

    def use_stack_mask_reference_fit_term(self, flag):
        self._use_stack_mask_reference_fit_term = flag

    def use_stack_mask_neighbour_fit_term(self, flag):
        self._use_stack_mask_neighbour_fit_term = flag

    def get_final_cost(self):
        if self._final_cost is None:
            self._compute_statistics_residuals_ell2()
        return self._final_cost

    def print_statistics(self):

        # Compute ell2-norm of residuals
        self._compute_statistics_residuals_ell2()

        StackRegistrationBase.print_statistics(self)

        if self._alpha_reference > self._ZERO:
            print("\tell^2-residual sum_k ||slice_k(T(theta_k)) - ref||_2^2 = %.3e" %
                  (self._residual_reference_fit_ell2))

        if self._alpha_neighbour > self._ZERO:
            print("\tell^2-residual sum_k ||slice_k(T(theta_k)) - slice_{k+1}(T(theta_{k+1}))||_2^2 = %.3e" % (
                self._residual_slice_neighbours_ell2))

        if self._alpha_parameter > self._ZERO:
            print("\tell^2-residual sum_k ||theta_k - theta_k0||_2^2 = %.3e" %
                  (self._residual_paramters_ell2))

        print("\tFinal cost: %.3e" % (self._final_cost))

    def get_setting_specific_filename(self, prefix="_"):

        dictionary_method = {
            "trf": "TRF",
            "dogbox": "DogBox",
            "lm": "LM"
        }
        dictionary_loss = {
            "linear": "Linear",
            "soft_l1": "Softl1",
            "huber": "Huber"
        }

        # Build filename
        filename = prefix
        filename += self._transform_type.capitalize()
        filename += "_IC" + \
            str(self._intensity_correction_type_slice_neighbour_fit)
        filename += "_Opt"
        filename += dictionary_method[self._optimizer_method]
        filename += dictionary_loss[self._optimizer_loss]
        filename += "_maskStack" + str(int(self._use_stack_mask))
        if self._reference is not None:
            filename += "_maskRef" + str(int(self._use_reference_mask))
        filename += "_Nfevmax" + str(self._optimizer_iter_max)
        filename += "_alphaR" + "%.g" % (self._alpha_reference)
        filename += "_alphaN" + "%.g" % (self._alpha_neighbour)
        filename += "_alphaP" + "%.g" % (self._alpha_parameter)

        # Replace dots by 'p'
        filename = filename.replace(".", "p")

        return filename

    def _print_info_text_least_squares(self):
        print("Minimization via least_squares solver (scipy.optimize.least_squares)")
        print("\tMethod: " + self._optimizer_method)
        print("\tLoss: " + self._optimizer_loss)
        print("\tMaximum number of function evaluations: " +
              str(self._optimizer_iter_max))
        self._print_into_text_common()

    def _print_info_text_minimize(self):
        print("Minimization via %s solver (scipy.optimize.minimize)" % (self._optimizer))
        print("\tLoss: " + self._optimizer_loss)
        print("\tMaximum number of iterations: " +
              str(self._optimizer_iter_max))
        self._print_into_text_common()

    def _print_into_text_common(self):
        print("\tTransform type: " + self._transform_type +
              " (Initialization: " + str(self._transform_initializer_type) + ")")
        if self._alpha_neighbour > self._ZERO:
            print("\tSlice neighbour fit term:")
            print("\t\tIntensity correction type: " +
                  str(self._intensity_correction_type_slice_neighbour_fit) +
                  " (Initialization: " +
                  str(self._intensity_correction_initializer_type) + ")")
            print("\t\tStack mask used: " +
                  str(self._use_stack_mask_neighbour_fit_term))
        if self._alpha_reference > self._ZERO:
            print("\tReference fit term:")
            print("\t\tIntensity correction type: " +
                  str(self._intensity_correction_type_reference_fit) +
                  " (Initialization: " +
                  str(self._intensity_correction_initializer_type) + ")")
            print("\t\tImage transform: " +
                  self._image_transform_reference_fit_term)
            print("\t\tStack mask used: " +
                  str(self._use_stack_mask_reference_fit_term))
            print("\t\tReference mask used: " + str(self._use_reference_mask))
        print("\tRegularization coefficients: %.g (reference), %.g (neighbour), %.g (parameter)" % (
            self._alpha_reference,
            self._alpha_neighbour,
            self._alpha_parameter))

    ##
    # { function_description }
    # \date       2016-11-08 14:59:26+0000
    #
    # \param      self  The object
    #
    def _run_registration_pipeline_initialization(self):

        self._transform_type_dofs = len(
            self._new_transform_sitk[self._transform_type]().GetParameters())

        # Get number of voxels in the x-y image plane
        self._N_slice_voxels = self._stack.sitk.GetWidth() * \
            self._stack.sitk.GetHeight()

        # Get projected 2D slices onto x-y image plane
        self._slices_2D = self._get_projected_2D_slices_of_stack(
            self._stack, registration_image_type="identity")

        # If reference is given, precompute required data
        if self._reference is not None:

            # Get numpy data arrays from reference image mask
            self._reference_nda_mask = sitk.GetArrayFromImage(
                self._reference.sitk_mask)

            # Since self._intensity_correction_type_slice_neighbour_fit defines
            # the used intensity correction type, i.e. the intensity parameters
            # for optimisation, set them equal in case no neighbour desired.
            if abs(self._alpha_neighbour) < self._ZERO:
                self._intensity_correction_type_slice_neighbour_fit = self._intensity_correction_type_reference_fit

            # slice_i(T(theta_i, x)) - ref(x))
            if self._image_transform_reference_fit_term in ["identity"]:

                # Used to get initial intensity correction
                # parameters/coefficients
                self._init_stack = self._stack
                self._init_reference = self._reference

                # Used to get initial transform parameters
                self._init_slices_2D_stack_reference_term = \
                    self._get_projected_2D_slices_of_stack(
                        self._stack, registration_image_type="identity")
                self._init_slices_2D_reference = \
                    self._get_projected_2D_slices_of_stack(
                        self._reference, registration_image_type="identity")

                # Used to compare slice data arrays against in residual
                # evaluation
                self._reference_nda = sitk.GetArrayFromImage(
                    self._reference.sitk)

            # |grad slice_i|(T(theta_i, x)) - |grad ref|(x))
            elif self._image_transform_reference_fit_term in ["gradient_magnitude"]:

                # Used to get initial intensity correction
                # parameters/coefficients
                gradient_magnitude_stack_sitk = \
                    self._gradient_magnitude_filter_sitk.Execute(
                        self._stack.sitk)
                self._init_stack = st.Stack.from_sitk_image(
                    gradient_magnitude_stack_sitk,
                    "GradMagn_" + self._stack.get_filename(),
                    self._stack.sitk_mask)
                gradient_magnitude_reference_sitk = \
                    self._gradient_magnitude_filter_sitk.Execute(
                        self._reference.sitk)
                self._init_reference = st.Stack.from_sitk_image(
                    gradient_magnitude_reference_sitk,
                    "GradMagn_" + self._reference.get_filename(),
                    self._stack.sitk_mask)

                # Used to get initial transform parameters
                self._init_slices_2D_stack_reference_term = \
                    self._get_projected_2D_slices_of_stack(
                        self._stack,
                        registration_image_type="gradient_magnitude")
                self._init_slices_2D_reference = \
                    self._get_projected_2D_slices_of_stack(
                        self._reference,
                        registration_image_type="gradient_magnitude")

                # Used to compare slice data arrays against in residual
                # evaluation
                self._gradient_magnitude_reference_nda = np.zeros(
                    np.array(self._reference.sitk.GetSize())[::-1])
                for i in range(0, self._N_slices):
                    self._gradient_magnitude_reference_nda[i, :, :] = \
                        sitk.GetArrayFromImage(
                            self._init_slices_2D_reference[i].sitk)

            # ||dx(slice_i)(T(theta_i)) - dx(ref)|| + ||dy(slice_i)(T(theta_i)) - dy(ref)||
            elif self._image_transform_reference_fit_term in ["partial_derivative"]:

                # Used to get initial intensity correction
                # parameters/coefficients
                gradient_magnitude_stack_sitk = \
                    self._gradient_magnitude_filter_sitk.Execute(
                        self._stack.sitk)
                self._init_stack = st.Stack.from_sitk_image(
                    gradient_magnitude_stack_sitk,
                    "GradMagn_"+self._stack.get_filename(),
                    self._stack.sitk_mask)
                gradient_magnitude_reference_sitk = \
                    self._gradient_magnitude_filter_sitk.Execute(
                        self._reference.sitk)
                self._init_reference = st.Stack.from_sitk_image(
                    gradient_magnitude_reference_sitk,
                    "GradMagn_" + self._reference.get_filename(),
                    self._stack.sitk_mask)

                # Used to get initial transform parameters
                self._init_slices_2D_stack_reference_term = \
                    self._get_projected_2D_slices_of_stack(
                        self._stack,
                        registration_image_type="gradient_magnitude")
                self._init_slices_2D_reference = \
                    self._get_projected_2D_slices_of_stack(
                        self._reference,
                        registration_image_type="gradient_magnitude")

                # Used to compare slice data arrays against in residual
                # evaluation
                dx_slices_2D_reference, dy_slices_2D_reference = \
                    self._get_projected_2D_slices_of_stack(
                        self._reference,
                        registration_image_type="partial_derivative")
                self._dx_reference_nda = np.zeros(
                    np.array(self._reference.sitk.GetSize())[::-1])
                self._dy_reference_nda = np.zeros_like(self._dx_reference_nda)
                for i in range(0, self._N_slices):
                    self._dx_reference_nda[i, :, :] = sitk.GetArrayFromImage(
                        dx_slices_2D_reference[i].sitk)
                    self._dy_reference_nda[i, :, :] = sitk.GetArrayFromImage(
                        dy_slices_2D_reference[i].sitk)

            # Resampling grid, i.e. the fixed image space during registration
            self._slice_grid_2D_sitk = sitk.Image(
                self._init_slices_2D_reference[0].sitk)

        else:
            # Resampling grid, i.e. the fixed image space during registration
            self._slice_grid_2D_sitk = sitk.Image(self._slices_2D[0].sitk)

        # Get inital transform and the respective initial transform parameters
        # used for further optimisation
        self._transforms_2D_sitk, parameters = \
            self._get_initial_transforms_and_parameters[
                self._transform_initializer_type]()

        if self._intensity_correction_type_slice_neighbour_fit is not None:
            parameters_intensity = \
                self._get_initial_intensity_correction_parameters[
                    self._intensity_correction_initializer_type]()
            parameters = np.concatenate(
                (parameters, parameters_intensity),
                axis=1)

        # Parameters for initialization and for regularization term
        self._parameters0_vec = parameters.flatten()

        # Create copy for member variable
        self._parameters = np.array(parameters)

        # Store number of degrees of freedom for overall optimization
        self._optimization_dofs = self._parameters.shape[1]

    ##
    # Based on the residual functions below and the chosen settings, this
    # function returns the residual call used for the least_squares method
    # \date       2016-11-21 20:02:32+0000
    #
    # \param      self  The object
    #
    # \return     The residual call.
    #
    def _get_residual_call(self):

        alpha_neighbour = abs(float(self._alpha_neighbour))
        alpha_parameter = abs(float(self._alpha_parameter))
        alpha_reference = abs(float(self._alpha_reference))

        # ---------------------------------------------------------------------
        # 1) Defines the prior term on the parameters
        if alpha_parameter > self._ZERO:
            if self._transform_type in ["similarity"]:
                self._get_residual_parameters = lambda x: np.concatenate((
                    self._get_residual_scale(x),
                    self._get_residual_intensity_coefficients[
                        self._intensity_correction_type_slice_neighbour_fit](x)
                ))
            else:
                self._get_residual_parameters = \
                    lambda x: self._get_residual_intensity_coefficients[
                        self._intensity_correction_type_slice_neighbour_fit](x)

        # ---------------------------------------------------------------------
        # 2) Construct overall residual
        if self._reference is None:
            if alpha_neighbour < self._ZERO:
                raise ValueError(
                    "A weight of alpha_neighbour <= 0 is not meaningful.")

            if alpha_parameter < self._ZERO:
                residual = lambda x: self._get_residual_slice_neighbours_fit(x)

            else:
                residual = lambda x: np.concatenate((
                    self._get_residual_slice_neighbours_fit(x),
                    alpha_parameter/alpha_neighbour *
                    self._get_residual_parameters(x)
                ))
        else:
            # Build total residual for reference fit
            if self._image_transform_reference_fit_term in ["identity"]:
                self._get_residual_reference_fit_total = lambda x: \
                    self._get_residual_reference_fit(
                        self._slices_2D,
                        self._reference_nda,
                        "identity",
                        x)

            if self._image_transform_reference_fit_term in ["gradient_magnitude"]:
                self._get_residual_reference_fit_total = \
                    lambda x: self._get_residual_reference_fit(
                        self._slices_2D,
                        self._gradient_magnitude_reference_nda,
                        "gradient_magnitude",
                        x)

            elif self._image_transform_reference_fit_term in ["partial_derivative"]:
                self._get_residual_reference_fit_total = \
                    lambda x: np.concatenate((
                        self._get_residual_reference_fit(
                            self._slices_2D,
                            self._dx_reference_nda,
                            "dx",
                            x),
                        self._get_residual_reference_fit(
                            self._slices_2D,
                            self._dy_reference_nda,
                            "dy",
                            x)
                    ))

            # Combine all the residuals
            if alpha_reference < self._ZERO:
                raise ValueError(
                    "A weight of alpha_reference <= 0 is not meaningful in case reference is given")

            if alpha_neighbour < self._ZERO and alpha_parameter < self._ZERO:
                residual = lambda x: self._get_residual_reference_fit_total(x)

            elif alpha_neighbour > self._ZERO and alpha_parameter < self._ZERO:
                residual = lambda x: np.concatenate((
                    self._get_residual_reference_fit_total(x),
                    alpha_neighbour/alpha_reference *
                    self._get_residual_slice_neighbours_fit(x)
                ))

            elif alpha_neighbour < self._ZERO and alpha_parameter > self._ZERO:
                residual = lambda x: np.concatenate((
                    self._get_residual_reference_fit_total(x),
                    alpha_parameter/alpha_reference *
                    self._get_residual_parameters(x)
                ))

            elif alpha_neighbour > self._ZERO and alpha_parameter > self._ZERO:
                residual = lambda x: np.concatenate((
                    self._get_residual_reference_fit_total(x),
                    alpha_neighbour/alpha_reference *
                    self._get_residual_slice_neighbours_fit(x),
                    alpha_parameter/alpha_reference *
                    self._get_residual_parameters(x)
                ))

        return residual

    ##
    # Based on the Jacobian of the residual functions below and the chosen
    # settings, this function returns the Jacobian call used for the
    # least_squares method.
    # \date       2016-11-21 20:04:37+0000
    #
    # \param      self  The object
    #
    # \return     The Jacobian call.
    #
    def _get_jacobian_residual_call(self):

        alpha_neighbour = abs(float(self._alpha_neighbour))
        alpha_parameter = abs(float(self._alpha_parameter))
        alpha_reference = abs(float(self._alpha_reference))

        # ---------------------------------------------------------------------
        # 1) Define Jacobian of the prior term on the parameters
        if alpha_parameter > self._ZERO:
            if self._transform_type in ["similarity"]:
                self._get_jacobian_residual_parameters = \
                    lambda x: np.concatenate((
                        self._get_jacobian_residual_scale(x),
                        self._get_jacobian_residual_intensity_coefficients[
                            self._intensity_correction_type_slice_neighbour_fit](x)
                    ))
            else:
                self._get_jacobian_residual_parameters = \
                    lambda x: self._get_jacobian_residual_intensity_coefficients[
                        self._intensity_correction_type_slice_neighbour_fit](x)

        # ---------------------------------------------------------------------
        # 2) Construct overall Jacobian of residual
        if self._reference is None:
            self._alpha_reference = 0

            if alpha_neighbour < self._ZERO:
                raise ValueError(
                    "A weight of alpha_neighbour <= 0 is not meaningful.")

            if alpha_parameter < self._ZERO:
                jacobian = \
                    lambda x: self._get_jacobian_residual_slice_neighbours_fit(
                        x)

            else:
                jacobian = lambda x: np.concatenate((
                    self._get_jacobian_residual_slice_neighbours_fit(x),
                    alpha_parameter / alpha_neighbour *
                    self._get_jacobian_residual_parameters(x)
                ))

        else:

            if self._image_transform_reference_fit_term in ["identity"]:
                self._get_jacobian_residual_reference_fit_total = \
                    lambda x: self._get_jacobian_residual_reference_fit(
                        self._slices_2D, "identity", x)

            elif self._image_transform_reference_fit_term in ["gradient_magnitude"]:
                self._get_jacobian_residual_reference_fit_total = \
                    lambda x: self._get_jacobian_residual_reference_fit(
                        self._slices_2D, "gradient_magnitude", x)

            elif self._image_transform_reference_fit_term in ["partial_derivative"]:
                self._get_jacobian_residual_reference_fit_total = \
                    lambda x: np.concatenate((
                        self._get_jacobian_residual_reference_fit(
                            self._slices_2D, "dx", x),
                        self._get_jacobian_residual_reference_fit(
                            self._slices_2D, "dy", x)
                    ))

            if alpha_reference < self._ZERO:
                raise ValueError(
                    "A weight of alpha_reference <= 0 is not meaningful in case reference is given")

            if alpha_neighbour < self._ZERO and alpha_parameter < self._ZERO:
                jacobian = \
                    lambda x: self._get_jacobian_residual_reference_fit_total(
                        x)

            elif alpha_neighbour > self._ZERO and alpha_parameter < self._ZERO:
                jacobian = lambda x: np.concatenate((
                    self._get_jacobian_residual_reference_fit_total(x),
                    alpha_neighbour / alpha_reference *
                    self._get_jacobian_residual_slice_neighbours_fit(x)
                ))

            elif alpha_neighbour < self._ZERO and alpha_parameter > self._ZERO:
                jacobian = lambda x: np.concatenate((
                    self._get_jacobian_residual_reference_fit_total(x),
                    alpha_parameter / alpha_reference *
                    self._get_jacobian_residual_parameters(x)
                ))

            elif alpha_neighbour > self._ZERO and alpha_parameter > self._ZERO:
                jacobian = lambda x: np.concatenate((
                    self._get_jacobian_residual_reference_fit_total(x),
                    alpha_neighbour / alpha_reference *
                    self._get_jacobian_residual_slice_neighbours_fit(x),
                    alpha_parameter / alpha_reference *
                    self._get_jacobian_residual_parameters(x)
                ))

        return jacobian

    ##
    # Gets the residual indicating the alignment between slices and reference.
    # \date       2016-11-08 20:37:49+0000
    #
    # It returns the stacked residual of slice_i(T(theta_i, x)) - ref(x)) for
    # all slices i.
    #
    # \param      self            The object
    # \param      slices_2D       The slices 2d
    # \param      reference_nda   The reference nda
    # \param      trafo           The trafo
    # \param      parameters_vec  The parameters vector
    #
    # \return     The residual reference fit as (N_slices * N_slice_voxels)
    #             numpy array
    #
    def _get_residual_reference_fit(self,
                                    slices_2D,
                                    reference_nda,
                                    trafo,
                                    parameters_vec):

        # Allocate memory for residual
        residual = np.zeros((self._N_slices, self._N_slice_voxels))

        # Reshape parameters for easier access
        parameters = parameters_vec.reshape(-1, self._optimization_dofs)

        # Compute residuals between each slice and reference
        for i in range(0, self._N_slices):

            # Get slice_i(T(theta_i, x))
            self._transforms_2D_sitk[i].SetParameters(
                parameters[i, 0:self._transform_type_dofs])
            slice_i_sitk = sitk.Resample(
                slices_2D[i].sitk,
                self._slice_grid_2D_sitk,
                self._transforms_2D_sitk[i],
                self._interpolator_sitk)

            # Apply image transform, i.e. gradients etc
            slice_i_sitk = self._apply_image_transform[trafo](slice_i_sitk)

            # Extract data array
            slice_i_nda = sitk.GetArrayFromImage(slice_i_sitk)

            # Correct intensities according to chosen model
            slice_i_nda = self._apply_intensity_correction[
                self._intensity_correction_type_reference_fit](
                slice_i_nda, parameters[i, self._transform_type_dofs:])

            # Compute residual slice_i(T(theta_i, x)) - ref(x))
            residual_slice_nda = slice_i_nda - reference_nda[i, :, :]

            # Incorporate mask computations
            if self._use_stack_mask_reference_fit_term:
                slice_i_sitk_mask = sitk.Resample(
                    slices_2D[i].sitk_mask,
                    self._slice_grid_2D_sitk,
                    self._transforms_2D_sitk[i],
                    sitk.sitkNearestNeighbor)
                slice_i_nda_mask = sitk.GetArrayFromImage(slice_i_sitk_mask)
                residual_slice_nda *= slice_i_nda_mask

            if self._use_reference_mask:
                residual_slice_nda *= self._reference_nda_mask[i, :, :]

            # ph.show_2D_array_list([residual_slice_nda, slice_i_nda_mask, self._reference_nda_mask[i,:,:]])
            # ph.pause()

            # Set residual for current slice difference
            residual[i, :] = residual_slice_nda.flatten()

        return residual.flatten()

    ##
    # Gets the Jacobian to \p _get_residual_reference_fit used for the
    # least_squares method.
    # \date       2016-11-21 20:09:36+0000
    #
    # \param      self            The object
    # \param      slices_2D       The slices 2d
    # \param      trafo           The trafo
    # \param      parameters_vec  The parameters vector
    #
    # \return     The jacobian residual reference fit as [N_slices *
    #             N_slice_voxels] x [transform_type_dofs * N_slices] numpy
    #             array
    #
    def _get_jacobian_residual_reference_fit(self,
                                             slices_2D,
                                             trafo,
                                             parameters_vec):

        # Allocate memory for Jacobian of residual
        jacobian = np.zeros(
            (self._N_slices * self._N_slice_voxels,
                self._optimization_dofs * self._N_slices))

        jacobian_slice_i = np.zeros(
            (self._N_slice_voxels, self._optimization_dofs))

        # Reshape parameters for easier access
        parameters = parameters_vec.reshape(-1, self._optimization_dofs)

        # Compute Jacobian of residuals between each slice and reference
        for i in range(0, self._N_slices):

            # Update transforms
            parameters_slice = parameters[i, 0:self._transform_type_dofs]
            self._transforms_2D_sitk[i].SetParameters(parameters_slice)
            self._transforms_2D_itk[i].SetParameters(
                itk.OptimizerParameters[itk.D](parameters_slice))

            # Get slice_i(T(theta, x))
            slice_i_sitk = sitk.Resample(
                slices_2D[i].sitk,
                self._slice_grid_2D_sitk,
                self._transforms_2D_sitk[i],
                self._interpolator_sitk)

            # Apply image transform, i.e. gradients etc
            slice_i_sitk = self._apply_image_transform[trafo](slice_i_sitk)

            # Get d[slice(T(theta, x))]/dx as (Ny x Nx x dim)-array
            dslice_i_nda = self._get_gradient_image_nda_from_sitk_image(
                slice_i_sitk)

            # Get slice data array (used for intensity correction parameter
            # gradient)
            slice_i_nda = sitk.GetArrayFromImage(slice_i_sitk)

            # Incorporate mask computations
            if self._use_stack_mask_reference_fit_term:
                # Slice mask
                slice_i_sitk_mask = sitk.Resample(
                    slices_2D[i].sitk_mask,
                    self._slice_grid_2D_sitk,
                    self._transforms_2D_sitk[i],
                    sitk.sitkNearestNeighbor)
                slice_i_nda_mask = sitk.GetArrayFromImage(slice_i_sitk_mask)

                # Mask data
                slice_i_nda *= slice_i_nda_mask

                # Mask gradient data
                dslice_i_nda *= slice_i_nda_mask[:, :, np.newaxis]

                # ph.show_2D_array_list

            if self._use_reference_mask:
                # Reference mask
                reference_i_nda_mask = self._reference_nda_mask[i, :, :]

                # Mask data
                slice_i_nda *= reference_i_nda_mask

                # Mask gradient data
                dslice_i_nda *= reference_i_nda_mask[:, :, np.newaxis]

            # Get Jacobian of slice w.r.t to transform parameters
            jacobian_slice_nda = \
                self._get_gradient_with_respect_to_transform_parameters(
                    dslice_i_nda, self._transforms_2D_itk[i], slice_i_sitk)

            # Get d[slice_i(T(theta_i, x))]/dtheta_i:
            # Add Jacobian w.r.t. to intensity correction parameters
            jacobian_slice_i_tmp = \
                self._add_gradient_with_respect_to_intensity_correction_parameters[
                    self._intensity_correction_type_reference_fit](
                        jacobian_slice_nda, slice_i_nda)

            # Second dimension is decided by intensity_correction_type_slice_neighbour_fit
            # as being of "higher order"
            # (e.g. affine for slice fit term and linear for reference fit term)
            jacobian_slice_i[:, 0:jacobian_slice_i_tmp.shape[
                1]] = jacobian_slice_i_tmp

            # Set elements in Jacobian for entire stack
            jacobian[
                i * self._N_slice_voxels:
                (i + 1) * self._N_slice_voxels,
                i * self._optimization_dofs:
                (i + 1) * self._optimization_dofs] = jacobian_slice_i

        return jacobian

    ##
    # Gets the residual indicating the alignment between neighbouring slices.
    # \date       2016-11-21 20:07:41+0000
    #
    # It returns the stacked residual of slice_i(T(theta_i, x)) -
    # slice_{i+1}(T(theta_{i+1}, x)) for all voxels x of all slices i.
    #
    # \param      self            The object
    # \param      parameters_vec  The parameters vector
    #
    # \return     The residual slice neighbours fit as
    #             (N_slices-1) * N_slice_voxels numpy array
    #
    def _get_residual_slice_neighbours_fit(self, parameters_vec):

        # Allocate memory for residual
        residual = np.zeros((self._N_slices-1, self._N_slice_voxels))

        # Reshape parameters for easier access
        parameters = parameters_vec.reshape(-1, self._optimization_dofs)

        # Update transform
        i = 0
        parameters_slice_i = parameters[i, 0:self._transform_type_dofs]
        self._transforms_2D_sitk[i].SetParameters(parameters_slice_i)

        # Get slice_i(T(theta_i, x)) for i=0
        slice_i_sitk = sitk.Resample(
            self._slices_2D[i].sitk,
            self._slice_grid_2D_sitk,
            self._transforms_2D_sitk[i],
            self._interpolator_sitk)
        slice_i_nda = sitk.GetArrayFromImage(slice_i_sitk)

        # Correct intensities according to chosen model
        slice_i_nda = self._apply_intensity_correction[
            self._intensity_correction_type_slice_neighbour_fit](
            slice_i_nda, parameters[i, self._transform_type_dofs:])

        if self._use_stack_mask_neighbour_fit_term:
            slice_i_sitk_mask = sitk.Resample(
                self._slices_2D[i].sitk_mask,
                self._slice_grid_2D_sitk,
                self._transforms_2D_sitk[i],
                sitk.sitkNearestNeighbor)
            slice_i_nda_mask = sitk.GetArrayFromImage(slice_i_sitk_mask)

        # Compute residuals for neighbouring slices
        for i in range(0, self._N_slices-1):

            # Update transform
            parameters_slice_ip1 = parameters[i+1, 0:self._transform_type_dofs]
            self._transforms_2D_sitk[i+1].SetParameters(parameters_slice_ip1)

            # Get slice_{i+1}(T(theta_{i+1}, x))
            slice_ip1_sitk = sitk.Resample(
                self._slices_2D[i+1].sitk,
                self._slice_grid_2D_sitk,
                self._transforms_2D_sitk[i+1],
                self._interpolator_sitk)
            slice_ip1_nda = sitk.GetArrayFromImage(slice_ip1_sitk)

            # Correct intensities according to chosen model
            slice_ip1_nda = self._apply_intensity_correction[
                self._intensity_correction_type_slice_neighbour_fit](
                slice_ip1_nda, parameters[i+1, self._transform_type_dofs:])

            # Compute residual slice_i(T(theta_i, x)) -
            # slice_{i+1}(T(theta_{i+1}, x))
            residual_slice_nda = slice_i_nda - slice_ip1_nda

            # Eliminate residual for non-masked regions
            if self._use_stack_mask_neighbour_fit_term:
                slice_ip1_sitk_mask = sitk.Resample(
                    self._slices_2D[i+1].sitk_mask,
                    self._slice_grid_2D_sitk,
                    self._transforms_2D_sitk[i+1],
                    sitk.sitkNearestNeighbor)
                slice_ip1_nda_mask = sitk.GetArrayFromImage(
                    slice_ip1_sitk_mask)

                residual_slice_nda = residual_slice_nda * slice_i_nda_mask * \
                    slice_ip1_nda_mask

                slice_i_nda_mask = slice_ip1_nda_mask

            # Set residual for current slice difference
            residual[i, :] = residual_slice_nda.flatten()

            # Prepare for next iteration
            slice_i_nda = slice_ip1_nda

        return residual.flatten()

    ##
    # Gets the Jacobian to \p _get_residual_slice_neighbours_fit used for the
    # least_squares method.
    # \date       2016-11-21 20:08:48+0000
    #
    # \param      self            The object
    # \param      parameters_vec  The parameters vector
    #
    # \return     The Jacobian residual slice neighbours fit as [(N_slices-1) *
    #             N_slice_voxels] x [transform_type_dofs * N_slices] numpy
    #             array
    #
    def _get_jacobian_residual_slice_neighbours_fit(self, parameters_vec):

        # Allocate memory for Jacobian of residual
        jacobian = np.zeros((
            (self._N_slices-1)*self._N_slice_voxels,
            self._optimization_dofs*self._N_slices))

        # Reshape parameters for easier access
        parameters = parameters_vec.reshape(-1, self._optimization_dofs)

        # Update transforms
        i = 0
        parameters_slice_i = parameters[i, 0:self._transform_type_dofs]
        self._transforms_2D_sitk[i].SetParameters(parameters_slice_i)
        self._transforms_2D_itk[i].SetParameters(
            itk.OptimizerParameters[itk.D](parameters_slice_i))

        # Get d[slice_i(T(theta_i, x))]/dtheta_i
        jacobian_slice_i = self._get_jacobian_slice_in_slice_neighbours_fit(
            self._slices_2D[i],
            self._transforms_2D_sitk[i],
            self._transforms_2D_itk[i])

        # Compute Jacobian of residuals
        for i in range(0, self._N_slices-1):

            # Update transforms
            parameters_slice_ip1 = parameters[i+1, 0:self._transform_type_dofs]
            self._transforms_2D_sitk[i+1].SetParameters(parameters_slice_ip1)
            self._transforms_2D_itk[i+1].SetParameters(
                itk.OptimizerParameters[itk.D](parameters_slice_ip1))

            # Get d[slice_{i+1}(T(theta_{i+1}, x))]/dtheta_{i+1}
            jacobian_slice_ip1 = \
                self._get_jacobian_slice_in_slice_neighbours_fit(
                    self._slices_2D[i+1],
                    self._transforms_2D_sitk[i+1],
                    self._transforms_2D_itk[i+1])

            # Set elements in Jacobian for entire stack
            jacobian[i * self._N_slice_voxels:
                     (i + 1) * self._N_slice_voxels,
                     i * self._optimization_dofs:
                     (i + 1) * self._optimization_dofs] = jacobian_slice_i
            jacobian[i * self._N_slice_voxels:
                     (i + 1) * self._N_slice_voxels,
                     (i + 1) * self._optimization_dofs:
                     (i + 2) * self._optimization_dofs] = -jacobian_slice_ip1

            # Prepare for next iteration
            jacobian_slice_i = jacobian_slice_ip1

        return jacobian

    ##
    # Gets the Jacobian of a slice based on the spatial transformation.
    # \date       2016-11-21 18:23:53+0000
    #
    # Compute the Jacobian
    # \f$ \frac{dI(T(\theta, x))}{d\theta} =
    # \frac{dI}{dy}(T(\theta,x))\,\frac{dT}{d\theta}(\theta, x)
    # \f$. It also considers the (affine) intensity correction model
    #
    # \param      self            The object
    # \param      slice           The slice
    # \param      transform_sitk  The transform sitk
    # \param      transform_itk   The transform itk
    #
    # \return     The Jacobian of a slice as (N_slice_voxels x
    #             transform_type_dofs)-array.
    #
    def _get_jacobian_slice_in_slice_neighbours_fit(self,
                                                    slice,
                                                    transform_sitk,
                                                    transform_itk):

        # Get slice(T(theta, x))
        slice_sitk = sitk.Resample(
            slice.sitk,
            self._slice_grid_2D_sitk,
            transform_sitk,
            self._interpolator_sitk)

        # Get d[slice(T(theta, x))]/dx as (Ny x Nx x dim)-array
        dslice_nda = self._get_gradient_image_nda_from_sitk_image(slice_sitk)

        # Get slice data array (used for intensity correction parameter
        # gradient)
        slice_nda = sitk.GetArrayFromImage(slice_sitk)

        if self._use_stack_mask_neighbour_fit_term:
            slice_sitk_mask = sitk.Resample(
                slice.sitk_mask,
                self._slice_grid_2D_sitk,
                transform_sitk,
                sitk.sitkNearestNeighbor)
            slice_nda_mask = sitk.GetArrayFromImage(slice_sitk_mask)

            # slice_nda *= slice_nda_mask[:,:,np.newaxis]
            slice_nda *= slice_nda_mask

        # Get Jacobian of slice w.r.t to transform parameters
        jacobian_slice_nda = \
            self._get_gradient_with_respect_to_transform_parameters(
                dslice_nda, transform_itk, slice_sitk)

        # Add Jacobian w.r.t. to intensity correction parameters
        jacobian_slice_nda = \
            self._add_gradient_with_respect_to_intensity_correction_parameters[
                self._intensity_correction_type_slice_neighbour_fit](
                    jacobian_slice_nda, slice_nda)

        return jacobian_slice_nda

    ##
    # Gets the gradient with respect to transform parameters of all voxels
    # within a slice.
    # \date       2017-07-15 23:03:10+0100
    #
    # \param      self           The object
    # \param      dslice_nda     The dslice nda
    # \param      transform_itk  The transform itk
    # \param      slice_sitk     The slice sitk
    #
    # \return     The gradient with respect to transform parameters;
    #             (N_slice_voxels x transform_type_dofs) numpy array
    #
    def _get_gradient_with_respect_to_transform_parameters(self,
                                                           dslice_nda,
                                                           transform_itk,
                                                           slice_sitk):

        # Reshape to (N_slice_voxels x dim)-array
        dslice_nda = dslice_nda.reshape(self._N_slice_voxels, -1)

        # Get d[T(theta, x)]/dtheta as (N_slice_voxels x dim x
        # transform_type_dofs)-array
        dT_nda = \
            sitkh.get_numpy_array_of_jacobian_itk_transform_applied_on_sitk_image(
                transform_itk, slice_sitk)

        # Compute Jacobian for slice as (N_slice_voxels x
        # transform_type_dofs)-array
        jacobian_slice = np.sum(dslice_nda[:, :, np.newaxis]*dT_nda, axis=1)

        return jacobian_slice

    def _get_gradient_image_nda_from_sitk_image(self, slice_sitk):

        # Compute d[slice(T(theta, x))]/dx
        dslice_sitk = self._gradient_image_filter_sitk.Execute(slice_sitk)

        # Get associated (Ny x Nx x dim)-array
        dslice_nda = sitk.GetArrayFromImage(dslice_sitk)

        return dslice_nda

    # ##
    # # Gets the residual parameters for all optimization parameters
    # # \date       2016-11-21 18:09:24+0000
    # #
    # # \param      self            The object
    # # \param      parameters_vec  The parameters vector
    # #
    # # \return     The residual parameters.
    # #
    # def _get_residual_parameters(self, parameters_vec):

    #     ## Reshape parameters for easier access
    #     parameters = parameters_vec.reshape(-1, self._optimization_dofs)

    #     parameters_prior = np.zeros(parameters.shape)

    #     ## Prior for transform parameters
    #     parameters_prior[:, 0: self._transform_type_dofs] = self._parameters_prior_transform[self._transform_type]

    #     ## Prior for intensity correction parameters
    #     parameters_prior[:,self._transform_type_dofs:] = self._parameters_prior_intensity_correction[self._intensity_correction_type_slice_neighbour_fit]

    #     # return parameters_vec/self._parameters0_vec
    #     return parameters_vec - parameters_prior.flatten()

    # def _get_jacobian_residual_parameters(self, parameters_vec):
    #     ## Reshape parameters for easier access
    #     parameters = parameters_vec.reshape(-1, self._optimization_dofs)

    #     parameters_prior = np.zeros(parameters.shape)

    #     # parameters_prior[:, self._transform_type_dofs:] = np.array([10.,50.])

    #     ## Allocate memory for Jacobian of residual
    #     jacobian = np.eye(self._N_slices*self._optimization_dofs)
    #     # jacobian = np.diag(1/parameters_prior.flatten())

    #     return jacobian

    ##
    # Gets the residual scale.
    # \date       2016-11-21 18:09:42+0000
    #
    # \param      self            The object
    # \param      parameters_vec  The parameters vector
    #
    # \return     The residual scale.
    #
    def _get_residual_scale(self, parameters_vec):

        # Reshape parameters for easier access
        parameters = parameters_vec.reshape(-1, self._optimization_dofs)

        parameters_scale = parameters[:, 0]

        return parameters_scale - self._prior_scale

    def _get_jacobian_residual_scale(self, parameters_vec):

        jacobian = np.zeros(
            (self._N_slices, self._N_slices*self._optimization_dofs))

        for i in range(0, self._N_slices):
            jacobian[i, i*self._optimization_dofs] = 1

        return jacobian

    ##
    # Gets the residual intensity coefficients for different intensity
    # correction models.
    # \date       2016-11-21 18:12:27+0000
    #
    # \param      self            The object
    # \param      parameters_vec  The parameters vector
    #
    # \return     The residual intensity coefficients for different types.
    #
    def _get_residual_intensity_coefficients_None(self, parameters_vec):
        return np.zeros(1)

    def _get_jacobian_residual_intensity_coefficients_None(self,
                                                           parameters_vec):
        return np.zeros((1, self._N_slices*self._optimization_dofs))

    def _get_residual_intensity_coefficients_linear(self, parameters_vec):

        # Reshape parameters for easier access
        parameters = parameters_vec.reshape(-1, self._optimization_dofs)

        parameters_coefficients = parameters[:, self._transform_type_dofs]

        return parameters_coefficients - \
            self._prior_intensity_correction_coefficients[0]

    def _get_jacobian_residual_intensity_coefficients_linear(self,
                                                             parameters_vec):

        jacobian = np.zeros(
            (self._N_slices, self._N_slices*self._optimization_dofs))

        for i in range(0, self._N_slices):
            jacobian[i, self._transform_type_dofs +
                     i * self._optimization_dofs] = 1

        return jacobian

    def _get_residual_intensity_coefficients_affine(self, parameters_vec):

        # Reshape parameters for easier access
        parameters = parameters_vec.reshape(-1, self._optimization_dofs)

        parameters_coefficients = parameters[:, self._transform_type_dofs:]

        return (parameters_coefficients -
                self._prior_intensity_correction_coefficients).flatten()

    def _get_jacobian_residual_intensity_coefficients_affine(self,
                                                             parameters_vec):

        jacobian = np.zeros(
            (2*self._N_slices, self._N_slices*self._optimization_dofs))

        for i in range(0, self._N_slices):
            jacobian[2*i, self._transform_type_dofs +
                     i*self._optimization_dofs] = 1
            jacobian[2*i+1, self._transform_type_dofs +
                     i*self._optimization_dofs+1] = 1

        return jacobian

    ##
    # Compute several transforms on image like identity, \f$ \partial_x \f$,
    # \f$ \partial_y \f$ and \f$ |\nabla | \f$.
    # \date       2016-12-01 03:08:50+0000
    #
    # \param      self           The object
    # \param      slice_2D_sitk  The slice 2d sitk
    #
    def _apply_image_transform_identity(self, slice_2D_sitk):
        return slice_2D_sitk

    def _apply_image_transform_dx(self, slice_2D_sitk):
        dx_slice_2D_sitk = self._get_dx_image_sitk(slice_2D_sitk)
        # Debug
        # sitkh.show_sitk_image([slice_2D_sitk, dx_slice_2D_sitk], title=["original", "dx"])
        return dx_slice_2D_sitk

    def _apply_image_transform_dy(self, slice_2D_sitk):
        dy_slice_2D_sitk = self._get_dy_image_sitk(slice_2D_sitk)
        # Debug
        # sitkh.show_sitk_image([slice_2D_sitk, dy_slice_2D_sitk], title=["original", "dy"])
        return dy_slice_2D_sitk

    def _apply_image_transform_gradient_magnitude(self, slice_2D_sitk):
        gradient_magnitude_slice_2D_sitk = \
            self._gradient_magnitude_filter_sitk.Execute(
                slice_2D_sitk)
        # Debug
        # sitkh.show_sitk_image([slice_2D_sitk, gradient_magnitude_slice_2D_sitk], title=["original", "gradient_magnitude"])
        return gradient_magnitude_slice_2D_sitk

    def _get_dx_image_sitk(self, image_sitk):
        dimage_sitk = self._gradient_image_filter_sitk.Execute(image_sitk)
        dx_image_sitk = sitk.VectorIndexSelectionCast(dimage_sitk, 0)

        return dx_image_sitk

    def _get_dy_image_sitk(self, image_sitk):
        dimage_sitk = self._gradient_image_filter_sitk.Execute(image_sitk)
        dy_image_sitk = sitk.VectorIndexSelectionCast(dimage_sitk, 1)

        return dy_image_sitk

    ##
    # Calculates the statistics of residuals based on ell^2 norm
    # \date       2016-11-30 14:16:20+0000
    #
    # \param      self  The object
    #
    # \return     The statistics residuals ell 2.
    #
    def _compute_statistics_residuals_ell2(self):

        self._final_cost = 0

        if self._alpha_reference > self._ZERO:
            self._residual_reference_fit_ell2 = np.sum(
                self._get_residual_reference_fit_total(
                    self._parameters.flatten())**2)
            self._final_cost += self._alpha_reference * \
                self._residual_reference_fit_ell2

        if self._alpha_neighbour > self._ZERO:
            self._residual_slice_neighbours_ell2 = np.sum(
                self._get_residual_slice_neighbours_fit(
                    self._parameters.flatten())**2)
            self._final_cost += self._alpha_neighbour * \
                self._residual_slice_neighbours_ell2

        if self._alpha_parameter > self._ZERO:
            self._residual_paramters_ell2 = np.sum(
                self._get_residual_parameters(self._parameters.flatten())**2)
            self._final_cost += self._alpha_parameter * \
                self._residual_paramters_ell2

    ##
    #       Gets the initial parameters for 'None', i.e. for identity
    #             transform.
    # \date       2016-11-08 15:06:54+0000
    #
    # \param      self  The object
    #
    # \return     The initial parameters corresponding to identity transform as
    #             (N_slices x DOF)-array
    #
    def _get_initial_transforms_and_parameters_identity(self):

        # Create list of identity transforms for all slices
        transforms_2D_sitk = [None] * self._N_slices

        # Get list of identity transform parameters for all slices
        parameters = np.zeros((self._N_slices, self._transform_type_dofs))
        for i in range(0, self._N_slices):
            transforms_2D_sitk[i] = self._new_transform_sitk[
                self._transform_type]()
            parameters[i, :] = transforms_2D_sitk[i].GetParameters()

        return transforms_2D_sitk, parameters

    ##
    # Gets the initial parameters for either 'GEOMETRY' or 'MOMENTS'.
    # \date       2016-11-08 15:08:07+0000
    #
    # \param      self  The object
    #
    # \return     The initial parameters corresponding to 'GEOMETRY' or
    #             'MOMENTS' as (N_slices x DOF)-array
    #
    def _get_initial_transforms_and_parameters_geometry_moments(self):

        transform_initializer_type_sitk = \
            self._dictionary_transform_initializer_type_sitk[
                self._transform_initializer_type]

        # Create list of identity transforms
        transforms_2D_sitk = [self._new_transform_sitk[
            self._transform_type]()] * self._N_slices

        # Get list of identity transform parameters for all slices
        parameters = np.zeros((self._N_slices, self._transform_type_dofs))

        # Set identity parameters for first slice
        parameters[0, :] = transforms_2D_sitk[0].GetParameters()

        # No reference is given and slices are initialized to align with
        # neighbouring slice
        if self._reference is None:

            # Create identity transform for first slice
            compensation_transform_sitk = self._new_transform_sitk[
                self._transform_type]()

            # First slice is kept at position and others are aligned
            # accordingly
            for i in range(1, self._N_slices):

                # Take into account the initialization of slice i-1
                slice_im1_sitk = sitk.Image(self._slices_2D[i-1].sitk)
                if self._use_stack_mask_neighbour_fit_term:
                    slice_im1_sitk *= sitk.Cast(
                        self._slices_2D[i-1].sitk_mask,
                        slice_im1_sitk.GetPixelIDValue())
                slice_im1_sitk = sitkh.get_transformed_sitk_image(
                    slice_im1_sitk, compensation_transform_sitk)

                # Use sitk.CenteredTransformInitializerFilter to get initial
                # transform
                fixed_sitk = slice_im1_sitk
                moving_sitk = sitk.Image(self._slices_2D[i].sitk)
                if self._use_stack_mask_neighbour_fit_term:
                    moving_sitk *= sitk.Cast(self._slices_2D[i].sitk_mask,
                                             moving_sitk.GetPixelIDValue())
                initial_transform_sitk = self._new_transform_sitk[
                    self._transform_type]()
                operation_mode_sitk = eval(
                    "sitk.CenteredTransformInitializerFilter." +
                    transform_initializer_type_sitk)

                # Get transform
                try:
                    # For operation_mode_sitk="MOMENTS" errors can occur!
                    initial_transform_sitk = sitk.CenteredTransformInitializer(
                        fixed_sitk, moving_sitk, initial_transform_sitk, operation_mode_sitk)
                except:
                    print("WARNING: Slice %d/%d" % (i, self._N_slices-1))
                    print("\tsitk.CenteredTransformInitializerFilter with " +
                          transform_initializer_type_sitk +
                          " does not work. Identity transform is used instead for initialization")
                    initial_transform_sitk = self._new_transform_sitk[
                        self._transform_type]()
                transforms_2D_sitk[i] = eval(
                    "sitk." + initial_transform_sitk.GetName() +
                    "(initial_transform_sitk)")

                # Get parameters
                parameters[i, :] = transforms_2D_sitk[i].GetParameters()

                # Store compensation transform for subsequent slice
                compensation_transform_sitk.SetParameters(
                    transforms_2D_sitk[i].GetParameters())
                compensation_transform_sitk.SetFixedParameters(
                    transforms_2D_sitk[i].GetFixedParameters())
                compensation_transform_sitk = eval(
                    "sitk." + compensation_transform_sitk.GetName() +
                    "(compensation_transform_sitk.GetInverse())")

        # Initialize transform to match each slice with the reference
        else:

            # print self._use_reference_mask
            # print self._use_stack_mask_reference_fit_term
            for i in range(0, self._N_slices):

                # Use sitk.CenteredTransformInitializerFilter to get initial
                # transform
                fixed_sitk = self._init_slices_2D_reference[i].sitk
                if self._use_reference_mask:
                    fixed_sitk *= sitk.Cast(
                        self._init_slices_2D_reference[i].sitk_mask,
                        fixed_sitk.GetPixelIDValue())
                moving_sitk = self._init_slices_2D_stack_reference_term[i].sitk
                if self._use_stack_mask_reference_fit_term:
                    moving_sitk *= sitk.Cast(
                        self._init_slices_2D_stack_reference_term[i].sitk_mask,
                        moving_sitk.GetPixelIDValue())
                initial_transform_sitk = self._new_transform_sitk[
                    self._transform_type]()
                operation_mode_sitk = eval(
                    "sitk.CenteredTransformInitializerFilter." +
                    transform_initializer_type_sitk)

                # Get transform
                try:
                    # For operation_mode_sitk="MOMENTS" errors can occur!
                    initial_transform_sitk = sitk.CenteredTransformInitializer(
                        fixed_sitk, moving_sitk, initial_transform_sitk, operation_mode_sitk)
                except:
                    print("WARNING: Slice %d/%d" % (i, self._N_slices-1))
                    print("\tsitk.CenteredTransformInitializerFilter with " +
                          transform_initializer_type_sitk +
                          " does not work. Identity transform is used instead for initialization")
                    initial_transform_sitk = \
                        self._new_transform_sitk[self._transform_type]()
                transforms_2D_sitk[i] = eval(
                    "sitk." + initial_transform_sitk.GetName() +
                    "(initial_transform_sitk)")

                # Get parameters
                parameters[i, :] = transforms_2D_sitk[i].GetParameters()

        return transforms_2D_sitk, parameters

    ##
    # Gets the initial intensity correction parameters.
    # \date       2016-11-10 02:38:17+0000
    #
    # \param      self  The object
    #
    # \return     The initial intensity correction parameters as (N_slices x
    #             DOF)-array with DOF being either 1 (linear) or 2 (affine)
    #
    def _get_initial_intensity_correction_parameters_None(self):

        # Set intensity correction parameters to identity
        if self._intensity_correction_type_slice_neighbour_fit in ["linear"]:
            return np.ones((self._N_slices, 1))

        # affine intensity correction type requires additional column (but set
        # to zero)
        elif self._intensity_correction_type_slice_neighbour_fit in ["affine"]:
            return np.concatenate((np.ones((self._N_slices, 1)),
                                   np.zeros((self._N_slices, 1))),
                                  axis=1)

    def _get_initial_intensity_correction_parameters_linear(self):

        if self._reference is None:
            print(
                "No reference given. Initial intensity correction parameters are set to identity")
            intensity_corrections_coefficients = \
                self._get_initial_intensity_correction_parameters_None()

        else:
            intensity_correction = ic.IntensityCorrection(
                stack=self._init_stack,
                reference=self._init_reference.get_resampled_stack_from_slices(
                    resampling_grid=self._init_stack.sitk),
                use_individual_slice_correction=True,
                use_verbose=False)
            intensity_correction.run_linear_intensity_correction()
            intensity_corrections_coefficients = intensity_correction.get_intensity_correction_coefficients()

            # affine intensity correction type requires additional column (but
            # set to zero)
            if self._intensity_correction_type_slice_neighbour_fit in ["affine"]:
                intensity_corrections_coefficients = np.concatenate(
                    (intensity_corrections_coefficients,
                        np.zeros((self._N_slices, 1))),
                    axis=1)

        return intensity_corrections_coefficients

    def _get_initial_intensity_correction_parameters_affine(self):

        if self._reference is not None:
            intensity_correction = ic.IntensityCorrection(
                stack=self._init_stack,
                reference=self._init_reference.get_resampled_stack_from_slices(
                    resampling_grid=self._init_stack.sitk),
                use_individual_slice_correction=False,
                use_verbose=False)
            intensity_correction.run_affine_intensity_correction()
            intensity_corrections_coefficients = \
                intensity_correction.get_intensity_correction_coefficients()

        else:
            print(
                "No reference given. Initial intensity correction parameters are set to identity")
            intensity_corrections_coefficients = np.ones((self._N_slices, 1))

        return intensity_corrections_coefficients

    ##
    # Correct intensity implementations
    # \date       2016-11-10 23:01:34+0000
    #
    # \param      self                     The object
    # \param      slice_nda                The slice nda
    # \param      correction_coefficients  The correction coefficients
    #
    # \return     intensity corrected slice / 2D data array
    #
    def _apply_intensity_correction_None(self,
                                         slice_nda,
                                         correction_coefficients):
        return slice_nda

    def _apply_intensity_correction_linear(self,
                                           slice_nda,
                                           correction_coefficients):
        return slice_nda * correction_coefficients[0]

    def _apply_intensity_correction_affine(self,
                                           slice_nda,
                                           correction_coefficients):
        return slice_nda * correction_coefficients[0] + \
            correction_coefficients[1]

    ##
    # Adds the Jacobian w.r.t to the intensity correction coefficients
    # depending on the chosen correction model to the existing Jacobian.
    # \date       2016-11-21 19:47:41+0000
    #
    # \param      self            The object
    # \param      jacobian_slice  The jacobian slice
    # \param      slice_sitk      The slice sitk
    # \param      mask_nda        The mask nda
    #
    # \return     Jacobian including intensity correction parameters
    #
    def _add_gradient_with_respect_to_intensity_correction_parameters_None(
            self,
            jacobian_slice_nda,
            slice_nda):
        return jacobian_slice_nda

    def _add_gradient_with_respect_to_intensity_correction_parameters_linear(
            self,
            jacobian_slice_nda,
            slice_nda):

        # Add the Jacobian w.r.t. intensity correction parameter (slope) to
        # existing Jacobian
        jacobian_slice_nda = np.concatenate(
            (jacobian_slice_nda, slice_nda.reshape(self._N_slice_voxels, -1)),
            axis=1)

        return jacobian_slice_nda

    def _add_gradient_with_respect_to_intensity_correction_parameters_affine(
            self,
            jacobian_slice_nda,
            slice_nda):

        # Add the Jacobian w.r.t. intensity correction parameter (slope) to
        # existing Jacobian
        jacobian_slice_nda = \
            self._add_gradient_with_respect_to_intensity_correction_parameters_linear(
                jacobian_slice_nda, slice_nda)

        # Add the Jacobian w.r.t. intensity correction parameter (bias) to
        # existing Jacobian
        jacobian_slice_nda = np.concatenate(
            (jacobian_slice_nda, np.ones((self._N_slice_voxels, 1))), axis=1)

        return jacobian_slice_nda

    ##
    # Gets the projected 2d slices of stack.
    # \date       2016-11-21 19:59:13+0000
    #
    # \param      self                     The object
    # \param      stack                    The stack
    # \param      image_transform_reference_fit_term  Either "identity" or "gradient_magnitude"
    #
    # \return     The projected 2d slices of stack.
    #
    def _get_projected_2D_slices_of_stack(self,
                                          stack,
                                          registration_image_type="identity"):

        slices_3D = stack.get_slices()
        slices_2D = [None]*self._N_slices

        if registration_image_type in ["partial_derivative"]:
            dy_slices_2D = [None]*self._N_slices

        for i in range(0, self._N_slices):

            # Create copy of the slices (since its header will be updated)
            slice_3D = sl.Slice.from_slice(slices_3D[i])

            # Get transform to get axis aligned slice of original stack
            # T_PP = self._get_TPP_transform(slice_3D.sitk)
            T_PP = self._get_TPP_transform(slices_3D[0].sitk)

            # Get current transform from image to physical space of slice
            T_PI = sitkh.get_sitk_affine_transform_from_sitk_image(
                slice_3D.sitk)

            # Get transform to align slice with physical coordinate system
            # (perhaps already shifted there)
            T_PI_align = sitkh.get_composite_sitk_affine_transform(T_PP, T_PI)

            # Set direction and origin of image accordingly
            origin_3D_sitk = \
                sitkh.get_sitk_image_origin_from_sitk_affine_transform(
                    T_PI_align, slice_3D.sitk)
            direction_3D_sitk = \
                sitkh.get_sitk_image_direction_from_sitk_affine_transform(
                    T_PI_align, slice_3D.sitk)

            slice_3D.sitk.SetDirection(direction_3D_sitk)
            slice_3D.sitk.SetOrigin(origin_3D_sitk)
            slice_3D.sitk_mask.SetDirection(direction_3D_sitk)
            slice_3D.sitk_mask.SetOrigin(origin_3D_sitk)

            # Get filename and slice number for name propagation
            filename = slice_3D.get_filename()
            slice_number = slice_3D.get_slice_number()

            slice_2D_sitk = slice_3D.sitk[:, :, 0]
            slice_2D_sitk_mask = slice_3D.sitk_mask[:, :, 0]

            if registration_image_type in ["identity"]:

                slices_2D[i] = sl.Slice.from_sitk_image(
                    slice_2D_sitk,
                    filename=filename,
                    slice_number=slice_number,
                    slice_sitk_mask=slice_2D_sitk_mask)

            elif registration_image_type in ["gradient_magnitude"]:
                # print("Gradient magnitude of image")
                gradient_magnitude_slice_2D_sitk = \
                    self._gradient_magnitude_filter_sitk.Execute(slice_2D_sitk)

                slices_2D[i] = sl.Slice.from_sitk_image(
                    gradient_magnitude_slice_2D_sitk,
                    filename="GradMagn_"+filename,
                    slice_number=slice_number,
                    slice_sitk_mask=slice_2D_sitk_mask)

            elif registration_image_type in ["partial_derivative"]:
                # print("Partial derivatives of image")

                dx_slice_2D_sitk = self._get_dx_image_sitk(slice_2D_sitk)
                dy_slice_2D_sitk = self._get_dy_image_sitk(slice_2D_sitk)

                slices_2D[i] = sl.Slice.from_sitk_image(
                    dx_slice_2D_sitk,
                    dir_input=None,
                    filename="dx_"+filename,
                    slice_number=slice_number,
                    slice_sitk_mask=slice_2D_sitk_mask)
                dy_slices_2D[i] = sl.Slice.from_sitk_image(
                    dy_slice_2D_sitk,
                    dir_input=None,
                    filename="dy_"+filename,
                    slice_number=slice_number,
                    slice_sitk_mask=slice_2D_sitk_mask)

                # Debug
                # sitkh.show_sitk_image([slice_3D.sitk[:,:,0],slice_2D_sitk], title=["standard_slice"+str(i), "gradient_magnitude_slice"+str(i)])
                # ph.pause()
                # ph.killall_itksnap()

        if registration_image_type in ["partial_derivative"]:
            return slices_2D, dy_slices_2D
        else:
            return slices_2D

    ##
    # Get the 3D rigid transforms to arrive at the positions of original 3D
    # slices starting from the physically aligned space with the main image
    # axes.
    # \date       2016-09-20 23:37:05+0100
    #
    # The rigid transform is given as composed translation and rotation
    # transform, i.e. T_PP = (T_t \c irc T_rot)^{-1}.
    #
    # \param      self  The object
    #
    # \return     List of 3D rigid transforms (sitk.AffineTransform(3) objects)
    #             to arrive at the positions of the original 3D slices.
    #
    # TODO: Change to make simpler
    #
    def _get_TPP_transform(self, slice_sitk):

        origin_3D_sitk = np.array(slice_sitk.GetOrigin())
        direction_3D_sitk = np.array(slice_sitk.GetDirection())
        T_PP = sitk.AffineTransform(3)
        T_PP.SetMatrix(direction_3D_sitk)
        T_PP.SetTranslation(origin_3D_sitk)
        T_PP = sitk.AffineTransform(T_PP.GetInverse())

        return T_PP

    """
    Transform specific parts from here
    """

    def _new_rigid_transform_sitk(self):
        return sitk.Euler2DTransform()

    def _new_rigid_transform_itk(self):
        return itk.Euler2DTransform.New()

    def _new_similarity_transform_sitk(self):
        return sitk.Similarity2DTransform()

    def _new_similarity_transform_itk(self):
        return itk.Similarity2DTransform.New()

    def _new_affine_transform_sitk(self):
        return sitk.AffineTransform(2)

    def _new_affine_transform_itk(self):
        return itk.AffineTransform.D2.New()

    ##
    # Perform motion correction based on performed registration to get motion
    # corrected stack and associated slice transforms.
    # \date       2016-11-21 20:11:53+0000
    #
    # \param      self  The object
    # \post       self._stack_corrected updated
    # \post       self._slice_transforms_sitk updated
    #
    def _apply_motion_correction(self):
        self._apply_motion_correction_and_compute_slice_transforms[
            self._transform_type]()

    ##
    # Apply motion correction after rigid registration
    # \date       2016-11-21 20:14:05+0000
    #
    # \param      self  The object
    # \post       self._stack_corrected updated
    # \post       self._slice_transforms_sitk updated
    #
    def _apply_rigid_motion_correction_and_compute_slice_transforms(self):

        stack_corrected = st.Stack.from_stack(self._stack)
        slices_corrected = stack_corrected.get_slices()

        slices = self._stack.get_slices()

        slice_transforms_sitk = [None] * self._N_slices

        for i in range(0, self._N_slices):

            # Set transform for the 2D slice based on registration transform
            self._transforms_2D_sitk[i].SetParameters(
                self._parameters[i, 0:self._transform_type_dofs])

            # Invert it to physically move the slice
            transform_2D_sitk = sitk.Euler2DTransform(
                self._transforms_2D_sitk[i].GetInverse())

            # Expand to 3D transform
            transform_3D_sitk = self._get_3D_from_2D_rigid_transform_sitk(
                transform_2D_sitk)

            # Get transform to get axis aligned slice
            # T_PP = self._get_TPP_transform(slices[i].sitk)
            T_PP = self._get_TPP_transform(slices[0].sitk)

            # Compose to 3D in-plane transform
            affine_transform_sitk = sitkh.get_composite_sitk_affine_transform(
                transform_3D_sitk, T_PP)
            affine_transform_sitk = sitkh.get_composite_sitk_affine_transform(
                sitk.AffineTransform(T_PP.GetInverse()), affine_transform_sitk)

            # Update motion correction of slice
            slices_corrected[i].update_motion_correction(affine_transform_sitk)

            # Keep slice transform
            slice_transforms_sitk[i] = affine_transform_sitk

        self._stack_corrected = stack_corrected
        self._slice_transforms_sitk = slice_transforms_sitk

    ##
    # Apply motion correction after similarity registration
    # \date       2016-11-21 20:14:42+0000
    #
    # \param      self  The object
    # \post       self._stack_corrected updated
    # \post       self._slice_transforms_sitk updated
    # \return     { description_of_the_return_value }
    #
    def _apply_similarity_motion_correction_and_compute_slice_transforms(self):

        stack_corrected = st.Stack.from_stack(self._stack)
        slices_corrected = stack_corrected.get_slices()

        slices = self._stack.get_slices()

        slice_transforms_sitk = [None] * self._N_slices

        for i in range(0, self._N_slices):

            # Set transform for the 2D slice based on registration transform
            self._transforms_2D_sitk[i].SetParameters(
                self._parameters[i, 0:self._transform_type_dofs])

            # Invert it to physically move the slice
            similarity_2D_sitk = sitk.Similarity2DTransform(
                self._transforms_2D_sitk[i].GetInverse())

            # Convert to 2D rigid registration transform
            scale = similarity_2D_sitk.GetScale()
            origin = np.array(self._slices_2D[i].sitk.GetOrigin())
            center = np.array(similarity_2D_sitk.GetCenter())
            angle = similarity_2D_sitk.GetAngle()
            translation = np.array(similarity_2D_sitk.GetTranslation())
            R = np.array(similarity_2D_sitk.GetMatrix()).reshape(2, 2)/scale

            # if self._use_verbose:
            #     print("Slice %2d/%d: in-plane scaling factor = %.3f" %(i, self._N_slices-1, 1/scale))

            rigid_2D_sitk = sitk.Euler2DTransform()
            rigid_2D_sitk.SetAngle(angle)
            rigid_2D_sitk.SetTranslation(
                scale*R.dot(origin-center) - R.dot(origin) + translation + center)

            # Expand to 3D rigid transform
            rigid_3D_sitk = self._get_3D_from_2D_rigid_transform_sitk(
                rigid_2D_sitk)

            # Get transform to get axis aligned slice
            # T_PP = self._get_TPP_transform(slices[i].sitk)
            T_PP = self._get_TPP_transform(slices[0].sitk)

            # Compose to 3D in-plane transform
            affine_transform_sitk = sitkh.get_composite_sitk_affine_transform(
                rigid_3D_sitk, T_PP)
            affine_transform_sitk = sitkh.get_composite_sitk_affine_transform(
                sitk.AffineTransform(T_PP.GetInverse()), affine_transform_sitk)

            # Update motion correction of slice
            slices_corrected[i].update_motion_correction(affine_transform_sitk)

            # Update spacing of slice accordingly
            spacing = np.array(slices[i].sitk.GetSpacing())
            spacing[0:-1] *= scale

            slices_corrected[i].sitk.SetSpacing(spacing)
            slices_corrected[i].sitk_mask.SetSpacing(spacing)
            slices_corrected[i].itk = sitkh.get_itk_from_sitk_image(
                slices_corrected[i].sitk)
            slices_corrected[i].itk_mask = \
                sitkh.get_itk_from_sitk_image(slices_corrected[i].sitk_mask)

            # Update affine transform (including scaling information)
            affine_3D_sitk = sitk.AffineTransform(3)
            affine_matrix_sitk = np.array(
                rigid_3D_sitk.GetMatrix()).reshape(3, 3)
            affine_matrix_sitk[0:-1, 0:-1] *= scale
            affine_3D_sitk.SetMatrix(affine_matrix_sitk.flatten())
            affine_3D_sitk.SetCenter(rigid_3D_sitk.GetCenter())
            affine_3D_sitk.SetTranslation(rigid_3D_sitk.GetTranslation())

            affine_3D_sitk = sitkh.get_composite_sitk_affine_transform(
                affine_3D_sitk, T_PP)
            affine_3D_sitk = sitkh.get_composite_sitk_affine_transform(
                sitk.AffineTransform(T_PP.GetInverse()), affine_3D_sitk)

            # Keep affine slice transform
            slice_transforms_sitk[i] = affine_3D_sitk

        self._stack_corrected = stack_corrected
        self._slice_transforms_sitk = slice_transforms_sitk

    ##
    # Apply motion correction after affine registration
    # \date       2016-11-21 20:14:05+0000
    #
    # \param      self  The object
    # \post       self._stack_corrected updated
    # \post       self._slice_transforms_sitk updated
    #
    def _apply_affine_motion_correction_and_compute_slice_transforms(self):

        stack_corrected = st.Stack.from_stack(self._stack)
        slices_corrected = stack_corrected.get_slices()

        slices = self._stack.get_slices()

        slice_transforms_sitk = [None] * self._N_slices

        for i in range(0, self._N_slices):

            # Set transform for the 2D slice based on registration transform
            self._transforms_2D_sitk[i].SetParameters(
                self._parameters[i, 0:self._transform_type_dofs])

            # Invert it to physically move the slice
            transform_2D_sitk = sitk.AffineTransform(
                self._transforms_2D_sitk[i].GetInverse())

            # Expand to 3D transform
            transform_3D_sitk = self._get_3D_from_2D_affine_transform_sitk(
                transform_2D_sitk)

            # Get transform to get axis aligned slice
            # T_PP = self._get_TPP_transform(slices[i].sitk)
            T_PP = self._get_TPP_transform(slices[0].sitk)

            # Compose to 3D in-plane transform
            affine_transform_sitk = sitkh.get_composite_sitk_affine_transform(
                transform_3D_sitk, T_PP)
            affine_transform_sitk = sitkh.get_composite_sitk_affine_transform(
                sitk.AffineTransform(T_PP.GetInverse()), affine_transform_sitk)

            # Update motion correction of slice
            slices_corrected[i].update_motion_correction(affine_transform_sitk)

            # Keep slice transform
            slice_transforms_sitk[i] = affine_transform_sitk

        self._stack_corrected = stack_corrected
        self._slice_transforms_sitk = slice_transforms_sitk

    ##
    # Create 3D from 2D transform.
    # \date       2016-09-20 23:18:55+0100
    #
    # The generated 3D transform performs in-plane operations in case the
    # physical coordinate system is aligned with the axis of the stack/slice
    #
    # \param      self                     The object
    # \param      rigid_transform_2D_sitk  sitk.Euler2DTransform object
    #
    # \return     sitk.Euler3DTransform object.
    #
    def _get_3D_from_2D_rigid_transform_sitk(self, rigid_transform_2D_sitk):

        # Get parameters of 2D registration
        angle_z, translation_x, translation_y = \
            rigid_transform_2D_sitk.GetParameters()
        center_x, center_y = rigid_transform_2D_sitk.GetCenter()

        # Expand obtained translation to 3D vector
        translation_3D = (translation_x, translation_y, 0)
        center_3D = (center_x, center_y, 0)

        # Create 3D rigid transform based on 2D
        rigid_transform_3D = sitk.Euler3DTransform()
        rigid_transform_3D.SetRotation(0, 0, angle_z)
        rigid_transform_3D.SetTranslation(translation_3D)

        # Append zero for m_ComputeZYX = 0 (part of fixed params in SimpleITK
        # 1.0.0)
        rigid_transform_3D.SetFixedParameters(center_3D + (0.,))

        return rigid_transform_3D

    ##
    # Create 3D from 2D transform.
    # \date       2016-09-20 23:18:55+0100
    #
    # The generated 3D transform performs in-plane operations in case the
    # physical coordinate system is aligned with the axis of the stack/slice
    #
    # \param      self                     The object
    # \param      rigid_transform_2D_sitk  sitk.Euler2DTransform object
    #
    # \return     sitk.Euler3DTransform object.
    #
    def _get_3D_from_2D_affine_transform_sitk(self, affine_transform_2D_sitk):

        # Get parameters of 2D registration
        a00, a01, a10, a11, translation_x, translation_y = \
            affine_transform_2D_sitk.GetParameters()
        center_x, center_y = affine_transform_2D_sitk.GetCenter()

        # Expand obtained translation to 3D vector
        translation_3D = (translation_x, translation_y, 0)
        center_3D = (center_x, center_y, 0)
        matrix_3D = np.eye(3).flatten()
        matrix_3D[0] = a00
        matrix_3D[1] = a01
        matrix_3D[3] = a10
        matrix_3D[4] = a11

        # Create 3D affine transform based on 2D
        affine_transform_3D = sitk.AffineTransform(3)
        affine_transform_3D.SetMatrix(matrix_3D)
        affine_transform_3D.SetTranslation(translation_3D)
        affine_transform_3D.SetFixedParameters(center_3D)

        return affine_transform_3D
