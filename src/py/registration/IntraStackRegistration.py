#!/usr/bin/python

##
# \file IntraStackRegistration.py
# \brief      Abstract class used for intra-stack registration steps. Slices
#             are only transformed in-plane. Hence, only 2D transforms are
#             applied
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Nov 2016
#


## Import libraries
import sys
import SimpleITK as sitk
import itk
import numpy as np
from scipy.optimize import least_squares
import time
from datetime import timedelta

## Import modules
import base.Slice as sl
import base.Stack as st
import utilities.SimpleITKHelper as sitkh
import utilities.IntensityCorrection as ic
import utilities.PythonHelper as ph
import utilities.ParameterNormalization as pn
from registration.StackRegistrationBase import StackRegistrationBase


class IntraStackRegistration(StackRegistrationBase):

    def __init__(self, stack=None, reference=None, use_stack_mask=False, use_reference_mask=False, use_verbose=False, initializer_type="identity", interpolator="Linear", alpha_neighbour=1, alpha_reference=1, alpha_parameter=0, transform_type="rigid", intensity_correction_type=None):

        ## Run constructor of superclass
        StackRegistrationBase.__init__(self, stack=stack, reference=reference, use_stack_mask=use_stack_mask, use_reference_mask=use_reference_mask, use_verbose=use_verbose, initializer_type=initializer_type, interpolator=interpolator, alpha_neighbour=alpha_neighbour, alpha_reference=alpha_reference, alpha_parameter=alpha_parameter)

        self._transform_type = transform_type

        self._intensity_correction_type = intensity_correction_type
        self._correct_intensity = {
            None        :  self._correct_intensity_None,
            "linear"    :  self._correct_intensity_linear,
            "affine"    :  self._correct_intensity_affine,
        }

        self._apply_motion_correction_and_compute_slice_transforms = {
            "rigid"     :  self._apply_rigid_motion_correction_and_compute_slice_transforms,    
            "similarity":  self._apply_similarity_motion_correction_and_compute_slice_transforms
        }

        self._new_transform = {
            "rigid"     :   self._new_rigid_transform,
            "similarity":   self._new_similarity_transform
        }

    ##
    #       Sets the transform type.
    # \date       2016-11-10 01:53:58+0000
    #
    # \param      self            The object
    # \param      transform_type  The transform type
    #
    #
    def set_transform_type(self, transform_type):
        if transform_type not in ["rigid", "similarity"]:
            raise ErrorValue("Transform type must either be 'rigid' or 'similarity'")
        self._transform_type = transform_type

    def get_transform_type(self):
        return self._transform_type


    ##
    #       { function_description }
    # \date       2016-11-10 01:58:39+0000
    #
    # \param      self  The object
    # \param      flag  The flag
    #
    # \return     { description_of_the_return_value }
    #
    def set_intensity_correction_type(self, intensity_correction_type):
        if intensity_correction_type not in [None, "linear", "affine"]:
            raise ErrorValue("Transform type must either be None, 'linear' or 'affine'")
        self._intensity_correction_type = intensity_correction_type


    def get_intensity_correction_type(self):
        return self._intensity_correction_type


    ##
    #       { function_description }
    # \date       2016-11-08 14:59:26+0000
    #
    # \param      self  The object
    #
    # \return     { description_of_the_return_value }
    #
    def _run_registration_pipeline_initialization(self):

        self._transform_type_dofs = len(self._new_transform[self._transform_type]().GetParameters())

        ## Get number of voxels in the x-y image plane
        self._N_slice_voxels = self._stack.sitk.GetWidth() * self._stack.sitk.GetHeight()
        
        ## Get projected 2D slices onto x-y image plane
        self._slices_2D = self._get_projected_2D_slices_of_stack(self._stack)

        ## If reference is given, precompute required data
        if self._reference is not None:
            self._slices_2D_reference = self._get_projected_2D_slices_of_stack(self._reference)
            self._reference_nda = sitk.GetArrayFromImage(self._reference.sitk)
            self._reference_nda_mask = sitk.GetArrayFromImage(self._reference.sitk_mask)

        ## Get inital transform and the respective initial transform parameters
        ## used for further optimisation
        self._transforms_2D_sitk, parameters = self._get_initial_transforms_and_parameters[self._initializer_type]()

        if self._intensity_correction_type is not None:
            parameters_intensity = self._get_initial_intensity_correction_parameters()
            parameters = np.concatenate((parameters, parameters_intensity), axis=1)

        if self._use_verbose:
            print("Initial values = ")
            print parameters
            # for i in range(0, self._N_slices):
                # print self._transforms_2D_sitk[i].GetParameters()

        ## Parameters for initialization and for regularization term
        self._parameters0_vec = parameters.flatten()

        ## Create copy for member variable
        self._parameters = np.array(parameters)

        ## Store number of degrees of freedom for overall optimization
        self._optimization_dofs = self._parameters.shape[1]

        ## Resampling grid, i.e. the fixed image space during registration
        self._slice_grid_2D_sitk = sitk.Image(self._slices_2D[0].sitk)


    def _apply_motion_correction(self):
        self._apply_motion_correction_and_compute_slice_transforms[self._transform_type]()


    def _get_residual_call(self):

        alpha_neighbour = self._alpha_neighbour
        alpha_parameter = self._alpha_parameter
        alpha_reference = self._alpha_reference

        if self._reference is None:
            if alpha_neighbour <= 0:
                raise ErrorValue("A weight of alpha_neighbour <= 0 is not meaningful.")

            if alpha_parameter is 0:
                residual = lambda x: self._get_residual_slice_neighbours_fit(x)

            else:
                residual = lambda x: np.concatenate((
                          self._get_residual_slice_neighbours_fit(x),
                          alpha_parameter/alpha_neighbour * self._get_residual_regularization(x)
                    ))
            
        else:

            if alpha_reference <= 0:
                raise ErrorValue("A weight of alpha_reference <= 0 is not meaningful in case reference is given")
            
            if alpha_neighbour is 0 and alpha_parameter is 0:
                residual = lambda x: self._get_residual_reference_fit(x)
            
            elif alpha_neighbour > 0 and alpha_parameter is 0:
                residual = lambda x: np.concatenate((
                          self._get_residual_reference_fit(x),
                          alpha_neighbour/alpha_reference * self._get_residual_slice_neighbours_fit(x)
                    ))
            
            elif alpha_neighbour is 0 and alpha_parameter > 0:
                residual = lambda x: np.concatenate((
                          self._get_residual_reference_fit(x),
                          alpha_paramete/ralpha_reference * self._get_residual_regularization(x)
                    ))
            

            elif alpha_neighbour > 0 and alpha_parameter > 0:
                residual = lambda x: np.concatenate((
                          self._get_residual_reference_fit(x),
                          alpha_neighbour/alpha_reference * self._get_residual_slice_neighbours_fit(x),
                          alpha_parameter/alpha_reference * self._get_residual_regularization(x)
                    ))

        return residual


    def _get_residual_regularization(self, parameters_vec):
        return parameters_vec - self._parameters0_vec


    def _get_residual_slice_neighbours_fit(self, parameters_vec):

        ## Allocate memory for residual
        residual = np.zeros((self._N_slices-1, self._N_slice_voxels))
        
        ## Reshape parameters for easier access
        parameters = parameters_vec.reshape(-1, self._optimization_dofs)            

        ## Get slice_i(T(theta_i, x))
        self._transforms_2D_sitk[0].SetParameters(parameters[0,0:self._transform_type_dofs])
        slice_i_sitk = sitk.Resample(self._slices_2D[0].sitk, self._slice_grid_2D_sitk, self._transforms_2D_sitk[0], self._interpolator_sitk)
        slice_i_nda = sitk.GetArrayFromImage(slice_i_sitk)

        if self._use_stack_mask:
            slice_i_sitk_mask = sitk.Resample(self._slices_2D[0].sitk_mask, self._slice_grid_2D_sitk, self._transforms_2D_sitk[0], sitk.sitkNearestNeighbor)
            slice_i_nda_mask = sitk.GetArrayFromImage(slice_i_sitk_mask)

        ## Compute residuals for neighbouring slices
        for i in range(0, self._N_slices-1):

            ## Get slice_{i+1}(T(theta_{i+1}, x))
            self._transforms_2D_sitk[i+1].SetParameters(parameters[i+1,0:self._transform_type_dofs])
            slice_ip1_sitk = sitk.Resample(self._slices_2D[i+1].sitk, self._slice_grid_2D_sitk, self._transforms_2D_sitk[i+1], self._interpolator_sitk)
            slice_ip1_nda = sitk.GetArrayFromImage(slice_ip1_sitk)

            ## Correct intensities according to chosen model
            slice_i_nda = self._correct_intensity[self._intensity_correction_type](slice_i_nda, parameters[i, self._transform_type_dofs:])
            slice_ip1_nda = self._correct_intensity[self._intensity_correction_type](slice_ip1_nda, parameters[i+1, self._transform_type_dofs:])

            ## Compute residual slice_i(T(theta_i, x)) - slice_{i+1}(T(theta_{i+1}, x))
            residual_slice_nda = slice_i_nda - slice_ip1_nda
            
            ## Eliminate residual for non-masked regions
            if self._use_stack_mask:
                slice_ip1_sitk_mask = sitk.Resample(self._slices_2D[i+1].sitk_mask, self._slice_grid_2D_sitk, self._transforms_2D_sitk[i+1], sitk.sitkNearestNeighbor)
                slice_ip1_nda_mask = sitk.GetArrayFromImage(slice_ip1_sitk_mask)

                residual_slice_nda = residual_slice_nda * slice_i_nda_mask * slice_ip1_nda_mask

                slice_i_nda_mask = slice_ip1_nda_mask

            ## Set residual for current slice difference
            residual[i,:] = residual_slice_nda.flatten()

            ## Prepare for next iteration
            slice_i_nda = slice_ip1_nda

        return residual.flatten()


    ##
    #       Gets the residual reference fit.
    # \date       2016-11-08 20:37:49+0000
    #
    # \param      self            The object
    # \param      parameters_vec  The parameters vector
    #
    # \return     The residual reference fit.
    #
    def _get_residual_reference_fit(self, parameters_vec):

        ## Allocate memory for residual
        residual = np.zeros((self._N_slices, self._N_slice_voxels))
        
        ## Reshape parameters for easier access
        parameters = parameters_vec.reshape(-1, self._optimization_dofs)

        ## Compute residuals between each slice and reference
        for i in range(0, self._N_slices):
            
            ## Get slice_i(T(theta_i, x))
            self._transforms_2D_sitk[i].SetParameters(parameters[i,0:self._transform_type_dofs])
            slice_i_sitk = sitk.Resample(self._slices_2D[i].sitk, self._slice_grid_2D_sitk, self._transforms_2D_sitk[i], self._interpolator_sitk)
            slice_i_nda = sitk.GetArrayFromImage(slice_i_sitk)

            ## Correct intensities according to chosen model
            slice_i_nda = self._correct_intensity[self._intensity_correction_type](slice_i_nda, parameters[i, self._transform_type_dofs:])

            ## Compute residual slice_i(T(theta_i, x)) - ref(x))
            residual_slice_nda = slice_i_nda - self._reference_nda[i,:,:]
            
            if self._use_stack_mask:
                slice_i_sitk_mask = sitk.Resample(self._slices_2D[i].sitk_mask, self._slice_grid_2D_sitk, self._transforms_2D_sitk[i], sitk.sitkNearestNeighbor)
                slice_i_nda_mask = sitk.GetArrayFromImage(slice_i_sitk_mask)

                residual_slice_nda *= slice_i_nda_mask

            if self._use_reference_mask:
                residual_slice_nda *= self._reference_nda_mask[i,:,:]

            # ph.plot_2D_array_list([residual_slice_nda, slice_i_nda_mask, self._reference_nda_mask[i,:,:]]) 
            # ph.pause()

            ## Set residual for current slice difference
            residual[i,:] = residual_slice_nda.flatten()

        return residual.flatten()


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
        
        ## Create list of identity transforms for all slices
        transforms_2D_sitk = [None] * self._N_slices
        
        ## Get list of identity transform parameters for all slices
        parameters = np.zeros((self._N_slices, self._transform_type_dofs))
        for i in range(0, self._N_slices):
            transforms_2D_sitk[i] = self._new_transform[self._transform_type]()
            parameters[i, :] = transforms_2D_sitk[i].GetParameters()

        return transforms_2D_sitk, parameters


    ##
    #       Gets the initial parameters for either 'GEOMETRY' or
    #             'MOMENTS'.
    # \date       2016-11-08 15:08:07+0000
    #
    # \param      self  The object
    #
    # \return     The initial parameters corresponding to 'GEOMETRY' or
    #             'MOMENTS' as (N_slices x DOF)-array
    #
    def _get_initial_transforms_and_parameters_geometry_moments(self):

        initializer_type_sitk = self._dictionary_initializer_type_sitk[self._initializer_type]

        ## Create list of identity transforms
        transforms_2D_sitk = [self._new_transform[self._transform_type]()] * self._N_slices

        ## Get list of identity transform parameters for all slices
        parameters = np.zeros((self._N_slices, self._transform_type_dofs))

        ## Set identity parameters for first slice
        parameters[0,:] = transforms_2D_sitk[0].GetParameters()

        ## No reference is given and slices are initialized to align with 
        ## neighbouring slice
        if self._reference is None:
            
            ## Create identity transform for first slice
            compensation_transform_sitk = self._new_transform[self._transform_type]()

            ## First slice is kept at position and others are aligned accordingly
            for i in range(1, self._N_slices):

                ## Take into account the initialization of slice i-1
                slice_im1_sitk = sitkh.get_transformed_sitk_image(self._slices_2D[i-1].sitk, compensation_transform_sitk)

                ## Use sitk.CenteredTransformInitializerFilter to get initial transform
                fixed_sitk = slice_im1_sitk
                moving_sitk = self._slices_2D[i].sitk
                initial_transform_sitk = self._new_transform[self._transform_type]()
                operation_mode_sitk = eval("sitk.CenteredTransformInitializerFilter." + initializer_type_sitk)
                
                ## Get transform
                initial_transform_sitk = sitk.CenteredTransformInitializer(fixed_sitk, moving_sitk, initial_transform_sitk, operation_mode_sitk)
                transforms_2D_sitk[i] = eval("sitk." + initial_transform_sitk.GetName() + "(initial_transform_sitk)")

                ## Get parameters
                parameters[i,:] = transforms_2D_sitk[i].GetParameters()

                ## Store compensation transform for subsequent slice
                compensation_transform_sitk.SetParameters(transforms_2D_sitk[i].GetParameters())
                compensation_transform_sitk.SetFixedParameters(transforms_2D_sitk[i].GetFixedParameters())
                compensation_transform_sitk = eval("sitk." + compensation_transform_sitk.GetName() + "(compensation_transform_sitk.GetInverse())")

        ## Initialize transform to match each slice with the reference
        else:
            for i in range(0, self._N_slices):

                ## Use sitk.CenteredTransformInitializerFilter to get initial transform
                fixed_sitk = self._slices_2D_reference[i].sitk
                moving_sitk = self._slices_2D[i].sitk
                initial_transform_sitk = self._new_transform[self._transform_type]()
                operation_mode_sitk = eval("sitk.CenteredTransformInitializerFilter." + initializer_type_sitk)
                
                ## Get transform
                initial_transform_sitk = sitk.CenteredTransformInitializer(fixed_sitk, moving_sitk, initial_transform_sitk, operation_mode_sitk)
                transforms_2D_sitk[i] = eval("sitk." + initial_transform_sitk.GetName() + "(initial_transform_sitk)")

                ## Get parameters
                parameters[i,:] = transforms_2D_sitk[i].GetParameters()
        
        return transforms_2D_sitk, parameters


    ##
    #       Gets the initial intensity correction parameters.
    # \date       2016-11-10 02:38:17+0000
    #
    # \param      self  The object
    #
    # \return     The initial intensity correction parameters as (N_slices x
    #             DOF)-array with DOF being either 1 (linear) or 2 (affine)
    #
    def _get_initial_intensity_correction_parameters(self):
        
        if self._reference is None:
            if self._intensity_correction_type in ["linear"]:
                return np.ones((self._N_slices,1))
            elif self._intensity_correction_type in ["affine"]:
                return np.ones((self._N_slices,2))

        else:
            intensity_correction = ic.IntensityCorrection(stack=self._stack, reference=self._reference, use_individual_slice_correction=True, use_verbose=False)

            if self._intensity_correction_type in ["linear"]:
                intensity_correction.run_linear_intensity_correction()

            elif self._intensity_correction_type in ["affine"]:
                intensity_correction.run_affine_intensity_correction()

            intensity_corrections_coefficients = intensity_correction.get_intensity_correction_coefficients()

            return intensity_corrections_coefficients


    ##
    #       Correct intensity implementations
    # \date       2016-11-10 23:01:34+0000
    #
    # \param      self                     The object
    # \param      slice_nda                The slice nda
    # \param      correction_coefficients  The correction coefficients
    #
    # \return     intensity corrected slice / 2D data array
    #
    def _correct_intensity_None(self, slice_nda, correction_coefficients):
        return slice_nda

    def _correct_intensity_linear(self, slice_nda, correction_coefficients):
        return slice_nda * correction_coefficients[0]

    def _correct_intensity_affine(self, slice_nda, correction_coefficients):
        return slice_nda * correction_coefficients[0] + correction_coefficients[1]


    def _get_projected_2D_slices_of_stack(self, stack):

        slices_3D = stack.get_slices()
        slices_2D = [None]*self._N_slices

        for i in range(0, self._N_slices):

            ## Create copy of the slices (since its header will be updated)
            slice_3D = sl.Slice.from_slice(slices_3D[i])

            ## Get transform to get axis aligned slice
            origin_3D_sitk = np.array(slice_3D.sitk.GetOrigin())
            direction_3D_sitk = np.array(slice_3D.sitk.GetDirection())
            T_PP = sitk.AffineTransform(3)
            T_PP.SetMatrix(direction_3D_sitk)
            T_PP.SetTranslation(origin_3D_sitk)
            T_PP = sitk.AffineTransform(T_PP.GetInverse())

            ## Get current transform from image to physical space of slice
            T_PI = sitkh.get_sitk_affine_transform_from_sitk_image(slice_3D.sitk)

            ## Get transform to align slice with physical coordinate system (perhaps already shifted there) 
            T_PI_align = sitkh.get_composite_sitk_affine_transform(T_PP, T_PI)

            ## Set direction and origin of image accordingly
            origin_3D_sitk = sitkh.get_sitk_image_origin_from_sitk_affine_transform(T_PI_align, slice_3D.sitk)
            direction_3D_sitk = sitkh.get_sitk_image_direction_from_sitk_affine_transform(T_PI_align, slice_3D.sitk)

            slice_3D.sitk.SetDirection(direction_3D_sitk)
            slice_3D.sitk.SetOrigin(origin_3D_sitk)
            slice_3D.sitk_mask.SetDirection(direction_3D_sitk)
            slice_3D.sitk_mask.SetOrigin(origin_3D_sitk)

            ## Get filename and slice number for name propagation
            filename = slice_3D.get_filename()
            slice_number = slice_3D.get_slice_number()

            slice_2D_sitk = slice_3D.sitk[:,:,0]
            slice_2D_sitk_mask = slice_3D.sitk_mask[:,:,0]

            slices_2D[i] = sl.Slice.from_sitk_image(slice_2D_sitk, dir_input=None, filename=filename, slice_number=slice_number, slice_sitk_mask=slice_2D_sitk_mask)

        return slices_2D

    """
    Transform specific parts from here
    """

    def _new_rigid_transform(self):
        return sitk.Euler2DTransform()

    def _new_similarity_transform(self):
        return sitk.Similarity2DTransform()

    def _apply_rigid_motion_correction_and_compute_slice_transforms(self):
        
        stack_corrected = st.Stack.from_stack(self._stack)
        slices_corrected = stack_corrected.get_slices()

        slices = self._stack.get_slices()

        slice_transforms_sitk = [None] * self._N_slices

        for i in range(0, self._N_slices):

            ## Set transform for the 2D slice based on registration transform
            self._transforms_2D_sitk[i].SetParameters(self._parameters[i,0:self._transform_type_dofs])

            ## Invert it to physically move the slice
            transform_2D_sitk = sitk.Euler2DTransform(self._transforms_2D_sitk[i].GetInverse())

            ## Expand to 3D transform
            transform_3D_sitk = self._get_3D_from_2D_rigid_transform_sitk(transform_2D_sitk)

            ## Get transform to get axis aligned slice
            origin_3D_sitk = np.array(slices[i].sitk.GetOrigin())
            direction_3D_sitk = np.array(slices[i].sitk.GetDirection())
            T_PP = sitk.AffineTransform(3)
            T_PP.SetMatrix(direction_3D_sitk)
            T_PP.SetTranslation(origin_3D_sitk)
            T_PP = sitk.AffineTransform(T_PP.GetInverse())

            ## Compose to 3D in-plane transform
            affine_transform_sitk = sitkh.get_composite_sitk_affine_transform(transform_3D_sitk, T_PP)
            affine_transform_sitk = sitkh.get_composite_sitk_affine_transform(sitk.AffineTransform(T_PP.GetInverse()), affine_transform_sitk)

            ## Update motion correction of slice
            slices_corrected[i].update_motion_correction(affine_transform_sitk)
            
            ## Keep slice transform
            slice_transforms_sitk[i] = affine_transform_sitk

        self._stack_corrected = stack_corrected
        self._slice_transforms_sitk = slice_transforms_sitk


    def _apply_similarity_motion_correction_and_compute_slice_transforms(self):

        stack_corrected = st.Stack.from_stack(self._stack)
        slices_corrected = stack_corrected.get_slices()

        slices = self._stack.get_slices()

        slice_transforms_sitk = [None] * self._N_slices

        for i in range(0, self._N_slices):

            ## Set transform for the 2D slice based on registration transform
            self._transforms_2D_sitk[i].SetParameters(self._parameters[i,0:self._transform_type_dofs])

            ## Invert it to physically move the slice
            similarity_2D_sitk = sitk.Similarity2DTransform(self._transforms_2D_sitk[i].GetInverse())

            ## Convert to 2D rigid registration transform
            scale = similarity_2D_sitk.GetScale()
            origin = np.array(self._slices_2D[i].sitk.GetOrigin())
            center = np.array(similarity_2D_sitk.GetCenter())
            angle = similarity_2D_sitk.GetAngle()
            translation = np.array(similarity_2D_sitk.GetTranslation())
            R = np.array(similarity_2D_sitk.GetMatrix()).reshape(2,2)/scale

            if self._use_verbose:
                print("Slice %2d/%d: in-plane scaling factor = %.3f" %(i, self._N_slices-1, scale))

            rigid_2D_sitk = sitk.Euler2DTransform()
            rigid_2D_sitk.SetAngle(angle)
            rigid_2D_sitk.SetTranslation(scale*R.dot(origin-center) - R.dot(origin) + translation + center)

            ## Expand to 3D rigid transform
            rigid_3D_sitk = self._get_3D_from_2D_rigid_transform_sitk(rigid_2D_sitk)

            ## Get transform to get axis aligned slice
            origin_3D_sitk = np.array(slices[i].sitk.GetOrigin())
            direction_3D_sitk = np.array(slices[i].sitk.GetDirection())
            T_PP = sitk.AffineTransform(3)
            T_PP.SetMatrix(direction_3D_sitk)
            T_PP.SetTranslation(origin_3D_sitk)
            T_PP = sitk.AffineTransform(T_PP.GetInverse())

            ## Compose to 3D in-plane transform
            affine_transform_sitk = sitkh.get_composite_sitk_affine_transform(rigid_3D_sitk, T_PP)
            affine_transform_sitk = sitkh.get_composite_sitk_affine_transform(sitk.AffineTransform(T_PP.GetInverse()), affine_transform_sitk)

            ## Update motion correction of slice
            slices_corrected[i].update_motion_correction(affine_transform_sitk)
            

            ## Update spacing of slice accordingly
            spacing = np.array(slices[i].sitk.GetSpacing())
            spacing[0:-1] *= scale

            slices_corrected[i].sitk.SetSpacing(spacing)
            slices_corrected[i].sitk_mask.SetSpacing(spacing)
            slices_corrected[i].itk = sitkh.get_itk_from_sitk_image(slices_corrected[i].sitk)
            slices_corrected[i].itk_mask = sitkh.get_itk_from_sitk_image(slices_corrected[i].sitk_mask)

            ## Update affine transform (including scaling information)
            affine_3D_sitk = sitk.AffineTransform(3)
            affine_matrix_sitk = np.array(rigid_3D_sitk.GetMatrix()).reshape(3,3)
            affine_matrix_sitk[0:-1,0:-1] *= scale
            affine_3D_sitk.SetMatrix(affine_matrix_sitk.flatten())
            affine_3D_sitk.SetCenter(rigid_3D_sitk.GetCenter())
            affine_3D_sitk.SetTranslation(rigid_3D_sitk.GetTranslation())

            affine_3D_sitk = sitkh.get_composite_sitk_affine_transform(affine_3D_sitk, T_PP)
            affine_3D_sitk = sitkh.get_composite_sitk_affine_transform(sitk.AffineTransform(T_PP.GetInverse()), affine_3D_sitk)

            ## Keep affine slice transform
            slice_transforms_sitk[i] = affine_3D_sitk

        self._stack_corrected = stack_corrected
        self._slice_transforms_sitk = slice_transforms_sitk


    ##
    #       Create 3D from 2D transform.
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
        angle_z, translation_x, translation_y = rigid_transform_2D_sitk.GetParameters()
        center_x, center_y = rigid_transform_2D_sitk.GetCenter()

        # Expand obtained translation to 3D vector
        translation_3D = (translation_x, translation_y, 0)
        center_3D = (center_x, center_y, 0)

        # Create 3D rigid transform based on 2D
        rigid_transform_3D = sitk.Euler3DTransform()
        rigid_transform_3D.SetRotation(0,0,angle_z)
        rigid_transform_3D.SetTranslation(translation_3D)
        rigid_transform_3D.SetFixedParameters(center_3D)
        
        return rigid_transform_3D
