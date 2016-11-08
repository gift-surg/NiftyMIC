#!/usr/bin/python

##-----------------------------------------------------------------------------
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
import base.PSF as psf
import base.Slice as sl
import base.Stack as st
import utilities.SimpleITKHelper as sitkh
from registration.StackRegistrationBase import StackRegistrationBase


## TODOs: 
##  - reference init
##  - regularization parameters
##  - different transforms


class IntraStackRegistration(StackRegistrationBase):

    def __init__(self, stack=None, reference=None, use_stack_mask=False, use_reference_mask=False, use_verbose=False, initializer_type="identity", interpolator="Linear", transform_type="rigid"):

        ## Run constructor of superclass
        StackRegistrationBase.__init__(self, stack=stack, reference=reference, use_stack_mask=use_stack_mask, use_reference_mask=use_reference_mask, use_verbose=use_verbose, initializer_type=initializer_type, interpolator=interpolator)

        if transform_type in ["rigid"]:
            self._transform_type_sitk_new = sitk.Euler2DTransform()
            self._transform_type_dofs = 3


    ##-------------------------------------------------------------------------
    # \brief      { function_description }
    # \date       2016-11-08 14:59:26+0000
    #
    # \param      self  The object
    #
    # \return     { description_of_the_return_value }
    #
    def _run_registration_pipeline_initialization(self):

        self._N_slice_voxels = self._stack.sitk.GetWidth() * self._stack.sitk.GetHeight()

        self._slices_2D = self._get_projected_2D_slices_of_stack(self._stack)


        self._parameters = self._get_initial_transform_parameters[self._initializer_type]()

        ## Grid used for resampling
        self._slice_grid_2D_sitk = sitk.Image(self._slices_2D[0].sitk)

        if self._reference is not None:
            self._reference_nda = sitk.GetArrayFromImage(self._reference.sitk)
            self._reference_nda_mask = sitk.GetArrayFromImage(self._reference.sitk_mask)

        ##
        self._transforms_2D_sitk = [self._transform_type_sitk_new] * self._N_slices


    def _get_residual_call(self):

        alpha = 0.5

        if self._reference is None:
            residual = lambda x: self._get_residual_slice_neighbours_fit(x)

        else:
            # residual = lambda x: self._get_residual_reference_fit(x)
            residual = lambda x: np.concatenate((
                      self._get_residual_reference_fit(x)
                    , alpha * self._get_residual_slice_neighbours_fit(x)
                ))

        return residual


    ## TODO: reference init
    def _get_residual_slice_neighbours_fit(self, parameters_vec):

        ## Allocate memory for residual
        residual = np.zeros((self._N_slices-1, self._N_slice_voxels))
        
        ## Reshape parameters for easier access
        parameters = parameters_vec.reshape(-1, self._transform_type_dofs)            

        ## Get slice_i(T(theta_i, x))
        self._transforms_2D_sitk[0].SetParameters(parameters[0,:])
        slice_i_sitk = sitk.Resample(self._slices_2D[0].sitk, self._slice_grid_2D_sitk, self._transforms_2D_sitk[0], self._interpolator_sitk)
        slice_i_nda = sitk.GetArrayFromImage(slice_i_sitk)

        if self._use_stack_mask:
            slice_i_sitk_mask = sitk.Resample(self._slices_2D[0].sitk_mask, self._slice_grid_2D_sitk, self._transforms_2D_sitk[0], sitk.sitkNearestNeighbor)
            slice_i_nda_mask = sitk.GetArrayFromImage(slice_i_sitk_mask)

        ## Compute residuals for neighbouring slices
        for i in range(0, self._N_slices-1):

            ## Get slice_{i+1}(T(theta_{i+1}, x))
            self._transforms_2D_sitk[i+1].SetParameters(parameters[i+1,:])
            slice_ip1_sitk = sitk.Resample(self._slices_2D[i+1].sitk, self._slice_grid_2D_sitk, self._transforms_2D_sitk[i+1], self._interpolator_sitk)
            slice_ip1_nda = sitk.GetArrayFromImage(slice_ip1_sitk)

            ## Compute residual slice_i(T(theta_i, x)) - slice_{i+1}(T(theta_{i+1}, x))
            residual_slice_nda = slice_i_nda - slice_ip1_nda

            slice_i_nda = slice_ip1_nda
            
            ## Eliminate residual for non-masked regions
            if self._use_stack_mask:
                slice_ip1_sitk_mask = sitk.Resample(self._slices_2D[i+1].sitk_mask, self._slice_grid_2D_sitk, self._transforms_2D_sitk[i+1], sitk.sitkNearestNeighbor)
                slice_ip1_nda_mask = sitk.GetArrayFromImage(slice_ip1_sitk_mask)

                residual_slice_nda = residual_slice_nda * slice_i_nda_mask * slice_ip1_nda_mask

                slice_i_nda_mask = slice_ip1_nda_mask

            ## Set residual for current slice difference
            residual[i,:] = residual_slice_nda.flatten()

        return residual.flatten()


    ##-------------------------------------------------------------------------
    # \brief      Gets the residual reference fit.
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
        parameters = parameters_vec.reshape(-1, self._transform_type_dofs)

        ## Compute residuals between each slice and reference
        for i in range(0, self._N_slices):
            
            ## Get slice_i(T(theta_i, x))
            self._transforms_2D_sitk[i].SetParameters(parameters[i,:])
            slice_i_sitk = sitk.Resample(self._slices_2D[i].sitk, self._slice_grid_2D_sitk, self._transforms_2D_sitk[i], self._interpolator_sitk)
            slice_i_nda = sitk.GetArrayFromImage(slice_i_sitk)

            ## Compute residual slice_i(T(theta_i, x)) - ref(x))
            residual_slice_nda = slice_i_nda - self._reference_nda[i,:,:]
            
            if self._use_stack_mask:
                slice_i_sitk_mask = sitk.Resample(self._slices_2D[i].sitk_mask, self._slice_grid_2D_sitk, self._transforms_2D_sitk[i], sitk.sitkNearestNeighbor)
                slice_i_nda_mask = sitk.GetArrayFromImage(slice_i_sitk_mask)

                residual_slice_nda *= slice_i_nda_mask

            if self._use_reference_mask:
                residual_slice_nda *= self._reference_nda_mask[i,:,:]

            ## Set residual for current slice difference
            residual[i,:] = residual_slice_nda.flatten()

        return residual.flatten()


    ##-------------------------------------------------------------------------
    # \brief      Gets the initial parameters for 'None', i.e. for identity
    #             transform.
    # \date       2016-11-08 15:06:54+0000
    #
    # \param      self  The object
    #
    # \return     The initial parameters corresponding to identity transform as
    #             (N_slices x DOF)-array
    #
    def _get_initial_transform_parameters_identity(self):
        return np.zeros((self._N_slices, self._transform_type_dofs))


    ##-------------------------------------------------------------------------
    # \brief      Gets the initial parameters for either 'GEOMETRY' or
    #             'MOMENTS'.
    # \date       2016-11-08 15:08:07+0000
    #
    # \param      self  The object
    #
    # \return     The initial parameters corresponding to 'GEOMETRY' or
    #             'MOMENTS' as (N_slices x DOF)-array
    #
    def _get_initial_transform_parameters_geometry_moments(self):

        initializer_type_sitk = self._dictionary_initializer_type_sitk[self._initializer_type]

        ## Get identity transform parameters for all slices
        parameters = np.zeros((self._N_slices, self._transform_type_dofs))

        ## No reference is given and slices are initialized to align with 
        ## neighbouring slice
        if self._reference is None:
            
            ## Create identity transform for first slice
            compensation_transform_sitk = self._transform_type_sitk_new

            ## First slice is kept at position and others are aligned accordingly
            for i in range(1, self._N_slices):

                ## Take into account the initialization of slice i-1
                slice_im1_sitk = sitkh.get_transformed_sitk_image(self._slices_2D[i-1].sitk, compensation_transform_sitk)

                ## Use sitk.CenteredTransformInitializerFilter to get initial transform
                fixed_sitk = slice_im1_sitk
                moving_sitk = self._slices_2D[i].sitk
                initial_transform_sitk = self._transform_type_sitk_new
                operation_mode_sitk = eval("sitk.CenteredTransformInitializerFilter." + initializer_type_sitk)
                
                initial_transform_sitk = sitk.CenteredTransformInitializer(fixed_sitk, moving_sitk, initial_transform_sitk, operation_mode_sitk)

                ## Get parameters
                parameters[i,:] = initial_transform_sitk.GetParameters()

                ## Store compensation transform for subsequent slice
                compensation_transform_sitk.SetParameters(parameters[i,:])
                compensation_transform_sitk = eval("sitk." + compensation_transform_sitk.GetName() + "(compensation_transform_sitk.GetInverse())")


            return parameters
        ## TODO
        else:
            return np.zeros((self._N_slices, self._transform_type_dofs))
            

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


    def _apply_motion_correction_and_compute_slice_transforms(self):
        
        stack_corrected = st.Stack.from_stack(self._stack)
        slices_corrected = stack_corrected.get_slices()

        slices = self._stack.get_slices()

        transform_2D_sitk = self._transform_type_sitk_new
        slice_transforms_sitk = [None] * self._N_slices

        for i in range(0, self._N_slices):

            ## Set transform for the 2D slice based on registration transform
            transform_2D_sitk.SetParameters(self._parameters[i,:])

            ## Invert it to physically move the slice
            transform_2D_sitk = sitk.Euler2DTransform(transform_2D_sitk.GetInverse())
            # transform_2D_sitk = eval("sitk." + transform_2D_sitk.GetName() + "(transform_2D_sitk.GetInverse())")

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
        self._registration_transforms_sitk = slice_transforms_sitk



    ##-------------------------------------------------------------------------
    # \brief      Create 3D from 2D transform.
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
