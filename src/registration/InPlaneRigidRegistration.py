## \file InPlaneRigidRegistration.py
#  \brief  
# 
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date September 2016


## Import libraries
import os                       # used to execute terminal commands in python
import sys
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

## Add directories to import modules
DIR_SRC_ROOT = "../../src/"
sys.path.append(DIR_SRC_ROOT + "base/")
sys.path.append(DIR_SRC_ROOT + "registration/")

## Import modules from src-folder
import SimpleITKHelper as sitkh
import Stack as st
import Slice as sl
import RegistrationSimpleITK as regsitk

##-----------------------------------------------------------------------------
# \brief      Class to perform in-plane rigid registration
# \date       2016-09-20 15:59:21+0100
#
class InPlaneRigidRegistration:

    def __init__(self):
        self._registration_2D = regsitk.RegistrationSimpleITK()
        self._registration_2D.set_registration_type("Rigid")
        # self._registration_2D.use_multiresolution_framework(True)
        self._registration_2D.set_centered_transform_initializer(None)
        self._registration_2D.set_optimizer_scales_from("PhysicalShift")
        # self._registration_2D.set_metric("MattesMutualInformation")
        # self._registration_2D.set_metric("MeanSquares")
        self._registration_2D.set_metric("Correlation")
        self._registration_2D.use_fixed_mask(True)
        self._registration_2D.use_moving_mask(True)
        self._registration_2D.use_verbose(False)


    ##-------------------------------------------------------------------------
    # \brief      Sets the stack.
    # \date       2016-09-20 22:30:54+0100
    #
    # \param      self   The object
    # \param      stack  stack as Stack object
    #
    def set_stack(self, stack):
        self._stack = st.Stack.from_stack(stack)
        self._slices = self._stack.get_slices()
        self._N_slices = self._stack.get_number_of_slices()


    ##-------------------------------------------------------------------------
    # \brief      Gets the stack.
    # \date       2016-09-20 22:38:53+0100
    #
    # \param      self  The object
    #
    # \return     The stack as Stack object.
    #
    def get_stack(self):
        return self._stack


    ##-------------------------------------------------------------------------
    # \brief      Run in-plane rigid registration algorithm
    # \date       2016-09-21 02:19:31+0100
    #
    # \param      self  The object
    #
    def run_registration(self):

        ## Get list of 3D affine transforms to arrive at the positions of the
        ## original 3D slices
        transforms_PP_3D_sitk = self._get_list_of_3D_rigid_transforms_of_slices()

        # for i in range(self._N_slices-2,-1,-1):
            # slice_2D_moving = self._get_2D_slice(self._slices[i+1], transforms_PP_3D_sitk[i+1])
        for i in range(1, self._N_slices):
            slice_2D_moving = self._get_2D_slice(self._slices[i-1], transforms_PP_3D_sitk[i-1])

            slice_2D_fixed  = self._get_2D_slice(self._slices[i], transforms_PP_3D_sitk[i])

            # print("slice_2D_fixed.sitk.GetDirection() = " + str(slice_2D_fixed.sitk.GetDirection()))
            # print("slice_2D_moving.sitk.GetDirection() = " + str(slice_2D_moving.sitk.GetDirection()))

            ## Perform in-plane rigid registration
            self._registration_2D.set_fixed(slice_2D_fixed)
            self._registration_2D.set_moving(slice_2D_moving)
            self._registration_2D.run_registration()

            rigid_registration_transform_2D_sitk = self._registration_2D.get_registration_transform_sitk()

            # foo_2D_sitk = sitkh.get_transformed_image(slice_2D_fixed.sitk, rigid_registration_transform_2D_sitk)
            # foo_2D_sitk = sitk.Resample(foo_2D_sitk, slice_2D_moving.sitk, sitk.Euler2DTransform(), sitk.sitkNearestNeighbor, 0.0, slice_2D_moving.sitk.GetPixelIDValue())
            # before_2D_sitk = sitk.Resample(slice_2D_fixed.sitk, slice_2D_moving.sitk, sitk.Euler2DTransform(), sitk.sitkNearestNeighbor, 0.0, slice_2D_moving.sitk.GetPixelIDValue())

            # sitkh.show_sitk_image([slice_2D_moving.sitk, before_2D_sitk, foo_2D_sitk],["moving","fixed_before", "fixed_after"])

            ## Expand to 3D transform
            rigid_registration_transform_3D_sitk = self._get_3D_from_2D_rigid_transform_sitk(rigid_registration_transform_2D_sitk)

            ## Compose to 3D in-plane transform
            affine_transform_sitk = sitkh.get_composite_sitk_affine_transform(rigid_registration_transform_3D_sitk, transforms_PP_3D_sitk[i])
            affine_transform_sitk = sitkh.get_composite_sitk_affine_transform(sitk.AffineTransform(transforms_PP_3D_sitk[i].GetInverse()), affine_transform_sitk)

            self._slices[i].update_motion_correction(affine_transform_sitk)            


    ##-------------------------------------------------------------------------
    # \brief      Get 2D slice for in-plane operations and
    # \date       2016-09-20 22:42:45+0100
    #
    # Get 2D slice for in-plane operations, i.e. where in-plane motion can be
    # captured by only using 2D transformations. Depending on previous
    # operations the slice can already be shifted due to previous in-plane
    # registration steps. The 3D affine transformation transform_PP_sitk
    # indicates the original position of the corresponding original 3D slice.
    # Given that it operates from the physical 3D space to the physical 3D
    # space the name 'PP' is given.
    #
    # \param      self  The object
    #
    # \return     2D slice (Slice objects)
    #
    def _get_2D_slice(self, slice_3D, transform_PP_sitk):

        ## Create copy of stack
        slice_3D_copy = sl.Slice.from_slice(slice_3D)

        ## Get current transform from image to physical space of slice
        T_PI = sitkh.get_sitk_affine_transform_from_sitk_image(slice_3D.sitk)

        ## Get transform to align slice with physical coordinate system (perhaps already shifted there) 
        T_PI_align = sitkh.get_composite_sitk_affine_transform(transform_PP_sitk, T_PI)

        ## Set direction and origin of image accordingly
        direction = sitkh.get_sitk_image_direction_from_sitk_affine_transform(T_PI_align, slice_3D.sitk)
        origin = sitkh.get_sitk_image_origin_from_sitk_affine_transform(T_PI_align, slice_3D.sitk)

        slice_3D_copy.sitk.SetDirection(direction)
        slice_3D_copy.sitk.SetOrigin(origin)
        slice_3D_copy.sitk_mask.SetDirection(direction)
        slice_3D_copy.sitk_mask.SetOrigin(origin)

        ## Get filename and slice number for name propagation
        filename = slice_3D.get_filename()
        slice_number = slice_3D.get_slice_number()

        slice_2D_sitk = slice_3D_copy.sitk[:,:,0]
        slice_2D_sitk_mask = slice_3D_copy.sitk_mask[:,:,0]

        slice_2D = sl.Slice.from_sitk_image(slice_2D_sitk, dir_input=None, filename=filename, slice_number=slice_number, slice_sitk_mask=slice_2D_sitk_mask)

        return slice_2D


    ##-------------------------------------------------------------------------
    # \brief      Get the 3D rigid transforms to arrive at the positions of
    #             original 3D slices starting from the physically aligned
    #             space.
    # \date       2016-09-20 23:37:05+0100
    #
    # The rigid transform is given as composed translation and rotation
    # transform, i.e. T_PP = (T_t \circ T_rot)^{-1}.
    #
    # \param      self  The object
    #
    # \return     List of 3D rigid transforms (sitk.AffineTransform(3) objects)
    #             to arrive at the positions of the original 3D slices.
    #
    def _get_list_of_3D_rigid_transforms_of_slices(self):

        transforms_PP_3D_sitk = [None]*self._N_slices

        for i in range(0, self._N_slices):
            slice_3D_sitk = self._slices[i].sitk

            ## Extract origin and direction matrix from slice:
            origin_3D_sitk = np.array(slice_3D_sitk.GetOrigin())
            direction_3D_sitk = np.array(slice_3D_sitk.GetDirection())

            ## Define rigid transformation
            transform_PP_sitk = sitk.AffineTransform(3)
            transform_PP_sitk.SetMatrix(direction_3D_sitk)
            transform_PP_sitk.SetTranslation(origin_3D_sitk)

            transforms_PP_3D_sitk[i] = sitk.AffineTransform(transform_PP_sitk.GetInverse())

        return transforms_PP_3D_sitk


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
        center_x, center_y = rigid_transform_2D_sitk.GetFixedParameters()

        # Expand obtained translation to 3D vector
        translation_3D = (translation_x, translation_y, 0)
        center_3D = (center_x, center_y, 0)

        # Create 3D rigid transform based on 2D
        rigid_transform_3D = sitk.Euler3DTransform()
        rigid_transform_3D.SetRotation(0,0,angle_z)
        rigid_transform_3D.SetTranslation(translation_3D)
        rigid_transform_3D.SetFixedParameters(center_3D)
        
        return rigid_transform_3D

