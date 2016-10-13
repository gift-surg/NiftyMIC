## \file StackInPlaneAlignment.py
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
class StackInPlaneAlignment:

    ##-------------------------------------------------------------------------
    # \brief      TODO
    # \date       2016-09-26 10:20:03+0100
    #
    # \param      self                The object
    # \param      stack               The stack
    # \param      reference           The reference
    # \param      use_stack_mask      The use stack mask
    # \param      use_reference_mask  The use reference mask
    # \param      alignment_approach  The alignment approach
    #
    def __init__(self, stack=None, reference=None, use_stack_mask=False, use_reference_mask=False, alignment_approach="rigid_inplane_within_stack"):
        

        self._alignment_approach = alignment_approach
        self._use_stack_mask = use_stack_mask
        self._use_reference_mask = use_reference_mask
        self._run_in_plane_registration = {
            "rigid_inplane_within_stack"        : self._run_rigid_in_plane_registration_within_stack,
            "rigid_inplane_to_reference"        : self._run_rigid_in_plane_registration_to_reference,
            "similarity_inplane_to_reference"   : self._run_similarity_in_plane_registration_to_reference
        }

        if stack is not None:
            self._stack = st.Stack.from_stack(stack, filename=stack.get_filename())
            self._slices = self._stack.get_slices()
            self._N_slices = self._stack.get_number_of_slices()

        if reference is not None:
            try:
                self._stack.sitk - reference.sitk
            except:
                raise ValueError("Reference and stack are not in the same space")
            self._reference = reference
            self._alignment_approach = "rigid_inplane_to_reference"


    ##-------------------------------------------------------------------------
    # \brief      Sets the stack.
    # \date       2016-09-20 22:30:54+0100
    #
    # \param      self   The object
    # \param      stack  stack as Stack object
    #
    def set_stack(self, stack):
        self._stack = st.Stack.from_stack(stack, filename=stack.get_filename())
        self._slices = self._stack.get_slices()
        self._N_slices = self._stack.get_number_of_slices()


    ##-------------------------------------------------------------------------
    # \brief      Sets the reference for in-plane alignment. Reference stack
    #             must be in the same physical space as the stack to be
    #             aligned.
    # \date       2016-10-12 15:32:24+0100
    #
    # \param      self       The object
    # \param      reference  The reference
    #
    def set_reference(self, reference):
        try:
            self._stack.sitk - reference.sitk
        except:
            raise ValueError("Reference and stack are not in the same space")

        self._reference = reference
        self._alignment_approach = "rigid_inplane_to_reference"


    ##-------------------------------------------------------------------------
    # \brief      Sets the alignment approach which is used for the slice
    #             in-plane alignment
    # \date       2016-10-13 21:40:09+0100
    #
    # \param      self                The object
    # \param      alignment_approach  The alignment approach as string in
    #                                 either 'rigid_inplane_within_stack',
    #                                 'rigid_inplane_to_reference' or
    #                                 'similarity_inplane_to_reference'
    #
    def set_alignment_approach(self, alignment_approach):
        if alignment_approach not in ["rigid_inplane_within_stack", "rigid_inplane_to_reference", "similarity_inplane_to_reference"]:
            raise ValueError("Alignment approach must be either 'rigid_inplane_within_stack', 'rigid_inplane_to_reference' or 'similarity_inplane_to_reference'")
        self._alignment_approach = alignment_approach


    ##-------------------------------------------------------------------------
    # \brief      Gets the alignment approach.
    # \date       2016-10-13 21:41:55+0100
    #
    # \param      self  The object
    #
    # \return     The alignment approach.
    #
    def get_alignment_approach(self):
        return self._alignment_approach


    ##-------------------------------------------------------------------------
    # \brief      Set whether mask is used for stack of slices for registration
    # \date       2016-10-12 15:33:45+0100
    #
    # \param      self  The object
    # \param      flag  Flag indicating whether mask is used, bool
    #
    def use_stack_mask(self, flag):
        self._use_stack_mask = flag


    ##-------------------------------------------------------------------------
    # \brief      Set whether mask is used for reference image for registration
    # \date       2016-10-12 15:34:34+0100
    #
    # \param      self  The object
    # \param      flag  Flag indicating whether mask is used, bool
    #
    def use_reference_mask(self, flag):
        self._use_reference_mask = flag


    ##-------------------------------------------------------------------------
    # \brief      Gets the in-plane rigidly registered stack.
    # \date       2016-09-20 22:38:53+0100
    #
    # \param      self  The object
    #
    # \return     The stack as Stack object.
    #
    def get_inplane_registered_stack(self):
        return self._stack


    ##-------------------------------------------------------------------------
    # \brief      Run in-plane rigid registration algorithm
    # \date       2016-09-21 02:19:31+0100
    #
    # \param      self  The object
    #
    def run_registration(self):

        self._run_in_plane_registration[self._alignment_approach]()


    ##-------------------------------------------------------------------------
    # \brief      Run in-plane rigid alignment to match the reference
    # \date       2016-10-12 15:38:14+0100
    #
    # \param      self  The object
    #
    def _run_rigid_in_plane_registration_to_reference(self):

        print("*** Perform rigid in-plane registration based on reference ***")

        ## Set up registration
        registration_2D = regsitk.RegistrationSimpleITK()
        registration_2D.set_registration_type("Rigid")
        # registration_2D.use_multiresolution_framework(True)
        registration_2D.set_centered_transform_initializer(None)
        registration_2D.set_scales_estimator("PhysicalShift")
        registration_2D.set_metric("MattesMutualInformation")
        # registration_2D.set_metric("MeanSquares")
        # registration_2D.set_metric("Correlation")
        registration_2D.use_fixed_mask(self._use_stack_mask)
        registration_2D.use_moving_mask(self._use_reference_mask)
        registration_2D.use_verbose(False)

        ## Get list of 3D affine transforms to arrive at the positions of the
        ## original 3D slices
        transforms_PP_3D_sitk = self._get_list_of_3D_rigid_transforms_of_slices()

        slices_reference = self._reference.get_slices()

        for i in range(0, self._N_slices):
            slice_2D_fixed = self._get_2D_slice(self._slices[i], transforms_PP_3D_sitk[i])
            slice_2D_moving = self._get_2D_slice(slices_reference[i], transforms_PP_3D_sitk[i])

            ## Perform in-plane rigid registration
            registration_2D.set_fixed(slice_2D_fixed)
            registration_2D.set_moving(slice_2D_moving)
            registration_2D.run_registration()

            rigid_registration_transform_2D_sitk = registration_2D.get_registration_transform_sitk()

            ## Debug
            # foo_2D_sitk = sitkh.get_transformed_image(slice_2D_fixed.sitk, rigid_registration_transform_2D_sitk)
            # foo_2D_sitk = sitk.Resample(foo_2D_sitk, slice_2D_moving.sitk, sitk.Euler2DTransform(), sitk.sitkNearestNeighbor, 0.0, slice_2D_moving.sitk.GetPixelIDValue())
            # before_2D_sitk = sitk.Resample(slice_2D_fixed.sitk, slice_2D_moving.sitk, sitk.Euler2DTransform(), sitk.sitkNearestNeighbor, 0.0, slice_2D_moving.sitk.GetPixelIDValue())
            # sitkh.show_sitk_image([slice_2D_moving.sitk, before_2D_sitk, foo_2D_sitk], segmentation=slice_2D_moving.sitk_mask, title=["moving","fixed_before", "fixed_after"])

            ## Expand to 3D transform
            rigid_registration_transform_3D_sitk = self._get_3D_from_2D_rigid_transform_sitk(rigid_registration_transform_2D_sitk)

            ## Compose to 3D in-plane transform
            affine_transform_sitk = sitkh.get_composite_sitk_affine_transform(rigid_registration_transform_3D_sitk, transforms_PP_3D_sitk[i])
            affine_transform_sitk = sitkh.get_composite_sitk_affine_transform(sitk.AffineTransform(transforms_PP_3D_sitk[i].GetInverse()), affine_transform_sitk)

            self._slices[i].update_motion_correction(affine_transform_sitk)   


    #--------------------------------------------------------------------------
    # \brief      Run in-plane similarity alignment to match the reference
    # \date       2016-10-13 21:44:49+0100
    #
    # \param      self  The object
    #
    def _run_similarity_in_plane_registration_to_reference(self):

        print("*** Perform similarity in-plane registration based on reference ***")

        ## Set up registration
        registration_2D = regsitk.RegistrationSimpleITK()
        registration_2D.set_registration_type("Similarity")
        # registration_2D.use_multiresolution_framework(True)
        registration_2D.set_centered_transform_initializer(None)
        registration_2D.set_scales_estimator("Jacobian")
        # registration_2D.set_metric("MattesMutualInformation")
        # registration_2D.set_metric("MeanSquares")
        registration_2D.set_metric("Correlation")
        registration_2D.use_fixed_mask(self._use_stack_mask)
        registration_2D.use_moving_mask(self._use_reference_mask)
        registration_2D.use_verbose(False)

        ## Get list of 3D affine transforms to arrive at the positions of the
        ## original 3D slices
        transforms_PP_3D_sitk = self._get_list_of_3D_rigid_transforms_of_slices()

        slices_reference = self._reference.get_slices()

        for i in range(0, self._N_slices):
            slice_2D_fixed = self._get_2D_slice(self._slices[i], transforms_PP_3D_sitk[i])
            slice_2D_moving = self._get_2D_slice(slices_reference[i], transforms_PP_3D_sitk[i])

            ## Perform in-plane rigid registration
            registration_2D.set_fixed(slice_2D_fixed)
            registration_2D.set_moving(slice_2D_moving)
            registration_2D.run_registration()

            similarity_registration_transform_2D_sitk = registration_2D.get_registration_transform_sitk()

            # print similarity_registration_transform_2D_sitk

            ## Update slice scale
            scale = similarity_registration_transform_2D_sitk.GetScale()
            print("Slice %s/%s: scale = %s" % (i,self._N_slices-1, scale))
            # slice_2D_fixed_scaled = sitk.Image(slice_2D_fixed.sitk)
            # spacing = np.array(slice_2D_fixed_scaled.GetSpacing())
            # spacing *= scale
            # slice_2D_fixed_scaled.SetSpacing(spacing)

            ## Convert to rigid registration transform
            center = similarity_registration_transform_2D_sitk.GetCenter()
            angle = similarity_registration_transform_2D_sitk.GetAngle()
            translation = similarity_registration_transform_2D_sitk.GetTranslation()
            rigid_registration_transform_2D_sitk = sitk.Euler2DTransform(center, angle, translation)

            ## Debug
            # foo_2D_sitk = sitkh.get_transformed_image(slice_2D_moving.sitk, sitk.Euler2DTransform(rigid_registration_transform_2D_sitk.GetInverse()))

            # foo_2D_sitk_resample0 = sitk.Resample(foo_2D_sitk, slice_2D_fixed_scaled, sitk.Euler2DTransform())
            # foo_2D_sitk_resample1 = sitk.Resample(slice_2D_moving.sitk, slice_2D_fixed.sitk, similarity_registration_transform_2D_sitk)

            # sitkh.show_sitk_image([foo_2D_sitk_resample0, foo_2D_sitk_resample1])

            # # foo_2D_sitk = sitkh.get_transformed_image(foo_2D_sitk, rigid_registration_transform_2D_sitk)
            # foo_2D_sitk = sitk.Resample(foo_2D_sitk, slice_2D_moving.sitk, sitk.Euler2DTransform(), sitk.sitkNearestNeighbor, 0.0, slice_2D_moving.sitk.GetPixelIDValue())
            # before_2D_sitk = sitk.Resample(slice_2D_fixed.sitk, slice_2D_moving.sitk, sitk.Euler2DTransform(), sitk.sitkNearestNeighbor, 0.0, slice_2D_moving.sitk.GetPixelIDValue())
            # sitkh.show_sitk_image([slice_2D_moving.sitk, before_2D_sitk, foo_2D_sitk], segmentation=slice_2D_moving.sitk_mask, title=["moving","fixed_before", "fixed_after"])

            ## Expand to 3D transform
            rigid_registration_transform_3D_sitk = self._get_3D_from_2D_rigid_transform_sitk(rigid_registration_transform_2D_sitk)

            ## Compose to 3D in-plane transform
            affine_transform_sitk = sitkh.get_composite_sitk_affine_transform(rigid_registration_transform_3D_sitk, transforms_PP_3D_sitk[i])
            affine_transform_sitk = sitkh.get_composite_sitk_affine_transform(sitk.AffineTransform(transforms_PP_3D_sitk[i].GetInverse()), affine_transform_sitk)

            self._slices[i].update_motion_correction(affine_transform_sitk)   

            slice_sitk = self._slices[i].sitk
            slice_sitk_mask = self._slices[i].sitk_mask
            spacing = np.array(slice_sitk.GetSpacing())
            spacing[0:-1] *= scale
            slice_sitk.SetSpacing(spacing)
            slice_sitk_mask.SetSpacing(spacing)

            self._slices[i] = sl.Slice.from_sitk_image(slice_sitk, self._slices[i].get_directory(), self._slices[i].get_filename(), self._slices[i].get_slice_number(), slice_sitk_mask)



    ##-------------------------------------------------------------------------
    # \brief      Run in-plane rigid registration to align within stack
    # \date       2016-09-26 16:11:14+0100
    #
    # \param      self  The object
    #
    # \return     { description_of_the_return_value }
    #
    def _run_rigid_in_plane_registration_within_stack(self):

        print("*** Perform rigid in-plane registration within stack ***")

        ## Set up registration
        registration_2D = regsitk.RegistrationSimpleITK()
        registration_2D.set_registration_type("Rigid")
        # registration_2D.use_multiresolution_framework(True)
        registration_2D.set_centered_transform_initializer(None)
        registration_2D.set_scales_estimator("PhysicalShift")
        # registration_2D.set_metric("MattesMutualInformation")
        # registration_2D.set_metric("MeanSquares")
        registration_2D.set_metric("Correlation")
        registration_2D.use_fixed_mask(False)
        registration_2D.use_moving_mask(False)
        registration_2D.use_verbose(False)

        ## Get list of 3D affine transforms to arrive at the positions of the
        ## original 3D slices
        transforms_PP_3D_sitk = self._get_list_of_3D_rigid_transforms_of_slices()

        for i in range(self._N_slices-2,-1,-1):
            slice_2D_moving = self._get_2D_slice(self._slices[i+1], transforms_PP_3D_sitk[i+1])
        # for i in range(1, self._N_slices):
            # slice_2D_moving = self._get_2D_slice(self._slices[i-1], transforms_PP_3D_sitk[i-1])

            slice_2D_fixed  = self._get_2D_slice(self._slices[i], transforms_PP_3D_sitk[i])

            # print("slice_2D_fixed.sitk.GetDirection() = " + str(slice_2D_fixed.sitk.GetDirection()))
            # print("slice_2D_moving.sitk.GetDirection() = " + str(slice_2D_moving.sitk.GetDirection()))

            ## Perform in-plane rigid registration
            registration_2D.set_fixed(slice_2D_fixed)
            registration_2D.set_moving(slice_2D_moving)
            registration_2D.run_registration()

            rigid_registration_transform_2D_sitk = registration_2D.get_registration_transform_sitk()

            ## Debug
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
    # \brief      Get 2D slice for in-plane operations as projection from 3D
    #             space onto the x-y-plane.
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
    # \param      self               The object
    # \param      slice_3D           The slice 3d
    # \param      transform_PP_sitk  sitk
    #
    # \return     Projected 2D slice onto x-y-plane of 3D stack as Slice object
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
    #             space with the main image axes.
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

        N_slices = self._stack.get_number_of_slices()
        slices = self._stack.get_slices()

        transforms_PP_3D_sitk = [None]*N_slices

        for i in range(0, N_slices):
            slice_3D_sitk = slices[i].sitk

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

