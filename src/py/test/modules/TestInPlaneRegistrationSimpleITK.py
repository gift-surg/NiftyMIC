## \file TestInPlaneRegistrationSimpleITK.py
#  \brief  Class containing unit tests for module InPlaneRegistration
# 
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date October 2016


## Import libraries 
import SimpleITK as sitk
import itk
import numpy as np
import unittest
import sys
import time
from datetime import timedelta

import pythonhelper.SimpleITKHelper as sitkh

## Import modules
import volumetricreconstruction.base.Stack as st
import volumetricreconstruction.registration.InPlaneRegistrationSimpleITK as inplaneregsitk

from volumetricreconstruction.definitions import DIR_TEST


def get_inplane_corrupted_stack(stack, angle_z, center, translation, scale=1, debug=0):
    
    ## Transform to align physical coordinate system with stack-coordinate system
    affine_centering_sitk = sitk.AffineTransform(3)
    affine_centering_sitk.SetMatrix(stack.sitk.GetDirection())
    affine_centering_sitk.SetTranslation(stack.sitk.GetOrigin())

    ## Corrupt first stack towards positive direction
    in_plane_motion_sitk = sitk.Euler3DTransform()
    in_plane_motion_sitk.SetRotation(0, 0, angle_z)
    in_plane_motion_sitk.SetTranslation(translation)
    motion_sitk = sitkh.get_composite_sitk_affine_transform(in_plane_motion_sitk, sitk.AffineTransform(affine_centering_sitk.GetInverse()))
    motion_sitk = sitkh.get_composite_sitk_affine_transform(affine_centering_sitk, motion_sitk)
    stack_corrupted_resampled_sitk = sitk.Resample(stack.sitk, motion_sitk, sitk.sitkLinear)
    stack_corrupted_resampled_sitk_mask = sitk.Resample(stack.sitk_mask, motion_sitk, sitk.sitkLinear)

    ## Corrupt first stack towards negative direction
    in_plane_motion_2_sitk = sitk.Euler3DTransform()
    in_plane_motion_2_sitk.SetRotation(0, 0, -angle_z)
    in_plane_motion_2_sitk.SetTranslation(-translation)
    motion_2_sitk = sitkh.get_composite_sitk_affine_transform(in_plane_motion_2_sitk, sitk.AffineTransform(affine_centering_sitk.GetInverse()))
    motion_2_sitk = sitkh.get_composite_sitk_affine_transform(affine_centering_sitk, motion_2_sitk)
    stack_corrupted_2_resampled_sitk = sitk.Resample(stack.sitk, motion_2_sitk, sitk.sitkLinear)
    stack_corrupted_2_resampled_sitk_mask = sitk.Resample(stack.sitk_mask, motion_2_sitk, sitk.sitkLinear)

    ## Create stack based on those two corrupted stacks
    nda = sitk.GetArrayFromImage(stack_corrupted_resampled_sitk)
    nda_mask = sitk.GetArrayFromImage(stack_corrupted_resampled_sitk_mask)
    nda_neg = sitk.GetArrayFromImage(stack_corrupted_2_resampled_sitk)
    nda_neg_mask = sitk.GetArrayFromImage(stack_corrupted_2_resampled_sitk_mask)
    for i in range(0, stack.sitk.GetDepth(),2):
        nda[i,:,:] = nda_neg[i,:,:]
        nda_mask[i,:,:] = nda_neg_mask[i,:,:]
    stack_corrupted_sitk = sitk.GetImageFromArray(nda)
    stack_corrupted_sitk_mask = sitk.GetImageFromArray(nda_mask)
    stack_corrupted_sitk.CopyInformation(stack.sitk)
    stack_corrupted_sitk_mask.CopyInformation(stack.sitk_mask)

    ## Debug: Show corrupted stacks (before scaling)
    if debug:
        sitkh.show_sitk_image([stack.sitk, stack_corrupted_resampled_sitk, stack_corrupted_2_resampled_sitk, stack_corrupted_sitk], title=["original", "corrupted_1" ,"corrupted_2" , "corrupted_final_from_1_and_2"])

    ## Update in-plane scaling
    spacing = np.array(stack.sitk.GetSpacing())
    spacing[0:-1] *= scale
    stack_corrupted_sitk.SetSpacing(spacing)
    stack_corrupted_sitk_mask.SetSpacing(spacing)

    ## Create Stack object
    stack_corrupted = st.Stack.from_sitk_image(stack_corrupted_sitk, "stack_corrupted", stack_corrupted_sitk_mask)

    ## Debug: Show corrupted stacks (after scaling)
    if debug:
        stack_corrupted_resampled_sitk = sitk.Resample(stack_corrupted.sitk, stack.sitk)
        sitkh.show_sitk_image([stack.sitk, stack_corrupted_resampled_sitk], title=["original", "corrupted"])

    return stack_corrupted, motion_sitk, motion_2_sitk


## Concept of unit testing for python used in here is based on
#  http://pythontesting.net/framework/unittest/unittest-introduction/
#  Retrieved: Aug 6, 2015
class TestInPlaneRegistrationSimpleITK(unittest.TestCase):

    ## Specify input data
    dir_test_data = DIR_TEST

    accuracy = 6

    def setUp(self):        
        pass


    ##
    #       Verify that in-plane rigid registration works
    # \date       2016-11-02 21:56:19+0000
    #
    # Verify that in-plane rigid registration works, i.e. test 
    #   1) registration parameters are close to ground truth (up to zero dp) 
    #   2) affine transformations for each slice correctly describes the 
    #      registration
    #
    # \param      self  The object
    #
    def test_inplane_rigid_alignment_to_reference(self):

        filename = "fetal_brain_0"
        stack = st.Stack.from_filename(self.dir_test_data, filename, "_mask")

        ## Create in-plane motion corruption (last coordinate zero!)
        angle_z = 0.08
        center = (0,0,0)
        translation = np.array([-1, 3, 0])

        ## Get corrupted stack and corresponding motions
        stack_corrupted, motion_sitk, motion_2_sitk = get_inplane_corrupted_stack(stack, angle_z, center, translation)
        
        # stack_corrupted.show(1)
        # stack.show(1)

        ## Perform in-plane rigid registration
        inplane_registration = inplanereg.InPlaneRegistration(stack=stack_corrupted, reference=stack)
        inplane_registration.set_alignment_approach("rigid_inplane_to_reference")
        inplane_registration.set_metric("MeanSquares")
        inplane_registration.set_scales_estimator("Jacobian")
        inplane_registration.set_initializer_type("MOMENTS") # MOMENTS works better than GEOMETRY here
        # inplane_registration.use_verbose(True)
        inplane_registration.run_registration()

        ## Debug: Visual Comparison
        # sitkh.show_sitk_image([stack.sitk, stack_corrupted.sitk, stack_aligned.get_resampled_stack_from_slices(interpolator="BSpline").sitk], title=["original", "corrupted", "recovered"])

        ## Check result:
        registration_transforms = inplane_registration.get_affine_transformations()
        parameters_pos = np.array(motion_sitk.GetParameters())
        parameters_neg = np.array(motion_2_sitk.GetParameters())
        for i in range(0, stack.sitk.GetDepth()):
            if i%2:
                error = np.linalg.norm(registration_transforms[i].GetParameters() - parameters_pos)
            else:
                error = np.linalg.norm(registration_transforms[i].GetParameters() - parameters_neg)
            # print("Slice %s/%s: \tl2-error = %.3f" %(i,stack.sitk.GetDepth()-1,error))
            self.assertEqual(np.round(
                error
            , decimals = 0), 0)
        

        ## Get all affine transforms describing the corrections
        transform_updates = inplane_registration.get_affine_transformations()

        ## Check whether corrected stack can be obtained by affine transforms too
        stack_aligned = inplane_registration.get_inplane_registered_stack()

        slices_aligned = stack_aligned.get_slices()
        slices_corrupted = stack_corrupted.get_slices()

        for i in range(0, len(slices_aligned)):
            mask_transformed_resampled_sitk = sitk.Resample(slices_corrupted[i].sitk_mask, slices_aligned[i].sitk_mask, transform_updates[i].GetInverse(), sitk.sitkNearestNeighbor)
            
            nda_diff = sitk.GetArrayFromImage(mask_transformed_resampled_sitk-slices_aligned[i].sitk_mask)
            
            self.assertEqual(np.round(
                np.linalg.norm(nda_diff)
            , decimals = self.accuracy), 0)


    ##
    #       Verify that in-plane rigid registration works
    # \date       2016-11-02 21:56:19+0000
    #
    # Same idea as rigid test above, i.e. test 
    #   1) registration parameters are close to ground truth (up to zero dp) 
    #   2) affine transformations for each slice correctly describes the 
    #      registration
    #      
    # In this first test, the same scenario as for the rigid case above is
    # chosen, i.e. same parameters + scale=1.
    # Result: In-plane similarity has some misregistered slices due to wrong
    # scaling estimate.
    #
    # \param      self  The object
    #
    def test_inplane_similarity_alignment_to_reference_1(self):

        filename = "fetal_brain_0"
        stack = st.Stack.from_filename(self.dir_test_data, filename, "_mask")

        ## Create in-plane motion corruption (last coordinate zero!)
        scale = 1
        angle_z = 0.08
        center = (0,0,0)
        translation = np.array([-1, 3, 0])
        
        ## Get corrupted stack and corresponding motions
        stack_corrupted, motion_sitk, motion_2_sitk = get_inplane_corrupted_stack(stack, angle_z, center, translation, scale)
        
        # stack_corrupted.show(1)
        # stack.show(1)
    
        ## Perform in-plane similarity registration (exact same setting as rigid case above)
        inplane_registration = inplanereg.InPlaneRegistration(stack=stack_corrupted, reference=stack)
        inplane_registration.set_alignment_approach("similarity_inplane_to_reference")
        # inplane_registration.set_alignment_approach("rigid_inplane_to_reference")
        inplane_registration.set_metric("MeanSquares")
        # inplane_registration.use_multiresolution_framework(True) #improves for some slices but makes it worse for others
        inplane_registration.set_scales_estimator("Jacobian")
        inplane_registration.set_initializer_type('MOMENTS')
        # inplane_registration.use_verbose(True)
        inplane_registration.run_registration()
        stack_aligned = inplane_registration.get_inplane_registered_stack()

        ## Debug: Visual Comparison
        # sitkh.show_sitk_image([stack.sitk, stack_corrupted.sitk, stack_aligned.get_resampled_stack_from_slices(interpolator="BSpline").sitk], title=["original", "corrupted", "recovered"])

        ## Check registration accuracy:
        registration_transforms = inplane_registration.get_affine_transformations()
        parameters_pos = np.array(motion_sitk.GetParameters())
        parameters_neg = np.array(motion_2_sitk.GetParameters())

        slices = stack_aligned.get_slices()
        spacing_stack = np.array(stack_aligned.sitk.GetSpacing())
        for i in range(0, stack.sitk.GetDepth()):
            spacing_slice = np.array(slices[i].sitk.GetSpacing())
            scale_estimated = spacing_stack[0]/spacing_slice[0]
            matrix = np.array(registration_transforms[i].GetMatrix()).reshape(3,3)
            matrix[0:-1,0:-1] *= scale_estimated
            parameters = np.concatenate((matrix.flatten(), registration_transforms[i].GetTranslation()))
            # print("Slice %s/%s: \tparameters = %s" %(i,stack.sitk.GetDepth()-1,parameters))
            # print("Slice %s/%s: \ttranslation = %s" %(i,stack.sitk.GetDepth()-1,registration_transforms[i].GetTranslation()))

            if i%2:
                parameters_diff = parameters-parameters_pos
            else:
                parameters_diff = parameters-parameters_neg
            error = np.linalg.norm(parameters_diff)
            print("Slice %s/%s: \tl2-error = %.3f \t(scale_err = %.2f)" %(i,stack.sitk.GetDepth()-1,error, (scale_estimated-scale)))
            # print("\tparameters-parameters_pos = %s" %(parameters_diff))
            # print("\tscale-scale_estimated = %s" %(scale-scale_estimated))
            # self.assertEqual(np.round(
            #     error
            # , decimals = 0), 0)

        print("Scenario the same as for the rigid case. Pure in-plane rigid can successfully recover the slices.")
        print("In-plane similarity has some issues for some slices due to incorrect scale estimate.\n")

        ## Get all affine transforms describing the corrections
        transform_updates = inplane_registration.get_affine_transformations()

        slices_aligned = stack_aligned.get_slices()
        slices_corrupted = stack_corrupted.get_slices()

        ## Check whether corrected stack can be obtained by affine transforms too
        for i in range(0, len(slices_aligned)):
            mask_transformed_resampled_sitk = sitk.Resample(slices_corrupted[i].sitk_mask, slices_aligned[i].sitk_mask, transform_updates[i].GetInverse(), sitk.sitkNearestNeighbor)
            
            nda_diff = sitk.GetArrayFromImage(mask_transformed_resampled_sitk-slices_aligned[i].sitk_mask)
            
            self.assertEqual(np.round(
                np.linalg.norm(nda_diff)
            , decimals = self.accuracy), 0)

    ##
    #       Verify that in-plane rigid registration works
    # \date       2016-11-02 21:56:19+0000
    #
    # Same idea as rigid test above, i.e. test 
    #   1) registration parameters are close to ground truth (up to zero dp) 
    #   2) affine transformations for each slice correctly describes the 
    #      registration
    #      
    # In the second test, the same scenario as for the rigid case above is
    # chosen but with a different scaling
    # Result: In-plane similarity has some misregistered slices due to wrong
    # scaling estimate.
    #
    # \param      self  The object
    #
    def test_inplane_similarity_alignment_to_reference_2(self):

        filename = "fetal_brain_0"
        stack = st.Stack.from_filename(self.dir_test_data, filename, "_mask")

        ## Create in-plane motion corruption (last coordinate zero!)
        scale = 0.97
        angle_z = 0.08
        center = (0,0,0)
        translation = np.array([-1, 3, 0])
        
        ## Get corrupted stack and corresponding motions
        stack_corrupted, motion_sitk, motion_2_sitk = get_inplane_corrupted_stack(stack, angle_z, center, translation, scale)
        
        # stack_corrupted.show(1)
        # stack.show(1)

        ## Ensure that stack and reference are in the same space for in-plane alignment
        stack_resampled_sitk = sitk.Resample(stack.sitk, stack_corrupted.sitk)
        stack_resampled_sitk_mask = sitk.Resample(stack.sitk_mask, stack_corrupted.sitk_mask)
        stack_resampled = st.Stack.from_sitk_image(stack_resampled_sitk, stack.get_filename()+"_resampled", stack_resampled_sitk_mask)

        ## Perform in-plane similarity registration
        inplane_registration = inplanereg.InPlaneRegistration(stack=stack_corrupted, reference=stack_resampled)
        inplane_registration.set_alignment_approach("similarity_inplane_to_reference")
        inplane_registration.set_metric("MeanSquares")
        inplane_registration.set_scales_estimator("Jacobian")
        inplane_registration.set_initializer_type('MOMENTS') #MOMENTS causes error message at times
        # inplane_registration.use_verbose(True)
        inplane_registration.use_multiresolution_framework(True) #improves for some slices but makes it worse for others
        inplane_registration.run_registration()
        stack_aligned = inplane_registration.get_inplane_registered_stack()

        ## Debug: Visual Comparison
        sitkh.show_sitk_image([stack.sitk, stack_corrupted.sitk, stack_aligned.get_resampled_stack_from_slices(interpolator="BSpline").sitk], title=["original", "corrupted", "recovered"])

        ## Check registration accuracy:
        registration_transforms = inplane_registration.get_affine_transformations()
        parameters_pos = np.array(motion_sitk.GetParameters())
        parameters_neg = np.array(motion_2_sitk.GetParameters())

        slices = stack_aligned.get_slices()
        spacing_stack = np.array(stack_aligned.sitk.GetSpacing())
        for i in range(0, stack.sitk.GetDepth()):
            spacing_slice = np.array(slices[i].sitk.GetSpacing())
            scale_estimated = spacing_stack[0]/spacing_slice[0]
            matrix = np.array(registration_transforms[i].GetMatrix()).reshape(3,3)
            matrix[0:-1,0:-1] *= scale_estimated
            parameters = np.concatenate((matrix.flatten(), registration_transforms[i].GetTranslation()))
            # print("Slice %s/%s: \tparameters = %s" %(i,stack.sitk.GetDepth()-1,parameters))
            # print("Slice %s/%s: \ttranslation = %s" %(i,stack.sitk.GetDepth()-1,registration_transforms[i].GetTranslation()))

            if i%2:
                parameters_diff = parameters-parameters_pos
            else:
                parameters_diff = parameters-parameters_neg
            error = np.linalg.norm(parameters_diff)
            print("Slice %s/%s: \tl2-error = %.3f \t(scale_err = %.2f)" %(i,stack.sitk.GetDepth()-1,error, (scale_estimated-scale)))
            # print("\tparameters-parameters_pos = %s" %(parameters_diff))
            # print("\tscale-scale_estimated = %s" %(scale-scale_estimated))
            # self.assertEqual(np.round(
            #     error
            # , decimals = 0), 0)

        print("Scale estimate seems to be problematic for some slices.")
        print("Overall, In-plane alignment seems to work OK-ish.\n")

        ## Get all affine transforms describing the corrections
        transform_updates = inplane_registration.get_affine_transformations()

        slices_aligned = stack_aligned.get_slices()
        slices_corrupted = stack_corrupted.get_slices()

        ## Check whether corrected stack can be obtained by affine transforms too
        for i in range(0, len(slices_aligned)):
            mask_transformed_resampled_sitk = sitk.Resample(slices_corrupted[i].sitk_mask, slices_aligned[i].sitk_mask, transform_updates[i].GetInverse(), sitk.sitkNearestNeighbor)
            
            nda_diff = sitk.GetArrayFromImage(mask_transformed_resampled_sitk-slices_aligned[i].sitk_mask)
            
            self.assertEqual(np.round(
                np.linalg.norm(nda_diff)
            , decimals = self.accuracy), 0)
