## \file TestRegularizedInPlaneRegistration.py
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

## Add directories to import modules
dir_src_root = "../"
sys.path.append( dir_src_root )

## Import modules
import base.Stack as st
import registration.RegularizedInPlaneRegistration as inplanereg
import utilities.SimpleITKHelper as sitkh


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
class TestRegularizedInPlaneRegistration(unittest.TestCase):

    ## Specify input data
    dir_test_data = "../../../test-data/"

    accuracy = 6

    def setUp(self):        
        pass

    """
    ##-------------------------------------------------------------------------
    # \brief      Verify that in-plane rigid registration works
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

        filename_stack = "fetal_brain_0"
        filename_recon = "FetalBrain_reconstruction_3stacks_myAlg"

        stack_sitk = sitk.ReadImage(self.dir_test_data + filename_stack + ".nii.gz")
        recon_sitk = sitk.ReadImage(self.dir_test_data + filename_recon + ".nii.gz")

        recon_resampled_sitk = sitk.Resample(recon_sitk, stack_sitk)
        stack = st.Stack.from_sitk_image(recon_resampled_sitk)

        # stack = st.Stack.from_filename(self.dir_test_data, filename)

        ## Create in-plane motion corruption (last coordinate zero!)
        angle_z = 0.2
        center = (0,0,0)
        translation = np.array([1, -3, 0])

        ## Get corrupted stack and corresponding motions
        stack_corrupted, motion_sitk, motion_2_sitk = get_inplane_corrupted_stack(stack, angle_z, center, translation)

        ## Perform in-plane rigid registration
        inplane_registration = inplanereg.RegularizedInPlaneRegistration(stack_corrupted, stack)
        # inplane_registration.use_verbose(True)
        inplane_registration.run_regularized_rigid_inplane_registration()

        stack_registered = inplane_registration.get_registered_stack()
        parameters = inplane_registration.get_parameters()
            
        print parameters

        sitkh.show_stacks([stack, stack_corrupted, stack_registered.get_resampled_stack_from_slices(interpolator="Linear")])

        # print parameters
        
            # self.assertEqual(np.round(
            #     np.linalg.norm(nda_diff)
            # , decimals = self.accuracy), 0)

    """

    def test_get_initial_parameters(self):

        ## Create stack of slice with only a dot in the middle
        shape_xy = 15
        shape_z = 15

        # nda_2D = np.zeros((shape_xy, shape_xy))
        # nda_2D[25,25] = 1
        # nda_3D = np.tile(nda_2D, (shape_z, 1, 1))

        nda_3D = np.zeros((shape_z, shape_xy, shape_xy))
        for i in range(0, shape_z):
            nda_3D[i,i,i] = 1

        stack_sitk = sitk.GetImageFromArray(nda_3D)
        stack = st.Stack.from_sitk_image(stack_sitk)

        ## Create in-plane motion corruption (last coordinate zero!)
        angle_z = 0
        center = (0,0,0)
        translation = np.array([1, -3, 0])

        ## Get corrupted stack and corresponding motions
        stack_corrupted, motion_sitk, motion_2_sitk = get_inplane_corrupted_stack(stack, angle_z, center, translation)

        ## Perform in-plane rigid registration
        transforms_sitk = [sitk.Euler2DTransform()] * shape_z

        inplane_registration = inplanereg.RegularizedInPlaneRegistration(stack)
        inplane_registration.set_initializer_type("MOMENTS")
        inplane_registration.run_regularized_rigid_inplane_registration()
        stack_registered = inplane_registration.get_registered_stack()
        parameters = inplane_registration.get_parameters()
            
        print parameters

        sitkh.show_stacks([stack, stack_registered.get_resampled_stack_from_slices(interpolator="Linear")])


        # inplane_registration.use_verbose(True)


