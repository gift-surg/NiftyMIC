##
# \file TestIntraStackRegistration.py
#  \brief  Class containing unit tests for module IntraStackRegistration
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date October 2016


# Import libraries
import SimpleITK as sitk
import itk
import numpy as np
import unittest
import sys
import os
from scipy.ndimage import imread

import pysitk.SimpleITKHelper as sitkh
import pysitk.PythonHelper as ph

# Import modules
import niftymic.base.Stack as st
import niftymic.registration.IntraStackRegistration as inplanereg

from niftymic.definitions import DIR_TEST


def get_inplane_corrupted_stack(stack,
                                angle_z,
                                center_2D,
                                translation_2D,
                                scale=1,
                                intensity_scale=1,
                                intensity_bias=0,
                                debug=0,
                                random=False):

    # Convert to 3D:
    translation_3D = np.zeros(3)
    translation_3D[0:-1] = translation_2D

    center_3D = np.zeros(3)
    center_3D[0:-1] = center_2D

    # Transform to align physical coordinate system with stack-coordinate
    # system
    affine_centering_sitk = sitk.AffineTransform(3)
    affine_centering_sitk.SetMatrix(stack.sitk.GetDirection())
    affine_centering_sitk.SetTranslation(stack.sitk.GetOrigin())

    # Corrupt first stack towards positive direction
    if random:
        angle_z_1 = -angle_z*np.random.rand(1)[0]
    else:
        angle_z_1 = -angle_z

    in_plane_motion_sitk = sitk.Euler3DTransform()
    in_plane_motion_sitk.SetRotation(0, 0, angle_z_1)
    in_plane_motion_sitk.SetCenter(center_3D)
    in_plane_motion_sitk.SetTranslation(translation_3D)
    motion_sitk = sitkh.get_composite_sitk_affine_transform(
        in_plane_motion_sitk, sitk.AffineTransform(
            affine_centering_sitk.GetInverse()))
    motion_sitk = sitkh.get_composite_sitk_affine_transform(
        affine_centering_sitk, motion_sitk)
    stack_corrupted_resampled_sitk = sitk.Resample(
        stack.sitk, motion_sitk, sitk.sitkLinear)
    stack_corrupted_resampled_sitk_mask = sitk.Resample(
        stack.sitk_mask, motion_sitk, sitk.sitkLinear)

    # Corrupt first stack towards negative direction
    if random:
        angle_z_2 = -angle_z*np.random.rand(1)[0]
    else:
        angle_z_2 = -angle_z

    in_plane_motion_2_sitk = sitk.Euler3DTransform()
    in_plane_motion_2_sitk.SetRotation(0, 0, angle_z_2)
    in_plane_motion_2_sitk.SetCenter(center_3D)
    in_plane_motion_2_sitk.SetTranslation(-translation_3D)
    motion_2_sitk = sitkh.get_composite_sitk_affine_transform(
        in_plane_motion_2_sitk, sitk.AffineTransform(
            affine_centering_sitk.GetInverse()))
    motion_2_sitk = sitkh.get_composite_sitk_affine_transform(
        affine_centering_sitk, motion_2_sitk)
    stack_corrupted_2_resampled_sitk = sitk.Resample(
        stack.sitk, motion_2_sitk, sitk.sitkLinear)
    stack_corrupted_2_resampled_sitk_mask = sitk.Resample(
        stack.sitk_mask, motion_2_sitk, sitk.sitkLinear)

    # Create stack based on those two corrupted stacks
    nda = sitk.GetArrayFromImage(stack_corrupted_resampled_sitk)
    nda_mask = sitk.GetArrayFromImage(stack_corrupted_resampled_sitk_mask)
    nda_neg = sitk.GetArrayFromImage(stack_corrupted_2_resampled_sitk)
    nda_neg_mask = sitk.GetArrayFromImage(
        stack_corrupted_2_resampled_sitk_mask)
    for i in range(0, stack.sitk.GetDepth(), 2):
        nda[i, :, :] = nda_neg[i, :, :]
        nda_mask[i, :, :] = nda_neg_mask[i, :, :]
    stack_corrupted_sitk = sitk.GetImageFromArray(
        (nda-intensity_bias)/intensity_scale)
    stack_corrupted_sitk_mask = sitk.GetImageFromArray(nda_mask)
    stack_corrupted_sitk.CopyInformation(stack.sitk)
    stack_corrupted_sitk_mask.CopyInformation(stack.sitk_mask)

    # Debug: Show corrupted stacks (before scaling)
    if debug:
        sitkh.show_sitk_image(
            [stack.sitk,
             stack_corrupted_resampled_sitk,
             stack_corrupted_2_resampled_sitk,
             stack_corrupted_sitk],
            title=["original",
                   "corrupted_1",
                   "corrupted_2",
                   "corrupted_final_from_1_and_2"])

    # Update in-plane scaling
    spacing = np.array(stack.sitk.GetSpacing())
    spacing[0:-1] /= scale
    stack_corrupted_sitk.SetSpacing(spacing)
    stack_corrupted_sitk_mask.SetSpacing(spacing)

    # Create Stack object
    stack_corrupted = st.Stack.from_sitk_image(
        stack_corrupted_sitk, "stack_corrupted", stack_corrupted_sitk_mask)

    # Debug: Show corrupted stacks (after scaling)
    if debug:
        stack_corrupted_resampled_sitk = sitk.Resample(
            stack_corrupted.sitk, stack.sitk)
        sitkh.show_sitk_image(
            [stack.sitk,
             stack_corrupted_resampled_sitk],
            title=["original", "corrupted"])

    return stack_corrupted, motion_sitk, motion_2_sitk


# Concept of unit testing for python used in here is based on
#  http://pythontesting.net/framework/unittest/unittest-introduction/
#  Retrieved: Aug 6, 2015
class TestIntraStackRegistration(unittest.TestCase):

    # Specify input data
    dir_test_data = DIR_TEST

    accuracy = 6

    def setUp(self):
        pass

    ##
    #       Test whether the function
    #             _get_initial_transforms_and_parameters_geometry_moments
    #             works.
    # \date       2016-11-09 23:59:25+0000
    #
    # \param      self  The object
    #
    def test_initial_transform_computation_1(self):

        # Create stack of slice with only a dot in the middle
        shape_xy = 15
        shape_z = 15

        # Original stack
        nda_3D = np.zeros((shape_z, shape_xy, shape_xy))
        nda_3D[:, 0, 0] = 1
        stack_sitk = sitk.GetImageFromArray(nda_3D)
        stack = st.Stack.from_sitk_image(stack_sitk, "stack")

        # Create 'motion corrupted stack', i.e. point moves diagonally with
        # step one
        nda_3D_corruped = np.zeros_like(nda_3D)
        for i in range(0, shape_z):
            nda_3D_corruped[i, i, i] = 1
        stack_corrupted_sitk = sitk.GetImageFromArray(nda_3D_corruped)
        stack_corrupted = st.Stack.from_sitk_image(
            stack_corrupted_sitk, "stack_corrupted")
        # stack_corrupted.show_slices()
        # sitkh.show_stacks([stack, stack_corrupted])

        # Ground truth-parameter: zero angle but translation = (1, 1) from one
        # slice to the next
        parameters = np.ones((shape_z, 3))
        parameters[:, 0] = 0
        for i in range(0, shape_z):
            parameters[i, :] *= i

        # 1) Get initial transform in case no reference is given
        inplane_registration = inplanereg.IntraStackRegistration(
            stack_corrupted)
        inplane_registration.set_transform_initializer_type("moments")
        # inplane_registration.set_transform_initializer_type("identity")
        inplane_registration._run_registration_pipeline_initialization()

        parameters_est = inplane_registration.get_parameters()
        nda_diff = parameters - parameters_est
        self.assertEqual(np.round(
            np.linalg.norm(nda_diff), decimals=self.accuracy), 0)

        # 2) Get initial transform in case reference is given
        inplane_registration = inplanereg.IntraStackRegistration(
            stack_corrupted, stack)
        inplane_registration.set_transform_initializer_type("moments")
        # inplane_registration.set_image_transform_reference_fit_term("gradient_magnitude")
        # inplane_registration.set_transform_initializer_type("identity")
        inplane_registration._run_registration_pipeline_initialization()
        inplane_registration._apply_motion_correction()
        # stack_corrected = inplane_registration.get_corrected_stack()
        # sitkh.show_stacks([stack, stack_corrupted, stack_corrected.get_resampled_stack_from_slices(interpolator="Linear")])

        parameters_est = inplane_registration.get_parameters()
        nda_diff = parameters - parameters_est
        # print(nda_diff)
        # print(parameters)
        self.assertEqual(np.round(
            np.linalg.norm(nda_diff), decimals=self.accuracy), 0)

    ##
    #       Test whether the function
    #             _get_initial_transforms_and_parameters_geometry_moments
    #             works.
    # \date       2016-11-09 23:59:25+0000
    #
    # \param      self  The object
    #
    def test_initial_transform_computation_2(self):

        # Create stack of slice with a pyramid in the middle
        shape_xy = 250
        shape_z = 15

        intensity_mask = 10

        length = 50
        nda_2D = ph.read_image(os.path.join(
            DIR_TEST, "2D_Pyramid_Midpoint_" + str(length) + ".png"))

        # Original stack
        nda_3D = np.zeros((shape_z, shape_xy, shape_xy))
        i0 = (shape_xy - length) / 2

        for i in range(0, shape_z):
            nda_3D[i, i0:-i0, i0:-i0] = nda_2D

        stack_sitk = sitk.GetImageFromArray(nda_3D)
        stack = st.Stack.from_sitk_image(stack_sitk, "stack")

        # Create 'motion corrupted stack', i.e. in-plane translation, and
        # associated ground-truth parameters
        parameters = np.zeros((shape_z, 3))
        parameters[:, 0] = 0

        nda_3D_corrupted = np.zeros_like(nda_3D)
        nda_3D_corrupted[0, :, :] = nda_3D[0, :, :]
        for i in range(1, shape_z):
            # Get random translation
            [tx, ty] = np.random.randint(0, 50, 2)

            # Get image based on corruption
            inew = i0 + tx
            jnew = i0 + ty
            nda_3D_corrupted[i, inew:, jnew:] = \
                nda_3D[i, i0:2*i0+length-tx, i0:2*i0+length-ty]

            # Get ground-truth parameters
            parameters[i, 1] = ty
            parameters[i, 2] = tx

        stack_corrupted_sitk = sitk.GetImageFromArray(nda_3D_corrupted)
        stack_corrupted = st.Stack.from_sitk_image(
            stack_corrupted_sitk, "stack_corrupted")

        # stack_corrupted.show_slices()
        # sitkh.show_stacks([stack, stack_corrupted])

        # 1) Get initial transform in case no reference is given
        inplane_registration = inplanereg.IntraStackRegistration(
            stack_corrupted)
        inplane_registration.set_transform_initializer_type("moments")
        # inplane_registration.set_transform_initializer_type("identity")
        # inplane_registration.set_transform_initializer_type("geometry")
        inplane_registration._run_registration_pipeline_initialization()

        # Debug:
        # inplane_registration._apply_motion_correction()
        # stack_corrected = inplane_registration.get_corrected_stack()
        # sitkh.show_stacks(
        #     [stack,
        #      stack_corrupted,
        #      stack_corrected.get_resampled_stack_from_slices(
        #          interpolator="Linear", filename="stack_corrected")])

        parameters_est = inplane_registration.get_parameters()
        nda_diff = parameters - parameters_est
        self.assertEqual(np.round(
            np.linalg.norm(nda_diff), decimals=self.accuracy), 0)

        # 2) Get initial transform in case reference is given
        inplane_registration = inplanereg.IntraStackRegistration(
            stack_corrupted, stack)
        inplane_registration.set_transform_initializer_type("moments")
        # inplane_registration.set_transform_initializer_type("identity")
        inplane_registration._run_registration_pipeline_initialization()

        # Debug:
        # inplane_registration._apply_motion_correction()
        # stack_corrected = inplane_registration.get_corrected_stack()
        # sitkh.show_stacks(
        #     [stack,
        #      stack_corrupted,
        #      stack_corrected.get_resampled_stack_from_slices(
        #          interpolator="Linear", filename="stack_corrected")])

        parameters_est = inplane_registration.get_parameters()
        nda_diff = parameters - parameters_est
        # print(nda_diff)
        # print(parameters)
        self.assertEqual(np.round(
            np.linalg.norm(nda_diff), decimals=self.accuracy), 0)

    ##
    #       Test whether the function
    #             _get_initial_transforms_and_parameters_geometry_moments
    #             works.
    # \date       2016-11-09 23:59:25+0000
    #
    # \param      self  The object
    #
    def test_initial_transform_computation_3(self):

        # Create stack of slice with a pyramid in the middle
        shape_xy = 250
        shape_z = 15

        intensity_mask = 10

        length = 50
        nda_2D = ph.read_image(os.path.join(
            DIR_TEST, "2D_Pyramid_Midpoint_" + str(length) + ".png"))

        # Original stack
        nda_3D = np.zeros((shape_z, shape_xy, shape_xy))
        i0 = (shape_xy - length) / 2

        for i in range(0, shape_z):
            nda_3D[i, i0:-i0, i0:-i0] = nda_2D

        nda_3D_mask = np.array(nda_3D).astype(np.uint8)
        nda_3D_mask[np.where(nda_3D_mask <= intensity_mask)] = 0
        nda_3D_mask[np.where(nda_3D_mask > intensity_mask)] = 1

        # Add additional weight s.t. initialization without mask fails
        for i in range(0, shape_z):
            nda_3D[i, -i0:, -i0:] = 10

        stack_sitk = sitk.GetImageFromArray(nda_3D)
        stack_sitk_mask = sitk.GetImageFromArray(nda_3D_mask)
        stack = st.Stack.from_sitk_image(stack_sitk, "stack", stack_sitk_mask)

        # Create 'motion corrupted stack', i.e. in-plane translation, and
        # associated ground-truth parameters
        parameters = np.zeros((shape_z, 3))
        parameters[:, 0] = 0

        nda_3D_corrupted = np.zeros_like(nda_3D)
        nda_3D_corrupted[0, :, :] = nda_3D[0, :, :]
        nda_3D_corrupted_mask = np.zeros_like(nda_3D_mask)
        nda_3D_corrupted_mask[0, :, :] = nda_3D_mask[0, :, :]
        for i in range(1, shape_z):
            # Get random translation
            [tx, ty] = np.random.randint(0, 50, 2)

            # Get image based on corruption
            inew = i0 + tx
            jnew = i0 + ty
            nda_3D_corrupted[i, inew:, jnew:] = \
                nda_3D[i, i0:2*i0+length-tx, i0:2*i0+length-ty]

            nda_3D_corrupted_mask[i, inew:, jnew:] = \
                nda_3D_mask[i, i0:2*i0+length-tx, i0:2*i0+length-ty]

            # Get ground-truth parameters
            parameters[i, 1] = ty
            parameters[i, 2] = tx

        # nda_3D_corrupted = np.zeros_like(nda_3D)
        # nda_3D_corrupted[0, i0:-i0, i0:-i0] = nda_2D
        # for i in range(1, shape_z):
        #     # Get random translation
        #     [tx, ty] = np.random.randint(0, 50, 2)

        #     # Get image based on corruption
        #     inew = i0 + tx
        #     jnew = i0 + ty
        #     nda_3D_corrupted[i, inew:inew+length, jnew:jnew+length] = nda_2D

        #     # Get ground-truth parameters
        #     parameters[i, 1] = ty
        #     parameters[i, 2] = tx

        stack_corrupted_sitk = sitk.GetImageFromArray(nda_3D_corrupted)
        stack_corrupted_sitk_mask = sitk.GetImageFromArray(
            nda_3D_corrupted_mask)
        stack_corrupted = st.Stack.from_sitk_image(
            stack_corrupted_sitk, "stack_corrupted", stack_corrupted_sitk_mask)
        # stack_corrupted.show(1)
        # stack_corrupted.show_slices()
        # sitkh.show_stacks([stack, stack_corrupted],
        #     segmentation=stack)

        # 1) Get initial transform in case no reference is given
        inplane_registration = inplanereg.IntraStackRegistration(
            stack_corrupted,
            use_stack_mask=True,
        )
        inplane_registration.set_transform_initializer_type("moments")
        # inplane_registration.set_transform_initializer_type("identity")
        # inplane_registration.set_transform_initializer_type("geometry")
        inplane_registration._run_registration_pipeline_initialization()

        # Debug:
        # inplane_registration._apply_motion_correction()
        # stack_corrected = inplane_registration.get_corrected_stack()
        # sitkh.show_stacks(
        #     [stack,
        #      stack_corrupted,
        #      stack_corrected.get_resampled_stack_from_slices(
        #          interpolator="Linear", filename="stack_corrected")])

        parameters_est = inplane_registration.get_parameters()
        nda_diff = parameters - parameters_est
        self.assertEqual(np.round(
            np.linalg.norm(nda_diff), decimals=self.accuracy), 0)

        # 2) Get initial transform in case reference is given
        inplane_registration = inplanereg.IntraStackRegistration(
            stack_corrupted, stack)
        inplane_registration.set_transform_initializer_type("moments")
        # inplane_registration.set_transform_initializer_type("identity")
        inplane_registration.use_reference_mask(True)
        inplane_registration.use_stack_mask_reference_fit_term(True)
        inplane_registration._run_registration_pipeline_initialization()
        # Debug:
        # inplane_registration._apply_motion_correction()
        # stack_corrected = inplane_registration.get_corrected_stack()
        # sitkh.show_stacks(
        #     [stack,
        #      stack_corrupted,
        #      stack_corrected.get_resampled_stack_from_slices(
        #          interpolator="Linear", filename="stack_corrected")])

        parameters_est = inplane_registration.get_parameters()
        nda_diff = parameters - parameters_est
        # print(nda_diff)
        # print(parameters)
        self.assertEqual(np.round(
            np.linalg.norm(nda_diff), decimals=self.accuracy), 0)

    ##
    #       Test that initial intensity coefficients are computed
    #             correctly
    # \date       2016-11-10 04:28:06+0000
    #
    # \param      self  The object
    #
    def test_initial_intensity_coefficient_computation(self):
        # Create stack
        shape_z = 15
        nda_2D = imread(self.dir_test_data + "2D_Lena_256.png", flatten=True)
        nda_3D = np.tile(nda_2D, (shape_z, 1, 1)).astype('double')
        stack_sitk = sitk.GetImageFromArray(nda_3D)
        stack = st.Stack.from_sitk_image(stack_sitk, "Lena")

        # 1) Create linearly corrupted intensity stack
        nda_3D_corruped = np.zeros_like(nda_3D)
        for i in range(0, shape_z):
            nda_3D_corruped[i, :, :] = nda_3D[i, :, :]/(i+1.)
        stack_corrupted_sitk = sitk.GetImageFromArray(nda_3D_corruped)
        stack_corrupted = st.Stack.from_sitk_image(
            stack_corrupted_sitk, "stack_corrupted")
        # stack_corrupted.show_slices()
        # sitkh.show_stacks([stack, stack_corrupted])

        # Ground truth-parameter: zero angle but translation = (1, 1) from one
        # slice to the next
        parameters = np.zeros((shape_z, 4))
        parameters[:, 0] = 0
        for i in range(0, shape_z):
            parameters[i, 3:] = 1*(i+1.)  # intensity

        # Get initial transform in case no reference is given
        inplane_registration = inplanereg.IntraStackRegistration(
            stack_corrupted, stack)
        # inplane_registration.set_transform_initializer_type("moments")
        inplane_registration.set_intensity_correction_type_slice_neighbour_fit(
            "linear")
        inplane_registration.set_intensity_correction_initializer_type(
            "linear")
        inplane_registration._run_registration_pipeline_initialization()

        parameters_est = inplane_registration.get_parameters()
        nda_diff = parameters - parameters_est
        self.assertEqual(np.round(
            np.linalg.norm(nda_diff), decimals=self.accuracy), 0)

        # 2) Create affinely corrupted intensity stack
        # HINT: In case of individual slice correction is active!!
        nda_3D_corruped = np.zeros_like(nda_3D)
        for i in range(0, shape_z):
            nda_3D_corruped[i, :, :] = (nda_3D[i, :, :]-10*i)/(i+1.)
        stack_corrupted_sitk = sitk.GetImageFromArray(nda_3D_corruped)
        stack_corrupted = st.Stack.from_sitk_image(
            stack_corrupted_sitk, "stack_corrupted")
        # stack_corrupted.show_slices()
        # sitkh.show_stacks([stack, stack_corrupted])

        # Ground truth-parameter: zero angle but translation = (1, 1) from one
        # slice to the next
        parameters = np.zeros((shape_z, 5))
        parameters[:, 0] = 0
        for i in range(0, shape_z):
            parameters[i, 3:] = (i+1, 10*i)  # intensity

        # Get initial transform in case no reference is given
        inplane_registration = inplanereg.IntraStackRegistration(
            stack_corrupted, stack)
        # inplane_registration.set_transform_initializer_type("moments")
        inplane_registration.set_intensity_correction_type_slice_neighbour_fit(
            "affine")
        inplane_registration.set_intensity_correction_initializer_type(
            "affine")
        inplane_registration._run_registration_pipeline_initialization()

        parameters_est = inplane_registration.get_parameters()
        nda_diff = parameters - parameters_est
        self.assertEqual(np.round(
            np.linalg.norm(nda_diff), decimals=self.accuracy), 0)

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
    def test_inplane_rigid_alignment_to_neighbour(self):

        filename_stack = "fetal_brain_0"
        # filename_recon = "FetalBrain_reconstruction_3stacks_myAlg"

        # stack_sitk = sitk.ReadImage(self.dir_test_data + filename_stack + ".nii.gz")
        # recon_sitk = sitk.ReadImage(self.dir_test_data + filename_recon + ".nii.gz")

        # recon_resampled_sitk = sitk.Resample(recon_sitk, stack_sitk)
        # stack = st.Stack.from_sitk_image(recon_resampled_sitk, "original")

        stack = st.Stack.from_filename(
            os.path.join(self.dir_test_data, filename_stack + ".nii.gz"),
            os.path.join(self.dir_test_data, filename_stack + "_mask.nii.gz")
        )

        nda = sitk.GetArrayFromImage(stack.sitk)
        nda_mask = sitk.GetArrayFromImage(stack.sitk_mask)
        i = 5
        nda_slice = np.array(nda[i, :, :])
        nda_mask_slice = np.array(nda_mask[i, :, :])

        for i in range(0, nda.shape[0]):
            nda[i, :, :] = nda_slice
            nda_mask[i, :, :] = nda_mask_slice

        stack_sitk = sitk.GetImageFromArray(nda)
        stack_sitk_mask = sitk.GetImageFromArray(nda_mask)
        stack_sitk.CopyInformation(stack.sitk)
        stack_sitk_mask.CopyInformation(stack.sitk_mask)

        stack = st.Stack.from_sitk_image(
            stack_sitk, stack.get_filename(), stack_sitk_mask)

        # Create in-plane motion corruption
        angle_z = 0.1
        center_2D = (0, 0)
        translation_2D = np.array([1, -2])

        # Get corrupted stack and corresponding motions
        stack_corrupted, motion_sitk, motion_2_sitk = get_inplane_corrupted_stack(
            stack, angle_z, center_2D, translation_2D, random=True)

        # stack.show(1)
        # stack_corrupted.show(1)

        # Perform in-plane rigid registration
        inplane_registration = inplanereg.IntraStackRegistration(
            stack_corrupted, stack)
        # inplane_registration = inplanereg.IntraStackRegistration(stack_corrupted)
        inplane_registration.set_transform_initializer_type("moments")
        inplane_registration.set_optimizer_iter_max(20)
        inplane_registration.set_alpha_neighbour(1)
        inplane_registration.set_alpha_reference(2)
        # inplane_registration.use_parameter_normalization(True)
        inplane_registration.use_stack_mask(1)
        inplane_registration.use_reference_mask(0)
        # inplane_registration.set_optimizer_loss("linear") # linear, soft_l1,
        # huber
        inplane_registration.set_optimizer_method("trf")  # trf, lm, dogbox
        # inplane_registration._run_registration_pipeline_initialization()
        # inplane_registration._apply_motion_correction()
        inplane_registration.use_verbose(True)
        inplane_registration.run_registration()
        inplane_registration.print_statistics()

        stack_registered = inplane_registration.get_corrected_stack()
        parameters = inplane_registration.get_parameters()

        sitkh.show_stacks([stack, stack_corrupted, stack_registered.get_resampled_stack_from_slices(
            interpolator="Linear")])

        # self.assertEqual(np.round(
        #     np.linalg.norm(nda_diff)
        # , decimals = self.accuracy), 0)

        # 2) Test slice transforms
        slice_transforms_sitk = inplane_registration.get_slice_transforms_sitk()

        stack_tmp = st.Stack.from_stack(stack_corrupted)
        stack_tmp.update_motion_correction_of_slices(slice_transforms_sitk)

        stack_diff_sitk = stack_tmp.get_resampled_stack_from_slices(
            resampling_grid=stack.sitk).sitk - stack_registered.get_resampled_stack_from_slices(resampling_grid=stack.sitk).sitk

        stack_diff_nda = sitk.GetArrayFromImage(stack_diff_sitk)

        self.assertEqual(np.round(
            np.linalg.norm(stack_diff_nda), decimals=8), 0)

    def test_inplane_rigid_alignment_to_reference(self):

        filename_stack = "fetal_brain_0"
        # filename_recon = "FetalBrain_reconstruction_3stacks_myAlg"

        # stack_sitk = sitk.ReadImage(self.dir_test_data + filename_stack + ".nii.gz")
        # recon_sitk = sitk.ReadImage(self.dir_test_data + filename_recon + ".nii.gz")

        # recon_resampled_sitk = sitk.Resample(recon_sitk, stack_sitk)
        # stack = st.Stack.from_sitk_image(recon_resampled_sitk, "original")

        stack = st.Stack.from_filename(
            os.path.join(self.dir_test_data, filename_stack + ".nii.gz"),
            os.path.join(self.dir_test_data, filename_stack + "_mask.nii.gz")
        )

        # Create in-plane motion corruption
        angle_z = 0.1
        center_2D = (0, 0)
        translation_2D = np.array([1, -2])

        # Get corrupted stack and corresponding motions
        stack_corrupted, motion_sitk, motion_2_sitk = get_inplane_corrupted_stack(
            stack, angle_z, center_2D, translation_2D)

        # stack.show(1)
        # stack_corrupted.show(1)

        # Perform in-plane rigid registration
        inplane_registration = inplanereg.IntraStackRegistration(
            stack_corrupted, stack)
        # inplane_registration = inplanereg.IntraStackRegistration(stack_corrupted)
        inplane_registration.set_transform_initializer_type("moments")
        inplane_registration.set_optimizer_iter_max(10)
        inplane_registration.set_alpha_neighbour(0)
        inplane_registration.set_alpha_parameter(0)
        inplane_registration.use_stack_mask(1)
        inplane_registration.use_reference_mask(0)
        inplane_registration.set_optimizer_loss("linear")
        # inplane_registration.set_optimizer_method("trf")
        # inplane_registration._run_registration_pipeline_initialization()
        # inplane_registration._apply_motion_correction()
        # inplane_registration.use_verbose(True)
        inplane_registration.run_registration()
        inplane_registration.print_statistics()

        stack_registered = inplane_registration.get_corrected_stack()
        parameters = inplane_registration.get_parameters()

        sitkh.show_stacks([stack, stack_corrupted, stack_registered.get_resampled_stack_from_slices(
            interpolator="Linear", resampling_grid=stack.sitk)])

        print(parameters)

        # self.assertEqual(np.round(
        #     np.linalg.norm(nda_diff)
        # , decimals = self.accuracy), 0)

        # 2) Test slice transforms
        slice_transforms_sitk = inplane_registration.get_slice_transforms_sitk()

        stack_tmp = st.Stack.from_stack(stack_corrupted)
        stack_tmp.update_motion_correction_of_slices(slice_transforms_sitk)

        stack_diff_sitk = stack_tmp.get_resampled_stack_from_slices(
            resampling_grid=stack.sitk).sitk - stack_registered.get_resampled_stack_from_slices(resampling_grid=stack.sitk).sitk

        stack_diff_nda = sitk.GetArrayFromImage(stack_diff_sitk)

        self.assertEqual(np.round(
            np.linalg.norm(stack_diff_nda), decimals=8), 0)

    def test_inplane_rigid_alignment_to_reference_with_intensity_correction_linear(self):

        filename_stack = "fetal_brain_0"
        filename_recon = "FetalBrain_reconstruction_3stacks_myAlg"

        stack_sitk = sitk.ReadImage(
            self.dir_test_data + filename_stack + ".nii.gz")
        recon_sitk = sitk.ReadImage(
            self.dir_test_data + filename_recon + ".nii.gz")

        recon_resampled_sitk = sitk.Resample(recon_sitk, stack_sitk)
        stack = st.Stack.from_sitk_image(recon_resampled_sitk, "original")

        # Create in-plane motion corruption
        angle_z = 0.05
        center_2D = (0, 0)
        translation_2D = np.array([1, -2])

        intensity_scale = 10
        intensity_bias = 0

        # Get corrupted stack and corresponding motions
        stack_corrupted, motion_sitk, motion_2_sitk = get_inplane_corrupted_stack(
            stack, angle_z, center_2D, translation_2D, intensity_scale=intensity_scale, intensity_bias=intensity_bias)

        # Perform in-plane rigid registration
        inplane_registration = inplanereg.IntraStackRegistration(
            stack_corrupted, stack)
        # inplane_registration = inplanereg.IntraStackRegistration(stack_corrupted)
        inplane_registration.set_transform_initializer_type("moments")
        inplane_registration.set_transform_type("rigid")
        inplane_registration.set_intensity_correction_initializer_type(
            "linear")
        inplane_registration.set_intensity_correction_type_slice_neighbour_fit(
            "linear")
        inplane_registration.set_intensity_correction_type_reference_fit(
            "linear")
        inplane_registration.set_optimizer_loss(
            "linear")  # linear, soft_l1, huber
        inplane_registration.use_parameter_normalization(True)
        inplane_registration.use_verbose(True)
        inplane_registration.set_alpha_reference(1)
        inplane_registration.set_alpha_neighbour(0)
        inplane_registration.set_alpha_parameter(0)
        inplane_registration.set_optimizer_iter_max(30)
        inplane_registration.use_verbose(True)
        inplane_registration.run_registration()
        inplane_registration.print_statistics()

        stack_registered = inplane_registration.get_corrected_stack()
        parameters = inplane_registration.get_parameters()

        sitkh.show_stacks([stack, stack_corrupted, stack_registered.get_resampled_stack_from_slices(
            resampling_grid=None, interpolator="Linear")])

        print("Final parameters:")
        print(parameters)

        self.assertEqual(np.round(
            np.linalg.norm(parameters[:, -1] - intensity_scale), decimals=0), 0)

        # 2) Test slice transforms
        slice_transforms_sitk = inplane_registration.get_slice_transforms_sitk()

        stack_tmp = st.Stack.from_stack(stack_corrupted)
        stack_tmp.update_motion_correction_of_slices(slice_transforms_sitk)

        stack_diff_sitk = stack_tmp.get_resampled_stack_from_slices(
            resampling_grid=stack.sitk).sitk - stack_registered.get_resampled_stack_from_slices(resampling_grid=stack.sitk).sitk

        stack_diff_nda = sitk.GetArrayFromImage(stack_diff_sitk)

        self.assertEqual(np.round(
            np.linalg.norm(stack_diff_nda), decimals=8), 0)

    ##
    # \bug There is some issue with slice based and uniform intensity correction.
    # Unit test needs to be fixed at some point
    # \date       2017-07-12 12:40:01+0100
    #
    # \param      self  The object
    #
    def test_inplane_rigid_alignment_to_reference_with_intensity_correction_affine(self):

        filename_stack = "fetal_brain_0"
        filename_recon = "FetalBrain_reconstruction_3stacks_myAlg"

        stack_sitk = sitk.ReadImage(
            self.dir_test_data + filename_stack + ".nii.gz")
        recon_sitk = sitk.ReadImage(
            self.dir_test_data + filename_recon + ".nii.gz")

        recon_resampled_sitk = sitk.Resample(recon_sitk, stack_sitk)
        stack = st.Stack.from_sitk_image(recon_resampled_sitk, "original")

        # Create in-plane motion corruption
        angle_z = 0.01
        center_2D = (0, 0)
        translation_2D = np.array([1, 0])

        intensity_scale = 5
        intensity_bias = 5

        # Get corrupted stack and corresponding motions
        stack_corrupted, motion_sitk, motion_2_sitk = get_inplane_corrupted_stack(
            stack, angle_z, center_2D, translation_2D, intensity_scale=intensity_scale, intensity_bias=intensity_bias)

        # Perform in-plane rigid registration
        inplane_registration = inplanereg.IntraStackRegistration(
            stack_corrupted, stack)
        # inplane_registration = inplanereg.IntraStackRegistration(stack_corrupted)
        inplane_registration.set_transform_type("rigid")
        inplane_registration.set_transform_initializer_type("identity")
        inplane_registration.set_optimizer_loss("linear")
        inplane_registration.set_intensity_correction_initializer_type(
            "affine")
        inplane_registration.set_intensity_correction_type_slice_neighbour_fit(
            "affine")
        inplane_registration.use_parameter_normalization(True)
        inplane_registration.use_verbose(True)
        inplane_registration.use_stack_mask(True)
        inplane_registration.set_prior_intensity_coefficients(
            (intensity_scale-0.4, intensity_bias+0.7))
        inplane_registration.set_alpha_reference(1)
        inplane_registration.set_alpha_neighbour(1)
        inplane_registration.set_alpha_parameter(1e3)
        inplane_registration.set_optimizer_iter_max(15)
        inplane_registration.use_verbose(True)
        inplane_registration.run_registration()
        inplane_registration.print_statistics()

        stack_registered = inplane_registration.get_corrected_stack()
        parameters = inplane_registration.get_parameters()

        sitkh.show_stacks([stack, stack_corrupted, stack_registered.get_resampled_stack_from_slices(
            resampling_grid=None, interpolator="Linear")])

        self.assertEqual(np.round(
            np.linalg.norm(parameters[:, -2:] - np.array([intensity_scale, intensity_bias])), decimals=0), 0)

        # 2) Test slice transforms
        slice_transforms_sitk = inplane_registration.get_slice_transforms_sitk()

        stack_tmp = st.Stack.from_stack(stack_corrupted)
        stack_tmp.update_motion_correction_of_slices(slice_transforms_sitk)

        stack_diff_sitk = stack_tmp.get_resampled_stack_from_slices(
            resampling_grid=stack.sitk).sitk - stack_registered.get_resampled_stack_from_slices(resampling_grid=stack.sitk).sitk

        stack_diff_nda = sitk.GetArrayFromImage(stack_diff_sitk)

        self.assertEqual(np.round(
            np.linalg.norm(stack_diff_nda), decimals=8), 0)

    def test_inplane_similarity_alignment_to_reference(self):

        filename_stack = "fetal_brain_0"
        # filename_stack = "3D_SheppLoganPhantom_64"

        stack = st.Stack.from_filename(
            os.path.join(self.dir_test_data, filename_stack + ".nii.gz"),
            os.path.join(self.dir_test_data, filename_stack + "_mask.nii.gz")
        )
        # stack.show(1)

        nda = sitk.GetArrayFromImage(stack.sitk)
        nda_mask = sitk.GetArrayFromImage(stack.sitk_mask)
        i = 5
        nda_slice = np.array(nda[i, :, :])
        nda_mask_slice = np.array(nda_mask[i, :, :])

        for i in range(0, nda.shape[0]):
            nda[i, :, :] = nda_slice
            nda_mask[i, :, :] = nda_mask_slice

        stack_sitk = sitk.GetImageFromArray(nda)
        stack_sitk_mask = sitk.GetImageFromArray(nda_mask)
        stack_sitk.CopyInformation(stack.sitk)
        stack_sitk_mask.CopyInformation(stack.sitk_mask)

        stack = st.Stack.from_sitk_image(
            stack_sitk, stack.get_filename(), stack_sitk_mask)

        # Create in-plane motion corruption
        scale = 1.2
        angle_z = 0.05
        center_2D = (0, 0)
        # translation_2D = np.array([0,0])
        translation_2D = np.array([1, -1])

        intensity_scale = 10
        intensity_bias = 50

        # Get corrupted stack and corresponding motions
        stack_corrupted, motion_sitk, motion_2_sitk = get_inplane_corrupted_stack(
            stack, angle_z, center_2D, translation_2D, scale=scale, intensity_scale=intensity_scale, intensity_bias=intensity_bias, debug=0)

        # stack_corrupted.show(1)
        # stack.show(1)

        # Perform in-plane rigid registrations
        inplane_registration = inplanereg.IntraStackRegistration(
            stack=stack_corrupted, reference=stack)
        # inplane_registration = inplanereg.IntraStackRegistration(stack_corrupted)
        inplane_registration.set_transform_initializer_type("geometry")
        # inplane_registration.set_transform_initializer_type("identity")
        inplane_registration.set_intensity_correction_initializer_type(
            "affine")
        inplane_registration.set_transform_type("similarity")
        inplane_registration.set_interpolator("Linear")
        inplane_registration.set_optimizer_loss("linear")
        # inplane_registration.use_reference_mask(True)
        inplane_registration.use_stack_mask(True)
        inplane_registration.use_parameter_normalization(True)
        inplane_registration.set_prior_scale(1/scale)
        inplane_registration.set_prior_intensity_coefficients(
            (intensity_scale, intensity_bias))
        inplane_registration.set_intensity_correction_type_slice_neighbour_fit(
            "affine")
        inplane_registration.set_intensity_correction_type_reference_fit(
            "affine")
        inplane_registration.use_verbose(True)
        inplane_registration.set_alpha_reference(1)
        inplane_registration.set_alpha_neighbour(0)
        inplane_registration.set_alpha_parameter(1e10)
        inplane_registration.set_optimizer_iter_max(20)
        inplane_registration.use_verbose(True)
        inplane_registration.run_registration()
        inplane_registration.print_statistics()

        # inplane_registration._run_registration_pipeline_initialization()
        # inplane_registration._apply_motion_correction()

        stack_registered = inplane_registration.get_corrected_stack()
        parameters = inplane_registration.get_parameters()

        sitkh.show_sitk_image([stack.sitk, stack_corrupted.get_resampled_stack_from_slices(interpolator="Linear", resampling_grid=stack.sitk).sitk,
                               stack_registered.get_resampled_stack_from_slices(interpolator="Linear", resampling_grid=stack.sitk).sitk], label=["original", "corrupted", "recovered"])

        # self.assertEqual(np.round(
        #     np.linalg.norm(nda_diff)
        # , decimals = self.accuracy), 0)

        # 2) Test slice transforms
        slice_transforms_sitk = inplane_registration.get_slice_transforms_sitk()

        stack_tmp = st.Stack.from_stack(stack_corrupted)
        stack_tmp.update_motion_correction_of_slices(slice_transforms_sitk)

        stack_diff_sitk = stack_tmp.get_resampled_stack_from_slices(
            resampling_grid=stack.sitk).sitk - stack_registered.get_resampled_stack_from_slices(resampling_grid=stack.sitk).sitk

        stack_diff_nda = sitk.GetArrayFromImage(stack_diff_sitk)

        self.assertEqual(np.round(
            np.linalg.norm(stack_diff_nda), decimals=8), 0)

    def test_inplane_rigid_alignment_to_reference_multimodal(self):

        filename_stack = "fetal_brain_0"
        filename_recon = "FetalBrain_reconstruction_3stacks_myAlg"

        stack_tmp = st.Stack.from_filename(
            os.path.join(self.dir_test_data, filename_stack + ".nii.gz"),
            os.path.join(self.dir_test_data, filename_stack + "_mask.nii.gz")
        )

        recon = st.Stack.from_filename(
            os.path.join(self.dir_test_data, filename_recon)
        )

        recon_sitk = recon.get_resampled_stack_from_slices(
            resampling_grid=stack_tmp.sitk, interpolator="Linear").sitk

        stack = st.Stack.from_sitk_image(
            recon_sitk, "original", stack_tmp.sitk_mask)

        # recon_resampled_sitk = sitk.Resample(recon_sitk, stack_sitk)
        # stack = st.Stack.from_sitk_image(recon_resampled_sitk, "original")

        # Create in-plane motion corruption
        scale = 1.05
        angle_z = 0.05
        center_2D = (0, 0)
        translation_2D = np.array([1, -2])

        intensity_scale = 1
        intensity_bias = 0

        # Get corrupted stack and corresponding motions
        stack_corrupted, motion_sitk, motion_2_sitk = get_inplane_corrupted_stack(
            stack, angle_z, center_2D, translation_2D, intensity_scale=intensity_scale, scale=scale, intensity_bias=intensity_bias)

        # stack_corrupted.show(1)
        # stack.show(1)

        # Perform in-plane rigid registration
        inplane_registration = inplanereg.IntraStackRegistration(
            stack_corrupted, stack)
        # inplane_registration = inplanereg.IntraStackRegistration(stack_corrupted)
        # inplane_registration.set_image_transform_reference_fit_term("gradient_magnitude")
        inplane_registration.set_image_transform_reference_fit_term(
            "partial_derivative")
        inplane_registration.set_transform_initializer_type("moments")
        # inplane_registration.set_transform_type("similarity")
        inplane_registration.set_intensity_correction_initializer_type(None)
        inplane_registration.set_intensity_correction_type_slice_neighbour_fit(
            None)
        inplane_registration.set_intensity_correction_type_reference_fit(None)
        inplane_registration.use_parameter_normalization(True)
        inplane_registration.use_verbose(True)
        inplane_registration.set_optimizer_loss(
            "linear")  # linear, soft_l1, huber
        inplane_registration.set_alpha_reference(100)
        inplane_registration.set_alpha_neighbour(0)
        inplane_registration.set_alpha_parameter(1)
        # inplane_registration.use_stack_mask(True)
        # inplane_registration.use_reference_mask(True)
        inplane_registration.set_optimizer_iter_max(10)
        inplane_registration.run_registration()
        inplane_registration.print_statistics()

        stack_registered = inplane_registration.get_corrected_stack()
        parameters = inplane_registration.get_parameters()

        sitkh.show_stacks([stack, stack_corrupted, stack_registered.get_resampled_stack_from_slices(
            resampling_grid=None, interpolator="Linear")])

        # print("Final parameters:")
        # print(parameters)

        # self.assertEqual(np.round(
        #     np.linalg.norm(parameters[:,-1] - intensity_scale)
        # , decimals = 0), 0)

        # 2) Test slice transforms
        slice_transforms_sitk = inplane_registration.get_slice_transforms_sitk()

        stack_tmp = st.Stack.from_stack(stack_corrupted)
        stack_tmp.update_motion_correction_of_slices(slice_transforms_sitk)

        stack_diff_sitk = stack_tmp.get_resampled_stack_from_slices(
            resampling_grid=stack.sitk).sitk - stack_registered.get_resampled_stack_from_slices(resampling_grid=stack.sitk).sitk

        stack_diff_nda = sitk.GetArrayFromImage(stack_diff_sitk)

        self.assertEqual(np.round(
            np.linalg.norm(stack_diff_nda), decimals=8), 0)

    def test_inplane_uniform_scale_similarity_alignment_to_reference(self):

        filename_stack = "fetal_brain_0"
        # filename_stack = "3D_SheppLoganPhantom_64"

        stack = st.Stack.from_filename(
            os.path.join(self.dir_test_data, filename_stack + ".nii.gz"),
            os.path.join(self.dir_test_data, filename_stack + "_mask.nii.gz")
        )
        # stack.show(1)

        nda = sitk.GetArrayFromImage(stack.sitk)
        nda_mask = sitk.GetArrayFromImage(stack.sitk_mask)
        i = 5
        nda_slice = np.array(nda[i, :, :])
        nda_mask_slice = np.array(nda_mask[i, :, :])

        for i in range(0, nda.shape[0]): #23 slices
            nda[i, :, :] = nda_slice
            nda_mask[i, :, :] = nda_mask_slice

        stack_sitk = sitk.GetImageFromArray(nda)
        stack_sitk_mask = sitk.GetImageFromArray(nda_mask)
        stack_sitk.CopyInformation(stack.sitk)
        stack_sitk_mask.CopyInformation(stack.sitk_mask)

        stack = st.Stack.from_sitk_image(
            stack_sitk, stack.get_filename(), stack_sitk_mask)

        # Create in-plane motion corruption
        # scale = 1.2
        scale = 1
        angle_z = 0.05
        center_2D = (0, 0)
        # translation_2D = np.array([0,0])
        translation_2D = np.array([1, -1])

        intensity_scale = 1
        intensity_bias = 0

        # Get corrupted stack and corresponding motions
        stack_corrupted, motion_sitk, motion_2_sitk = get_inplane_corrupted_stack(
            stack, angle_z, center_2D, translation_2D, scale=scale, intensity_scale=intensity_scale, intensity_bias=intensity_bias, debug=0)

        # stack_corrupted.show(1)
        # stack.show(1)

        # Perform in-plane rigid registrations
        inplane_registration = inplanereg.IntraStackRegistration(
            stack=stack_corrupted,
            reference=stack,
            use_stack_mask=True,
            use_reference_mask=True,
            interpolator="Linear",
            use_verbose=True,
            )
        # inplane_registration = inplanereg.IntraStackRegistration(stack_corrupted)
        inplane_registration.set_transform_initializer_type("geometry")
        # inplane_registration.set_transform_initializer_type("identity")
        inplane_registration.set_intensity_correction_initializer_type(
            "affine")
        # inplane_registration.set_transform_type("similarity")
        inplane_registration.set_transform_type("rigid")
        # inplane_registration.set_optimizer("least_squares")
        # inplane_registration.set_optimizer("BFGS")
        # inplane_registration.set_optimizer("L-BFGS-B")
        inplane_registration.set_optimizer("TNC")
        # inplane_registration.set_optimizer("Powell")
        # inplane_registration.set_optimizer("CG")
        # inplane_registration.set_optimizer("Newton-CG")
        inplane_registration.set_optimizer_loss("linear")
        # inplane_registration.set_optimizer_loss("soft_l1")
        # inplane_registration.set_optimizer_loss("arctan")
        # inplane_registration.use_parameter_normalization(True)
        inplane_registration.set_prior_scale(1/scale)
        inplane_registration.set_prior_intensity_coefficients(
            (intensity_scale, intensity_bias))
        # inplane_registration.set_intensity_correction_type_slice_neighbour_fit(
            # "affine")
        # inplane_registration.set_intensity_correction_type_reference_fit(
            # "affine")
        inplane_registration.set_alpha_reference(1)
        inplane_registration.set_alpha_neighbour(0)
        inplane_registration.set_alpha_parameter(0)
        inplane_registration.set_optimizer_iter_max(30)
        inplane_registration.run_registration()
        inplane_registration.print_statistics()

        # inplane_registration._run_registration_pipeline_initialization()
        # inplane_registration._apply_motion_correction()

        stack_registered = inplane_registration.get_corrected_stack()
        parameters = inplane_registration.get_parameters()

        sitkh.show_sitk_image([stack.sitk, stack_corrupted.get_resampled_stack_from_slices(interpolator="Linear", resampling_grid=stack.sitk).sitk,
                               stack_registered.get_resampled_stack_from_slices(interpolator="Linear", resampling_grid=stack.sitk).sitk], label=["original", "corrupted", "recovered"])

        # self.assertEqual(np.round(
        #     np.linalg.norm(nda_diff)
        # , decimals = self.accuracy), 0)

        # 2) Test slice transforms
        slice_transforms_sitk = inplane_registration.get_slice_transforms_sitk()

        stack_tmp = st.Stack.from_stack(stack_corrupted)
        stack_tmp.update_motion_correction_of_slices(slice_transforms_sitk)

        stack_diff_sitk = stack_tmp.get_resampled_stack_from_slices(
            resampling_grid=stack.sitk).sitk - stack_registered.get_resampled_stack_from_slices(resampling_grid=stack.sitk).sitk

        stack_diff_nda = sitk.GetArrayFromImage(stack_diff_sitk)

        self.assertEqual(np.round(
            np.linalg.norm(stack_diff_nda), decimals=8), 0)