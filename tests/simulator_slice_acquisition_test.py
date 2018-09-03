# \file simulator_slice_acquisition_test.py
#  \brief  Class containing unit tests for module SimulatorSliceAcqusition
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date May 2016


import unittest

# Import libraries
import SimpleITK as sitk
import itk
import numpy as np
import os

import niftymic.base.psf as psf
# Import modules
import niftymic.base.stack as st
import niftymic.prototyping.simulator_slice_acqusition as sa
import pysitk.simple_itk_helper as sitkh
from niftymic.definitions import DIR_TEST

# Pixel type of used 3D ITK image
pixel_type = itk.D

# ITK image type
image_type = itk.Image[pixel_type, 3]


class SimulatorSliceAcqusitionTest(unittest.TestCase):

    # Specify input data
    dir_test_data = DIR_TEST

    accuracy = 7

    def setUp(self):
        pass

    def test_conversion_image_direction(self):

        filename_HR_volume = "HR_volume_postmortem"
        HR_volume = st.Stack.from_filename(
            os.path.join(self.dir_test_data, filename_HR_volume + ".nii.gz"),
            os.path.join(self.dir_test_data,
                         filename_HR_volume + "_mask.nii.gz")
        )
        # Get unit vectors defining image grid in physical space and construct
        # direction matrix
        origin_HR_volume = np.array(HR_volume.sitk.GetOrigin())
        a_x = HR_volume.sitk.TransformIndexToPhysicalPoint(
            (1, 0, 0)) - origin_HR_volume
        a_y = HR_volume.sitk.TransformIndexToPhysicalPoint(
            (0, 1, 0)) - origin_HR_volume
        a_z = HR_volume.sitk.TransformIndexToPhysicalPoint(
            (0, 0, 1)) - origin_HR_volume

        e_x = a_x/np.linalg.norm(a_x)
        e_y = a_y/np.linalg.norm(a_y)
        e_z = a_z/np.linalg.norm(a_z)

        direction_matrix_test = np.array([e_x, e_y, e_z]).transpose()
        direction_test = direction_matrix_test.flatten()

        # Get respective vectors from image direction
        direction = np.array(HR_volume.sitk.GetDirection())

        e_x_test = direction[0::3]
        e_y_test = direction[1::3]
        e_z_test = direction[2::3]

        # Check correspondences
        self.assertEqual(np.round(np.linalg.norm(
            e_x_test - e_x), decimals=self.accuracy), 0)
        self.assertEqual(np.round(np.linalg.norm(
            e_y_test - e_y), decimals=self.accuracy), 0)
        self.assertEqual(np.round(np.linalg.norm(
            e_z_test - e_z), decimals=self.accuracy), 0)

        self.assertEqual(np.round(np.linalg.norm(
            direction - direction_test), decimals=self.accuracy), 0)

    # Test whether the slices, and hence stacks, are correctly simulated
    #  in each orthogonal direction by assuming no subject motion
    def test_run_simulation_view(self):

        filename_HR_volume = "HR_volume_postmortem"
        HR_volume = st.Stack.from_filename(
            os.path.join(self.dir_test_data, filename_HR_volume + ".nii.gz"),
            os.path.join(self.dir_test_data,
                         filename_HR_volume + "_mask.nii.gz")
        )

        # 1) Test for Nearest Neighbor Interpolator
        slice_acquistion = sa.SliceAcqusition(HR_volume)
        slice_acquistion.set_interpolator_type("NearestNeighbor")
        slice_acquistion.set_motion_type("NoMotion")

        slice_acquistion.run_simulation_view_1()
        slice_acquistion.run_simulation_view_2()
        slice_acquistion.run_simulation_view_3()

        stacks_simulated = slice_acquistion.get_simulated_stacks()

        for i in range(0, len(stacks_simulated)):
            HR_volume_resampled_sitk = sitk.Resample(
                HR_volume.sitk, stacks_simulated[i].sitk, sitk.Euler3DTransform(
                ), sitk.sitkNearestNeighbor, 0.0, stacks_simulated[i].sitk.GetPixelIDValue()
            )

        self.assertEqual(np.round(
            np.linalg.norm(sitk.GetArrayFromImage(stacks_simulated[i].sitk - HR_volume_resampled_sitk)), decimals=self.accuracy), 0)

        # 2) Test for Linear Interpolator
        slice_acquistion = sa.SliceAcqusition(HR_volume)
        slice_acquistion.set_interpolator_type("Linear")
        slice_acquistion.set_motion_type("NoMotion")

        slice_acquistion.run_simulation_view_1()
        slice_acquistion.run_simulation_view_2()
        slice_acquistion.run_simulation_view_3()

        stacks_simulated = slice_acquistion.get_simulated_stacks()

        for i in range(0, len(stacks_simulated)):
            HR_volume_resampled_sitk = sitk.Resample(
                HR_volume.sitk, stacks_simulated[i].sitk, sitk.Euler3DTransform(
                ), sitk.sitkLinear, 0.0, stacks_simulated[i].sitk.GetPixelIDValue()
            )

        self.assertEqual(np.round(
            np.linalg.norm(sitk.GetArrayFromImage(stacks_simulated[i].sitk - HR_volume_resampled_sitk)), decimals=self.accuracy), 0)

        # 3) Test for Oriented Gaussian Interpolator
        alpha_cut = 3

        slice_acquistion = sa.SliceAcqusition(HR_volume)
        slice_acquistion.set_interpolator_type("OrientedGaussian")
        slice_acquistion.set_motion_type("NoMotion")
        slice_acquistion.set_alpha_cut(alpha_cut)

        slice_acquistion.run_simulation_view_1()
        slice_acquistion.run_simulation_view_2()
        slice_acquistion.run_simulation_view_3()

        stacks_simulated = slice_acquistion.get_simulated_stacks()

        resampler = itk.ResampleImageFilter[image_type, image_type].New()
        resampler.SetDefaultPixelValue(0.0)
        resampler.SetInput(HR_volume.itk)

        interpolator = itk.OrientedGaussianInterpolateImageFunction[
            image_type, pixel_type].New()
        interpolator.SetAlpha(alpha_cut)
        resampler.SetInterpolator(interpolator)

        PSF = psf.PSF()

        for i in range(0, len(stacks_simulated)):
            resampler.SetOutputParametersFromImage(stacks_simulated[i].itk)

            # Set covariance based on oblique PSF
            Cov_HR_coord = PSF.get_covariance_matrix_in_reconstruction_space(
                stacks_simulated[i], HR_volume)
            interpolator.SetCovariance(Cov_HR_coord.flatten())

            resampler.UpdateLargestPossibleRegion()
            resampler.Update()

            HR_volume_resampled_itk = resampler.GetOutput()
            HR_volume_resampled_sitk = sitkh.get_sitk_from_itk_image(
                HR_volume_resampled_itk)

        self.assertEqual(np.round(
            np.linalg.norm(sitk.GetArrayFromImage(stacks_simulated[i].sitk - HR_volume_resampled_sitk)), decimals=self.accuracy), 0)

    # Test whether the ground truth affine transforms set during the
    #  simulation correspond to the actually acquired positions within the
    #  sliced volume whereby no motion is applied to the HR volume
    def test_ground_truth_affine_transforms_no_motion(self):

        filename_HR_volume = "HR_volume_postmortem"
        HR_volume = st.Stack.from_filename(
            os.path.join(self.dir_test_data, filename_HR_volume + ".nii.gz"),
            os.path.join(self.dir_test_data,
                         filename_HR_volume + "_mask.nii.gz")
        )

        # Run simulation for Nearest Neighbor interpolation (shouldn't not
        # matter anyway and is quicker)
        slice_acquistion = sa.SliceAcqusition(HR_volume)
        slice_acquistion.set_interpolator_type("NearestNeighbor")
        slice_acquistion.set_motion_type("NoMotion")

        slice_acquistion.run_simulation_view_1()
        slice_acquistion.run_simulation_view_2()
        slice_acquistion.run_simulation_view_3()

        # Get simulated stack of slices + corresponding ground truth affine
        #  transforms indicating the correct acquisition of the slice
        #  within the volume
        stacks_simulated = slice_acquistion.get_simulated_stacks()
        affine_transforms_ground_truth, rigid_motion_transforms_ground_truth = slice_acquistion.get_ground_truth_data()

        for i in range(0, len(stacks_simulated)):
            stack = st.Stack.from_stack(stacks_simulated[i])

            slices = stack.get_slices()
            N_slices = stack.get_number_of_slices()

            for j in range(0, N_slices):
                # print("Stack %s: Slice %s/%s" %(i,j,N_slices-1))
                slice = slices[j]
                # slice.update_motion_correction(rigid_motion_transforms_ground_truth[i][j])

                HR_volume_resampled_slice_sitk = sitk.Resample(
                    HR_volume.sitk, slice.sitk, sitk.Euler3DTransform(
                    ), sitk.sitkNearestNeighbor, 0.0, slice.sitk.GetPixelIDValue()
                )

                self.assertEqual(np.round(
                    np.linalg.norm(sitk.GetArrayFromImage(slice.sitk - HR_volume_resampled_slice_sitk)), decimals=self.accuracy), 0)

    # Test whether the ground truth affine transforms set during the
    #  simulation correspond to the actually acquired positions within the
    #  sliced volume whereby motion is applied to the HR volume
    def test_ground_truth_affine_transforms_with_motion_NearestNeighbor(self):

        filename_HR_volume = "HR_volume_postmortem"
        HR_volume = st.Stack.from_filename(
            os.path.join(self.dir_test_data, filename_HR_volume + ".nii.gz"),
            os.path.join(self.dir_test_data,
                         filename_HR_volume + "_mask.nii.gz")
        )

        # Run simulation for Nearest Neighbor interpolation (shouldn't not
        # matter anyway and is quicker)
        slice_acquistion = sa.SliceAcqusition(HR_volume)
        slice_acquistion.set_interpolator_type("NearestNeighbor")
        slice_acquistion.set_motion_type("Random")

        slice_acquistion.run_simulation_view_1()
        slice_acquistion.run_simulation_view_2()
        slice_acquistion.run_simulation_view_3()

        # Get simulated stack of slices + corresponding ground truth affine
        #  transforms indicating the correct acquisition of the slice
        #  within the volume
        stacks_simulated = slice_acquistion.get_simulated_stacks()
        affine_transforms_ground_truth, rigid_motion_transforms_ground_truth = slice_acquistion.get_ground_truth_data()

        for i in range(0, len(stacks_simulated)):
            stack = stacks_simulated[i]

            slices = stack.get_slices()
            N_slices = stack.get_number_of_slices()

            for j in range(0, N_slices):
                # print("Stack %s: Slice %s/%s" %(i,j,N_slices-1))
                slice = slices[j]
                slice.update_motion_correction(
                    rigid_motion_transforms_ground_truth[i][j])

                HR_volume_resampled_slice_sitk = sitk.Resample(
                    HR_volume.sitk, slice.sitk, sitk.Euler3DTransform(
                    ), sitk.sitkNearestNeighbor, 0.0, slice.sitk.GetPixelIDValue()
                )

                self.assertEqual(np.round(
                    np.linalg.norm(sitk.GetArrayFromImage(slice.sitk - HR_volume_resampled_slice_sitk)), decimals=self.accuracy), 0)

    # Test whether the ground truth affine transforms set during the
    #  simulation correspond to the actually acquired positions within the
    #  sliced volume whereby motion is applied to the HR volume
    def test_ground_truth_affine_transforms_with_motion_OrientedGaussian(self):

        filename_HR_volume = "HR_volume_postmortem"
        HR_volume = st.Stack.from_filename(
            os.path.join(self.dir_test_data, filename_HR_volume + ".nii.gz"),
            os.path.join(self.dir_test_data,
                         filename_HR_volume + "_mask.nii.gz")
        )

        alpha_cut = 3

        # Run simulation for Oriented Gaussian interpolation, hence more
        # "realistic" case
        slice_acquistion = sa.SliceAcqusition(HR_volume)
        slice_acquistion.set_interpolator_type("OrientedGaussian")
        # slice_acquistion.set_interpolator_type("NearestNeighbor")
        # slice_acquistion.set_interpolator_type("Linear")
        slice_acquistion.set_motion_type("Random")
        # slice_acquistion.set_motion_type("NoMotion")
        slice_acquistion.set_alpha_cut(alpha_cut)

        slice_acquistion.run_simulation_view_1()
        slice_acquistion.run_simulation_view_2()
        slice_acquistion.run_simulation_view_3()

        # Get simulated stack of slices + corresponding ground truth affine
        #  transforms indicating the correct acquisition of the slice
        #  within the volume
        stacks_simulated = slice_acquistion.get_simulated_stacks()
        affine_transforms_ground_truth, rigid_motion_transforms_ground_truth = slice_acquistion.get_ground_truth_data()

        resampler = itk.ResampleImageFilter[image_type, image_type].New()
        resampler.SetDefaultPixelValue(0.0)
        resampler.SetInput(HR_volume.itk)

        interpolator = itk.OrientedGaussianInterpolateImageFunction[
            image_type, pixel_type].New()
        interpolator.SetAlpha(alpha_cut)
        # interpolator = itk.LinearInterpolateImageFunction[image_type, pixel_type].New()
        # interpolator = itk.NearestNeighborInterpolateImageFunction[image_type, pixel_type].New()
        resampler.SetInterpolator(interpolator)

        PSF = psf.PSF()

        for i in range(0, len(stacks_simulated)):
            stack = stacks_simulated[i]

            slices = stack.get_slices()
            N_slices = stack.get_number_of_slices()

            for j in range(0, N_slices):
                # print("Stack %s: Slice %s/%s" %(i,j,N_slices-1))
                slice = slices[j]
                slice.update_motion_correction(
                    rigid_motion_transforms_ground_truth[i][j])

                # Get covariance based on oblique PSF
                Cov_HR_coord = PSF.get_covariance_matrix_in_reconstruction_space(
                    slice, HR_volume)

                # Update resampler
                interpolator.SetCovariance(Cov_HR_coord.flatten())
                resampler.SetOutputParametersFromImage(slice.itk)
                resampler.UpdateLargestPossibleRegion()
                resampler.Update()

                HR_volume_resampled_slice_itk = resampler.GetOutput()
                HR_volume_resampled_slice_sitk = sitkh.get_sitk_from_itk_image(
                    HR_volume_resampled_slice_itk)

                # HR_volume_resampled_slice_sitk = sitk.Resample(
                #     HR_volume.sitk, slice.sitk, sitk.Euler3DTransform(), sitk.sitkNearestNeighbor, 0.0, slice.sitk.GetPixelIDValue()
                #     )

                norm_diff = np.linalg.norm(sitk.GetArrayFromImage(
                    slice.sitk - HR_volume_resampled_slice_sitk))
                # try:
                self.assertEqual(
                    np.round(norm_diff, decimals=self.accuracy), 0)
                # except:
                #     print("Stack %s: Slice %s/%s" %(i,j,N_slices-1))

                #     print(norm_diff)
