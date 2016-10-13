#!/usr/bin/python

## \file 
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date Aug 2016

## Import libraries 
import SimpleITK as sitk
import itk
import numpy as np
import sys

## Add directories to import modules
dir_src_root = "src/"
sys.path.append(dir_src_root + "base/")
sys.path.append(dir_src_root + "preprocessing/")
sys.path.append(dir_src_root + "registration/")
sys.path.append(dir_src_root + "reconstruction/")
sys.path.append(dir_src_root + "reconstruction/solver/")
sys.path.append(dir_src_root + "simulation/")
# sys.path.append(dir_src_root + "reconstruction/regularization_parameter_estimator/")

## Import modules
import SimpleITKHelper as sitkh
import DataPreprocessing as dp
import Stack as st
import Slice as sl
import StackManager as sm
import ScatteredDataApproximation as sda
import TikhonovSolver as tk
import SimulatorSliceAcqusition as sa
import Registration as myreg
import StackInPlaneAlignment as sipa
import DataPreprocessing as dp
import BrainStripping as bs

## Pixel type of used 3D ITK image
PIXEL_TYPE = itk.D

## ITK image type
IMAGE_TYPE_2D = itk.Image[PIXEL_TYPE, 2]
IMAGE_TYPE_3D = itk.Image[PIXEL_TYPE, 3]
IMAGE_TYPE_3D_CV18 = itk.Image.CVD183
IMAGE_TYPE_3D_CV3 = itk.Image.CVD33


"""
Main Function
"""
if __name__ == '__main__':

    np.set_printoptions(precision=3)

    # dir_input = "data/test/"
    # filename_2D = "2D_BrainWeb"
    # filename_HRVolume = "FetalBrain_reconstruction_4stacks"
    # filename_stack = "fetal_brain_0"
    # filename_slice = "FetalBrain_stack1_registered_midslice"

    # dir_input = "test/data/"
    # filename_HRVolume = "recon_fetal_neck_mass_brain_cycles0_SRR_TK0_itermax20_alpha0.1"
    # filename_stack = "stack1_rotated_angle_z_is_pi_over_10"

    # HR_volume = st.Stack.from_filename(dir_input, filename_HRVolume)
    # stack = st.Stack.from_filename(dir_input, filename_stack, suffix_mask="_mask")

    # registration = myreg.Registration(fixed, moving)
    # registration.use_verbose(True)
    # registration.run_registration()

    # print registration.get_parameters()
    # transform_sitk = registration.get_registration_transform_sitk()

    # moving_resampled_sitk = sitk.Resample(moving.sitk, fixed.sitk, sitk.Euler3DTransform(), sitk.sitkBSpline)
    # moving_registered_sitk = sitk.Resample(moving.sitk, fixed.sitk, transform_sitk, sitk.sitkBSpline)

    # # sitkh.show_sitk_image([fixed.sitk, moving_resampled_sitk, moving_registered_sitk], ["fixed", "moving_orig", "moving_registered"])

    # sitkh.show_sitk_image(fixed.sitk,"fixed")
    # sitkh.show_sitk_image(moving_resampled_sitk,"moving_resampled")
    # sitkh.show_sitk_image(moving_registered_sitk,"moving_registered")

    # # image_2D_itk = sitkh.read_itk_image(dir_input + filename_2D + ".nii.gz", dim=2, pixel_type=PIXEL_TYPE)
    # # HRvolume_itk = sitkh.read_itk_image(dir_input + filename_HRVolume + ".nii.gz", dim=3, pixel_type=PIXEL_TYPE)
    # # slice_itk = sitkh.read_itk_image(dir_input + filename_slice + ".nii.gz", dim=3, pixel_type=PIXEL_TYPE)
    # # slice_itk = HRvolume_itk

    # image_2D_sitk = sitk.ReadImage(dir_input + filename_2D + ".nii.gz")

    # DIR_INPUT = "data/placenta/";                   filename_stack = "a23_05"
    # # DIR_INPUT = "data/fetal_neck_mass_brain/";      filename_stack = "0"
    # stack = st.Stack.from_filename(DIR_INPUT, filename_stack, suffix_mask="_mask")

    # data_preprocessing = dp.DataPreprocessing.from_stacks([stack])
    # # data_preprocessing.set_dilation_radius(0)
    # data_preprocessing.run_preprocessing()
    # stack = data_preprocessing.get_preprocessed_stacks()[0]

    # inplane_reg = sipa.StackInPlaneAlignment()
    # inplane_reg.set_stack(stack)

    # inplane_reg.run_registration()

    # stack_inplane_reg = inplane_reg.get_stack()

    # # stack_inplane_reg.get_resampled_stack_from_slices().show(1)
    # stack_registered_sitk = stack_inplane_reg.get_resampled_stack_from_slices(interpolator="Linear").sitk
    # sitkh.show_sitk_image([stack.sitk, stack_registered_sitk], ["original", "inplane-registered"])

    subject = "2"
    filename = "002-30yr-AxT2"
    DIR_ROOT_DIRECTORY = "/Users/mebner/UCL/Data/30_year_old_data/"
    dir_input = DIR_ROOT_DIRECTORY + "Subject_" + subject + "/"

    dir_input = "studies/30YearMSData/Subject" + subject + "/data_preprocessing/"
    filename = "5year_2_downsampled_factor10"
    
    # reference_image = st.Stack.from_filename(dir_input, filename, "_mask")
    brain_stripping = bs.BrainStripping.from_filename(dir_input, filename)
    # brain_stripping = bs.BrainStripping.from_sitk_image(reference_image.sitk)
    # brain_stripping = bs.BrainStripping()
    # brain_stripping.set_input_image_sitk(reference_image.sitk)
    brain_stripping.compute_brain_mask(1)
    # brain_stripping.compute_brain_image(0)
    # brain_stripping.compute_skull_image(0)
    # brain_stripping.set_bet_options("-f 0.3")

    brain_stripping.run_stripping()
    original_sitk = brain_stripping.get_input_image_sitk()
    brain_mask_sitk = brain_stripping.get_brain_mask_sitk()
    # brain_sitk = brain_stripping.get_brain_image_sitk()
    # skull_mask_sitk = brain_stripping.get_skull_image_sitk()


    sitkh.show_sitk_image([original_sitk], segmentation=brain_mask_sitk)
    # sitkh.show_sitk_image([original_sitk, brain_sitk], segmentation=skull_mask_sitk)


    reg = regsitk.RegistrationSimpleITK()
    reg.set_registration_type("Similarity")
    reg.set_scales_estimator("Jacobian")
    reg.set_fixed(HR_volume_init)
    reg.set_moving(reference_image)
    reg.use_verbose(True)
    reg.set_metric("MattesMutualInformation")
    reg.run_registration()
    trafo = reg.get_registration_transform_sitk()
    print trafo


