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
import InPlaneRigidRegistration as iprr
import DataPreprocessing as dp

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

    dir_input = "data/test/"
    # filename_2D = "2D_BrainWeb"
    filename_HRVolume = "FetalBrain_reconstruction_4stacks"
    filename_stack = "fetal_brain_0"
    # filename_slice = "FetalBrain_stack1_registered_midslice"

    # dir_input = "test/data/"
    # filename_HRVolume = "recon_fetal_neck_mass_brain_cycles0_SRR_TK0_itermax20_alpha0.1"
    # filename_stack = "stack1_rotated_angle_z_is_pi_over_10"

    HR_volume = st.Stack.from_filename(dir_input, filename_HRVolume)

    stack = st.Stack.from_filename(dir_input, filename_stack, suffix_mask="_mask")
    transform_sitk = sitk.Euler3DTransform()
    transform_sitk.SetParameters((0.1,0.1,0.2,-1,3,2))
    stack_sitk = sitkh.get_transformed_image(stack.sitk, transform_sitk)
    stack_sitk_mask = sitkh.get_transformed_image(stack.sitk_mask, transform_sitk)

    stack = st.Stack.from_sitk_image(stack_sitk, name=stack.get_filename(), image_sitk_mask=stack_sitk_mask)

    fixed = stack
    moving = HR_volume


    registration = myreg.Registration(fixed, moving)
    registration.use_verbose(True)
    registration.run_registration()

    print registration.get_parameters()
    transform_sitk = registration.get_registration_transform_sitk()

    moving_resampled_sitk = sitk.Resample(moving.sitk, fixed.sitk, sitk.Euler3DTransform(), sitk.sitkBSpline)
    moving_registered_sitk = sitk.Resample(moving.sitk, fixed.sitk, transform_sitk, sitk.sitkBSpline)

    # sitkh.show_sitk_image([fixed.sitk, moving_resampled_sitk, moving_registered_sitk], ["fixed", "moving_orig", "moving_registered"])

    sitkh.show_sitk_image(fixed.sitk,"fixed")
    sitkh.show_sitk_image(moving_resampled_sitk,"moving_resampled")
    sitkh.show_sitk_image(moving_registered_sitk,"moving_registered")

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

    # inplane_reg = iprr.InPlaneRigidRegistration()
    # inplane_reg.set_stack(stack)

    # inplane_reg.run_registration()

    # stack_inplane_reg = inplane_reg.get_stack()

    # # stack_inplane_reg.get_resampled_stack_from_slices().show(1)
    # stack_registered_sitk = stack_inplane_reg.get_resampled_stack_from_slices(interpolator="Linear").sitk
    # sitkh.show_sitk_image([stack.sitk, stack_registered_sitk], ["original", "inplane-registered"])

                







