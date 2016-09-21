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
    filename_2D = "2D_BrainWeb"
    # filename_HRVolume = "FetalBrain_reconstruction_4stacks"
    # filename_slice = "FetalBrain_stack1_registered_midslice"

    # # image_2D_itk = sitkh.read_itk_image(dir_input + filename_2D + ".nii.gz", dim=2, pixel_type=PIXEL_TYPE)
    # # HRvolume_itk = sitkh.read_itk_image(dir_input + filename_HRVolume + ".nii.gz", dim=3, pixel_type=PIXEL_TYPE)
    # # slice_itk = sitkh.read_itk_image(dir_input + filename_slice + ".nii.gz", dim=3, pixel_type=PIXEL_TYPE)
    # # slice_itk = HRvolume_itk

    image_2D_sitk = sitk.ReadImage(dir_input + filename_2D + ".nii.gz")

    DIR_INPUT = "data/placenta/";                   filename_stack = "a23_05"
    # DIR_INPUT = "data/fetal_neck_mass_brain/";      filename_stack = "0"
    stack = st.Stack.from_filename(DIR_INPUT, filename_stack, suffix_mask="_mask")

    data_preprocessing = dp.DataPreprocessing.from_stacks([stack])
    # data_preprocessing.set_dilation_radius(0)
    data_preprocessing.run_preprocessing()
    stack = data_preprocessing.get_preprocessed_stacks()[0]

    inplane_reg = iprr.InPlaneRigidRegistration()
    inplane_reg.set_stack(stack)

    inplane_reg.run_registration()

    stack_inplane_reg = inplane_reg.get_stack()

    # stack_inplane_reg.get_resampled_stack_from_slices().show(1)
    stack_registered_sitk = stack_inplane_reg.get_resampled_stack_from_slices(interpolator="Linear").sitk
    sitkh.show_sitk_image([stack.sitk, stack_registered_sitk], ["original", "inplane-registered"])

                







