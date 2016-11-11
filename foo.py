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
dir_src_root = "src/py/"
sys.path.append(dir_src_root)
# sys.path.append(dir_src_root + "reconstruction/regularization_parameter_estimator/")

## Import modules
import utilities.SimpleITKHelper as sitkh
import base.Stack as st
import base.Slice as sl
import utilities.StackManager as sm
import reconstruction.ScatteredDataApproximation as sda

import reconstruction.solver.TikhonovSolver as tk
import simulation.SimulatorSliceAcqusition as sa
import registration.Registration as myreg
import registration.InPlaneRegistrationSimpleITK as inplaneregsitk
import preprocessing.DataPreprocessing as dp
import preprocessing.BrainStripping as bs
import utilities.IntensityCorrection as ic
import registration.RegistrationSimpleITK as regsitk

import utilities.ScanExtractor as se
import utilities.FilenameParser as fp
import registration.IntraStackRegistration as inplanereg
import utilities.ParameterNormalization as pn

## Pixel type of used 3D ITK image
PIXEL_TYPE = itk.D

## ITK image type
IMAGE_TYPE_2D = itk.Image[PIXEL_TYPE, 2]
IMAGE_TYPE_3D = itk.Image[PIXEL_TYPE, 3]
IMAGE_TYPE_3D_CV18 = itk.Image.CVD183
IMAGE_TYPE_3D_CV3 = itk.Image.CVD33



def get_inplane_corrupted_stack(stack, angle_z, center_2D, translation_2D, scale=1, intensity_scale=1, intensity_bias=0, debug=0):
    
    ## Convert to 3D:
    translation_3D = np.zeros(3)
    translation_3D[0:-1] = translation_2D

    center_3D = np.zeros(3)
    center_3D[0:-1] = center_2D

    ## Transform to align physical coordinate system with stack-coordinate system
    affine_centering_sitk = sitk.AffineTransform(3)
    affine_centering_sitk.SetMatrix(stack.sitk.GetDirection())
    affine_centering_sitk.SetTranslation(stack.sitk.GetOrigin())

    ## Corrupt first stack towards positive direction
    in_plane_motion_sitk = sitk.Euler3DTransform()
    in_plane_motion_sitk.SetRotation(0, 0, angle_z)
    in_plane_motion_sitk.SetCenter(center_3D)
    in_plane_motion_sitk.SetTranslation(translation_3D)
    motion_sitk = sitkh.get_composite_sitk_affine_transform(in_plane_motion_sitk, sitk.AffineTransform(affine_centering_sitk.GetInverse()))
    motion_sitk = sitkh.get_composite_sitk_affine_transform(affine_centering_sitk, motion_sitk)
    stack_corrupted_resampled_sitk = sitk.Resample(stack.sitk, motion_sitk, sitk.sitkLinear)
    stack_corrupted_resampled_sitk_mask = sitk.Resample(stack.sitk_mask, motion_sitk, sitk.sitkLinear)

    ## Corrupt first stack towards negative direction
    in_plane_motion_2_sitk = sitk.Euler3DTransform()
    in_plane_motion_2_sitk.SetRotation(0, 0, -angle_z)
    in_plane_motion_2_sitk.SetCenter(center_3D)
    in_plane_motion_2_sitk.SetTranslation(-translation_3D)
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
    stack_corrupted_sitk = sitk.GetImageFromArray((nda-intensity_bias)/intensity_scale)
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


"""
Main Function
"""
if __name__ == '__main__':

    np.set_printoptions(precision=3)

    dir_input = "test-data/"

    filename_stack = "FetalBrain_reconstruction_3stacks_myAlg"
    filename_stack_corrupted = "FetalBrain_reconstruction_3stacks_myAlg_corrupted_inplane"

    stack_sitk = sitk.ReadImage(dir_input + filename_stack + ".nii.gz")
    stack_corrupted_sitk = sitk.ReadImage(dir_input + filename_stack_corrupted + ".nii.gz")

    stack_corrupted = st.Stack.from_sitk_image(stack_corrupted_sitk, "stack_corrupted")
    stack = st.Stack.from_sitk_image(sitk.Resample(stack_sitk, stack_corrupted.sitk),"stack")

    # sitkh.show_stacks([stack, stack_corrupted])

    inplane_registration = inplanereg.IntraStackRegistration(stack_corrupted, stack)
    inplane_registration.set_initializer_type("moments")
    inplane_registration.set_intensity_correction_type("affine")
    inplane_registration.set_transform_type("rigid")
    inplane_registration._run_registration_pipeline_initialization()
    parameters = inplane_registration.get_parameters()

    parameters_array = np.array(parameters)

    parameter_normalization = pn.ParameterNormalization(parameters)

    parameter_normalization.compute_normalization_coefficients()
    print parameter_normalization.get_normalization_coefficients()

    parameter_normalization.normalize_parameters(parameters_array)

    print parameter_normalization.denormalize_parameters(parameters_array)

    print np.linalg.norm(parameters_array-parameters)


