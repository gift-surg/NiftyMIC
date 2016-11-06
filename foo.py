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
import registration.InPlaneRegistration as inplanereg
import preprocessing.DataPreprocessing as dp
import preprocessing.BrainStripping as bs
import utilities.IntensityCorrection as ic
import registration.RegistrationSimpleITK as regsitk

import utilities.ScanExtractor as se
import utilities.FilenameParser as fp
import registration.RegularizedInPlaneRegistration as reginplanereg


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

    dir_input = "test-data/"
    # filename_HRVolume = "FetalBrain_reconstruction_4stacks"
    filename_stack = "fetal_brain_0"
    # filename_slice = "FetalBrain_stack1_registered_midslice"

    # HR_volume = st.Stack.from_filename(dir_input, filename_HRVolume)
    stack = st.Stack.from_filename(dir_input, filename_stack, suffix_mask="_mask")

    timepoint = "1year"
    dir_input = "studies/30YearMSData/Subject2/" + timepoint + "/PDref/"
    filename_stack = timepoint + "_2_intensity_corrected"
    stack_sitk = sitk.ReadImage(dir_input + filename_stack + ".nii.gz")

    brain_stripping = bs.BrainStripping()
    brain_stripping.set_input_image_sitk(stack_sitk)
    brain_stripping.run_stripping()
    stack_sitk_mask = brain_stripping.get_mask_around_skull()
    stack = st.Stack.from_sitk_image(stack_sitk, filename_stack, stack_sitk_mask)

    registration = reginplanereg.RegularizedInPlaneRegistration(fixed=stack)
    registration.use_fixed_mask(True)
    registration.run_regularized_rigid_inplane_registration()
    registration.print_statistics()

    stack_registered = registration.get_registered_stack()
    parameters = registration.get_parameters()
    print parameters
    # transforms = registration.get_registration_transform_sitk()

    sitkh.show_stacks([stack, stack_registered.get_resampled_stack_from_slices(interpolator="Linear")])
