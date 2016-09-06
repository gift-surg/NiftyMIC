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
dir_src_root = "../../src/"
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
import StackManager as sm
import ScatteredDataApproximation as sda
import TikhonovSolver as tk
import SimulatorSliceAcqusition as sa
import Registration as myreg

## Pixel type of used 3D ITK image
PIXEL_TYPE = itk.D

## ITK image type
IMAGE_TYPE = itk.Image[PIXEL_TYPE, 2]


"""
Main Function
"""
if __name__ == '__main__':

    np.set_printoptions(precision=3)

    dir_input = "data/test/"
    filename_2D = "2D_BrainWeb"

    image_itk = sitkh.read_itk_image(dir_input + filename_2D + ".nii.gz", dim=2)

    filter_derivative = itk.GradientImageFilter[IMAGE_TYPE, itk.F, itk.F].New()
    # filter_derivative = itk.DerivativeImageFilter[IMAGE_TYPE, IMAGE_TYPE].New()

    filter_derivative.SetInput(image_itk)
    filter_derivative.Update()

    dimage_itk = filter_derivative.GetOutput()









