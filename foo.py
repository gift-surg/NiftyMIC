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
import StackManager as sm
import ScatteredDataApproximation as sda
import TikhonovSolver as tk
import SimulatorSliceAcqusition as sa
import Registration as myreg

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
    filename_HRVolume = "FetalBrain_reconstruction_4stacks"
    filename_slice = "FetalBrain_stack1_registered_midslice"

    image_2D_itk = sitkh.read_itk_image(dir_input + filename_2D + ".nii.gz", dim=2, pixel_type=PIXEL_TYPE)
    HRvolume_itk = sitkh.read_itk_image(dir_input + filename_HRVolume + ".nii.gz", dim=3, pixel_type=PIXEL_TYPE)
    slice_itk = sitkh.read_itk_image(dir_input + filename_slice + ".nii.gz", dim=3, pixel_type=PIXEL_TYPE)
    # slice_itk = HRvolume_itk
    slice_sitk = sitkh.convert_itk_to_sitk_image(slice_itk)

    filter_derivative = itk.GradientImageFilter[IMAGE_TYPE_2D, PIXEL_TYPE, PIXEL_TYPE].New()
    # filter_derivative = itk.DerivativeImageFilter[IMAGE_TYPE_2D, IMAGE_TYPE_2D].New()

    filter_derivative.SetInput(image_2D_itk)
    filter_derivative.Update()
    dimage_2D_itk = filter_derivative.GetOutput()
     
    itk2np = itk.PyBuffer[IMAGE_TYPE_3D]
    itk2np_CVD183 = itk.PyBuffer[IMAGE_TYPE_3D_CV18]
    itk2np_CVD33 = itk.PyBuffer[IMAGE_TYPE_3D_CV3]

    # sys.exit()

    filter_OrientedGaussian_3D = itk.OrientedGaussianInterpolateImageFilter[IMAGE_TYPE_3D, IMAGE_TYPE_3D].New()
    filter_OrientedGaussian_3D.SetInput(HRvolume_itk)
    filter_OrientedGaussian_3D.SetOutputParametersFromImage(slice_itk)
    filter_OrientedGaussian_3D.SetUseJacobian(True)
    filter_OrientedGaussian_3D.Update()

    tmp = filter_OrientedGaussian_3D.GetOutput()
    gradient_OrientedGaussian_3D_itk = filter_OrientedGaussian_3D.GetJacobian()

    # sitkh.show_itk_image(tmp, overlay=slice_itk)

    filter_GradientEuler3D = itk.GradientEuler3DTransformImageFilter[IMAGE_TYPE_3D, PIXEL_TYPE, PIXEL_TYPE].New()
    filter_GradientEuler3D.SetInput(slice_itk)
    filter_GradientEuler3D.Update()
    gradient_Euler3DTransform_itk = filter_GradientEuler3D.GetOutput()

    nda_GradientOrientedGaussian = itk2np_CVD33.GetArrayFromImage(gradient_OrientedGaussian_3D_itk)
    nda_GradientEuler3D = itk2np_CVD183.GetArrayFromImage(gradient_Euler3DTransform_itk)

    dF = nda_GradientOrientedGaussian
    shape = np.array(slice_itk.GetBufferedRegion().GetSize())
    dT = nda_GradientEuler3D
    nda0 = nda_GradientOrientedGaussian.reshape(-1,3)
    nda1 = nda_GradientEuler3D.reshape(-1,3,6)

    Euler3DTransform_itk = itk.Euler3DTransform[PIXEL_TYPE].New()
    filter_GradientEuler3D.SetTransform(Euler3DTransform_itk)

    N = shape.prod()
    res = np.zeros((N,6))
    res_ = np.zeros((shape[2],shape[1],shape[0],6))

    for i in range(0, N):
        res[i,:] = nda0[i,:].dot(nda1[i,:,:])

    for i in range(0, shape[2]):
        for j in range(0, shape[1]):
            for k in range(0, shape[0]):
                res_[i,j,k,:] = dF[i,j,k,:].dot(dT[i,j,k,:].reshape(3,6))

    res_ = res_.reshape(-1,6)
    print (res - res_).sum()
                







