#!/usr/bin/python

## \file ITK_Registration.py
#  \brief Figure out how to register two images within WrapITK
#
#  \author: Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date: 

## import librarties
import itk
import numpy as np

import sys
sys.path.append("../src")

import SimpleITKHelper as sitkh
import PSF as psf

pixel_type = itk.D
image_type = itk.Image[pixel_type, 3]

"""
Functions
"""
def read_itk_image(filename):
    # image_IO_type = itk.NiftiImageIO

    reader = itk.ImageFileReader[image_type].New()
    reader.SetFileName(filename)
    reader.Update()
    image_itk = reader.GetOutput()
    image_itk.DisconnectPipeline()

    return image_itk

def get_rigid_registration_transform_3D(fixed_itk, moving_itk):

    registration = itk.ImageRegistrationMethod[image_type, image_type].New()

    # initial_transform = itk.CenteredTransformInitializer[fixed_itk, moving_itk, itk.Euler3DTransform.New()].New()
    initial_transform = itk.Euler3DTransform.New()

    # interpolator = itk.LinearInterpolateImageFunction[image_type, pixel_type].New()
    interpolator = itk.OrientedGaussianInterpolateImageFunction[image_type, pixel_type].New()
    Cov = np.eye(3)*3;
    interpolator.SetCovariance(Cov.flatten())

    
    # metric = itk.MeanSquaresImageToImageMetric[image_type, image_type].New()
    metric = itk.NormalizedCorrelationImageToImageMetric[image_type, image_type].New()

    optimizer = itk.RegularStepGradientDescentOptimizer.New()
    # optimizer.SetMaximumStepLength( 4.00 )
    # optimizer.SetMinimumStepLength( 0.01 )
    # optimizer.SetNumberOfIterations( 200 )

    registration.SetInitialTransformParameters(initial_transform.GetParameters())
    registration.SetOptimizer(optimizer)
    registration.SetTransform(initial_transform)
    registration.SetInterpolator(interpolator)
    registration.SetMetric(metric)
    registration.SetMovingImage(moving_itk)
    registration.SetFixedImage(fixed_itk)

    ## Execute registration
    registration.Update()

    ## Get registration transform
    rigid_registration_3D = registration.GetOutput().Get()

    return rigid_registration_3D


def get_resampled_image(fixed_itk, moving_itk, transformation):
    resampler = itk.ResampleImageFilter[image_type, image_type].New()

    resampler.SetInput(moving_itk)
    resampler.SetTransform(transformation)
    resampler.SetOutputParametersFromImage(fixed_itk)
    # resampler.SetSize(fixed_itk.GetLargestPossibleRegion().GetSize())
    # resampler.SetOutputOrigin(fixed_itk.GetOrigin())
    # resampler.SetOutputSpacing(fixed_itk.GetSpacing())
    # resampler.SetOutputDirection(fixed_itk.GetDirection())
    resampler.SetDefaultPixelValue(0.0)

    warped_itk = resampler.GetOutput()

    return warped_itk


"""
Main
"""
## define input data
dir_input = "data/"
dir_output = "results/"
filename = "fetal_brain_a"

fixed_itk = read_itk_image(dir_input + filename + ".nii.gz")
moving_itk = read_itk_image(dir_input + filename + "_rotated_angle_z.nii.gz")

## Register images
rigid_transform_3D = get_rigid_registration_transform_3D(fixed_itk, moving_itk)

## Resample image
warped_itk = get_resampled_image(fixed_itk, moving_itk, rigid_transform_3D)

# trafo = itk.Euler3DTransform.New()
# writer_transformation = itk.TransformFileWriterTemplate[itk.D].New()


## Write warped image
# writer.SetFileName(dir_output + filename + "_test.nii.gz")
# writer.SetInput(warped_itk)
# writer.Update()

sitkh.show_itk_image(fixed_itk,overlay=warped_itk)

