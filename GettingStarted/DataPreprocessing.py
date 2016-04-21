## \file DataPreprocessing.py
#  \brief  
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date April 2016

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import leastsq
from scipy import ndimage

import itk
import SimpleITK as sitk
import numpy as np
import unittest
import matplotlib.pyplot as plt

import sys
import time
sys.path.append("../src")

import SimpleITKHelper as sitkh


## Change viewer for sitk.Show command
#%env SITK_SHOW_COMMAND /Applications/ITK-SNAP.app/Contents/MacOS/ITK-SNAP


## Define types of input and output pixels and state dimension of images
pixel_type = itk.D

## Define type of input and output image
image_type_2D = itk.Image[itk.D, 2]
image_type_3D = itk.Image[itk.D, 3]


"""
Functions
"""
def show_2D_example():
    dir_input = "data/"

    # filename_2D = "2D_SingleDot_50"
    filename_2D = "2D_Cross_50"

    ## Read image
    mask = read_sitk_image(dir_input + filename_2D + ".nii.gz")

    #/ Create binary mask (some values are set to 100)
    nda = sitk.GetArrayFromImage(mask)
    nda[nda>0]=1
    nda = nda.astype(np.uint8)
    mask = sitk.GetImageFromArray(nda)

    ## Dilate mask
    mask_dilated = dilate_mask(mask)

    sitkh.show_sitk_image(mask, overlay=mask_dilated)


def read_itk_image(filename):
    # image_IO_type = itk.NiftiImageIO

    # try:
    # Works for 2D and 3D for any reason
        reader = itk.ImageFileReader[image_type_3D].New()
        reader.SetFileName(filename)
        reader.Update()
        image_itk = reader.GetOutput()
        image_itk.DisconnectPipeline()

        return image_itk


    # except:
    #     reader = itk.ImageFileReader[image_type_2D].New()
    #     reader.SetFileName(filename)
    #     reader.Update()

    #     image_itk = reader.GetOutput()

    #     return image_itk

def read_sitk_image(filename):
    return sitk.ReadImage(filename,sitk.sitkFloat64)


def get_sitk_rigid_registration_transform_3D(fixed_3D, moving_3D):

    ## Instantiate interface method to the modular ITKv4 registration framework
    registration_method = sitk.ImageRegistrationMethod()

    ## Select between using the geometrical center (GEOMETRY) of the images or using the center of mass (MOMENTS) given by the image intensities
    # initial_transform = sitk.CenteredTransformInitializer(fixed_slice_3D._sitk_upsampled, moving_HR_volume_3D.sitk, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
    initial_transform = sitk.Euler3DTransform()

    ## Set the initial transform and parameters to optimize
    registration_method.SetInitialTransform(initial_transform)

    ## Set an image masks in order to restrict the sampled points for the metric
    # registration_method.SetMetricFixedMask(fixed_slice_3D._sitk_mask_upsampled)
    # registration_method.SetMetricMovingMask(moving_HR_volume_3D.sitk_mask)

    ## Set percentage of pixels sampled for metric evaluation
    # registration_method.SetMetricSamplingStrategy(registration_method.NONE)

    ## Set interpolator to use
    registration_method.SetInterpolator(sitk.sitkLinear)

    """
    similarity metric settings
    """
    ## Use normalized cross correlation using a small neighborhood for each voxel between two images, with speed optimizations for dense registration
    # registration_method.SetMetricAsANTSNeighborhoodCorrelation(radius=5)
    
    ## Use negative normalized cross correlation image metric
    # registration_method.SetMetricAsCorrelation()

    ## Use demons image metric
    # registration_method.SetMetricAsDemons(intensityDifferenceThreshold=1e-3)

    ## Use mutual information between two images
    # registration_method.SetMetricAsJointHistogramMutualInformation(numberOfHistogramBins=100, varianceForJointPDFSmoothing=1)
    
    ## Use the mutual information between two images to be registered using the method of Mattes2001
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=100)

    ## Use negative means squares image metric
    # registration_method.SetMetricAsMeanSquares()
    
    """
    optimizer settings
    """
    ## Set optimizer to Nelder-Mead downhill simplex algorithm
    # registration_method.SetOptimizerAsAmoeba(simplexDelta=0.1, numberOfIterations=100, parametersConvergenceTolerance=1e-8, functionConvergenceTolerance=1e-4, withStarts=false)

    ## Conjugate gradient descent optimizer with a golden section line search for nonlinear optimization
    # registration_method.SetOptimizerAsConjugateGradientLineSearch(learningRate=1, numberOfIterations=100, convergenceMinimumValue=1e-8, convergenceWindowSize=10)

    ## Set the optimizer to sample the metric at regular steps
    # registration_method.SetOptimizerAsExhaustive(numberOfSteps=50, stepLength=1.0)

    ## Gradient descent optimizer with a golden section line search
    # registration_method.SetOptimizerAsGradientDescentLineSearch(learningRate=1, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10)

    ## Limited memory Broyden Fletcher Goldfarb Shannon minimization with simple bounds
    # registration_method.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=500, maximumNumberOfCorrections=5, maximumNumberOfFunctionEvaluations=200, costFunctionConvergenceFactor=1e+7)

    ## Regular Step Gradient descent optimizer
    registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=1, minStep=0.05, numberOfIterations=2000)

    ## Estimating scales of transform parameters a step sizes, from the maximum voxel shift in physical space caused by a parameter change
    ## (Many more possibilities to estimate scales)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    
    """
    setup for the multi-resolution framework            
    """
    ## Set the shrink factors for each level where each level has the same shrink factor for each dimension
    # registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])

    ## Set the sigmas of Gaussian used for smoothing at each level
    # registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])

    ## Enable the smoothing sigmas for each level in physical units (default) or in terms of voxels (then *UnitsOff instead)
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    ## Connect all of the observers so that we can perform plotting during registration
    # registration_method.AddCommand(sitk.sitkStartEvent, start_plot)
    # registration_method.AddCommand(sitk.sitkEndEvent, end_plot)
    # registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, update_multires_iterations) 
    # registration_method.AddCommand(sitk.sitkIterationEvent, lambda: plot_values(registration_method))

    # print('  Final metric value: {0}'.format(registration_method.GetMetricValue()))
    # print('  Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))
    # print("\n")

    ## Execute 3D registration
    final_transform_3D_sitk = registration_method.Execute(fixed_3D, moving_3D) 
    print("SimpleITK Image Registration Method:")
    print('  Final metric value: {0}'.format(registration_method.GetMetricValue()))
    print('  Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))

    return final_transform_3D_sitk


def get_propagated_sitk_mask_3D(fixed, moving, moving_mask):
    trafo = get_sitk_rigid_registration_transform_3D(fixed, moving)
    # print trafo

    fixed_mask_prop = sitk.Resample(moving_mask, fixed, trafo, sitk.sitkLinear, 0.0, 1)
    # fixed_mask_prop = sitk.Resample(moving_mask, fixed, trafo, sitk.sitkNearestNeighbor, 0.0, 1)

    return fixed_mask_prop


def dilate_mask(image_sitk):
    filter = sitk.BinaryDilateImageFilter()
    filter.SetKernelType(sitk.sitkBall)
    # filter.SetKernelType(sitk.sitkBox)
    # filter.SetKernelType(sitk.sitkAnnulus)
    # filter.SetKernelType(sitk.sitkCross)
    filter.SetKernelRadius(1)
    filter.SetForegroundValue(1)
    dilated = filter.Execute(image_sitk)

    return dilated


"""
Main Function
"""
if __name__ == '__main__':

    # show_2D_example()


    # dir_input = "data/"
    dir_input = "../data/fetal_neck/"
    dir_output = "results/"

    # filename_2D = "2D_BrainWeb"
    # filename_2D = "2D_SingleDot_50"
    filename_2D = "2D_Cross_50"
    # filename_2D = "2D_Text"
    # filename_2D = "2D_Cameraman_256"
    # filename_2D = "2D_House_256"
    # filename_2D = "2D_SheppLoganPhantom_512"
    # filename_2D = "2D_Lena_512"
    # filename_2D = "2D_Boat_512"
    # filename_2D = "2D_Man_1024"

    # filename_3D = "FetalBrain_reconstruction_4stacks"
    # filename_3D = "3D_SingleDot_50"
    # filename_3D = "3D_Cross_50"
    # filename_3D = "3D_SheppLoganPhantom_64"
    # filename_3D = "3D_SheppLoganPhantom_128"


    # image_itk = read_itk_image(dir_input + filename_3D + ".nii.gz")
    # sitkh.show_itk_image(image_itk)

    
    fixed_sitk = read_sitk_image(dir_input + "0.nii.gz") 
    fixed_sitk_mask = read_sitk_image(dir_input + "0_mask.nii.gz") 

    moving_sitk = read_sitk_image(dir_input + "1.nii.gz") 
    moving_sitk_mask = read_sitk_image(dir_input + "1_mask.nii.gz") 


    # trafo = get_sitk_rigid_registration_transform_3D(fixed_sitk, moving_sitk)

    fixed_sitk_mask_prop = get_propagated_sitk_mask_3D(fixed_sitk,moving_sitk,moving_sitk_mask)

    # mask_dilated = dilate_mask(fixed_sitk_mask)

    # sitkh.show_sitk_image(fixed_sitk_mask, overlay=fixed_sitk_mask_prop)
    sitkh.show_sitk_image(dilate_mask(fixed_sitk_mask_prop), overlay=fixed_sitk_mask_prop,title="dilate")

    sitkh.show_sitk_image(fixed_sitk, dilate_mask(fixed_sitk_mask_prop))




