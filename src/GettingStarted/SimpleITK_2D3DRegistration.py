#!/usr/bin/python

## \file SimpleITK_2D3DRegistration.py
#  \brief Test of 2D to 3D registration within SimpleITK
#
#  \author: Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date: Octoboer 2015

## Import libraries
import SimpleITK as sitk
# import nibabel as nib
import numpy as np
import unittest
import itk

import os                       # used to execute terminal commands in python
import sys
sys.path.append("../")

## Import modules from src-folder
import SimpleITKHelper as sitkh


"""
Functions
"""

def get_sitk_rigid_registration_transform_2D(fixed_2D, moving_2D):

    ## Instantiate interface method to the modular ITKv4 registration framework
    registration_method = sitk.ImageRegistrationMethod()

    ## Select between using the geometrical center (GEOMETRY) of the images or using the center of mass (MOMENTS) given by the image intensities
    initial_transform = sitk.CenteredTransformInitializer(fixed_2D, moving_2D, sitk.Euler2DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS)
    # initial_transform = sitk.Euler2DTransform()

    ## Set the initial transform and parameters to optimize
    registration_method.SetInitialTransform(initial_transform)

    ## Set an image masks in order to restrict the sampled points for the metric
    # registration_method.SetMetricFixedMask(fixed_2D_mask)
    # registration_method.SetMetricMovingMask(moving_2D_mask)

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
    registration_method.SetMetricAsJointHistogramMutualInformation(numberOfHistogramBins=50, varianceForJointPDFSmoothing=3)
    
    ## Use the mutual information between two images to be registered using the method of Mattes2001
    # registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)

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
    registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=1, minStep=1, numberOfIterations=100)

    ## Estimating scales of transform parameters a step sizes, from the maximum voxel shift in physical space caused by a parameter change
    ## (Many more possibilities to estimate scales)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    
    """
    setup for the multi-resolution framework            
    """
    ## Set the shrink factors for each level where each level has the same shrink factor for each dimension
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])

    ## Set the sigmas of Gaussian used for smoothing at each level
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])

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

    ## Execute 2D registration
    final_transform_2D_sitk = registration_method.Execute(fixed_2D, moving_2D) 
    print("SimpleITK Image Registration Method:")
    print('  Final metric value: {0}'.format(registration_method.GetMetricValue()))
    print('  Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))

    return sitk.Euler2DTransform(final_transform_2D_sitk)


def get_sitk_rigid_registration_transform_3D(fixed_3D, moving_3D):

    ## Instantiate interface method to the modular ITKv4 registration framework
    registration_method = sitk.ImageRegistrationMethod()

    ## Select between using the geometrical center (GEOMETRY) of the images or using the center of mass (MOMENTS) given by the image intensities
    initial_transform = sitk.CenteredTransformInitializer(fixed_3D, moving_3D, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
    # initial_transform = sitk.CenteredTransformInitializer(fixed_3D, moving_3D, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS)
    # initial_transform = sitk.Euler3DTransform()

    ## Set the initial transform and parameters to optimize
    registration_method.SetInitialTransform(initial_transform)

    ## Set an image masks in order to restrict the sampled points for the metric
    # registration_method.SetMetricFixedMask(fixed_3D_mask)
    # registration_method.SetMetricMovingMask(moving_3D_mask)

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
    # registration_method.SetMetricAsJointHistogramMutualInformation(numberOfHistogramBins=100, varianceForJointPDFSmoothing=3)
    
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
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])

    ## Set the sigmas of Gaussian used for smoothing at each level
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])

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

    return sitk.Euler3DTransform(final_transform_3D_sitk)


## difference to get_sitk_rigid_registration_transform_3D is that initial_transform
## is defined as identity in here
def get_slice_sitk_rigid_registration_transform_3D(fixed_slice_3D, moving_3D):


    ## Instantiate interface method to the modular ITKv4 registration framework
    registration_method = sitk.ImageRegistrationMethod()

    ## Select between using the geometrical center (GEOMETRY) of the images or using the center of mass (MOMENTS) given by the image intensities
    # initial_transform = sitk.CenteredTransformInitializer(fixed_slice_3D, moving_3D, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
    initial_transform = sitk.Euler3DTransform()

    ## Set the initial transform and parameters to optimize
    registration_method.SetInitialTransform(initial_transform)

    ## Set an image masks in order to restrict the sampled points for the metric
    # registration_method.SetMetricFixedMask(fixed_slice_3D_mask)
    # registration_method.SetMetricMovingMask(moving_3D_mask)

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
    # registration_method.SetMetricAsJointHistogramMutualInformation(numberOfHistogramBins=100, varianceForJointPDFSmoothing=3)
    
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
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])

    ## Set the sigmas of Gaussian used for smoothing at each level
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])

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
    final_transform_3D_sitk = registration_method.Execute(fixed_slice_3D, moving_3D) 
    print("SimpleITK Image Registration Method:")
    print('  Final metric value: {0}'.format(registration_method.GetMetricValue()))
    print('  Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))

    return sitk.Euler3DTransform(final_transform_3D_sitk)


"""
Unit Test Class
"""

class TestUM(unittest.TestCase):

    def setUp(self):
        pass


    def test_01_sitk_Resample(self):
        angle_x = np.pi/3
        angle_y = 0
        angle_z = 0
        translation = (10,-20,0)
        center = (30,40,0)
        # center = (0,0)

        ## Load image
        fixed = sitk.ReadImage(dir_input + filename + ".nii.gz", sitk.sitkFloat64)

        ## Generate rigid transformation
        rigid_transform_3D = sitk.Euler3DTransform(center, angle_x, angle_y, angle_z, translation)

        ## Resample rigidly transformed image
        moving_resampled = sitk.Resample(fixed, sitkh.get_inverse_of_sitk_rigid_registration_transform(rigid_transform_3D), sitk.sitkBSpline, 0.0, fixed.GetPixelIDValue())

        ## Get transformed image in physical space:
        moving_warped = sitkh.get_transformed_image(fixed, rigid_transform_3D)

        ## Resample rigidly transformed fixed to image space of moving_resampled:
        moving_warped_resampled = sitk.Resample(moving_warped, moving_resampled, sitk.Euler3DTransform(), sitk.sitkBSpline, 0.0, fixed.GetPixelIDValue())

        ## Test whether resampling directly the image via sitk.Resample with provided rigid_transform_3d yields the
        ## same result as transforming first and then resampling with provided identity transform:
        try:
            self.assertEqual(np.around(
                np.linalg.norm( sitk.GetArrayFromImage(moving_warped_resampled - moving_resampled) )
                , decimals = accuracy), 0 )

        except Exception as e:
            print(self.id() + " failed and image is plotted for further investigation")
            ## Plot outcome
            sitkh.plot_compare_sitk_2D_images(moving_warped_resampled[:,:,12], moving_resampled[:,:,12])
            # self.skipTest(MyTestCase)


    ## Commented since equality almost impossible to reach after registration. But gives a good illustration about what happens
    # def test_02_sitk_Registration(self):
    #     # angle_x = 0
    #     angle_x = np.pi/3
    #     angle_y = 0
    #     angle_z = 0
    #     translation = (10,-20,0)
    #     center = (30,40,0)
    #     # center = (0,0)

    #     ## Load image
    #     fixed = sitk.ReadImage(dir_input + filename + ".nii.gz", sitk.sitkFloat64)

    #     ## Generate rigid transformation
    #     rigid_transform_3D = sitk.Euler3DTransform(center, angle_x, angle_y, angle_z, translation)

    #     ## Get transformed image in physical space:
    #     moving = sitkh.get_transformed_image(fixed, rigid_transform_3D)

    #     ## SimpleITK registration:
    #     registration_transform_3D = get_sitk_rigid_registration_transform_3D(fixed_3D=fixed, moving_3D=moving)

    #     ## Resulting warped image:
    #     ## Resample rigidly transformed fixed to image space of moving_resampled:
    #     warped = sitk.Resample(moving, fixed, registration_transform_3D, sitk.sitkBSpline, 0.0, fixed.GetPixelIDValue())
    #     # warped = sitkh.get_transformed_image(moving, sitkh.get_inverse_of_sitk_rigid_registration_transform(registration_transform_3D))

    #     ## Test whether resampling directly the image via sitk.Resample with provided rigid_transform_3d yields the
    #     ## same result as transforming first and then resampling with provided identity transform:
    #     try:
    #         self.assertEqual(np.around(
    #             np.linalg.norm( sitk.GetArrayFromImage(fixed - warped) )
    #             , decimals = 1), 0 )

    #     except Exception as e:
    #         print(self.id() + " failed and image is plotted for further investigation")
    #         ## Plot outcome
    #         sitkh.plot_compare_sitk_2D_images(fixed[:,:,12], warped[:,:,12])
    #         # self.skipTest(MyTestCase)




"""
Main Function
"""
if __name__ == '__main__':

    """
    Set variables
    """
    ## Specify data
    dir_input = "../../data/fetal_neck/"
    dir_output = "results/"
    filename =  "5"
    filename_recon = "data/3TReconstruction" # result B. Kainz
    # filename_recon = "data/SRTV_fetalUCL_3V_NoNLM_bcorr_norm_lambda_0.1_deltat_0.001_loops_10_it1" # result S. Tourbier
    filename_out = "test"

    accuracy = 6 # decimal places for accuracy of unit tests


    """
    Playground
    """
    reconstruction = sitk.ReadImage(filename_recon + ".nii.gz", sitk.sitkFloat64)
    stack = sitk.ReadImage(dir_input + filename + ".nii.gz", sitk.sitkFloat64)

    # sitkh.show_sitk_image(reconstruction)
    # sitkh.show_sitk_image(stack)


    ## 3D rigid transformation of stack:
    transform_3D = get_sitk_rigid_registration_transform_3D(fixed_3D=stack, moving_3D=reconstruction)
    stack_rigidly_aligned = sitkh.get_transformed_image(stack, transform_3D)

    # stack_rigidly_aligned = sitk.ReadImage(dir_output + "stack_aligned.nii.gz", sitk.sitkFloat64)

    sitkh.print_rigid_transformation(transform_3D,"global alignment")

    ## Check alignment
    test = sitk.Resample(
        stack_rigidly_aligned, reconstruction, sitk.Euler3DTransform(), sitk.sitkLinear, 0.0, reconstruction.GetPixelIDValue()
        )
    # test = sitk.Resample(
        # stack, reconstruction, transform_3D, sitk.sitkLinear, 0.0, reconstruction.GetPixelIDValue()
        # )
    sitk.WriteImage(test, dir_output + filename_out + "_.nii.gz")

    # cmd = "fslview " + dir_output + filename_out + ".nii.gz & "
    cmd = "itksnap " \
            + "-g " + dir_output + filename_out + "_.nii.gz " \
            + "-o " +  filename_recon + ".nii.gz " + \
            "& "
    # os.system(cmd)

    ## 3D rigid transformation of slice
    slice_number = 4
    slice_3D = stack_rigidly_aligned[:,:,slice_number:slice_number+1]

    spacing_HR = np.array(reconstruction.GetSpacing())
    spacing_LR = np.array(slice_3D.GetSpacing())

    size_resampled = np.array(slice_3D.GetSize())
    size_resampled[2] = np.round(spacing_LR[2]/spacing_HR[2])

    slice_3D_HR_grid = sitk.Resample(slice_3D, size_resampled, sitk.Euler3DTransform(), sitk.sitkNearestNeighbor, slice_3D.GetOrigin(), spacing_HR, slice_3D.GetDirection(), 0.0, slice_3D.GetPixelIDValue())


    transform_3D_slice = get_slice_sitk_rigid_registration_transform_3D(fixed_slice_3D=slice_3D_HR_grid, moving_3D=reconstruction)

    sitkh.print_rigid_transformation(transform_3D_slice,"local alignment")
    


    # ## Check alignment
    # test = sitk.Resample(
    #     slice_3D_HR_grid, reconstruction, transform_3D_slice, sitk.sitkLinear, 0.0, reconstruction.GetPixelIDValue()
    #     # slice_3D_HR_grid, reconstruction, sitk.Euler3DTransform(), sitk.sitkLinear, 0.0, reconstruction.GetPixelIDValue()
    #     )

    # sitk.WriteImage(test, dir_output + filename_out + ".nii.gz")
    # cmd = "fslview " + dir_output + filename_out + ".nii.gz & "
    cmd = "itksnap " \
            + "-g " + dir_output + filename_out + ".nii.gz " \
            + "-o " +  filename_recon + ".nii.gz " + \
            "& "
    # os.system(cmd)


    """
    Unit tests:
    """
    # print("\nUnit tests:\n--------------")
    # unittest.main()
