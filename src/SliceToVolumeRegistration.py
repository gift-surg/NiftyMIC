## \file SliceToVolumeRegistration.py
#  \brief Perform slice to volume registration 
# 
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date December 2015


## Import libraries
import os                       # used to execute terminal commands in python
import sys
import SimpleITK as sitk
import numpy as np


## Import modules from src-folder
import SimpleITKHelper as sitkh
# import Stack as st
# import Slice as sl


## Class implementing the slice to volume registration algorithm
class SliceToVolumeRegistration:

    ## Constructor
    #  \param[in,out] stack_manager instance of StackManager containing all stacks and additional information
    def __init__(self, stack_manager):
        self._stack_manager = stack_manager
        self._stacks = stack_manager.get_stacks()
        self._N_stacks = stack_manager.get_number_of_stacks()

        ## HACK:
        #  Upsample slices in k-direction to in-plane resolution in order to perform
        #  slice-to-volume registration
        default_pixel_value = 0.0

        for i in range(0, self._N_stacks):
            stack = self._stacks[i]
            slices = stack.get_slices()
            N_slices = stack.get_number_of_slices()

            spacing = np.array(stack.sitk.GetSpacing())
            size = np.array(stack.sitk.GetSize())

            ## Set dimension of each slice in k-direction accordingly
            size[2] = np.round(spacing[2]/spacing[0])

            ## Update spacing in k-direction to be equal in-plane spacing
            spacing[2] = spacing[0]

            for j in range(0, N_slices):
                ## Upsample slice
                slice = slices[j]
                slice._sitk_upsampled = sitk.Resample(
                    slice.sitk, 
                    size, 
                    sitk.Euler3DTransform(), 
                    sitk.sitkNearestNeighbor, 
                    slice.sitk.GetOrigin(), 
                    spacing, 
                    slice.sitk.GetDirection(), 
                    default_pixel_value,
                    slice.sitk.GetPixelIDValue())

                ## Upsample slice mask
                slice._sitk_mask_upsampled = sitk.Resample(
                    slice.sitk_mask, 
                    size, 
                    sitk.Euler3DTransform(), 
                    sitk.sitkNearestNeighbor, 
                    slice.sitk.GetOrigin(), 
                    spacing, 
                    slice.sitk.GetDirection(), 
                    default_pixel_value,
                    slice.sitk_mask.GetPixelIDValue())


    ## Perform slice-to-volume registration of all slices to current estimate of HR volume reconstruction
    #  \param HR_volume current estimate of reconstructed HR volume (Stack object)
    def run_slice_to_volume_registration(self, HR_volume):
        print("Slice-to-volume registration")
        for i in range(0, self._N_stacks):
        # for i in range(0,1):
            print("  Stack %s/%s" %(i,self._N_stacks-1))
            stack = self._stacks[i]
            slices = stack.get_slices()
            N_slices = stack.get_number_of_slices()
            
            for j in range(0, N_slices):
                slice = slices[j]

                ## Register slice to current volume
                rigid_transform = self._get_rigid_registration_transform_3D_sitk(slice, HR_volume,1)

                ## print if translation is strange
                translation = rigid_transform.GetTranslation()
                if np.linalg.norm(translation)>10:
                    print("Rigid registration of slice %s/%s within stack %s is odd:" %(j,N_slices-1,i))
                    sitkh.print_rigid_transformation(rigid_transform)

                    continue
                ## Trafo from physical origin to origin of slice j
                slice_trafo = slice.get_affine_transform()

                ## New affine transform of slice j with respect to rigid registration
                affine_transform = sitkh.get_composited_sitk_affine_transform(rigid_transform, slice_trafo)

                ## Update affine transform of slice j
                slice.set_affine_transform(affine_transform)

                # sitkh.print_rigid_transformation(rigid_transform)
                
        return None


    ## Rigid registration routine based on SimpleITK
    #  \param fixed_slice_3D upsampled fixed Slice
    #  \param moving_3D moving Stack
    #  \param display_registration_info display registration summary at the end of execution (default=0)
    #  \return Rigid registration as sitk.Euler3DTransform object
    def _get_rigid_registration_transform_3D_sitk(self, fixed_slice_3D, moving_3D, display_registration_info=0):

        ## Instantiate interface method to the modular ITKv4 registration framework
        registration_method = sitk.ImageRegistrationMethod()

        ## Select between using the geometrical center (GEOMETRY) of the images or using the center of mass (MOMENTS) given by the image intensities
        # initial_transform = sitk.CenteredTransformInitializer(fixed_slice_3D._sitk_upsampled, moving_3D.sitk, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
        initial_transform = sitk.Euler3DTransform()

        ## Set the initial transform and parameters to optimize
        registration_method.SetInitialTransform(initial_transform)

        ## Set an image masks in order to restrict the sampled points for the metric
        # registration_method.SetMetricFixedMask(fixed_slice_3D._sitk_mask_upsampled)
        # registration_method.SetMetricMovingMask(moving_3D.sitk_mask)

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
        final_transform_3D_sitk = registration_method.Execute(fixed_slice_3D._sitk_upsampled, moving_3D.sitk) 

        if display_registration_info:
            print("SimpleITK Image Registration Method:")
            print('  Final metric value: {0}'.format(registration_method.GetMetricValue()))
            print('  Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))

        return sitk.Euler3DTransform(final_transform_3D_sitk)

