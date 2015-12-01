## \file FirstEstimateOfHRVolume.py
#  \brief  
# 
#  \author Michael Ebner
#  \date November 2015


## Import libraries
import os                       # used to execute terminal commands in python
import sys
import SimpleITK as sitk
import numpy as np

import matplotlib.pyplot as plt


## Import modules from src-folder
import SimpleITKHelper as sitkh
import InPlaneRigidRegistration as iprr
import Stack as stack


class FirstEstimateOfHRVolume:

    def __init__(self, stack_manager, dir_results, filename_reconstructed_volume):
        self._stack_manager = stack_manager
        self._stacks = stack_manager.get_stacks()
        self._N_stacks = stack_manager.get_number_of_stacks()

        self._target_stack_number = 0
        self._HR_volume = None

        self._dir_results = dir_results
        self._filename_reconstructed_volume = filename_reconstructed_volume

        ## Compute first estimate of HR volume
        self._compute_first_estimate_of_HR_volume()

        return None


    def _compute_first_estimate_of_HR_volume(self):

        rigid_registrations = [None]*self._N_stacks

        ## Run in-plane rigid registration of all stacks
        in_plane_rigid_registration =  iprr.InPlaneRigidRegistration(self._stack_manager)

        ## Get resampled stacks of planarly aligned slices as Stack objects
        stacks_aligned = in_plane_rigid_registration.get_resampled_planarly_aligned_stacks()

        ## Get target stack as Stack object
        
        ## Resample chosen target volume and its mask to isotropic grid
        ## TODO: replace self._target_stack_number with "best choice" of stack
        self._HR_volume = self._get_isotropically_resampled_stack(stacks_aligned[self._target_stack_number])

        ## Register all planarly aligned stacks to resampled target volume
        rigid_registrations = self._get_rigid_registrations_of_all_stacks_to_HR_volume()

        return None


    def _get_isotropically_resampled_stack(self, target_stack):
        
        # HR_volume = stack.Stack(target_stack.sitk)

        ## Read original spacing (voxel dimension) and size of target stack:
        spacing = np.array(target_stack.sitk.GetSpacing())
        size = np.array(target_stack.sitk.GetSize())

        ## Update information according to isotropic resolution
        size[2] = np.round(spacing[2]/spacing[0]*size[2])
        spacing[2] = spacing[0]

        ## Resample to isotropic grid
        default_pixel_value = 0.0

        HR_volume_sitk =  sitk.Resample(
            target_stack.sitk, 
            size, 
            sitk.Euler3DTransform(), 
            sitk.sitkNearestNeighbor, 
            target_stack.sitk.GetOrigin(), 
            spacing,
            target_stack.sitk.GetDirection(),
            default_pixel_value,
            target_stack.sitk.GetPixelIDValue())

        HR_volume_sitk_mask =  sitk.Resample(
            target_stack.sitk_mask, 
            size, 
            sitk.Euler3DTransform(), 
            sitk.sitkNearestNeighbor, 
            target_stack.sitk.GetOrigin(), 
            spacing,
            target_stack.sitk.GetDirection(),
            default_pixel_value,
            target_stack.sitk.GetPixelIDValue())

        HR_volume = stack.Stack(HR_volume_sitk, self._dir_results, self._filename_reconstructed_volume)
        HR_volume.add_mask(HR_volume_sitk_mask)

        return HR_volume


    def _get_rigid_registrations_of_all_stacks_to_HR_volume(self):
        rigid_registrations = [None]*self._N_stacks

        # self._stacks[0].show_stack(1)
        # self._stacks[1].show_stack(1)
        # self._HR_volume.show_stack(1)

        # self._HR_volume.sitk = sitk.ReadImage("GettingStarted/data/3TReconstruction.nii.gz")


        for i in xrange(0, self._N_stacks):

            rigid_registrations[i] = self._get_rigid_registration_transform_3D_sitk(self._stacks[i], self._HR_volume)

            test = sitk.Resample(sitkh.get_transformed_image(self._stacks[i].sitk, rigid_registrations[i]),
                self._HR_volume.sitk, sitk.Euler3DTransform(), sitk.sitkLinear, 0.0, self._HR_volume.sitk.GetPixelIDValue())

            full_file_name = os.path.join("/tmp/", self._stacks[i].get_filename() + ".nii.gz")
            sitk.WriteImage(test, full_file_name)

            sitkh.print_rigid_transformation(rigid_registrations[i])

        sitk.WriteImage(self._HR_volume.sitk, "/tmp/HR_volume.nii.gz")

            

    def _get_rigid_registration_transform_3D_sitk(self, fixed_3D, moving_3D, print_registration_info=0):

        ## Instantiate interface method to the modular ITKv4 registration framework
        registration_method = sitk.ImageRegistrationMethod()

        ## Select between using the geometrical center (GEOMETRY) of the images or using the center of mass (MOMENTS) given by the image intensities
        initial_transform = sitk.CenteredTransformInitializer(fixed_3D.sitk, moving_3D.sitk, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)

        # initial_transform = sitk.Euler3DTransform()

        ## Set the initial transform and parameters to optimize
        registration_method.SetInitialTransform(initial_transform)

        ## Set an image masks in order to restrict the sampled points for the metric
        # registration_method.SetMetricFixedMask(fixed_3D.sitk_mask)
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
        registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=0.5, minStep=0.05, numberOfIterations=2000)

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
        final_transform_3D_sitk = registration_method.Execute(fixed_3D.sitk, moving_3D.sitk) 

        if print_registration_info:
            print("SimpleITK Image Registration Method:")
            print('  Final metric value: {0}'.format(registration_method.GetMetricValue()))
            print('  Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))

        return sitk.Euler3DTransform(final_transform_3D_sitk)



    def get_first_estimate_of_HR_volume(self):
        return self._HR_volume