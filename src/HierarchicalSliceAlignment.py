## \file HierarchicalSliceAlignment.py
#  \brief Compute first estimate of HR volume based on given stacks
# 
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date April 2016


## Import libraries
import os                       # used to execute terminal commands in python
import sys
import SimpleITK as sitk
import numpy as np

## Import modules from src-folder
import SimpleITKHelper as sitkh
import StackManager as sm
import ScatteredDataApproximation as sda
import StackAverage as sa
import Stack as st


## Class to implement hierarchical alignment of slices before slice to volume
#  registration. Idea: Take advantage of knowledge of interleaved acquisition.
#  The updated slice locations are directly encoded in their respective stacks
class HierarchicalSliceAlignment:

    ## Constructor
    #  \param[in] stack_manager instance of StackManager containing all stacks and additional information
    #  \param[in] HR_volume Stack object containing the current estimate of the HR volume (required for defining HR space)
    def __init__(self, stack_manager, HR_volume=None):

        ## Initialize variables
        self._stack_manager = stack_manager
        self._stacks = stack_manager.get_stacks()
        self._N_stacks = stack_manager.get_number_of_stacks()

        if HR_volume is not None:
            self._HR_volume = HR_volume
        else:
            self._HR_volume = self._stacks[0].get_isotropically_resampled_stack_from_slices()

        ## Define dictionary to choose computational approach for estimating first HR volume
        self._get_volume_estimate = {
            "SDA"       :   self._get_volume_estimate_SDA,
            "Average"   :   self._get_volume_estimate_averaging
        }
        # self._volume_estimate_approach = "SDA"        # default reconstruction approach
        self._volume_estimate_approach = "Average"        # default reconstruction approach

        ## SDA reconstruction settings:
        self._SDA_sigma = 1                 # sigma for recursive Gaussian smoothing
        self._SDA_type = 'Shepard-YVV'      # Either 'Shepard-YVV' or 'Shepard-Deriche'


    ## Perform hierarchical alignment for each stack
    def run_hierarchical_alignment(self):

        step = 2

        HR_volume = st.Stack.from_stack(self._HR_volume)

        stacks_ind_all = np.arange(0, self._N_stacks)

        ## Obtain first HR volume estimate for comparison
        self._SDA = sda.ScatteredDataApproximation(self._stack_manager, HR_volume)
        self._SDA.set_sigma(1)
        self._SDA.set_approach(self._SDA_type)
        self._SDA.run_reconstruction()    
        vol0_sitk = sitk.Image(self._SDA.get_HR_volume().sitk)

        # for i in range(0, 1):
        # for i in range(1, self._N_stacks):
        for i in range(0, self._N_stacks):

            ## Get all stacks apart from current one
            stacks_ind = list(set(stacks_ind_all) - set([i]))
            stacks = [ self._stacks[j] for j in stacks_ind ]

            ## Obtain estimated volume based on those stacks
            volume_estimate = self._get_volume_estimate[self._volume_estimate_approach](stacks, HR_volume)

            # volume_estimate.show(title="VolumeEstimate_"+str(i))

            self._hierarchically_align_stack(self._stacks[i], volume_estimate, step)


        # self._SDA = sda.ScatteredDataApproximation(self._stack_manager, HR_volume)
        self._SDA.run_reconstruction()    
        vol = self._SDA.get_HR_volume()

        sitkh.show_sitk_image(vol.sitk, overlay=vol0_sitk,title="comparison")
        
        ## Perform reconstruction via SDA
        print("\n\t--- Run Scattered Data Approximation algorithm ---")

        return st.Stack.from_stack(self._SDA.get_HR_volume())

        return None

    ## Perform hierarchical strategy to align slices within stack
    #  \param[in] stack stack as Stack object whose slices will be aligned
    #  \param[in] volume_estimate stack as Stack object which will serve as moving object for registration
    #  \param[in] step interleaved acquisition step
    #  \post Slice objects of group carry updated affine transformation
    def _hierarchically_align_stack(self, stack, volume_estimate, step):

        slices = stack.get_slices()

        i_min = 0
        i_max = len(slices)

        ## 1) Register entire stack with volume estimate
        sitk.WriteImage(stack.sitk_mask, "/tmp/bla.nii.gz")
        ### WTF!?!?! The line below works when line above is there!? Otherwise not!?
        transform = self._get_rigid_registration_transform(stack, volume_estimate, 1)

        sitkh.print_rigid_transformation(transform)
        self._update_slice_transformations_of_group(stack, volume_estimate, range(0, len(slices)), transform)

        ## 2) Hierarchical Alignment Strategy
        for i in range(0, step):
            ## get indices of slices which are grouped together based on interleaved acquisition
            ind = np.arange(i_min+i, i_max, step)

            ## Perform recursive alignment strategy for those indices within chosen stack
            self._apply_recursive_alignment_of_group(stack, volume_estimate, ind)


    ## Perform recursive alignment in order to perform hierarchical registration strategy
    #  within chosen stack
    #  \param[in] stack stack as Stack object whose slices will be aligned
    #  \param[in] volume_estimate stack as Stack object which will serve as moving object for registration
    #  \param[in] ind list of indices specifying the hierarchical group of slices
    #  \post Slice objects of group carry updated affine transformation
    def _apply_recursive_alignment_of_group(self, stack, volume_estimate, ind):

        print("Register group of slices " + str(ind))

        ## Get number of slices within group
        N = len(ind)

        ## If more than one slice, register and half into two subgroups afterwards
        if N > 1:
            ## Register group
            transform = self._get_rigid_registration_transform_of_hierarchical_group(stack, volume_estimate, ind)
            sitkh.print_rigid_transformation(transform)
            self._update_slice_transformations_of_group(stack, volume_estimate, ind, transform)

            ## Half into subgroups and run recursive alignment
            mid = N/2

            self._apply_recursive_alignment_of_group(stack, volume_estimate, ind[0:mid])
            self._apply_recursive_alignment_of_group(stack, volume_estimate, ind[mid:])

        ## If only one slice in group, only register that single slice
        elif N is 1:
            ## Register single slice
            transform = self._get_rigid_registration_transform(stack.get_slice(ind[0]), volume_estimate,1)
            sitkh.print_rigid_transformation(transform)
            self._update_slice_transformations_of_group(stack, volume_estimate, ind, transform)



    ## Register group of slices to volume estimate to update
    #  \param[in] stack stack as Stack object whose slices will be aligned
    #  \param[in] volume_estimate stack as Stack object which will serve as moving object for registration
    #  \param[in] ind indices of slices within stack which will be registered to volume_estimate
    #  \return registration transforms aligning stack[ind] with volume_estimate
    def _get_rigid_registration_transform_of_hierarchical_group(self, stack, volume_estimate, ind):
        slices = stack.get_slices()

        ## Retrieve indices
        i_min = ind[0]
        i_max = ind[-1]+1
        step = ind[1]-ind[0]

        # print np.arange(i_min, i_max, step)

        ## Create image stack and mask based on group
        group_sitk = stack.sitk[:,:,i_min:i_max:step]
        group_sitk_mask = stack.sitk_mask[:,:,i_min:i_max:step]

        ## Update position of grouped stack based on "basis" slice
        origin = slices[i_min].sitk.GetOrigin()
        direction = slices[i_min].sitk.GetDirection()

        group_sitk.SetOrigin(origin)
        group_sitk.SetDirection(direction)

        group_sitk_mask.SetOrigin(origin)
        group_sitk_mask.SetDirection(direction)

        ## Create Stack object
        group = st.Stack.from_sitk_image(group_sitk, str(i_min)+"_"+str(step)+"_"+str(i_max), group_sitk_mask)

        ## Get rigid registration transform
        transform = self._get_rigid_registration_transform(group, volume_estimate, 1)
        
        return transform


    ## Update affine transforms, i.e. position and orientation, of grouped slices
    #  \param[in] stack stack as Stack object whose slices will be aligned
    #  \param[in] volume_estimate stack as Stack object which will serve as moving object for registration
    #  \param[in] ind indices of slices within stack which will be registered to volume_estimate
    #  \param[in] transform registration transform which aligns stack[ind] with volume_estimate
    #  \post Slice objects within stack carry updated information
    def _update_slice_transformations_of_group(self, stack, volume_estimate, ind, transform):
        slices = stack.get_slices()

        ## Update transforms within group of slices        
        for i in ind:
            slice = slices[i]

            ## Compute new affine transform for slice
            slice_transform = slice.get_affine_transform()
            affine_transform = sitkh.get_composited_sitk_affine_transform(transform, slice_transform)

            ## Update affine transform of slice
            slice.update_affine_transform(affine_transform)


    ## Compute average of all registered stacks
    #  \param[in] stacks stacks as Stack objects used for average
    #  \param[in] HR_volume Stack object used for specifying the phyiscal space for averaging
    #  \return averaged volume as Stack object
    def _get_volume_estimate_averaging(self, stacks, HR_volume):
        
        ## Create Stack Average instance
        stack_manager = sm.StackManager.from_stacks(stacks)
        self._sa = sa.StackAverage(stack_manager, HR_volume)

        ## Do not black out non-masked voxels
        self._sa.set_mask_volume_voxels(False)

        print("\n\t--- Run averaging of stacks ---")
        self._sa.run_averaging()

        return st.Stack.from_stack(self._sa.get_averaged_volume())


    ## Estimate the HR volume via SDA approach
    #  \param[in] stacks stacks as Stack objects used for average
    #  \param[in] HR_volume Stack object used for specifying the phyiscal space for SDA
    #  \return averaged volume as Stack object
    def _get_volume_estimate_SDA(self, stacks, HR_volume):

        stack_manager = sm.StackManager.from_stacks(stacks)

        self._SDA = sda.ScatteredDataApproximation(stack_manager, HR_volume)
        self._SDA.set_sigma(self._SDA_sigma)
        self._SDA.set_approach(self._SDA_type)
        
        ## Perform reconstruction via SDA
        print("\n\t--- Run Scattered Data Approximation algorithm ---")
        self._SDA.run_reconstruction()    

        return st.Stack.from_stack(self._SDA.get_HR_volume())


    ## Rigid registration routine based on SimpleITK
    #  \param[in] fixed_3D fixed Stack representing acquired stacks
    #  \param[in] moving_3D moving Stack representing current HR volume estimate
    #  \param[in] display_registration_info display registration summary at the end of execution (default=0)
    #  \return Rigid registration as sitk.Euler3DTransform object
    def _get_rigid_registration_transform(self, fixed_3D, moving_3D, display_registration_info=0):


        ## Instantiate interface method to the modular ITKv4 registration framework
        registration_method = sitk.ImageRegistrationMethod()

        ## Select between using the geometrical center (GEOMETRY) of the images or using the center of mass (MOMENTS) given by the image intensities
        # initial_transform = sitk.CenteredTransformInitializer(fixed_3D.sitk, moving_3D.sitk, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)

        initial_transform = sitk.Euler3DTransform()

        ## Set the initial transform and parameters to optimize
        registration_method.SetInitialTransform(initial_transform)

        ## Set an image masks in order to restrict the sampled points for the metric
        registration_method.SetMetricFixedMask(fixed_3D.sitk_mask)
        # registration_method.SetMetricMovingMask(moving_3D.sitk_mask)

        ## Set percentage of pixels sampled for metric evaluation
        # registration_method.SetMetricSamplingStrategy(registration_method.NONE)

        ## Set interpolator to use
        registration_method.SetInterpolator(sitk.sitkLinear)

        """
        similarity metric settings
        """
        ## Use normalized cross correlation using a small neighborhood for each voxel between two images, with speed optimizations for dense registration
        # registration_method.SetMetricAsANTSNeighborhoodCorrelation(radius=10)
        
        ## Use negative normalized cross correlation image metric
        registration_method.SetMetricAsCorrelation()

        ## Use demons image metric
        # registration_method.SetMetricAsDemons(intensityDifferenceThreshold=1e-3)

        ## Use mutual information between two images
        # registration_method.SetMetricAsJointHistogramMutualInformation(numberOfHistogramBins=100, varianceForJointPDFSmoothing=3)
        
        ## Use the mutual information between two images to be registered using the method of Mattes2001
        # registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=200)

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
        registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=0.1, minStep=0.01, numberOfIterations=20)

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
        final_transform_3D_sitk = registration_method.Execute(fixed_3D.sitk, moving_3D.sitk) 

        if display_registration_info:
            print("SimpleITK Image Registration Method:")
            print('  Final metric value: {0}'.format(registration_method.GetMetricValue()))
            print('  Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))

        return sitk.Euler3DTransform(final_transform_3D_sitk)
