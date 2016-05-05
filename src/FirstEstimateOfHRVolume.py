## \file FirstEstimateOfHRVolume.py
#  \brief Compute first estimate of HR volume based on given stacks
# 
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date November 2015
#  \version{0.1} Estimate first HR volume based on averaging stacks, Nov 2015
#  \version{0.2} Add possibility to not register stacks to target first, March 2016


## Import libraries
import os                       # used to execute terminal commands in python
import sys
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

## Import modules from src-folder
import SimpleITKHelper as sitkh
import InPlaneRigidRegistration as iprr
import StackManager as sm
import Stack as st
import ScatteredDataApproximation as sda
import StackAverage as sa


## Class to compute first estimate of HR volume. Steps included are:
#  -# In-plane registration of all stacks (optional)
#  -# (Resample in-plane registered stacks to 3D-volumes)
#  -# Pick one (planarly-aligned) stack and assign it as target volume
#  -# Resample target volume on isotropic grid
#  -# Register all (planarly-aligned) stacks to target-volume (optional)
#  -# Create first HR volume estimate: Average all registered (planarly-aligned) stacks
#  -# Update all slice transformations: Each slice position gets updated according to alignment with HR volume
class FirstEstimateOfHRVolume:

    ## Constructor
    #  \param[in] stack_manager instance of StackManager containing all stacks and additional information
    #  \param[in] filename_reconstructed_volume chosen filename for created HR volume, Stack object
    #  \param[in] target_stack_number stack chosen to define space and coordinate system of HR reconstruction, integer
    def __init__(self, stack_manager, filename_reconstructed_volume, target_stack_number):
        self._stack_manager = stack_manager
        self._stacks = stack_manager.get_stacks()
        self._N_stacks = stack_manager.get_number_of_stacks()

        self._filename_reconstructed_volume = filename_reconstructed_volume
        self._target_stack_number = target_stack_number

        ## additional boundary surrounding the stack in mm (used to get additional black frame)
        boundary = 5 

        ## Resample chosen target volume and its mask to isotropic grid with optional additional boundary
        ## \todo replace self._target_stack_number with "best choice" of stack
        self._HR_volume = self._get_isotropically_resampled_stack(self._stacks[self._target_stack_number], boundary)

        ## Flags indicating whether or not these options are selected
        self._flag_use_in_plane_rigid_registration_for_initial_volume_estimate = False
        self._flag_register_stacks_before_initial_volume_estimate = False

        ## Rigid registrations obtained after registering each stack to (upsampled) target stack
        #  Why? Every update on a slice base is not fed back to entire stack volume, cf. _run_averaging
        self._rigid_registrations = [sitk.Euler3DTransform()]*self._N_stacks

        ## Define dictionary to choose computational approach for estimating first HR volume
        self._update_HR_volume_estimate = {
            "SDA"       :   self._run_SDA,
            "Average"   :   self._run_averaging
        }
        self._recon_approach = "SDA"        # default reconstruction approach

        ## SDA reconstruction settings:
        self._SDA_sigma = 0.6                 # sigma for recursive Gaussian smoothing
        self._SDA_type = 'Shepard-YVV'      # Either 'Shepard-YVV' or 'Shepard-Deriche'


    ## Set flag to use in-plane of all slices to each other within their stacks
    def use_in_plane_registration_for_initial_volume_estimate(self, flag):
        self._flag_use_in_plane_rigid_registration_for_initial_volume_estimate = flag


    ## Set flag to globally register each stack with chosen target stack.
    #  Otherwise, the initial positions as given from the original files are used
    def register_stacks_before_initial_volume_estimate(self, flag):
        self._flag_register_stacks_before_initial_volume_estimate = flag


    ## Get current estimate of HR volume
    #  \return current estimate of HR volume, instance of Stack
    def get_HR_volume(self):
        return self._HR_volume


    ## Set approach for reconstructing the HR volume. It can be either 
    #  'SDA' or 'Average'
    #  \param[in] recon_approach either 'SDA' or 'Average', string
    def set_reconstruction_approach(self, recon_approach):
        if recon_approach not in ["SDA", "Average"]:
            raise ValueError("Error: regularization type can only be either 'SDA' or 'Average'")

        self._recon_approach = recon_approach


    ## Get chosen type of regularization.
    #  \return regularization type as string
    def get_reconstruction_approach(self):
        return self._recon_approach


    ## Execute computation for first estimate of HR volume.
    #  This function steers the estimation of first HR volume which then updates
    #  self._HR_volume
    #  The process consists of several steps:
    #  -# In-plane registration of all stacks (optional)
    #  -# (Resample in-plane registered stacks to 3D-volumes)
    #  -# Pick one (planarly-aligned) stack and assign it as target volume
    #  -# Resample target volume on isotropic grid
    #  -# Register all (planarly-aligned) stacks to target-volume (optional)
    #  -# Create first HR volume estimate: Average all registered (planarly-aligned) stacks
    #  -# Update all slice transformations: Each slice position gets updated according to alignment with HR volume
    #  \param[in] display_info display information of registration results as we go along
    def compute_first_estimate_of_HR_volume(self, display_info=0):

        ## Use stacks with in-plane aligned slices
        if self._flag_use_in_plane_rigid_registration_for_initial_volume_estimate:
            print("In-plane alignment of slices within each stack is performed")
            ## Run in-plane rigid registration of all stacks

            # self._stacks[1].show(1)

            self._in_plane_rigid_registration =  iprr.InPlaneRigidRegistration(self._stack_manager)
            self._in_plane_rigid_registration.run_in_plane_rigid_registration()
            # stacks = self._in_plane_rigid_registration.get_resampled_planarly_aligned_stacks()

            ## Update HR volume and its mask after planar alignment of slices
            foo = st.Stack.from_stack(self._HR_volume) ## only for show_sitk_image to have comparison before-after

            target_stack_aligned = self._stacks[self._target_stack_number].get_resampled_stack_from_slices()
            self._HR_volume = self._get_isotropically_resampled_stack(target_stack_aligned)

            sitkh.show_sitk_image(foo.sitk, overlay=self._HR_volume.sitk, title="upsampled_target_stack_before_and_after_in_plane_reg")

        ## Use "raw" stacks as given by their originally given physical positions
        else:
            print("In-plane alignment of slices within each stack is NOT performed")

        ## If desired: Register all (planarly) aligned stacks to resampled target volume
        if self._flag_register_stacks_before_initial_volume_estimate:
            print("Rigid registration between each stack and target is performed")
            self._rigidly_register_all_stacks_to_HR_volume(print_trafos=display_info)

        ## No rigid registration
        else:
            print("Rigid registration between each stack and target is NOT performed")

        ## Update HR volume: Compute average of all (registered) stacks
        self._update_HR_volume_estimate[self._recon_approach]()


    ## Resample stack to isotropic grid
    #  The image and its mask get resampled to isotropic grid 
    #  (in-plane resolution also in through-plane direction)
    #  \param[in] target_stack Stack being resampled
    #  \param[in] boundary additional boundary surrounding stack in mm 
    #  \return Isotropically resampled Stack
    def _get_isotropically_resampled_stack(self, target_stack, boundary):
        
        ## Read original spacing (voxel dimension) and size of target stack:
        spacing = np.array(target_stack.sitk.GetSpacing())
        size = np.array(target_stack.sitk.GetSize())
        origin = np.array(target_stack.sitk.GetOrigin())

        ## Isotropic resolution for HR volume
        spacing_HR_volume = spacing[0]*np.ones(3)

        ## Update information according to isotropic resolution if no boundary is given
        size_HR_volume_z_exact = spacing[2]/spacing[0]*size[2]
        size_HR_volume = np.array((size[0], size[1], np.ceil(size_HR_volume_z_exact))).astype('int')

        ## Compensate for residual in z-direction in physical space
        residual = size_HR_volume[2] - size_HR_volume_z_exact
        a_z = target_stack.sitk.TransformIndexToPhysicalPoint((0,0,1)) - origin
        e_z = a_z/np.linalg.norm(a_z) ## unit direction of z-axis in physical space

        ## Add spacing to residual (due to discretization) to get translation
        translation = e_z * (residual+spacing[0])

        ## Updated origin for HR "to not lose information due to discretization"
        origin_HR_volume = origin - translation


        ## Add additional boundary if desired
        if boundary is not 0:
            ## Get boundary in voxel space
            boundary_vox = np.round(boundary/spacing[0]).astype("int")
            
            ## Compute size of resampled stack by considering additional boundary
            size_HR_volume += + 2*boundary_vox

            ## Compute origin of resampled stack by considering additional boundary
            a_x = target_stack.sitk.TransformIndexToPhysicalPoint((1,0,0)) - origin
            a_y = target_stack.sitk.TransformIndexToPhysicalPoint((0,1,0)) - origin
            a_z = target_stack.sitk.TransformIndexToPhysicalPoint((0,0,1)) - origin
            e_x = a_x/np.linalg.norm(a_x)
            e_y = a_y/np.linalg.norm(a_y)
            e_z = a_z/np.linalg.norm(a_z)

            translation += (e_x + e_y + e_z)*boundary_vox*spacing[0]

            origin_HR_volume = origin - translation

        ## Resample image and its mask to isotropic grid
        default_pixel_value = 0.0

        HR_volume_sitk =  sitk.Resample(
            target_stack.sitk, 
            size_HR_volume, 
            sitk.Euler3DTransform(), 
            sitk.sitkNearestNeighbor, 
            origin_HR_volume, 
            spacing_HR_volume,
            target_stack.sitk.GetDirection(),
            default_pixel_value,
            target_stack.sitk.GetPixelIDValue())

        HR_volume_sitk_mask =  sitk.Resample(
            target_stack.sitk_mask, 
            size_HR_volume, 
            sitk.Euler3DTransform(), 
            sitk.sitkNearestNeighbor, 
            origin_HR_volume, 
            spacing_HR_volume,
            target_stack.sitk.GetDirection(),
            default_pixel_value,
            target_stack.sitk_mask.GetPixelIDValue())

        ## Create Stack instance of HR_volume
        HR_volume = st.Stack.from_sitk_image(HR_volume_sitk, self._filename_reconstructed_volume, HR_volume_sitk_mask)

        return HR_volume


    ## Register all stacks to chosen target stack (HR volume)
    #  \post each Slice is updated according to obtained registration
    def _rigidly_register_all_stacks_to_HR_volume(self, print_trafos):

        ## Compute rigid registrations aligning each stack with the HR volume
        for i in range(0, self._N_stacks):
            if self._flag_use_in_plane_rigid_registration_for_initial_volume_estimate:
                ## Get resampled stacks of planarly aligned slices as Stack objects (3D volume)
                stack = self._stacks[i].get_resampled_stack_from_slices()

            else:
                stack = self._stacks[i]

            self._rigid_registrations[i] = self._get_rigid_registration_transform_3D_sitk(stack, self._HR_volume)

            ## Print rigid registration results (optional)
            if print_trafos:
                sitkh.print_rigid_transformation(self._rigid_registrations[i])

        ## Update all slice transformations based on obtaine registration
        #  Note: trafos of self._stack do not get updated!
        self._update_slice_transformations()


    ## Update all slice transformations of each stack given the rigid transformations
    #  computed to align each stack with the HR volume
    def _update_slice_transformations(self):

        for i in range(0, self._N_stacks):
            stack = self._stacks[i]

            ## Rigid transformation to align stack i with target (HR volume)
            T = self._rigid_registrations[i]

            for j in range(0, stack.get_number_of_slices()):
                slice = stack._slices[j]
                
                ## Trafo from physical origin to origin of slice j
                slice_trafo = slice.get_affine_transform()

                ## New affine transform of slice j with respect to rigid registration
                affine_transform = sitkh.get_composited_sitk_affine_transform(T, slice_trafo)

                ## Update affine transform of slice j
                slice.update_affine_transform(affine_transform)


    ## Rigid registration routine based on SimpleITK
    #  \param[in] fixed_3D fixed Stack representing acquired stacks
    #  \param[in] moving_3D moving Stack representing current HR volume estimate
    #  \param[in] display_registration_info display registration summary at the end of execution (default=0)
    #  \return Rigid registration as sitk.Euler3DTransform object
    def _get_rigid_registration_transform_3D_sitk(self, fixed_3D, moving_3D, display_registration_info=0):

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
        registration_method.SetMetricAsANTSNeighborhoodCorrelation(radius=5)
        
        ## Use negative normalized cross correlation image metric
        # registration_method.SetMetricAsCorrelation()

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
        # registration_method.SetOptimizerAsAmoeba(simplexDelta=0.1, numberOfIterations=100, parametersConvergenceTolerance=1e-8, functionConvergenceTolerance=1e-4, withRestarts=False)

        ## Conjugate gradient descent optimizer with a golden section line search for nonlinear optimization
        # registration_method.SetOptimizerAsConjugateGradientLineSearch(learningRate=1, numberOfIterations=100, convergenceMinimumValue=1e-8, convergenceWindowSize=10)

        ## Set the optimizer to sample the metric at regular steps
        # registration_method.SetOptimizerAsExhaustive(numberOfSteps=50, stepLength=1.0)

        ## Gradient descent optimizer with a golden section line search
        # registration_method.SetOptimizerAsGradientDescentLineSearch(learningRate=1, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10)

        ## Limited memory Broyden Fletcher Goldfarb Shannon minimization with simple bounds
        # registration_method.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, maximumNumberOfIterations=500, maximumNumberOfCorrections=5, maximumNumberOfFunctionEvaluations=200, costFunctionConvergenceFactor=1e+7)

        ## Regular Step Gradient descent optimizer
        registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=0.5, minStep=0.05, numberOfIterations=2000)

        ## Estimating scales of transform parameters a step sizes, from the maximum voxel shift in physical space caused by a parameter change
        ## (Many more possibilities to estimate scales)
        # registration_method.SetOptimizerScalesFromPhysicalShift()
        registration_method.SetOptimizerScalesFromJacobian()
        
        
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

        if display_registration_info:
            print("SimpleITK Image Registration Method:")
            print('  Final metric value: {0}'.format(registration_method.GetMetricValue()))
            print('  Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))

        return sitk.Euler3DTransform(final_transform_3D_sitk)


    """
    Averaging of stacks
    """
    ## Compute average of all registered stacks and update self._HR_volume
    #  \post self._HR_volume is overwritten with new estimate
    def _run_averaging(self):
        
        ## Use planarly aligned stacks for average if given
        if self._flag_use_in_plane_rigid_registration_for_initial_volume_estimate:
            stacks = [None]*self._N_stacks
            for i in range(0, self._N_stacks):
                stacks[i] = self._stacks[i].get_resampled_stack_from_slices()

        else:
            stacks = self._stacks

        self._sa = sa.StackAverage(sm.StackManager.from_stacks(stacks), self._HR_volume)
        self._sa.set_averaged_volume_name(self._filename_reconstructed_volume)
        self._sa.set_stack_transformations(self._rigid_registrations)

        print("\n\t--- Run averaging of stacks ---")
        self._sa.run_averaging()
        self._HR_volume = self._sa.get_averaged_volume()
        


    """
    Scattered Data Approximation: Shepard's like reconstruction approaches
    """
    ## Set sigma used for recursive Gaussian smoothing
    #  \param[in] sigma, scalar
    def set_SDA_sigma(self, sigma):
        self._SDA_sigma = sigma


    ## Get sigma used for recursive Gaussian smoothing
    #  \return sigma, scalar
    def get_SDA_sigma(self):
        return self._SDA_sigma


    ## Set SDA approach. It can be either 'Shepard' or 'Shepard-Deriche'
    #  \param[in] SDA_approach either 'Shepard' or 'Shepard-Deriche', string
    def set_SDA_approach(self, SDA_approach):
        if SDA_approach not in ["Shepard-YVV", "Shepard-Deriche"]:
            raise ValueError("Error: SDA approach can only be either 'Shepard-YVV' or 'Shepard-Deriche'")

        self._SDA_approach = SDA_approach


    ## Estimate the HR volume via SDA approach
    #  \post self._HR_volume is overwritten with new estimate
    def _run_SDA(self):

        self._SDA = sda.ScatteredDataApproximation(self._stack_manager, self._HR_volume)
        self._SDA.set_sigma(self._SDA_sigma)
        self._SDA.set_approach(self._SDA_type)
        
        ## Perform reconstruction via SDA
        print("\n\t--- Run Scattered Data Approximation algorithm ---")
        self._SDA.run_reconstruction()    
