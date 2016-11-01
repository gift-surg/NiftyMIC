#!/usr/bin/python

## \file simulateAcquisition.py
#  \brief main-file used to simulate acquisitions
# 
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date May 2016

## for IPython: reload all changed modules every time before executing
## However, added in file ~/.ipython/profile_default/ipython_config.py
# %load_ext autoreload
# %autoreload 2

## Import libraries 
import pdb # set "pdb.set_trace()" to break into the debugger from a running program
import itk
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../src/")

## Import modules from src-folder
import preprocessing.DataPreprocessing as dp
import utilities.ReconstructionManager as rm
import utilities.StackManager as sm
import utilities.SimpleITKHelper as sitkh
import base.Stack as st
import base.Slice as sl
import reconstruction.InverseProblemSolver as ips

import simulation.SimulatorSliceAcqusition as sa
import validation.ValidationReconstruction as vrec
import validation.ValidationRegistration as vreg


## Rigid registration routine based on SimpleITK
#  \param fixed
#  \param moving
#  \param display_registration_info display registration summary at the end of execution (default=0)
#  \return Rigid registration as sitk.Euler3DTransform object
def get_rigid_registration_transform_sitk(fixed, moving, display_registration_info=0):

    # moving_3D_sitk = sitkh.convert_itk_to_sitk_image(moving_3D_itk)
    moving_3D_sitk = moving.sitk

    ## Instantiate interface method to the modular ITKv4 registration framework
    registration_method = sitk.ImageRegistrationMethod()

    ## Select between using the geometrical center (GEOMETRY) of the images or using the center of mass (MOMENTS) given by the image intensities
    # initial_transform = sitk.CenteredTransformInitializer(fixed.sitk, moving.sitk, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
    initial_transform = sitk.Euler3DTransform()

    ## Set the initial transform and parameters to optimize
    registration_method.SetInitialTransform(initial_transform)

    ## Set an image masks in order to restrict the sampled points for the metric
    # registration_method.SetMetricFixedMask(fixed.sitk_mask)

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
    registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=1, minStep=0.05, numberOfIterations=100)

    ## Estimating scales of transform parameters a step sizes, from the maximum voxel shift in physical space caused by a parameter change
    ## (Many more possibilities to estimate scales)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    # registration_method.SetOptimizerScalesFromJacobian()
    
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
    final_transform_3D_sitk = registration_method.Execute(fixed.sitk, moving_3D_sitk) 

    if display_registration_info:
        print("SimpleITK Image Registration Method:")
        print('  Final metric value: {0}'.format(registration_method.GetMetricValue()))
        print('  Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))

    return sitk.Euler3DTransform(final_transform_3D_sitk)



""" ###########################################################################
Main Function
"""
if __name__ == '__main__':
    
    np.set_printoptions(precision=3)

    # dir_input_HR_volume = "../data/test/"
    dir_input_HR_volume = "/Users/mebner/UCL/Data/UZL/invivo_inutero_postmortem/exutero_postmortem/nifti/"
    dir_output = "results/"
    dir_input = "data/"

    # filename_HR_volume = "recon_fetal_neck_mass_brain_cycles0_SRR_TK0_itermax20_alpha0.1"
    # filename_HR_volume = "recon_fetal_neck_mass_brain_cycles1_SRR_TK0_itermax20_alpha0.1"
    # filename_HR_volume = "recon_fetal_neck_mass_brain_cycles1_SRR_TK1_itermax20_alpha0.1"
    filename_HR_volume = "s801a1008_T2W_NEW_VISTA_SHC"


    # HR_volume_ref.show(1)

    """
    Simulation of Stacks
    """
    flag_simulate_stacks = 1

    if flag_simulate_stacks:
        ## Read stack
        HR_volume_ref = st.Stack.from_filename(dir_input_HR_volume, filename_HR_volume, suffix_mask="_mask_brain")

        ## Data preprocessing, i.e. crop to masked region
        data_preprocessing = dp.DataPreprocessing.from_stacks([HR_volume_ref])
        data_preprocessing.run_preprocessing(boundary=10)
        HR_volume_ref = data_preprocessing.get_preprocessed_stacks()[0]

        ## Define in- and through-plane spacing for slice acquisition simulator
        spacing_in_plane = 1.1
        spacing_trough_plance = 4

        ## Resample to 1mm isotropic grid
        # HR_volume_ref = HR_volume_ref.get_isotropically_resampled_stack(spacing_in_plane)

        ## Initialize simulator
        slice_acquistion = sa.SliceAcqusition(HR_volume_ref)
        # slice_acquistion.set_interpolator_type("NearestNeighbor")
        slice_acquistion.set_interpolator_type("OrientedGaussian")
        slice_acquistion.set_spacing((spacing_in_plane, spacing_in_plane, spacing_trough_plance))
        # slice_acquistion.set_motion_type("Random")

        ## Generate stacks of slices in the respective (three) orthogonal views
        slice_acquistion.run_simulation_view_1()
        # slice_acquistion.run_simulation_view_2()
        # slice_acquistion.run_simulation_view_3()
        # slice_acquistion.run_simulation_view_1()
        # slice_acquistion.run_simulation_view_2()
        # slice_acquistion.run_simulation_view_3()

        ## Get simulated stacks and their ground truth affine transforms +
        #  rigid motion transforms for the associated slices
        stacks_simulated = slice_acquistion.get_simulated_stacks()
        ground_truth_data = slice_acquistion.get_ground_truth_data()

        ## Show simulated stack and overlaid resampled initial HR volume 
        stacks_simulated[0].show(1)
        # HR_volume_ref.show()

        # HR_volume_resampled_sitk = sitk.Resample(
        #         HR_volume_ref.sitk, stacks_simulated[0].sitk, sitk.Euler3DTransform(), sitk.sitkNearestNeighbor, 0.0, stacks_simulated[0].sitk.GetPixelIDValue()
        #     )
        # sitkh.show_sitk_image(stacks_simulated[0].sitk, overlay=HR_volume_resampled_sitk, title="simulated_resampledHRvolume")

        ## Create stack manager and write all stacks to input directory
        # stack_manager = sm.StackManager.from_stacks(stacks_simulated)
        # stack_manager.write_stacks(directory=dir_input)

    ##
    """
    Registration Algorithm
    """
    flag_evaluate_registration = 0

    if flag_evaluate_registration:
        # stacks_simulated[0].get_isotropically_resampled_stack_from_slices().show()
        validation_registration = vreg.ValidationRegistration(HR_volume_ref, stacks_simulated, ground_truth_data)

        validation_registration.run_slice_to_volume_registration(iterations=5, display_info=1, save_figure=0)
        stacks = validation_registration.get_stacks()
        # stacks[0].get_isotropically_resampled_stack_from_slices().show()



    """
    Direct Reconstruction
    """
    flag_direct_recon = 0

    if flag_direct_recon:
        stack = stacks_simulated[0]
        stack_resampled_sitk = stack.get_isotropically_resampled_stack(interpolator="Linear").sitk
        HR_volume = st.Stack.from_sitk_image(stack_resampled_sitk, name="HR_volume")

        ## SRR parameters
        alpha_cut = 5
        SRR_approach = "TK1"
        DTD_computation_type = "FiniteDifference"
        tolerance = 1e-3
        iter_max = 30
        alpha = 0.01     #0.05 yields visually good results
        rho = None
        ADMM_iterations = None
        ADMM_iterations_output_dir = None
        ADMM_iterations_output_filename_prefix = None

        ## Initialize and parametrize SRR class
        SRR = ips.InverseProblemSolver([stack], HR_volume)
        SRR.set_alpha_cut(alpha_cut)
        SRR.set_regularization_type(SRR_approach)
        SRR.set_DTD_computation_type(DTD_computation_type)
        SRR.set_tolerance(tolerance)
        SRR.set_iter_max(iter_max)
        SRR.set_alpha(alpha)
        SRR.set_rho(rho)
        SRR.set_ADMM_iterations(ADMM_iterations)
        SRR.set_ADMM_iterations_output_dir(ADMM_iterations_output_dir)
        SRR.set_ADMM_iterations_output_filename_prefix(ADMM_iterations_output_filename_prefix)

        ## Run reconstruction
        SRR.run_reconstruction()
        # HR_volume.show()

        ## Compare results
        HR_volume_ref_resampled_sitk = sitk.Resample( HR_volume_ref.sitk, HR_volume.sitk, sitk.Euler3DTransform(), sitk.sitkLinear, 0.0, HR_volume.sitk.GetPixelIDValue())

        sitkh.show_sitk_image(HR_volume.sitk, overlay=stack.get_isotropically_resampled_stack(interpolator="Linear").sitk, title="HRrecon_StackResampled")
        sitkh.show_sitk_image(HR_volume.sitk, overlay=HR_volume_ref_resampled_sitk, title="HRrecon_HRref")

    
    """
    Reconstruction Algorithm
    """
    # sys.exit()
    flag_run_recon = 0

    if flag_run_recon:

        ## Prepare output directory
        reconstruction_manager = rm.ReconstructionManager(dir_output, target_stack_number=0, recon_name="brain")

        ## Read input stack data (including data preprocessing)
        reconstruction_manager.read_input_stacks_from_stacks(stacks_simulated[0:5])
        # reconstruction_manager.read_input_stacks_from_filenames("data/", [str(i) for i in range(0, 6)], suffix_mask="_mask")

        ## Compute first estimate of HR volume
        reconstruction_manager.set_on_registration_of_stacks_before_estimating_initial_volume()
        reconstruction_manager.compute_first_estimate_of_HR_volume_from_stacks(display_info=0)

        ## Run two step reconstruction alignment approach
        reconstruction_manager.run_two_step_reconstruction_alignment_approach(iterations=3, display_info=0)

        ## Get reconstruction
        HR_volume_recon = reconstruction_manager.get_HR_volume()

        ## Get stacks
        stacks = reconstruction_manager.get_stacks()

        ## Get rigid motion transforms
        rigid_motion_transforms = reconstruction_manager.get_slice_registration_history_of_stacks()[1]
        
        # HR_volume_recon.show()

        ## Compare reconstruction with "ground truth"
        # HR_volume_recon_comp = st.Stack.from_filename(dir_output+"SRR/","recon_brain_stacks3_cycles5_SRRTK0_itermax30_alpha0.1")
        # HR_volume_recon_comp = st.Stack.from_filename(dir_output+"SRR/","recon_brain_stacks6_cycles5_SRRTK0_itermax30_alpha0.1")
        
        recon = st.Stack.from_stack(HR_volume_recon)

        transform = get_rigid_registration_transform_sitk(HR_volume_ref, recon, 1)

        recon_sitk = sitk.Resample(
                recon.sitk, HR_volume_ref.sitk, transform, sitk.sitkLinear, 0.0, HR_volume_ref.sitk.GetPixelIDValue()
            )
        sitkh.show_sitk_image(recon_sitk, overlay=HR_volume_ref.sitk, title="Comparison_recon_original")

    
    """
    Validation of Reconstruction
    """
    flag_evaluate_reconstruction = 0

    if flag_evaluate_reconstruction:

        validation_reconstruction = vrec.ValidationReconstruction(HR_volume_ref)

        validation_reconstruction.reconstruct_volume_based_on_ground_truth_transforms(stacks_simulated, ground_truth_data)

        # validation_reconstruction.compute_target_registration_error_of_estimated_slice_positions_intensities(HR_volume_recon, stacks, rigid_motion_transforms)
        # validation_reconstruction.compute_target_registration_error_of_estimated_slice_positions_intensities(HR_volume_ref, stacks_simulated, ground_truth_data)

        # validation_reconstruction.reconstruct_volume_based_on_ground_truth_transforms(stacks_simulated, affine_transforms, show_comparison=True)

        # validation_reconstruction.compute_error_rigid_registration_parameters(HR_volume_recon, rigid_motion_transforms, ground_truth_data)

    plt.show()
