## \file ValidationReconstruction.py
#  \brief 
# 
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date May 2016


## Import libraries
import itk
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append("../src/")

## Import modules from src-folder
import pythonhelper.SimpleITKHelper as sitkh
import reconstruction.InverseProblemSolver as ips
import base.Stack as st
import base.PSF as psf


## Pixel type of used 3D ITK image
pixel_type = itk.D

## ITK image type 
image_type = itk.Image[pixel_type, 3]

## Class used to evaluate reconstruction results
class ValidationReconstruction:

    ## Initialize class for scenario of simulation
    # \param[in] HR_volume_ref HR volume considered as ground truth as Stack object
    def __init__(self, HR_volume_ref):

        self._HR_volume_ref = HR_volume_ref
        self._HR_volume_from_simulated_stacks = None

        self._HR_volume_estimated = None
        self._HR_volume_from_estimated_stacks = None


    ## Reconstruct HR volume which can be considered as "optimum" given the
    #  available stacks and used SRR approach
    #  \param[in] stacks_simulated Simulated stacks as list of Stack objects, 
    #       obtained via SliceAcquisition e.g.
    #  \param[in] rigid_motion_transforms_gd ground truth rigid
    #       motion transforms as list of sitk.Euler3DTransforms
    #  \param[in] show_comparison display comparison between HR_volume_ref and
    #       obtained "optimal" HR volume possible
    def reconstruct_volume_based_on_ground_truth_transforms(self, stacks_simulated, rigid_motion_transforms_gd, show_comparison=True):
        if self._HR_volume_from_simulated_stacks is not None:
            raise ValueError("Error: Reconstruction of volume based on ground truth transforms not possible")

        ## Get stacks with ground truth position of slices
        stacks_gd = self._get_simulated_stacks_with_ground_truth_data(stacks_simulated, rigid_motion_transforms_gd)

        ## Get initial volume for SRR algorithm
        gaussian = itk.SmoothingRecursiveYvvGaussianImageFilter[image_type, image_type].New()
        gaussian.SetSigma(1)
        gaussian.SetInput(self._HR_volume_ref.itk)
        gaussian.Update()

        HR_init_itk = gaussian.GetOutput()
        HR_init_itk.DisconnectPipeline()
        HR_init = st.Stack.from_sitk_image(sitkh.get_sitk_from_itk_image(HR_init_itk))
        sitkh.show_sitk_image(HR_init.sitk, overlay=self._HR_volume_ref.sitk, title="altered_HR_as_HR_init")

        ## Configure SRR reconstruction
        SRR = ips.InverseProblemSolver(stacks_gd, HR_init)

        SRR.set_regularization_type('TK0')
        # SRR.set_DTD_computation_type('FiniteDifference')
        SRR.set_alpha(0.1)
        SRR.set_iter_max(30)

        ## Compute "optimal" HR volume estimate possible
        SRR.run_reconstruction()

        self._HR_volume_from_simulated_stacks = SRR.get_HR_volume()

        if show_comparison:
            sitkh.show_sitk_image(self._HR_volume_ref.sitk, overlay=self._HR_volume_from_simulated_stacks.sitk, title="HRref_HRpossible")
            sitkh.show_sitk_image(self._HR_volume_ref.sitk-self._HR_volume_from_simulated_stacks.sitk, title="HRref-HRpossible")


    ## Get list of Stack objects containing slices with ground truth positions
    #  within HR volume space
    #  \param[in] stacks_simulated Simulated stacks as list of Stack objects, 
    #       obtained via SliceAcquisition e.g.
    #  \param[in] rigid_motion_transforms_gd ground truth rigid
    #       motion transforms as list of sitk.Euler3DTransforms
    #  \return list of Stack objects containing ground truth positioned slices
    def _get_simulated_stacks_with_ground_truth_data(self, stacks_simulated, rigid_motion_transforms_gd):

        N_stacks = len(stacks_simulated)
        stacks_gd = [None]*N_stacks

        for i in range(0, N_stacks):
            stack_gd = st.Stack.from_stack(stacks_simulated[i])

            slices = stack_gd.get_slices()

            N_slices = stack_gd.get_number_of_slices()

            ## Set physical position of each slice to ground truth
            for j in range(0, N_slices):
                slices[j].update_rigid_motion_estimate(rigid_motion_transforms_gd[i][j])

            stacks_gd[i] = stack_gd

        return stacks_gd


    ## Compute squared error of intensities based on the obtained estimated
    #  positions of the slices
    #  \param[in] HR_volume_est Reconstructed volume based on stacks
    #       (required to find correct alignment with reference volume)
    #  \param[in] stacks Stack objects resulting from reconstruction algorithm
    #  \param[in] rigid_motion_transforms associated to Stack objects containing the sitk.Euler3DTransform objects
    def compute_target_registration_error_of_estimated_slice_positions_intensities(self, HR_volume_est, stacks, rigid_motion_transforms):
        
        ## Compute transform to align estimated volume with reference volume
        offset_transform = self._get_rigid_registration_transform(HR_volume_est, self._HR_volume_ref)

        HR_ref_aligned_sitk = sitk.Resample(self._HR_volume_ref.sitk, HR_volume_est.sitk, offset_transform, sitk.sitkLinear, 0.0, self._HR_volume_ref.sitk.GetPixelIDValue())
        sitkh.show_sitk_image(HR_ref_aligned_sitk, overlay=HR_volume_est.sitk, title="HRref_HRrecon")

        N_stacks = len(stacks)

        try:
            N_cycles = len(rigid_motion_transforms[0][0])
        except:
            N_cycles = 1

        N_slices_total = 0

        ## Count total amount of slices
        for i in range(0, N_stacks):
            N_slices_total += stacks[i].get_number_of_slices()
                
        ## Squared Error of slices
        error_squared = np.zeros((N_slices_total, N_cycles))

        ind_slice = 0

        resampler = itk.ResampleImageFilter[image_type, image_type].New()
        resampler.SetDefaultPixelValue( 0.0 )
        resampler.SetInput( self._HR_volume_ref.itk )

        interpolator = itk.OrientedGaussianInterpolateImageFunction[image_type, pixel_type].New()
        alpha_cut = 3
        interpolator.SetAlpha(alpha_cut)
        resampler.SetInterpolator(interpolator)

        PSF = psf.PSF()


        for i in range(0, N_stacks):
            slices = stacks[i].get_slices()

            for j in range(0, stacks[i].get_number_of_slices()):
                slice = slices[j]

                for k in range(0, N_cycles):
                    ## Get rigid motion transform of k-th cycle of slice j within stack i
                    try:
                        rigid_motion_transform = rigid_motion_transforms[i][j][k]
                    except:
                        rigid_motion_transform = rigid_motion_transforms[i][j]

                    ## Take into consideration initial alignment of volumes
                    rigid_motion_transform_corrected = sitkh.get_composite_sitk_affine_transform(rigid_motion_transform, offset_transform)
                    slice.update_rigid_motion_estimate(rigid_motion_transform_corrected)

                    ## Set covariance based on oblique PSF
                    Cov_HR_coord = PSF.get_gaussian_PSF_covariance_matrix_HR_volume_coordinates(slice, self._HR_volume_ref)
                    interpolator.SetCovariance(Cov_HR_coord.flatten())

                    ## Resample reference volume to slice space
                    resampler.SetOutputParametersFromImage( slice.itk )
                    resampler.UpdateLargestPossibleRegion()
                    resampler.Update()

                    HR_slice_itk = resampler.GetOutput()
                    HR_slice_itk.DisconnectPipeline()
                    HR_slice_sitk = sitkh.get_sitk_from_itk_image(HR_slice_itk)

                    slice_diff_sitk = (HR_slice_sitk - slice.sitk)*sitk.Cast(slice.sitk_mask, slice.sitk.GetPixelIDValue())
                    # sitkh.show_sitk_image(slice_diff_sitk)

                    ## Compute Squared Error
                    nda = sitk.GetArrayFromImage(slice_diff_sitk)
                    error_squared[ind_slice, k] = np.sum(nda**2)

                ind_slice += 1
        

        fig = plt.figure()
        plt.title("Evolution of squared intensity error")
        for i in range(0, N_cycles):
            plt.plot(i*np.ones(N_slices_total), error_squared[:,i], 'rx')
            plt.plot(i, np.mean(error_squared[:,i]), 'bo')
        plt.xlim(-0.5,N_cycles-0.5)
        # plt.yscale("log")
        plt.show()


    # ## Compute squared error of intensities based on the obtained estimated
    # #  positions of the slices
    # #  \param[in] HR_volume_est Reconstructed volume based on stacks
    # #       (required to find correct alignment with reference volume)
    # #  \param[in] stacks Stack objects resulting from reconstruction algorithm
    # #  \param[in] rigid_motion_transforms associated to Stack objects containing the sitk.Euler3DTransform objects
    # def compute_error_rigid_registration_parameters(self, HR_volume_est, rigid_motion_transforms, rigid_motion_transforms_gd):
        
    #     ## Compute transform to align estimated volume with reference volume
    #     offset_transform = self._get_rigid_registration_transform(HR_volume_est, self._HR_volume_ref)

    #     HR_ref_aligned_sitk = sitk.Resample(self._HR_volume_ref.sitk, HR_volume_est.sitk, offset_transform, sitk.sitkLinear, 0.0, self._HR_volume_ref.sitk.GetPixelIDValue())
    #     # sitkh.show_sitk_image(HR_ref_aligned_sitk, overlay=HR_volume_est.sitk, title="HRref_HRrecon")

    #     N_stacks = len(rigid_motion_transforms)

    #     N_cycles = len(rigid_motion_transforms[0][0])

    #     N_slices_total = 0
    #     N_begin_new_stack = np.zeros(N_stacks)

    #     ## Count total amount of slices
    #     for i in range(0, N_stacks):
    #         N_slices_total += len(rigid_motion_transforms[i])
    #         N_begin_new_stack[i] = N_slices_total - len(rigid_motion_transforms[i])
                
    #     print(N_begin_new_stack)
    #     ## 6 Parameters for rigid registration
    #     N_params = 6
    #     parameters_est = np.zeros((N_slices_total, N_params, N_cycles))
    #     parameters_gd = np.zeros((N_slices_total, N_params))

    #     ind_slice = 0

    #     ## Loop through stacks
    #     for i in range(0, N_stacks):

    #         ## Loop through the slices of current stack
    #         for j in range(0, len(rigid_motion_transforms[i])):

    #             for k in range(0, N_cycles):
    #                 ## Get rigid motion transform of k-th cycle of slice j within stack i
    #                 rigid_motion_transform = rigid_motion_transforms[i][j][k]

    #                 ## Take into consideration initial alignment of volumes
    #                 rigid_motion_transform_corrected = sitkh.get_composite_sitk_euler_transform(rigid_motion_transform, offset_transform)

    #                 ## Get (angle_x, angle_y, angle_x, translation_x, translation_y, translation_z)
    #                 parameters_est[ind_slice, :, k] = rigid_motion_transform_corrected.GetParameters()

    #             parameters_gd[ind_slice,:] = rigid_motion_transforms_gd[i][j].GetParameters()
    #             ind_slice += 1
        

    #     titles = ["angle_x", "angle_y", "angle_z", "t_x", "t_y", "t_z"]
        
    #     fig = plt.figure()
    #     fig.suptitle("Accuracy of registration")

    #     ## Plot result for each parameter
    #     for i_param in range(0, N_params):

    #         ax = fig.add_subplot(N_params,1,i_param+1)
    #         plt.ylabel(titles[i_param])
    #         # plt.title("Accuracy of " + titles[i_param])

    #         ## Plot each all cycles whereby initial and final value are marked distinctly
    #         ax.plot(np.arange(0,N_slices_total), parameters_est[:,i_param,0] - parameters_gd[:,i_param], 'go', mfc='none', label=str(0) )
    #         for i_cycle in range(1, N_cycles-1):
    #             ax.plot(np.arange(0,N_slices_total), parameters_est[:,i_param,i_cycle] - parameters_gd[:,i_param], 'rx', label=str(i_cycle) )
    #         ax.plot(np.arange(0,N_slices_total), parameters_est[:,i_param,-1] - parameters_gd[:,i_param], 'bo', label=str(i_cycle) )
            
    #         ## Mark first slice of every stack
    #         for i_stack in range(0, N_stacks):
    #             ax.plot(N_begin_new_stack[i_stack]*np.ones(2), ax.get_ylim())

    #         ## Increase xlimit for easier reading
    #         plt.xlim(-0.5,N_slices_total-0.5)

    #         ## Draw zero level to show zero-error line
    #         ax.plot(ax.get_xlim(),(0,0),'k')

    #         ## Show grid
    #         ax.grid()

    #     # ax.yscale("log")
    #     plt.xlabel("slice")
    #     plt.show()


    ## Rigid registration routine based on SimpleITK
    #  \param[in] fixed_3D as Stack object
    #  \param[in] moving_3D as Stack object
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
        # registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=200)

        ## Use negative means squares image metric
        registration_method.SetMetricAsMeanSquares()
        
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
        # registration_method.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, maximumNumberOfIterations=100, maximumNumberOfCorrections=5, maximumNumberOfFunctionEvaluations=1000, costFunctionConvergenceFactor=1e+7)

        ## Regular Step Gradient descent optimizer
        registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=1, minStep=0.05, numberOfIterations=100)

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
            print("\t\tSimpleITK Image Registration Method:")
            print('\t\t\tFinal metric value: {0}'.format(registration_method.GetMetricValue()))
            print('\t\t\tOptimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))

        return sitk.Euler3DTransform(final_transform_3D_sitk)
