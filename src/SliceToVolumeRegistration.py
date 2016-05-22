## \file SliceToVolumeRegistration.py
#  \brief Perform slice to volume registration 
# 
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date December 2015


## Import libraries
import os                       # used to execute terminal commands in python
import sys
import itk
import SimpleITK as sitk
import numpy as np


## Import modules from src-folder
import SimpleITKHelper as sitkh
import PSF as psf
import ScatteredDataApproximation as sda
# import Stack as st
# import Slice as sl


## Pixel type of used 3D ITK image
pixel_type = itk.D

## ITK image type 
image_type = itk.Image[pixel_type, 3]

## Class implementing the slice to volume registration algorithm
#  \warning HACK: Upsample slices in k-direction to in-plane resolution
class SliceToVolumeRegistration:

    ## Idea is to use neighbouring slices to 
    #  -# initialize registration for current slice with "meaningful" parameters
    #  -# check resulting trafo and damp its effect in case it is notice as "outlier"
    class StackRegistrationStabilizer:

        # \param[in] N_slices
        # \param[in] interleave
        def __init__(self, N_slices, interleave=1):
            self._N_slices = N_slices
            self._rigid_transforms = [sitk.Euler3DTransform]*N_slices
            
            self._rigid_transform_parameters = np.zeros((N_slices,6))
            self._interleave = interleave

            ## Number of neighbouring slices (i.e. neighbours based on next/previous
            #  interleaved slice acquisition) left and right the current one
            #  considered to estimate initial values for the 6 DOF of 
            #  rigid parameter transform.
            neighbourhood_slices = 1

            self._neighbourhood = neighbourhood_slices*self._interleave


        ## Compute initial transform based on the rigid transform parameters
        #  of a local neighbourhood
        #  \param[in] fixed slice as Slice object
        #  \param[in] moving image as Stack object (HR volume estimate)
        #  \return initial transform as sitk.Euler3DTransform object
        def get_initial_transform_rigid_parameters(self, fixed, moving):

            initial_transform = sitk.Euler3DTransform()

            ## Use arithmetic mean of fixed voxels as initial transform center
            #  Remark: Center is taken into consideration for composited transforms,
            #  see SimpleITKHelper
            center = sitk.CenteredTransformInitializer(fixed.sitk, moving.sitk, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY).GetFixedParameters()
            initial_transform.SetCenter(center)

            ## Compute indices of neighbouring slices to be considered by taking
            #  into account interleaved acquisition
            j_slice = fixed.get_slice_number()

            j_min = np.max((j_slice-self._neighbourhood, 0))
            j_max = np.min((j_slice+self._neighbourhood, self._N_slices-1))
            
            neighbourhood_range = list(set(np.concatenate((np.arange(j_slice, j_max+1, self._interleave), np.arange(j_slice, j_min-1, -self._interleave)))) - set([j_slice]))

            # print("j_slice = " + str(j_slice))
            # print("j_min = " + str(j_min))
            # print("j_max = " + str(j_max))
            # print("neighbourhood_range = " + str(neighbourhood_range))

            ## Get average parameters of local neighbourhood
            parameters = np.mean( self._rigid_transform_parameters[neighbourhood_range,:], 0 )

            ## Update initial transform
            initial_transform.SetParameters( parameters )

            return initial_transform


        ## Check suggested transform obtained via \p _get_rigid_registration_transform
        #  and alleviate its parameters in case it is detected as outlier
        #  based on slice transforms obtained for chosen neighbourhood
        #  \param[in] fixed slice as Slice object
        #  \param[in] rigid_transform rigid transform as sitk.Euler3DTransform
        #  \return sitk.EulerTransform
        def check_and_return_rigid_transform(self, fixed, rigid_transform_sitk):
            
            ## Get parameters of suggested rigid transform
            parameters = np.array(rigid_transform_sitk.GetParameters())
            
            ## Compute indices of neighbouring slices to be considered by taking
            #  into account interleaved acquisition
            j_slice = fixed.get_slice_number()

            j_min = np.max((j_slice-self._neighbourhood, 0))
            j_max = np.min((j_slice+self._neighbourhood, self._N_slices-1))
            
            neighbourhood_range = list(set(np.concatenate((np.arange(j_slice, j_max+1, self._interleave), np.arange(j_slice, j_min-1, -self._interleave)))) - set([j_slice]))

            # range_right = np.minimum(j_slice + np.arange(interleave,neighbourhood,interleave), self._N_slices-1)
            # range_left = np.maximum(j_slice - np.arange(interleave,neighbourhood,interleave),0)
            # print("j_slice = " + str(j_slice))
            # neighbourhood_range = list(set(np.concatenate((range_left, range_right))))
            # print ("neighbourhood_range" + str(neighbourhood_range))

            ## Compute mean and standard deviation of parameters in neighbourhood
            rigid_params_mean = np.mean( self._rigid_transform_parameters[neighbourhood_range,:], 0 )
            rigid_params_std = np.std( self._rigid_transform_parameters[neighbourhood_range,:], 0 )

            ## In case stds are too small (at the beginning of the algorithm they are zero e.g.)
            std_angles_min = 0.1*np.ones(3)
            std_translation_min = 1*np.ones(3)
            rigid_params_std = np.maximum( rigid_params_std, np.concatenate((std_angles_min, std_translation_min)) )

            # print("rigid_params_mean = " + str(rigid_params_mean))
            # print("rigid_params_std = " + str(rigid_params_std))
            # print("parameters = " + str(parameters))

            ## Alleviate effect of transform in case suggested transform is 
            #  not within range (mean +- threshold*sigma)
            threshold = 3

            if ( parameters > rigid_params_mean + threshold*rigid_params_std ).sum() >0 \
                or ( parameters < rigid_params_mean - threshold*rigid_params_std ).sum() >0 :
                print ("Slice %s: Update trafo!" %(j_slice))
                sitkh.print_rigid_transformation(rigid_transform_sitk)

                ## Alleviate effect of detected "outlier" transform
                parameters /= 6;

                rigid_transform_sitk.SetParameters(parameters)
                sitkh.print_rigid_transformation(rigid_transform_sitk)
            
            ## Update database of obtained transforms    
            self._update_transform(j_slice, rigid_transform_sitk)

            return rigid_transform_sitk

        
        ## Update database of obtained transforms    
        #  \param[in] j_slice slice number within stack
        #  \param[in] rigid_transform_sitk transform as sitk.Euler3DTransform object
        def _update_transform(self, j_slice, rigid_transform_sitk):
            self._rigid_transforms[j_slice] = rigid_transform_sitk
            self._rigid_transform_parameters[j_slice,:] = rigid_transform_sitk.GetParameters()


    ## Constructor
    #  \param[in,out] stack_manager instance of StackManager containing all stacks and additional information
    def __init__(self, stack_manager, HR_volume):

        ## Initialize variables
        self._stack_manager = stack_manager
        self._stacks = stack_manager.get_stacks()
        self._N_stacks = stack_manager.get_number_of_stacks()
        self._HR_volume = HR_volume

        ## Used for PSF modelling and smoothign w.r.t. relative alignment between
        #  one single slice and HR volume grid
        self._psf = psf.PSF()
        self._gaussian_yvv = itk.SmoothingRecursiveYvvGaussianImageFilter[image_type, image_type].New()   # YVV-based Filter

        ## Define dictionary to choose registration approach
        self._get_rigid_registration_transform = {
            "SimpleITK" :   self._get_rigid_registration_transform_sitk,
            "ITK"       :   self._get_rigid_registration_transform_itk,
            "NiftyReg"  :   self._get_rigid_registration_transform_NiftyReg
        }

        self._registration_approach = "SimpleITK"    # default registration approach
        # self._registration_approach = "ITK"    # default registration approach
        # self._registration_approach = "NiftyReg"    # default registration approach


    ## Perform slice-to-volume registration of all slices to current estimate of HR volume reconstruction
    #  \param[in] display_info display information of registration results as we go along
    #  \param[in] interleave step of interleaved stack acquisition used for hierarchical alignment
    #       Used for "strategy" in StackRegistrationStabilizer
    def run_slice_to_volume_registration(self, interleave=2, display_info=0):
        print("\t--- Slice-to-Volume Registration ---")

        self._gaussian_yvv.SetInput(sitkh.convert_sitk_to_itk_image(self._HR_volume.sitk))

        for i in range(0, self._N_stacks):
            print("\tStack %s/%s" %(i,self._N_stacks-1))
            stack = self._stacks[i]
            slices = stack.get_slices()
            N_slices = stack.get_number_of_slices()
            
            self._stack_registration_stabilizer = self.StackRegistrationStabilizer(N_slices, interleave)

            ## Consider groups of acquired slices and start from non-border slice
            # for j in range(0, N_slices):
            for j_slice_groups in range(interleave-1,-1,-1):
                slice_group = np.arange(j_slice_groups, N_slices, interleave)

                for j in slice_group:
                    if display_info:
                        print("\t\tSVR of slice %s/%s:" %(j,N_slices-1))
                    
                    slice = slices[j]

                    rigid_transform = self._get_rigid_registration_transform[self._registration_approach](fixed_slice_3D=slice, moving_HR_volume_3D=self._HR_volume, display_registration_info=display_info)


                ## Update rigid motion estimate for current slice and update its 
                #  position in physical space accordingly
                slice.update_rigid_motion_estimate(rigid_transform)


    ## Rigid registration routine based on SimpleITK
    #  \param fixed_slice_3D upsampled fixed Slice
    #  \param moving_HR_volume_3D moving Stack
    #  \param display_registration_info display registration summary at the end of execution (default=0)
    #  \return Rigid registration as sitk.Euler3DTransform object
    def _get_rigid_registration_transform_sitk(self, fixed_slice_3D, moving_HR_volume_3D, display_registration_info=0):

        ## Blur 
        Cov_HR_coord = self._psf.get_gaussian_PSF_covariance_matrix_HR_volume_coordinates( fixed_slice_3D, moving_HR_volume_3D )

        self._gaussian_yvv.SetSigmaArray(np.sqrt(np.diagonal(Cov_HR_coord)))
        self._gaussian_yvv.Update()
        moving_3D_itk = self._gaussian_yvv.GetOutput()
        moving_3D_itk.DisconnectPipeline()

        moving_3D_sitk = sitkh.convert_itk_to_sitk_image(moving_3D_itk)
        # moving_3D_sitk = moving_HR_volume_3D.sitk

        ## Instantiate interface method to the modular ITKv4 registration framework
        registration_method = sitk.ImageRegistrationMethod()

        initial_transform = self._stack_registration_stabilizer.get_initial_transform_rigid_parameters(fixed_slice_3D, moving_HR_volume_3D)
        # initial_transform = sitk.Euler3DTransform()

        # ## Set the initial transform and parameters to optimize
        registration_method.SetInitialTransform(initial_transform)

        ## Set an image masks in order to restrict the sampled points for the metric
        registration_method.SetMetricFixedMask(fixed_slice_3D.sitk_mask)

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
        # registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=100)

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
        # registration_method.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, maximumNumberOfIterations=500, maximumNumberOfCorrections=5, maximumNumberOfFunctionEvaluations=200, costFunctionConvergenceFactor=1e+7)

        ## Regular Step Gradient descent optimizer
        registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=1, minStep=1e-6, numberOfIterations=500, gradientMagnitudeTolerance=1e-4)

        ## Estimating scales of transform parameters a step sizes, from the maximum voxel shift in physical space caused by a parameter change
        ## (Many more possibilities to estimate scales)
        # registration_method.SetOptimizerScalesFromIndexShift()
        # registration_method.SetOptimizerScalesFromPhysicalShift()
        registration_method.SetOptimizerScalesFromJacobian()
        
        """
        setup for the multi-resolution framework            
        """
        ## Set the shrink factors for each level where each level has the same shrink factor for each dimension
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4,2,1])

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
        final_transform_3D_sitk = sitk.Euler3DTransform(registration_method.Execute(fixed_slice_3D.sitk, moving_3D_sitk))

        final_transform_3D_sitk = self._stack_registration_stabilizer.check_and_return_rigid_transform(fixed_slice_3D, final_transform_3D_sitk)

        if display_registration_info:
            print("\t\tSimpleITK Image Registration Method:")
            print('\t\t\tFinal metric value: {0}'.format(registration_method.GetMetricValue()))
            print('\t\t\tOptimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))

            sitkh.print_rigid_transformation(final_transform_3D_sitk)


        return final_transform_3D_sitk


    ## Rigid registration routine based on ITK
    #  \param fixed_slice_3D upsampled fixed Slice
    #  \param moving_HR_volume_3D moving Stack
    #  \param display_registration_info display registration summary at the end of execution (default=0)
    #  \return Rigid registration as sitk.Euler3DTransform object
    def _get_rigid_registration_transform_itk(self, fixed_slice_3D, moving_HR_volume_3D, display_registration_info=0):
        ## Look at http://www.itk.org/Doxygen/html/Examples_2RegistrationITKv3_2ImageRegistration8_8cxx-example.html#_a10

        registration = itk.ImageRegistrationMethod[image_type, image_type].New()

        ## Create Spatial Objects for masks so that they can be used within metric
        caster = itk.CastImageFilter[itk.Image[itk.D,3],itk.Image[itk.UC,3]].New()
        caster.SetInput(fixed_slice_3D.itk_mask)
        caster.Update()

        fixed_mask_object = itk.ImageMaskSpatialObject[3].New()
        fixed_mask_object.SetImage(caster.GetOutput())

        ## Initial transform: Variant A
        # transform_type = itk.VersorRigid3DTransform.D
        # initial_transform = transform_type.New()

        # initializer = itk.CenteredTransformInitializer[transform_type, image_type, image_type].New()
        # initializer.SetTransform(initial_transform)
        # initializer.SetFixedImage(fixed_slice_3D.itk)
        # initializer.SetMovingImage(moving_HR_volume_3D.itk)
        # initializer.MomentsOn()
        # initializer.InitializeTransform()

        ## Initial transform: Variant B
        initial_transform = itk.Euler3DTransform.New()

        # interpolator = itk.LinearInterpolateImageFunction[image_type, pixel_type].New()
        interpolator = itk.OrientedGaussianInterpolateImageFunction[image_type, pixel_type].New()
        Cov = self._psf.get_gaussian_PSF_covariance_matrix_HR_volume_coordinates( fixed_slice_3D, moving_HR_volume_3D )
        interpolator.SetCovariance(Cov.flatten())

        
        # metric = itk.MeanSquaresImageToImageMetric[image_type, image_type].New()
        metric = itk.NormalizedCorrelationImageToImageMetric[image_type, image_type].New()
        
        # metric = itk.MutualInformationImageToImageMetric[image_type, image_type].New()
        
        # metric = itk.MattesMutualInformationImageToImageMetric[image_type, image_type].New()
        # metric.SetNumberOfHistogramBins(200)

        # metric = itk.NormalizedMutualInformationHistogramImageToImageMetric[image_type, image_type].New()
        # scales = np.ones(initial_transform.GetParameters().GetNumberOfElements())
        # metric.SetHistogramSize(100*np.ones(3))
        # metric.SetDerivativeStepLengthScales(scales)

        ## Add masks
        metric.SetFixedImageMask(fixed_mask_object)

        # optimizer = itk.ConjugateGradientOptimizer.New()
        optimizer = itk.RegularStepGradientDescentOptimizer.New()
        optimizer.SetMaximumStepLength(1.00)
        optimizer.SetMinimumStepLength(0.01)
        optimizer.SetNumberOfIterations(200)

        registration.SetInitialTransformParameters(initial_transform.GetParameters())
        registration.SetFixedImageRegion(fixed_slice_3D.itk.GetBufferedRegion())
        registration.SetOptimizer(optimizer)
        registration.SetTransform(initial_transform)
        # registration.SetTransform(transform_type.New())
        registration.SetInterpolator(interpolator)
        
        registration.SetMetric(metric)

        registration.SetMovingImage(moving_HR_volume_3D.itk)
        registration.SetFixedImage(fixed_slice_3D.itk)

        ## Execute registration
        registration.Update()

        ## Get registration transform
        rigid_registration_3D_itk = registration.GetOutput().Get()

        # final_parameters = registration.GetLastTransformParameters()
        # rigid_registration_3D = itk.Euler3DTransform.New()
        # rigid_registration_3D.SetParameters(final_parameters)


        return sitkh.get_sitk_Euler3DTransform_from_itk_Euler3DTransform(rigid_registration_3D_itk)


    def _get_rigid_registration_transform_NiftyReg(self, fixed_slice_3D, moving_HR_volume_3D, display_registration_info=0):
        ## Save images prior to the use of NiftyReg
        dir_tmp = "../results/tmp/" 
        os.system("mkdir -p " + dir_tmp)

        j = fixed_slice_3D.get_slice_number()

        moving_str = str(j) + "_moving" 
        fixed_str = str(j) + "_fixed"
        moving_mask_str = str(j) +"_moving_mask"
        fixed_mask_str = str(j) + "_fixed_mask"

        # sitk.WriteImage(fixed_slice_3D._sitk_upsampled, dir_tmp+fixed_str+".nii.gz")
        sitk.WriteImage(fixed_slice_3D.sitk, dir_tmp+fixed_str+".nii.gz")
        sitk.WriteImage(fixed_slice_3D.sitk_mask, dir_tmp+fixed_mask_str+".nii.gz")
        sitk.WriteImage(moving_HR_volume_3D.sitk, dir_tmp+moving_str+".nii.gz")
        sitk.WriteImage(moving_HR_volume_3D.sitk_mask, dir_tmp+moving_mask_str+".nii.gz")

        ## NiftyReg: Global affine registration:
        #  \param[in] -ref reference image
        #  \param[in] -flo floating image
        #  \param[out] -res affine registration of floating image
        #  \param[out] -aff affine transformation matrix
        res_affine_image = moving_str + "_warped_NiftyReg"
        res_affine_matrix = str(j) + "_affine_matrix_NiftyReg"

        # options = "-voff -rigOnly "
        options = "-rigOnly "
        # options = "-voff -platf 1 "
            # "-rmask " + dir_tmp + fixed_mask_str + ".nii.gz " + \
            # "-fmask " + dir_tmp + moving_mask_str + ".nii.gz " + \
        cmd = "reg_aladin " + options + \
            "-ref " + dir_tmp + fixed_str + ".nii.gz " + \
            "-flo " + dir_tmp + moving_str + ".nii.gz " + \
            "-res " + dir_tmp + res_affine_image + ".nii.gz " + \
            "-aff " + dir_tmp + res_affine_matrix + ".txt "
        print(cmd)
        sys.stdout.write("  Rigid registration (NiftyReg reg_aladin) " + str(j) + " ... ")

        sys.stdout.flush() #flush output; otherwise sys.stdout.write would wait until next newline before printing
        os.system(cmd)
        print "done"

        ## Read trafo and invert such that format fits within SimpleITK structure
        matrix = np.loadtxt(dir_tmp+res_affine_matrix+".txt")
        A = matrix[0:-1,0:-1]
        t = matrix[0:-1,-1]

        ## Convert to SimpleITK physical coordinate system
        ## TODO: Unit tests according to SimpleITK_NiftyReg_FLIRT.py
        R = np.array([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]])

        A = R.dot(A).dot(R)
        t = R.dot(t)

        return sitk.AffineTransform(A.flatten(),t)
