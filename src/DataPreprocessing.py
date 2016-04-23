## \file DataPreprocessing.py
#  \brief Performs preprocessing steps
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
# import Stack as st
# import Slice as sl


## Class implementing data preprocessing steps
#  -# Crop stacks to region marked by mask
class DataPreprocessing:

    ## Constructor
    #  \param[in] dir_input directory where data is stored for preprocessing
    #  \param[in] dir_output directory in which preprocessed data gets written
    #  \param[in] target_stack_number in case only one mask is given (optional)
    def __init__(self, dir_input, dir_output, target_stack_number=0):

        self._dir_input = dir_input
        self._dir_output = dir_output
        self._target_stack_number = target_stack_number


    ## Perform data preprocessing step
    #  \param[in] filenames list of filenames referring to the data in dir_input to be processed
    #  \param[in] suffix_mask extension of stack filename which indicates associated mask
    def run_preprocessing(self, filenames, suffix_mask):
        
        ## Number of stacks to be read
        N_stacks = len(filenames)

        ## Number of masked stacks provided
        filenames_masks = self._get_mask_filenames_in_directory(suffix_mask)
        number_of_masks = len(filenames_masks)

        ## Each stack is provided a mask
        if number_of_masks is N_stacks:
            print("All masks provided. Stack and associated mask are cropped to their masked region.")
            self._run_preprocessing_all_masks_provided(filenames, suffix_mask)

        ## No stack is provided a mask. Hence, mask entire region of stack
        elif number_of_masks is 0:
            print("No mask is provided. Consider entire stack for reconstruction pipeline.")
            self._run_preprocessing_no_mask_provided(filenames)

        ## Not all stacks are provided a mask. Propagate target stack mask to other stacks
        else:
            print("Not all stacks are provided a mask. Mask of target stack is propagated to other masks and cropped.")
            self._run_preprocessing_not_all_masks_provided(filenames, filenames_masks, suffix_mask)


    ## Get filenames of stacks with provided masks in input directory
    #  \param[in] suffix_mask extension of stack filename which indicates associated mask
    #  \return filenames as list of strings
    def _get_mask_filenames_in_directory(self, suffix_mask):

        filenames = []

        ## List of all files in directory
        all_files = os.listdir(self._dir_input)

        ## Count number of files labelled as masks
        for file in all_files:

            if file.endswith(suffix_mask + ".nii.gz"):

                filename = file.replace(suffix_mask + ".nii.gz","")
                filenames.append(filename)

        return filenames


    """
    All masks provided
    """
    ## Perform data preprocessing step for case that all masks are provided
    #  \param[in] filenames list of filenames referring to the data in dir_input to be processed
    #  \param[in] suffix_mask extension of stack filename which indicates associated mask
    def _run_preprocessing_all_masks_provided(self, filenames, suffix_mask):

        ## Number of stacks to be read
        N_stacks = len(filenames)

        for i in range(0, N_stacks):
            ## Read stack and mask from directory
            stack_sitk = sitk.ReadImage(self._dir_input + filenames[i] + ".nii.gz", sitk.sitkFloat64)
            mask_sitk = sitk.ReadImage(self._dir_input + filenames[i] + suffix_mask + ".nii.gz", sitk.sitkUInt8)

            ## Crop stack and mask based on the mask provided
            [stack_sitk, mask_sitk] = self._crop_stack_and_mask(stack_sitk, mask_sitk, boundary=0)

            ## Write preprocessed data to output directory
            self._write_preprocessed_stack_and_mask(stack_sitk, mask_sitk, filenames[i])


    """
    No mask provided
    """
    ## Perform data preprocessing step for case that no mask is provided
    #  \param[in] filenames list of filenames referring to the data in dir_input to be processed
    def _run_preprocessing_no_mask_provided(self, filenames):
        
        ## Number of stacks to be read
        N_stacks = len(filenames)

        for i in range(0, N_stacks):
            ## Read stack from directory
            stack_sitk = sitk.ReadImage(self._dir_input + filenames[i] + ".nii.gz", sitk.sitkFloat64)

            ## Create binary mask consisting of ones
            shape = sitk.GetArrayFromImage(stack_sitk).shape
            nda = np.ones(shape, dtype=np.uint8)

            mask_sitk = sitk.GetImageFromArray(nda)
            mask_sitk.CopyInformation(stack_sitk) 
        
            ## Write preprocessed data to output directory
            self._write_preprocessed_stack_and_mask(stack_sitk, mask_sitk, filenames[i])


    """
    Not all masks provided
    """
    ## Perform data preprocessing step for case that some masks are missing.
    #  mask stored in self._target_stack_number is used as template mask for
    #  mask propagation
    #  \param[in] filenames list of filenames referring to the data in dir_input to be processed
    #  \param[in] filenames_masks list of filenames which come with a mask
    #  \param[in] suffix_mask extension of stack filename which indicates associated mask
    def _run_preprocessing_not_all_masks_provided(self, filenames, filenames_masks, suffix_mask):

        ## Find stacks which require a mask. Target stack mask is used
        #  as template mask for mask propagation.
        range_prop_mask, range_prop_mask_complement = self._get_filename_indices_for_mask_propagation(filenames, filenames_masks)

        ##*** Propagate masks

        ## Read target stack and mask and use as template
        template_sitk = sitk.ReadImage(self._dir_input + str(self._target_stack_number) + ".nii.gz", sitk.sitkFloat64)
        template_mask_sitk = sitk.ReadImage(self._dir_input + str(self._target_stack_number) + suffix_mask + ".nii.gz", sitk.sitkUInt8)

        for i in range_prop_mask:
            ## Read stack from directory
            stack_sitk = sitk.ReadImage(self._dir_input + filenames[i] + ".nii.gz", sitk.sitkFloat64)

            ## Propagate mask
            mask_sitk = self._get_propagated_mask(stack_sitk, template_sitk, template_mask_sitk)

            ## Dilate propagated mask (to smooth mask)
            # mask_sitk = self._dilate_mask(mask_sitk)

            ## Crop stack and mask based on the mask provided
            [stack_sitk, mask_sitk] = self._crop_stack_and_mask(stack_sitk, mask_sitk, boundary=0)

            ## Write preprocessed data to output directory
            self._write_preprocessed_stack_and_mask(stack_sitk, mask_sitk, filenames[i])

        ##*** Do not propagate masks
        for i in range_prop_mask_complement:
            ## Read stack and mask from directory
            stack_sitk = sitk.ReadImage(self._dir_input + filenames[i] + ".nii.gz", sitk.sitkFloat64)
            mask_sitk = sitk.ReadImage(self._dir_input + filenames[i] + suffix_mask + ".nii.gz", sitk.sitkUInt8)

            ## Crop stack and mask based on the mask provided
            [stack_sitk, mask_sitk] = self._crop_stack_and_mask(stack_sitk, mask_sitk, boundary=0)

            ## Write preprocessed data to output directory
            self._write_preprocessed_stack_and_mask(stack_sitk, mask_sitk, filenames[i])


    ## Get filenames of stacks which require masks
    #  \param[in] filenames list of filenames referring to the data in dir_input to be processed
    #  \param[in] filenames_masks list of filenames which come with a mask
    #  \return list of indices which require a mask
    #  \return complementary list which already have a mask (no mask propagation required on those)
    def _get_filename_indices_for_mask_propagation(self, filenames, filenames_masks):

        ## Number of stacks to be read
        N_stacks = len(filenames)

        ## Indices of all stacks
        stacks_all = np.arange(0, N_stacks)

        ## Get indices of filenames where mask is provided
        indices = []

        for i in stacks_all:
            filename = filenames[i]

            ## if mask does not exist, add corresponding index
            if bool(len(list(set(filename)-set(filenames_masks)))):
                indices.append(i)

        indices_complement = list(set(stacks_all)-set(indices))

        print("Indices for mask propagation = " + str(indices))
        print("No mask propagation required = " + str(indices_complement))

        return indices, indices_complement


    ## Obtain propagated mask based on a given template (moving). The mask 
    #  propagation is obtained via rigid registration.
    #  \param[in] fixed_sitk stack as sitk.Image for which a mask is desired
    #  \param[in] moving_sitk stack as sitk.Image for which a mask is available
    #  \param[in] moving_mask_sitk mask of moving_sitk as sitk.Image to propagate
    #  \return approximate mask of fixed_sitk based on the given template
    def _get_propagated_mask(self, fixed_sitk, moving_sitk, moving_mask_sitk):

        ## Get transform which aligns fixed_sitk with the template moving_sitk
        transform = self._get_rigid_registration_transform_3D(fixed_sitk, moving_sitk)

        ## Resample propagated mask to fixed_sitk grid
        default_pixel_value = 0.0

        # fixed_mask_prop_sitk = sitk.Resample(moving_mask_sitk, fixed_sitk, transform, sitk.sitkLinear, default_pixel_value, moving_mask_sitk.GetPixelIDValue())
        fixed_mask_prop_sitk = sitk.Resample(moving_mask_sitk, fixed_sitk, transform, sitk.sitkNearestNeighbor, default_pixel_value, moving_mask_sitk.GetPixelIDValue())

        return fixed_mask_prop_sitk


    ## Rigid registration routine based on SimpleITK
    #  \param fixed_3D_sitk upsampled fixed Slice
    #  \param moving_3D_sitk moving Stack
    #  \param display_registration_info display registration summary at the end of execution (default=0)
    #  \return Rigid transform as sitk.Euler3DTransform object
    def _get_rigid_registration_transform_3D(self, fixed_3D_sitk, moving_3D_sitk, display_registration_info=0):

        ## Instantiate interface method to the modular ITKv4 registration framework
        registration_method = sitk.ImageRegistrationMethod()

        ## Select between using the geometrical center (GEOMETRY) of the images or using the center of mass (MOMENTS) given by the image intensities
        # initial_transform = sitk.CenteredTransformInitializer(fixed_3D_sitk._sitk_upsampled, moving_3D_sitk.sitk, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
        initial_transform = sitk.Euler3DTransform()

        ## Set the initial transform and parameters to optimize
        registration_method.SetInitialTransform(initial_transform)

        ## Set an image masks in order to restrict the sampled points for the metric
        # registration_method.SetMetricFixedMask(fixed_3D_sitk._sitk_mask_upsampled)
        # registration_method.SetMetricMovingMask(moving_3D_sitk.sitk_mask)

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
        final_transform_3D_sitk = registration_method.Execute(fixed_3D_sitk, moving_3D_sitk) 

        if display_registration_info:
            print("SimpleITK Image Registration Method:")
            print('  Final metric value: {0}'.format(registration_method.GetMetricValue()))
            print('  Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))

        return final_transform_3D_sitk


    ## Dilate mask
    #  \param[in] mask_sitk mask to be dilated
    #  \return dilated mask
    def _dilate_mask(self, mask_sitk):
        filter = sitk.BinaryDilateImageFilter()
        filter.SetKernelType(sitk.sitkBall)
        # filter.SetKernelType(sitk.sitkBox)
        # filter.SetKernelType(sitk.sitkAnnulus)
        # filter.SetKernelType(sitk.sitkCross)
        filter.SetKernelRadius(1)
        filter.SetForegroundValue(1)
        dilated = filter.Execute(mask_sitk)

        return dilated


    """
    Helpers for all "_run_preprocessing_*" types
    """
    ## Crop stack and mask to region given my mask
    #  \param[in] stack_sitk stack as sitk.Image object
    #  \param[in] mask_sitk mask as sitk.Image object
    #  \param[in] boundary additional boundary surrounding mask in mm (optional)
    #  \return cropped stack as sitk.Object
    #  \return cropped mask as sitk.Object
    def _crop_stack_and_mask(self, stack_sitk, mask_sitk, boundary=0):

        ## Get rectangular region surrounding the masked voxels
        [x_range, y_range, z_range] = self._get_rectangular_masked_region(mask_sitk, boundary)

        ## Crop stack and mask to defined image region
        stack_crop_sitk = self._crop_image_to_region(stack_sitk, x_range, y_range, z_range)
        mask_crop_sitk = self._crop_image_to_region(mask_sitk, x_range, y_range, z_range)

        return stack_crop_sitk, mask_crop_sitk
        

    ## Return rectangular region surrounding masked region. 
    #  \param[in] mask_sitk sitk.Image representing the mask
    #  \param[in] boundary additional boundary surrounding mask in mm (optional)
    #  \return range_x pair defining x interval of mask in voxel space 
    #  \return range_y pair defining y interval of mask in voxel space
    #  \return range_z pair defining z interval of mask in voxel space
    def _get_rectangular_masked_region(self, mask_sitk, boundary=0):

        spacing = np.array(mask_sitk.GetSpacing())

        ## Get mask array
        nda = sitk.GetArrayFromImage(mask_sitk)
        
        ## Get shape defining the dimension in each direction
        shape = nda.shape

        ## Set additional offset around identified masked region in voxels
        offset_x = np.round(boundary/spacing[2])
        offset_y = np.round(boundary/spacing[1])
        offset_z = np.round(boundary/spacing[0])

        ## Compute sum of pixels of each slice along specified directions
        sum_xy = np.sum(nda, axis=(0,1)) # sum within x-y-plane
        sum_xz = np.sum(nda, axis=(0,2)) # sum within x-z-plane
        sum_yz = np.sum(nda, axis=(1,2)) # sum within y-z-plane

        ## Find masked regions (non-zero sum!)
        range_x = np.zeros(2)
        range_y = np.zeros(2)
        range_z = np.zeros(2)

        ## Non-zero elements of numpy array nda defining x_range
        ran = np.nonzero(sum_yz)[0]
        range_x[0] = np.max( [0,         ran[0]-offset_x] )
        range_x[1] = np.min( [shape[0], ran[-1]+offset_x+1] )

        ## Non-zero elements of numpy array nda defining y_range
        ran = np.nonzero(sum_xz)[0]
        range_y[0] = np.max( [0,         ran[0]-offset_y] )
        range_y[1] = np.min( [shape[1], ran[-1]+offset_y+1] )

        ## Non-zero elements of numpy array nda defining z_range
        ran = np.nonzero(sum_xy)[0]
        range_z[0] = np.max( [0,         ran[0]-offset_z] )
        range_z[1] = np.min( [shape[2], ran[-1]+offset_z+1] )

        ## Numpy reads the array as z,y,x coordinates! So swap them accordingly
        return range_z.astype(int), range_y.astype(int), range_x.astype(int)


    ## Crop given image to region defined by voxel space ranges
    #  \param[in] image_sitk image which will be cropped
    #  \param[in] range_x pair defining x interval in voxel space for image cropping
    #  \param[in] range_y pair defining y interval in voxel space for image cropping
    #  \param[in] range_z pair defining z interval in voxel space for image cropping
    #  \return image cropped to defined region
    def _crop_image_to_region(self, image_sitk, range_x, range_y, range_z):

        image_cropped_sitk = image_sitk[\
                                range_x[0]:range_x[1],\
                                range_y[0]:range_y[1],\
                                range_z[0]:range_z[1]\
                            ]

        return image_cropped_sitk


    ## Write preprocessed stack and mask to given output folder
    #  \param[in] stack_sitk stack to be written
    #  \param[in] stack_sitk mask to be written
    #  \param[in] filename filename to be used
    def _write_preprocessed_stack_and_mask(self, stack_sitk, mask_sitk, filename):

        ## Write stack
        sitk.WriteImage(stack_sitk, self._dir_output + filename + ".nii.gz")

        ## Write mask
        sitk.WriteImage(mask_sitk, self._dir_output + filename + "_mask.nii.gz")


