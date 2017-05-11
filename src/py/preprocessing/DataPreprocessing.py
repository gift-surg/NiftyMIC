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
import base.Stack as st
import utilities.PythonHelper as ph
import utilities.SimpleITKHelper as sitkh
import preprocessing.N4BiasFieldCorrection as n4bfc


## Class implementing data preprocessing steps
class DataPreprocessing:

    ## Constructor
    def __init__(self, use_N4BiasFieldCorrector=False, segmentation_propagator=None, target_stack_index=0, use_crop_to_mask=True, boundary_i=0, boundary_j=0, boundary_k=0, unit="mm"):

        self._use_N4BiasFieldCorrector = use_N4BiasFieldCorrector
        self._segmentation_propagator = segmentation_propagator
        self._target_stack_index = target_stack_index
        self._use_crop_to_mask = use_crop_to_mask
        self._boundary_i = boundary_i
        self._boundary_j = boundary_j
        self._boundary_k = boundary_k
        self._unit = unit


    ## Initialize data preprocessing class based on filenames, i.e. data used
    #  is going to be read from the hard disk. 
    #  \param[in] dir_input directory where data is stored for preprocessing
    #  \param[in] filenames list of filenames referring to the data in dir_input to be processed (without .nii.gz)
    #  \param[in] suffix_mask extension of stack filename which indicates associated mask
    @classmethod
    def from_filenames(cls, dir_input, filenames, suffix_mask="_mask", use_N4BiasFieldCorrector=False, segmentation_propagator=None, target_stack_index=0, use_crop_to_mask=True, boundary_i=0, boundary_j=0, boundary_k=0, unit="mm"):

        self = cls(use_N4BiasFieldCorrector=use_N4BiasFieldCorrector, segmentation_propagator=segmentation_propagator, target_stack_index=target_stack_index, use_crop_to_mask=use_crop_to_mask, boundary_i=boundary_i, boundary_j=boundary_j, boundary_k=boundary_k, unit=unit)

        ## Number of stacks to be read
        self._N_stacks = len(filenames)

        ## Read stacks and their masks (if no mask is found a binary image is created automatically)
        self._stacks_preprocessed = [None]*self._N_stacks

        for i in range(0, self._N_stacks):
            self._stacks_preprocessed[i] = st.Stack.from_filename(dir_input, filenames[i], suffix_mask)

        print("%s stacks were read for data preprocessing." %(self._N_stacks))

        return self


    ## Initialize data preprocessing class based on stacks, i.e. those stacks
    #  are going to be preprocessed
    #  \param[in] stacks list of Stack objects
    #  \param[in] filenames list of filenames referring to the data in dir_input to be processed
    #  \param[in] suffix_mask extension of stack filename which indicates associated mask
    @classmethod
    def from_stacks(cls, stacks, use_N4BiasFieldCorrector=False, segmentation_propagator=None, target_stack_index=0, use_crop_to_mask=True, boundary_i=0, boundary_j=0, boundary_k=0, unit="mm"):

        self = cls(use_N4BiasFieldCorrector=use_N4BiasFieldCorrector, segmentation_propagator=segmentation_propagator, target_stack_index=target_stack_index, use_crop_to_mask=use_crop_to_mask, boundary_i=boundary_i, boundary_j=boundary_j, boundary_k=boundary_k, unit=unit)

        ## Number of stacks
        self._N_stacks = len(stacks)

        ## Use stacks provided
        self._stacks_preprocessed = [None]*self._N_stacks
        for i in range(0, self._N_stacks):
            self._stacks_preprocessed[i] = st.Stack.from_stack(stacks[i])

        print("%s stacks were loaded for data preprocessing." %(self._N_stacks))

        return self

    ## Specify whether bias field correction based on N4 Bias Field Correction 
    #  Filter shall be used
    #  \param[in] flag
    def use_N4BiasFieldCorrector(self, flag):
        self._use_N4BiasFieldCorrector = flag;


    ## Specify prefix which will be used for naming the stacks
    #  param[in] prefix as string
    def set_filename_prefix(self, prefix):
        self._filename_prefix = prefix


    ## Perform data preprocessing step by reading images from files
    #  \param[in] mask_template_number relevant in case not all masks are given (optional). Indicates stack for mask propagation.
    #  \param[in] boundary additional boundary surrounding mask in mm (optional). Capped by image domain.
    def run_preprocessing(self):


        ## Segmentation propagation
        if self._segmentation_propagator is not None:
            ph.print_subtitle("Segmentation propagation")
            
            stacks_to_propagate_indices = list(set(range(0,self._N_stacks)) - set([self._target_stack_index]))
            
            target = self._stacks_preprocessed[self._target_stack_index]
            self._stacks_preprocessed[self._target_stack_index] = st.Stack.from_stack(target)

            self._segmentation_propagator.set_template(target)

            for i in stacks_to_propagate_indices:
                self._segmentation_propagator.set_stack(self._stacks_preprocessed[i])
                self._segmentation_propagator.run_segmentation_propagation()
                self._stacks_preprocessed[i] = self._segmentation_propagator.get_segmented_stack()

        ## Crop to mask
        if self._use_crop_to_mask:
            ph.print_subtitle("Crop stack to mask")

            for i in range(0, self._N_stacks):
                self._stacks_preprocessed[i] = self._stacks_preprocessed[i].get_cropped_stack_based_on_mask(boundary_i=self._boundary_i, boundary_j=self._boundary_j, boundary_k=self._boundary_k, unit=self._unit)

        ## N4 Bias Field Correction
        if self._use_N4BiasFieldCorrector:
            ph.print_subtitle("N4 Bias Field Correction")
            bias_field_corrector = n4bfc.N4BiasFieldCorrection()

            for i in range(0, self._N_stacks):
                bias_field_corrector.set_stack(self._stacks_preprocessed[i])
                bias_field_corrector.run_bias_field_correction()
                self._stacks_preprocessed[i] = bias_field_corrector.get_bias_field_corrected_stack()
        

    ## Get preprocessed stacks
    #  \return preprocessed stacks as list of Stack objects
    def get_preprocessed_stacks(self):

        ## Return a copy of preprocessed stacks
        stacks_copy = [None]*self._N_stacks

        ## Move target stack to first position
        stacks_copy[0] = st.Stack.from_stack(self._stacks_preprocessed[self._target_stack_index])
        remaining_indices = list(set(range(0,self._N_stacks)) - set([self._target_stack_index]))

        i_ctr = 1
        for i in remaining_indices:
            stacks_copy[i_ctr] = st.Stack.from_stack(self._stacks_preprocessed[i])
            i_ctr = i_ctr +  1
        return stacks_copy


    ## Write preprocessed data to specified output directory
    #  \param[in] dir_output output directory
    def write_preprocessed_data(self, dir_output):
        if all(x is None for x in self._stacks_preprocessed):
            raise ValueError("Error: Run preprocessing first")

        ## Write all slices
        for i in range(0, self._N_stacks):
            slices = self._stacks_preprocessed[i].write(directory=dir_output, write_mask=True, write_slices=False)



    """
    All masks provided
    """
    ## Perform data preprocessing step for case that all masks are provided
    def _run_preprocessing_all_masks_provided(self):

        print("All masks provided. Stack and associated mask are cropped to their masked region.")

        ## Radius for dilation
        dilation_radius = self._dilation_radius

        for i in range(0, self._N_stacks):

            ## Dilate propagated mask (to smooth mask)
            if dilation_radius > 0:
                sys.stdout.write("\tStack %s: Dilate mask (radius=%s) ... " %(i, dilation_radius))
                sys.stdout.flush() #flush output; otherwise sys.stdout.write would wait until next newline before printing
                mask_sitk = self._dilate_mask(self._stacks[i].sitk_mask, dilation_radius)
                print "done"
            else:
                mask_sitk = self._stacks[i].sitk_mask


            ## Crop stack and mask based on the mask provided
            sys.stdout.write("\tStack %s: Crop stack and its mask ... " %(i))
            sys.stdout.flush()
            [stack_sitk, mask_sitk] = self._crop_stack_and_mask(self._stacks[i].sitk, mask_sitk, boundary=self._boundary, boundary_y=self._boundary_y, boundary_z=self._boundary_z)
            print "done"

            ## Create stack instance
            if self._filename_prefix is not None:
                filename = self._filename_prefix+str(i)
            else:
                filename = self._stacks[i].get_filename()
            stack = st.Stack.from_sitk_image(image_sitk=stack_sitk, name=filename, image_sitk_mask=mask_sitk)

            ## Perform Bias Field correction if desired
            if self._use_N4BiasFieldCorrector:
                sys.stdout.write("\tStack %s: Apply itkN4BiasFieldCorrectionImageFilter ... " %(i))
                sys.stdout.flush()
                self._stacks_preprocessed[i] = self._get_bias_field_corrected_stack(stack)
                print "done"

            else:
                self._stacks_preprocessed[i] = stack
            

    """
    No mask provided
    """
    ## Perform data preprocessing step for case that no mask is provided
    def _run_preprocessing_no_mask_provided(self):
        
        print("No mask is provided. Consider entire stack for binary mask.")
        
        for i in range(0, self._N_stacks):

            ## Preprocessed stack consists of untouched image and full binary mask
            if self._filename_prefix is not None:
                filename = self._filename_prefix+str(i)
            else:
                filename = self._stacks[i].get_filename()
            stack = st.Stack.from_sitk_image(image_sitk=self._stacks[i].sitk, name=filename, image_sitk_mask=self._stacks[i].sitk_mask)

            ## Perform Bias Field correction if desired
            if self._use_N4BiasFieldCorrector:
                sys.stdout.write("\tStack %s: Apply itkN4BiasFieldCorrectionImageFilter ... " %(i))
                sys.stdout.flush()
                self._stacks_preprocessed[i] = self._get_bias_field_corrected_stack(stack)
                print "done"

            else:
                self._stacks_preprocessed[i] = stack


    """
    Not all masks provided
    """
    ## Perform data preprocessing step for case that some masks are missing.
    #  Mask stored in self._mask_template_number is used as template mask for
    #  mask propagation
    def _run_preprocessing_not_all_masks_provided(self):

        print("Not all stacks are provided a mask. Mask of target stack is propagated to other masks and cropped.")

        ## Radius for dilation
        dilation_radius = self._dilation_radius

        ## Find stacks which require a mask. Target stack mask is used
        #  as template mask for mask propagation.
        range_prop_mask, range_prop_mask_complement = self._get_filename_indices_for_mask_propagation(self._filenames, self._filenames_masks)

        ##*** Propagate masks

        ## Specify target stack and mask to be used as template
        template = self._stacks[self._mask_template_number]

        for i in range_prop_mask:

            sys.stdout.write("\tStack %s: Rigidly align stack with template and propagate mask ... " %(i))
            sys.stdout.flush() #flush output; otherwise sys.stdout.write would wait until next newline before printing
            
            ## Rigidly align stack with template and propagate the mask
            self._stacks[i] = self._propagate_mask_from_template_to_stack(self._stacks[i], template)
            mask_sitk = self._stacks[i].sitk_mask
            print "done"

            ## Dilate propagated mask (to smooth mask)
            if dilation_radius > 0:
                sys.stdout.write("\tStack %s: Dilate mask (radius=%s) ... " %(i, dilation_radius))
                sys.stdout.flush() #flush output; otherwise sys.stdout.write would wait until next newline before printing
                mask_sitk = self._dilate_mask(mask_sitk, dilation_radius)
                print "done"

            ## Crop stack and mask based on the mask provided
            sys.stdout.write("\tStack %s: Crop stack and its mask ... " %(i))
            sys.stdout.flush()
            [stack_sitk, mask_sitk] = self._crop_stack_and_mask(self._stacks[i].sitk, mask_sitk, boundary=self._boundary, boundary_y=self._boundary_y, boundary_z=self._boundary_z)
            print "done"

            ## Create stack instance
            if self._filename_prefix is not None:
                filename = self._filename_prefix+str(i)
            else:
                filename = self._stacks[i].get_filename()
            stack = st.Stack.from_sitk_image(image_sitk=stack_sitk, name=filename, image_sitk_mask=mask_sitk)

            ## Perform Bias Field correction if desired
            if self._use_N4BiasFieldCorrector:
                sys.stdout.write("\tStack %s: Apply itkN4BiasFieldCorrectionImageFilter ... " %(i))
                sys.stdout.flush()
                self._stacks_preprocessed[i] = self._get_bias_field_corrected_stack(stack)
                print "done"

            else:
                self._stacks_preprocessed[i] = stack

        ##*** Do not propagate masks (includes template stack)
        for i in range_prop_mask_complement:

            ## Dilate propagated mask (to smooth mask)
            if dilation_radius > 0:
                sys.stdout.write("\tStack %s: Dilate mask (radius=%s) ... " %(i, dilation_radius))
                sys.stdout.flush() #flush output; otherwise sys.stdout.write would wait until next newline before printing
                mask_sitk = self._dilate_mask(self._stacks[i].sitk_mask, dilation_radius)
                print "done"
            else:
                mask_sitk = self._stacks[i].sitk_mask

            ## Crop stack and mask based on the mask provided
            sys.stdout.write("\tStack %s: Crop stack and its mask ... " %(i))
            sys.stdout.flush()
            [stack_sitk, mask_sitk] = self._crop_stack_and_mask(self._stacks[i].sitk, mask_sitk, boundary=self._boundary, boundary_y=self._boundary_y, boundary_z=self._boundary_z)
            print "done"

            ## Create stack instance
            if self._filename_prefix is not None:
                filename = self._filename_prefix+str(i)
            else:
                filename = self._stacks[i].get_filename()
            stack = st.Stack.from_sitk_image(image_sitk=stack_sitk, name=filename, image_sitk_mask=mask_sitk)

            ## Perform Bias Field correction if desired
            if self._use_N4BiasFieldCorrector:
                sys.stdout.write("\tStack %s: Apply itkN4BiasFieldCorrectionImageFilter ... " %(i))
                sys.stdout.flush()
                self._stacks_preprocessed[i] = self._get_bias_field_corrected_stack(stack)
                print "done"

            else:
                self._stacks_preprocessed[i] = stack


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
            if filename in filenames_masks:
                indices.append(i)

        indices_complement = list(set(stacks_all)-set(indices))

        print("Mask is provided for indices = " + str(indices))
        print("Propagate mask for indices = " + str(indices_complement))

        return indices_complement, indices


    ##
    # Rigidly align stack with template and propagate the template mask.
    # \date       2017-02-02 17:36:50+0000
    #
    # \param      self      The object
    # \param      stack     The stack
    # \param      template  The template
    #
    # \post       stack object is rigidly aligned with template
    # \return     The propagated mask sitk.
    #
    def _propagate_mask_from_template_to_stack(self, stack, template):

        # registration = regniftyreg.NiftyReg(fixed=template, moving=stack, registration_type="Rigid", verbose=False)
        registration = regitk.RegistrationITK(fixed=template, moving=stack, registration_type="Rigid", interpolator="Linear", verbose=False, metric="Correlation")
        # registration = regsitk.RegistrationSimpleITK(fixed=template, moving=stack, registration_type="Rigid", interpolator="Linear", verbose=True, metric="Correlation", use_centered_transform_initializer=True, scales_estimator="IndexShift", initializer_type="MOMENTS")
        registration.use_fixed_mask(True)
        registration.run_registration()

        ## Get transform which aligns
        # transform_sitk = sitk.AffineTransform(registration.get_registration_transform_sitk().GetInverse())
        transform_sitk = sitk.Euler3DTransform(registration.get_registration_transform_sitk().GetInverse())
        stack.update_motion_correction(transform_sitk)

        # ## Get transform which aligns fixed_sitk with the template moving_sitk
        # transform = self._get_rigid_registration_transform_3D(fixed_sitk, moving_sitk)

        ## Resample propagated mask to fixed_sitk grid
        stack_mask_sitk = sitk.Resample(template.sitk_mask, stack.sitk_mask, sitk.Euler3DTransform(), sitk.sitkNearestNeighbor, 0, template.sitk_mask.GetPixelIDValue())       

        stack_aligned_masked = st.Stack.from_sitk_image(stack.sitk, stack.get_filename(), stack_mask_sitk)
        # stack_aligned_masked = st.Stack.from_sitk_image(stack.sitk, stack.get_filename())

        # slice_k = stack_aligned_masked

        return stack_aligned_masked


    ## Dilate mask
    #  \param[in] mask_sitk mask to be dilated
    #  \param[in] radius radius for kernel with units in voxel space
    #  \return dilated mask
    def _dilate_mask(self, mask_sitk, radius):
        filter = sitk.BinaryDilateImageFilter()
        # filter.SetKernelType(sitk.sitkBall)
        # filter.SetKernelType(sitk.sitkBox)
        # filter.SetKernelType(sitk.sitkAnnulus)
        filter.SetKernelType(sitk.sitkCross)
        filter.SetKernelRadius(radius)
        filter.SetForegroundValue(1)
        dilated = filter.Execute(mask_sitk)

        return dilated


    """
    Helpers for all "_run_preprocessing_*" types
    """
    ## Crop stack and mask to region given my mask
    #  \param[in] stack_sitk stack as sitk.Image object
    #  \param[in] mask_sitk mask as sitk.Image object
    #  \param[in] boundary additional boundary surrounding mask in mm (optional). Capped by image domain.
    #  \return cropped stack as sitk.Object
    #  \return cropped mask as sitk.Object
    def _crop_stack_and_mask(self, stack_sitk, mask_sitk, boundary=0, boundary_y=None, boundary_z=None):

        ## Get rectangular region surrounding the masked voxels
        [x_range, y_range, z_range] = self._get_rectangular_masked_region(mask_sitk, boundary, boundary_y, boundary_z)

        ## Crop stack and mask to defined image region
        stack_crop_sitk = self._crop_image_to_region(stack_sitk, x_range, y_range, z_range)
        mask_crop_sitk = self._crop_image_to_region(mask_sitk, x_range, y_range, z_range)

        ## Update header of mask accordingly! Otherwise, not necessarily in the same space!!!
        mask_crop_sitk.CopyInformation(stack_crop_sitk)

        return stack_crop_sitk, mask_crop_sitk
        

    ## Return rectangular region surrounding masked region. 
    #  \param[in] mask_sitk sitk.Image representing the mask
    #  \param[in] boundary additional boundary surrounding mask in mm (optional). Capped by image domain.
    #  \return range_x pair defining x interval of mask in voxel space 
    #  \return range_y pair defining y interval of mask in voxel space
    #  \return range_z pair defining z interval of mask in voxel space
    def _get_rectangular_masked_region(self, mask_sitk, boundary=0, boundary_y=None, boundary_z=None):

        spacing = np.array(mask_sitk.GetSpacing())

        ## Get mask array
        nda = sitk.GetArrayFromImage(mask_sitk)
        
        ## Get shape defining the dimension in each direction
        shape = nda.shape

        if boundary_y is None and boundary_z is None:
            boundary_y = boundary
            boundary_z = boundary

        ## Set additional offset around identified masked region in voxels
        offset_x = np.round(boundary_z/spacing[2])
        offset_y = np.round(boundary_y/spacing[1])
        offset_z = np.round(boundary  /spacing[0])

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


    ## Apply itkN4BiasFieldCorrectionImageFilter onto the stack
    #  \param[in] stack as Stack object to
    #  \return bias field corrected stack
    def _get_bias_field_corrected_stack(self, stack):
        dir_tmp = "/tmp/N4BiasFieldCorrection/"
        filename_out = "N4BiasFieldCorrection_image"

        os.system("mkdir -p " + dir_tmp)
        os.system("rm -rf " + dir_tmp + "*")

        sitk.WriteImage(stack.sitk, dir_tmp + filename_out + ".nii.gz")
        sitk.WriteImage(stack.sitk_mask, dir_tmp + filename_out + "_mask.nii.gz")


        cmd =  "/Users/mebner/UCL/UCL/Volumetric\ Reconstruction/build/cpp/bin/runN4BiasFieldCorrectionImageFilter "
        cmd += "--f " + dir_tmp + filename_out + " "
        cmd += "--fmask " + dir_tmp + filename_out + "_mask "
        cmd += "--tout " + dir_tmp + " "
        cmd += "--m " + filename_out
        
        # print cmd
        os.system(cmd)

        stack_corrected_sitk = sitk.ReadImage(dir_tmp + filename_out + "_corrected.nii.gz", sitk.sitkFloat64)

        ## Read in again: Otherwise there can occur probems. Hence, read in 
        #  and force to be the same sitk.sitkFloat64 type!
        stack_corrected_sitk_mask = sitk.ReadImage(dir_tmp + filename_out + "_mask.nii.gz", sitk.sitkUInt8)

        # stack_corrected = st.Stack.from_sitk_image(stack_corrected_sitk, stack.get_filename(), stack.sitk_mask)
        stack_corrected = st.Stack.from_sitk_image(stack_corrected_sitk, stack.get_filename(), stack_corrected_sitk_mask)

        ## Debug
        # sitkh.show_sitk_image(stack.sitk, overlay=stack_corrected.sitk, title="StackOrig_StackBiasFieldCorrected")

        return stack_corrected

