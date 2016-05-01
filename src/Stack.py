## \file Stack.py
#  \brief  Class containing a stack as sitk.Image object with additional helpers
# 
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date September 2015


## Import libraries
import os                       # used to execute terminal commands in python
import sys
import itk
import SimpleITK as sitk
import numpy as np
import copy

## Import modules from src-folder
import Slice as sl
import SimpleITKHelper as sitkh

## In addition to the nifti-image as being stored as sitk.Image for the whole
#  stack volume \f$ \in R^3 \times R^3 \times R^3\f$ 
#  the class Stack also contains additional variables helpful to work with the data 
class Stack:

    # The constructor
    # def __init__(self):
    #     self.sitk = None
    #     self.sitk_mask = None
    #     self._dir = None
    #     self._filename = None
    #     self._N_slices = None


    ## Create Stack instance from file and add corresponding mask. Mask is
    #  either provided in the directory or created as binary mask consisting
    #  of ones.
    #  \param[in] dir_input string to input directory of nifti-file to read
    #  \param[in] filename string of nifti-file to read
    #  \param[in] suffix_mask extension of stack filename which indicates associated mask
    #  \return Stack object including its slices with corresponding masks
    @classmethod
    def from_filename(cls, dir_input, filename, suffix_mask=None):

        stack = cls()
        # stack = []

        stack._dir = dir_input
        stack._filename = filename

        ## Append stacks as SimpleITK and ITK Image objects
        stack.sitk = sitk.ReadImage(dir_input + filename + ".nii.gz", sitk.sitkFloat64)
        stack.itk = sitkh.convert_sitk_to_itk_image(stack.sitk)

        ## Append masks (either provided or binary mask)
        if suffix_mask is not None and os.path.isfile(dir_input + filename + suffix_mask + ".nii.gz"):
            stack.sitk_mask = sitk.ReadImage(dir_input + filename + suffix_mask + ".nii.gz", sitk.sitkUInt8)
            stack.itk_mask = sitkh.convert_sitk_to_itk_image(stack.sitk_mask)
        else:
            stack.sitk_mask = stack._generate_binary_mask()
            stack.itk_mask = sitkh.convert_sitk_to_itk_image(stack.sitk_mask)

        ## Extract all slices and their masks from the stack and store them 
        stack._N_slices = stack.sitk.GetSize()[-1]
        stack._slices = stack._extract_slices()

        return stack


    ## Create Stack instance from stack slices in specified directory and add corresponding mask.
    #  \param[in] dir_input string to input directory where bundle of slices are stored
    #  \param[in] prefix_stack prefix indicating the corresponding stack
    #  \param[in] suffix_mask extension of stack filename which indicates associated mask
    #  \return Stack object including its slices with corresponding masks
    #  \example mask (suffix_mask) of slice j of stack i (prefix_stack) reads: i_j_mask.nii.gz
    @classmethod
    def from_slice_filenames(cls, dir_input, prefix_stack, suffix_mask=None):

        stack = cls()

        stack._dir = dir_input
        stack._filename = prefix_stack

        ## Get filenames of slices
        filenames_slices = stack._get_slice_filenames(dir_input, prefix_stack, suffix_mask)

        ## There exists no entire image volume.
        stack.sitk = None
        stack.itk = None

        stack._N_slices = len(filenames_slices)
        stack._slices = [None] * stack._N_slices

        ## Append slices as Slice objects
        for i in range(0, stack._N_slices):
            filename_slice = prefix_stack + "_" + filenames_slices[i]
            stack._slices[i] = sl.Slice.from_filename(dir_input, filename_slice, stack._filename, i, suffix_mask)

        return stack


    ## Create Stack instance from exisiting sitk.Image instance. Slices are
    #  not extracted and stored separately in the object. The idea is to use
    #  this function when the stack is regarded as entire volume (like the 
    #  reconstructed HR volume). A mask can be added via self.add_mask then.
    #  \param[in] image_sitk sitk.Image created from nifti-file
    #  \param[in] name string containing the chosen name for the stack
    #  \param[in] image_sitk_mask associated mask of stack, sitk.Image object (optional)
    #  \return Stack object without slice information
    @classmethod
    def from_sitk_image(cls, image_sitk, name, image_sitk_mask=None):
        stack = cls()
        
        stack.sitk = sitk.Image(image_sitk)
        stack.itk = sitkh.convert_sitk_to_itk_image(stack.sitk)

        stack._filename = name

        stack._N_slices = stack.sitk.GetSize()[-1]
        stack._slices = [None]*stack._N_slices

        ## Append masks (if provided)
        if image_sitk_mask is not None:
            stack.sitk_mask = image_sitk_mask
            stack.itk_mask = sitkh.convert_sitk_to_itk_image(stack.sitk_mask)
        else:
            stack.sitk_mask = None
            stack.itk_mask = None

        return stack


    ## Copy constructor
    #  \param[in] stack_to_copy Stack object to be copied
    #  \return copied Stack object
    # TODO: That's not really well done!
    @classmethod
    def from_stack(cls, stack_to_copy):
        stack = cls()
        
        ## Copy image stack and mask
        stack.sitk = sitk.Image(stack_to_copy.sitk)
        stack.itk = sitkh.convert_sitk_to_itk_image(stack.sitk)

        stack.sitk_mask = sitk.Image(stack_to_copy.sitk_mask)
        stack.itk_mask = sitkh.convert_sitk_to_itk_image(stack.sitk_mask)

        stack._filename = "copy_" + stack_to_copy.get_filename()

        stack._N_slices = stack_to_copy.sitk.GetSize()[-1]
        stack._slices = [None]*stack._N_slices

        ## Extract all slices and their masks from the stack and store them if given
        if not all(x is None for x in stack_to_copy.get_slices()):
            slices_to_copy = stack_to_copy.get_slices()
            
            for i in range(0, stack._N_slices):
                stack._slices[i] = sl.Slice.from_slice(slices_to_copy[i])

        return stack


    # @classmethod
    # def from_Stack(cls, class_instance):
    #     data = copy.deepcopy(class_instance) # if deepcopy is necessary
    #     return cls(data)

    # def __deepcopy__(self, memo):
    #     print '__deepcopy__(%s)' % str(memo)
    #     return Stack(copy.deepcopy(memo))

    # def copy(self):
        # return copy.deepcopy(self)


    ## Get filenames of slices in specified directory.
    #  Assumption: Filenames constitute of subsequent numbering
    #  \param[in] dir_input string to input directory of nifti-file to read
    #  \param[in] prefix_stack prefix indicating the corresponding stack
    #  \param[in] suffix_mask extension of stack filename which indicates associated mask
    #  \return filenames as list of strings
    def _get_slice_filenames(self, dir_input, prefix_stack, suffix_mask):

        filenames = []

        ## List of all files in directory
        all_files = os.listdir(dir_input)

        ## Number of symbols of stack prefix which defines sought slices
        prefix_stack_len = len(prefix_stack)

        for file in all_files:

            ## Only consider nifti images without their mask
            if file.endswith(".nii.gz") and file.startswith(prefix_stack) and not file.endswith(suffix_mask + ".nii.gz"):
                
                filename = file

                ## Chop off prefix + underscore
                filename = filename[prefix_stack_len+1:]

                ## Chop off ending
                filename = filename.replace(".nii.gz","")

                ## Filename consits only of slice filename
                filenames.append(filename)

        ## Assumption of subsequent numbering
        N = len(filenames)

        ## Create filenames as list of strings
        filenames = [str(i) for i in range(0, N)]

        return filenames



    ## Add a mask to a existing Stack instance with no existing mask yet.
    #  \param[in] image_sitk_mask sitk.Image representing the mask
    def add_mask(self, image_sitk_mask):

        try:
            if self.sitk_mask is None:
                self.sitk_mask = image_sitk_mask
                self.itk_mask = sitkh.convert_sitk_to_itk_image(image_sitk_mask)

            else:
                raise ValueError("Error: Attempt to override already existing mask")

        except ValueError as err:
            print(err.args)


    ## Get all slices of current stack
    #  \return Array of sitk.Images containing slices in 3D space
    def get_slices(self):
        return self._slices


    ## Get one particular slice of current stack
    #  \return requested 3D slice of stack as Slice object
    def get_slice(self, index):
        
        index = int(index)
        if index > self._N_slices - 1 or index < 0:
            raise ValueError("Enter a valid index between 0 and %s. Tried: %s" %(self._N_slices-1, index))

        return self._slices[index]


    ## Get name of directory where nifti was read from
    #  \return string of directory wher nifti was read from
    #  \bug Does not exist for all created instances! E.g. Stack.from_sitk_image
    def get_directory(self):
        return self._dir


    #  \return string of filename
    def set_filename(self, filename):
        self._filename = filename


    ## Get filename of read/assigned nifti file (Stack.from_filename vs Stack.from_sitk_image)
    #  \return string of filename
    def get_filename(self):
        return self._filename


    ## Get number of slices of stack
    #  \return number of slices of stack
    def get_number_of_slices(self):
        return self._N_slices


    ## Display stack with external viewer (ITK-Snap)
    #  \param[in][in] show_segmentation display stack with or without associated segmentation (default=0)
    def show(self, show_segmentation=0, title=None):
        dir_output = "/tmp/"

        if title is None:
            title = self._filename

        if show_segmentation:
            sitk.WriteImage(self.sitk, dir_output + title + ".nii.gz")
            sitk.WriteImage(self.sitk_mask, dir_output + title + "_mask.nii.gz")

            cmd = "itksnap " \
                    + "-g " + dir_output + title + ".nii.gz " \
                    + "-s " +  dir_output + title + "_mask.nii.gz " + \
                    "& "

        else:
            sitk.WriteImage(self.sitk, dir_output + title + ".nii.gz")

            cmd = "itksnap " \
                    + "-g " + dir_output + title + ".nii.gz " \
                    "& "

        # cmd = "fslview " + dir_output + filename_out + ".nii.gz & "
        os.system(cmd)


    ## Write information of Stack to HDD to given directory: 
    #  - sitk.Image object as entire volume
    #  - each single slice with its associated spatial transformation (optional)
    #  \param[in] directory string specifying where the output will be written to (default="/tmp/")
    #  \param[in] filename string specifying the filename. If not given the assigned one within Stack will be chosen.
    #  \param[in] write_slices boolean indicating whether each Slice of the stack shall be written (default=False)
    def write(self, directory="/tmp/", filename=None, write_slices=False):
        if filename is None:
            filename = self._filename

        full_file_name = os.path.join(directory, filename + ".nii.gz")

        ## Write file to specified location
        sitk.WriteImage(self.sitk, full_file_name)
        print("Stack was successfully written to %s" %(full_file_name))

        ## Write each separate Slice of stack (if they exist)
        if write_slices:
            try:
                ## Check whether variable exists
                # if 'self._slices' not in locals() or all(i is None for i in self._slices):
                if not hasattr(self,'_slices'):
                    raise ValueError("Error occurred in attempt to write %s: No separate slices of object Slice are found" % (full_file_name))

                ## Write slices
                else:
                    for i in xrange(0,self._N_slices):
                        self._slices[i].write(directory=directory, filename=filename)

            except ValueError as err:
                print(err.message)


    ## After slice-based registrations slice j does not correspond to the physical
    #  space of stack[:,:,j:j+1] anymore. With this method resample all containing
    #  slices to the physical space defined by the stack. Overlapping slices get 
    #  averaged
    #  \return resampled stack based on current position of slices as Stack object
    def get_resampled_stack_from_slices(self):

        ## Get shape of image data array
        nda_shape = self.sitk.GetSize()[::-1]

        ## Create zero image and its mask aligned with sitk.Image
        nda = np.zeros(nda_shape)
        
        stack_resampled_sitk = sitk.GetImageFromArray(nda)
        stack_resampled_sitk.CopyInformation(self.sitk)

        stack_resampled_sitk_mask = sitk.GetImageFromArray(nda.astype("uint8"))
        stack_resampled_sitk_mask.CopyInformation(self.sitk_mask)

        ## Create helper used for normalization at the end
        nda_stack_covered_indices = np.zeros(nda_shape)


        default_pixel_value = 0.0

        for i in range(0, self._N_slices):
            slice = self._slices[i]

            ## Resample slice and its mask to stack space
            stack_resampled_slice_sitk = sitk.Resample(
                slice.sitk, 
                self.sitk, 
                sitk.Euler3DTransform(), 
                sitk.sitkNearestNeighbor, 
                default_pixel_value, 
                self.sitk.GetPixelIDValue())

            stack_resampled_slice_sitk_mask = sitk.Resample(
                slice.sitk_mask, 
                self.sitk_mask, 
                sitk.Euler3DTransform(), 
                sitk.sitkNearestNeighbor, 
                default_pixel_value, 
                self.sitk_mask.GetPixelIDValue())

            ## Add resampled slice and mask to stack space
            stack_resampled_sitk += stack_resampled_slice_sitk
            stack_resampled_sitk_mask += stack_resampled_slice_sitk_mask

            ## Get indices which are updated in stack space
            nda_stack_resampled_slice_ind = sitk.GetArrayFromImage(stack_resampled_slice_sitk)
            ind = np.nonzero(nda_stack_resampled_slice_ind)

            ## Increment counter for respective updated voxels
            nda_stack_covered_indices[ind] += 1
            
        ## Set voxels with zero counter to 1 so as to have well-defined normalization
        nda_stack_covered_indices[nda_stack_covered_indices==0] = 1

        ## Normalize resampled image
        stack_normalization = sitk.GetImageFromArray(nda_stack_covered_indices)
        stack_normalization.CopyInformation(self.sitk)
        stack_resampled_sitk /= stack_normalization

        ## Get valid binary mask
        stack_resampled_slice_sitk_mask /= stack_resampled_slice_sitk_mask

        stack = self.from_sitk_image(stack_resampled_sitk,"resampled_"+self._filename)
        stack.add_mask(stack_resampled_sitk_mask)

        return stack


    ## Get stack resampled on isotropic grid based on the actual position of
    #  its slices
    #  \return isotropically, resampled stack as Stack object
    def get_isotropically_resampled_stack_from_slices(self):
        resampled_stack = self.get_resampled_stack_from_slices()

        ## Read original spacing (voxel dimension) and size of target stack:
        spacing = np.array(resampled_stack.sitk.GetSpacing())
        size = np.array(resampled_stack.sitk.GetSize())

        ## Update information according to isotropic resolution
        size[2] = np.round(spacing[2]/spacing[0]*size[2])
        spacing[2] = spacing[0]

        ## Resample image and its mask to isotropic grid
        default_pixel_value = 0.0

        isotropic_resampled_stack_sitk =  sitk.Resample(
            resampled_stack.sitk, 
            size, 
            sitk.Euler3DTransform(), 
            sitk.sitkNearestNeighbor, 
            resampled_stack.sitk.GetOrigin(), 
            spacing,
            resampled_stack.sitk.GetDirection(),
            default_pixel_value,
            resampled_stack.sitk.GetPixelIDValue())

        isotropic_resampled_stack_sitk_mask =  sitk.Resample(
            resampled_stack.sitk_mask, 
            size, 
            sitk.Euler3DTransform(), 
            sitk.sitkNearestNeighbor, 
            resampled_stack.sitk.GetOrigin(), 
            spacing,
            resampled_stack.sitk.GetDirection(),
            default_pixel_value,
            resampled_stack.sitk_mask.GetPixelIDValue())

        ## Create Stack instance of HR_volume
        stack = self.from_sitk_image(isotropic_resampled_stack_sitk, "isotropic_resampled_"+self._filename)
        stack.add_mask(isotropic_resampled_stack_sitk_mask)

        return stack



    ## Burst the stack into its slices and return all slices of the stack
    #  return list of Slice objects
    def _extract_slices(self):

        slices = [None]*self._N_slices

        ## Extract slices and add masks
        for i in range(0, self._N_slices):
            slices[i] = sl.Slice.from_sitk_image(
                slice_sitk = self.sitk[:,:,i:i+1], 
                dir_input = self._dir, 
                filename = self._filename, 
                slice_number = i,
                slice_sitk_mask = self.sitk_mask[:,:,i:i+1])        

        return slices


    ## Create a binary mask consisting of ones
    #  \return binary_mask as sitk.Image object consisting of ones
    def _generate_binary_mask(self):
        shape = sitk.GetArrayFromImage(self.sitk).shape
        nda = np.ones(shape, dtype=np.uint8)

        binary_mask = sitk.GetImageFromArray(nda)
        binary_mask.CopyInformation(self.sitk)

        return binary_mask
