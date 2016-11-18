##-----------------------------------------------------------------------------
# \file Stack.py
# \brief      Class containing a stack as sitk.Image object with additional
#             helper methods
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       September 2015
#


## Import libraries
import os                       # used to execute terminal commands in python
import sys
import itk
import SimpleITK as sitk
import numpy as np
import copy

## Import modules from src-folder
import base.Slice as sl
import utilities.SimpleITKHelper as sitkh
import utilities.FilenameParser as fp

## In addition to the nifti-image (stored as sitk.Image object) this class 
## Stack also contains additional variables helpful to work with the data.
class Stack:

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
        stack.itk = sitkh.get_itk_from_sitk_image(stack.sitk)

        ## Append masks (either provided or binary mask)
        if suffix_mask is None:
            stack.sitk_mask = stack._generate_binary_mask()
        else:    
            if os.path.isfile(dir_input + filename + suffix_mask + ".nii.gz"):
                stack.sitk_mask = sitk.ReadImage(dir_input + filename + suffix_mask + ".nii.gz", sitk.sitkUInt8)
            else:
                print("Mask file for " + dir_input + filename + ".nii.gz" +  " not found. Binary mask created." )
                stack.sitk_mask = stack._generate_binary_mask()
        
        ## Append itk object
        stack.itk_mask = sitkh.get_itk_from_sitk_image(stack.sitk_mask)

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

        ## Get 3D images
        stack.sitk = sitk.ReadImage(dir_input + prefix_stack + ".nii.gz", sitk.sitkFloat64)
        stack.itk = sitkh.get_itk_from_sitk_image(stack.sitk)

        ## Append masks (either provided or binary mask)
        if suffix_mask is not None and os.path.isfile(dir_input + prefix_stack + suffix_mask + ".nii.gz"):
            stack.sitk_mask = sitk.ReadImage(dir_input + prefix_stack + suffix_mask + ".nii.gz", sitk.sitkUInt8)
            stack.itk_mask = sitkh.get_itk_from_sitk_image(stack.sitk_mask)
        else:
            stack.sitk_mask = stack._generate_binary_mask()
            stack.itk_mask = sitkh.get_itk_from_sitk_image(stack.sitk_mask)

        ## Get slices
        stack._N_slices = stack.sitk.GetDepth()
        stack._slices = [None] * stack._N_slices

        # ## Get filenames of slices
        # filename_parser = fp.FilenameParser()
        # filenames_slices = filename_parser.get_filenames_which_match_pattern_in_directory(dir_input, patterns=[prefix_stack+"_", ".nii"])
        # if suffix_mask is not None:
        #     filenames_slices = filename_parser.exclude_filenames_which_match_pattern(filenames_slices, suffix_mask)

        ## Append slices as Slice objects
        for i in range(0, stack._N_slices):
            filename_slice = prefix_stack + "_" + str(i)
            stack._slices[i] = sl.Slice.from_filename(dir_input, filename_slice, stack._filename, i, suffix_mask)

        return stack


    ## Create Stack instance from a bundle of Slice objects.
    #  \param[in] slices list of Slice objects
    #  \param[in] stack_sitk optional volumetric stack as sitk.Image object
    #  \param[in] mask_sitk optional associated mask of volumetric stack as sitk.Image object
    #  \return Stack object based on Slice objects
    @classmethod
    def from_slices(cls, slices, stack_sitk=None, mask_sitk=None):

        stack = cls()

        stack._dir = slices[0].get_directory()
        stack._filename = slices[0].get_filename()

        if stack_sitk is None:
            stack.sitk = None
            stack.itk = None
        else:
            stack.sitk = stack_sitk
            stack.itk = sitkh.get_itk_from_sitk_image(stack.sitk)

        stack._N_slices = len(slices)
        stack._slices = slices

        ## Append masks (if provided)
        if mask_sitk is not None:
            stack.sitk_mask = mask_sitk
            stack.itk_mask = sitkh.get_itk_from_sitk_image(stack.sitk_mask)
        else:
            stack.sitk_mask = stack._generate_binary_mask()
            stack.itk_mask = sitkh.get_itk_from_sitk_image(stack.sitk_mask)

        return stack


    ## Create Stack instance from exisiting sitk.Image instance. Slices are
    #  not extracted and stored separately in the object. The idea is to use
    #  this function when the stack is regarded as entire volume (like the 
    #  reconstructed HR volume).
    #  \param[in] image_sitk sitk.Image created from nifti-file
    #  \param[in] name string containing the chosen name for the stack
    #  \param[in] image_sitk_mask associated mask of stack, sitk.Image object (optional)
    #  \return Stack object without slice information
    @classmethod
    def from_sitk_image(cls, image_sitk, name=None, image_sitk_mask=None):
        stack = cls()
        
        stack.sitk = sitk.Image(image_sitk)
        stack.itk = sitkh.get_itk_from_sitk_image(stack.sitk)

        if name is None:
            stack._filename = "unknown"
        else:
            stack._filename = name
        stack._dir = None

        ## Append masks (if provided)
        if image_sitk_mask is not None:
            stack.sitk_mask = image_sitk_mask
            stack.itk_mask = sitkh.get_itk_from_sitk_image(stack.sitk_mask)
        else:
            stack.sitk_mask = stack._generate_binary_mask()
            stack.itk_mask = sitkh.get_itk_from_sitk_image(stack.sitk_mask)

        ## Extract all slices and their masks from the stack and store them 
        stack._N_slices = stack.sitk.GetSize()[-1]
        stack._slices = stack._extract_slices()

        return stack


    ## Copy constructor
    #  \param[in] stack_to_copy Stack object to be copied
    #  \return copied Stack object
    # TODO: That's not really well done!
    @classmethod
    def from_stack(cls, stack_to_copy, filename=None):
        stack = cls()
        
        ## Copy image stack and mask
        stack.sitk = sitk.Image(stack_to_copy.sitk)
        stack.itk = sitkh.get_itk_from_sitk_image(stack.sitk)

        stack.sitk_mask = sitk.Image(stack_to_copy.sitk_mask)
        stack.itk_mask = sitkh.get_itk_from_sitk_image(stack.sitk_mask)

        if filename is None:
            stack._filename = stack_to_copy.get_filename()
        else:
            stack._filename = filename
        stack._dir = stack_to_copy.get_directory()

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

    ## Get all slices of current stack
    #  \return Array of sitk.Images containing slices in 3D space
    def get_slices(self):
        return self._slices


    ## Get one particular slice of current stack
    #  \return requested 3D slice of stack as Slice object
    def get_slice(self, index):
        
        index = int(index)
        if abs(index) > self._N_slices - 1:
            raise ValueError("Enter a valid index between -%s and %s. Tried: %s" %(self._N_slices-1, self._N_slices-1))

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


    def show_slices(self):
        sitkh.plot_stack_of_slices(self.sitk, cmap="Greys_r", title=self.get_filename())



    ## Write information of Stack to HDD to given directory: 
    #  - sitk.Image object as entire volume
    #  - each single slice with its associated spatial transformation (optional)
    #  \param[in] directory string specifying where the output will be written to (default="/tmp/")
    #  \param[in] filename string specifying the filename. If not given the assigned one within Stack will be chosen.
    #  \param[in] write_slices boolean indicating whether each Slice of the stack shall be written (default=False)
    def write(self, directory="/tmp/", filename=None, write_mask=False, write_slices=False, write_transforms=False):

        ## Create directory if not existing
        os.system("mkdir -p " + directory)

        ## Construct filename
        if filename is None:
            filename = self._filename

        full_file_name = os.path.join(directory, filename)

        ## Write file to specified location
        sitk.WriteImage(self.sitk, full_file_name + ".nii.gz")

        ## Write mask to specified location if given
        if self.sitk_mask is not None:
            nda = sitk.GetArrayFromImage(self.sitk_mask)

            ## Write mask if it does not consist of only ones
            if not np.all(nda) and write_mask:
                sitk.WriteImage(self.sitk_mask, full_file_name + "_mask.nii.gz")

        print("Stack was successfully written to %s.nii.gz" %(full_file_name))

        ## Write each separate Slice of stack (if they exist)
        if write_slices:
            try:
                ## Check whether variable exists
                # if 'self._slices' not in locals() or all(i is None for i in self._slices):
                if not hasattr(self,'_slices'):
                    raise ValueError("Error occurred in attempt to write %s.nii.gz: No separate slices of object Slice are found" % (full_file_name))

                ## Write slices
                else:
                    for i in xrange(0,self._N_slices):
                        self._slices[i].write(directory=directory, filename=filename, write_transform=write_transforms)

            except ValueError as err:
                print(err.message)


    ##-------------------------------------------------------------------------
    # \brief      Apply transform on stack and all its slices
    # \date       2016-11-05 19:15:57+0000
    #
    # \param      self                   The object
    # \param      affine_transform_sitk  The affine transform sitk
    #
    def update_motion_correction(self, affine_transform_sitk):

        ## Apply transform to 3D image / stack of slices
        self.sitk = sitkh.get_transformed_sitk_image(self.sitk, affine_transform_sitk)
        
        ## Update header information of other associated images
        origin = self.sitk.GetOrigin()
        direction = self.sitk.GetDirection()

        self.sitk_mask.SetOrigin(orign)
        self.sitk_mask.SetDirection(direction)

        self.itk.SetOrigin(orign)
        self.itk.SetDirection(sitkh.get_itk_from_sitk_direction(direction))

        self.itk_mask.SetOrigin(orign)
        self.itk_mask.SetDirection(sitkh.get_itk_from_sitk_direction(direction))

        ## Update slices
        for i in range(0, self._N_slices):
            self._slices[i].update_motion_correction(affine_transform_sitk)


    ##-------------------------------------------------------------------------
    # \brief      Apply transforms on all the slices of the stack. Stack itself
    #             is not getting transformed
    # \date       2016-11-05 19:16:33+0000
    #
    # \param      self                    The object
    # \param      affine_transforms_sitk  List of sitk transform instances
    #
    def update_motion_correction_of_slices(self, affine_transforms_sitk):
        if type(affine_transforms_sitk) is list and len(affine_transforms_sitk) is self._N_slices:
            for i in range(0, self._N_slices):
                self._slices[i].update_motion_correction(affine_transforms_sitk[i])
                    
        else:
            raise ValueErr("Number of affine transforms does not match the number of slices")


    ##-------------------------------------------------------------------------
    # \brief      Gets the resampled stack from slices.
    # \date       2016-09-26 17:28:43+0100
    #
    # After slice-based registrations slice j does not correspond to the
    # physical space of stack[:,:,j:j+1] anymore. With this method resample all
    # containing slices to the physical space defined by the stack itself (or
    # by a given resampling_pace). Overlapping slices get averaged.
    #
    # \param      self              The object
    # \param      resampling_grid  Define the space to which the stack of
    #                               slices shall be resampled; given as Stack
    #                               object
    # \param      interpolator      The interpolator
    #
    # \return     resampled stack based on current position of slices as Stack
    #             object
    #
    def get_resampled_stack_from_slices(self, resampling_grid=None, interpolator="NearestNeighbor"):

        ## Choose interpolator
        try:
            interpolator_str = interpolator
            interpolator = eval("sitk.sitk" + interpolator_str)
        except:
            raise ValueError("Error: interpolator is not known")

        ## Use resampling grid defined by original volumetric image
        if resampling_grid is None:
            resampling_grid = Stack.from_sitk_image(self.sitk)

        else:
            ## Use resampling grid defined by first slice (which might be shifted already)
            if resampling_grid in ["on_first_slice"]:
                stack_sitk = sitk.Image(self.sitk)
                foo_sitk = sitk.Image(self._slices[0].sitk)
                stack_sitk.SetDirection(foo_sitk.GetDirection())
                stack_sitk.SetOrigin(foo_sitk.GetOrigin())
                stack_sitk.SetSpacing(foo_sitk.GetSpacing())
                resampling_grid = Stack.from_sitk_image(stack_sitk)

            ## Use resampling grid defined by given sitk.Image
            elif type(resampling_grid) is sitk.Image:
                resampling_grid = Stack.from_sitk_image(resampling_grid)

        ## Get shape of image data array
        nda_shape = resampling_grid.sitk.GetSize()[::-1]

        ## Create zero image and its mask aligned with sitk.Image
        nda = np.zeros(nda_shape)
        
        stack_resampled_sitk = sitk.GetImageFromArray(nda)
        stack_resampled_sitk.CopyInformation(resampling_grid.sitk)

        stack_resampled_sitk_mask = sitk.GetImageFromArray(nda.astype("uint8"))
        stack_resampled_sitk_mask.CopyInformation(resampling_grid.sitk_mask)

        ## Create helper used for normalization at the end
        nda_stack_covered_indices = np.zeros(nda_shape)


        default_pixel_value = 0.0

        for i in range(0, self._N_slices):
            slice = self._slices[i]

            ## Resample slice and its mask to stack space
            stack_resampled_slice_sitk = sitk.Resample(
                slice.sitk, 
                resampling_grid.sitk, 
                sitk.Euler3DTransform(), 
                interpolator, 
                default_pixel_value, 
                resampling_grid.sitk.GetPixelIDValue())

            stack_resampled_slice_sitk_mask = sitk.Resample(
                slice.sitk_mask, 
                resampling_grid.sitk_mask, 
                sitk.Euler3DTransform(), 
                sitk.sitkNearestNeighbor, 
                default_pixel_value, 
                resampling_grid.sitk_mask.GetPixelIDValue())

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
        stack_normalization.CopyInformation(resampling_grid.sitk)
        stack_resampled_sitk /= stack_normalization

        ## Get valid binary mask
        stack_resampled_slice_sitk_mask /= stack_resampled_slice_sitk_mask

        stack = self.from_sitk_image(stack_resampled_sitk, self._filename + "_" + interpolator_str, stack_resampled_sitk_mask)

        return stack


    ## Get stack resampled on isotropic grid based on the actual position of
    #  its slices
    #  \param[in] spacing_new_scalar length of voxel side, scalar
    #  \return isotropically, resampled stack as Stack object
    def get_isotropically_resampled_stack_from_slices(self, spacing_new_scalar=None, interpolator="NearestNeighbor"):
        resampled_stack = self.get_resampled_stack_from_slices()

        ## Choose interpolator
        try:
            interpolator_str = interpolator
            interpolator = eval("sitk.sitk" + interpolator_str)
        except:
            raise ValueError("Error: interpolator is not known")

        ## Read original spacing (voxel dimension) and size of target stack:
        spacing = np.array(resampled_stack.sitk.GetSpacing())
        size = np.array(resampled_stack.sitk.GetSize()).astype("int")

        if spacing_new_scalar is None:
            size_new = size
            spacing_new = spacing
            ## Update information according to isotropic resolution
            size_new[2] = np.round(spacing[2]/spacing[0]*size[2]).astype("int")
            spacing_new[2] = spacing[0]
        else:
            spacing_new = np.ones(3)*spacing_new_scalar
            size_new = np.round(spacing/spacing_new*size).astype("int")

        ## Resample image and its mask to isotropic grid
        default_pixel_value = 0.0

        isotropic_resampled_stack_sitk =  sitk.Resample(
            resampled_stack.sitk, 
            size_new, 
            sitk.Euler3DTransform(), 
            interpolator, 
            resampled_stack.sitk.GetOrigin(), 
            spacing_new,
            resampled_stack.sitk.GetDirection(),
            default_pixel_value,
            resampled_stack.sitk.GetPixelIDValue())

        isotropic_resampled_stack_sitk_mask =  sitk.Resample(
            resampled_stack.sitk_mask, 
            size_new, 
            sitk.Euler3DTransform(), 
            sitk.sitkNearestNeighbor, 
            resampled_stack.sitk.GetOrigin(), 
            spacing_new,
            resampled_stack.sitk.GetDirection(),
            default_pixel_value,
            resampled_stack.sitk_mask.GetPixelIDValue())

        ## Create Stack instance
        stack = self.from_sitk_image(isotropic_resampled_stack_sitk, self._filename + "_" + interpolator_str + "Iso", isotropic_resampled_stack_sitk_mask)

        return stack


    ## Get isotropically resampled grid
    #  \param[in] spacing_new_scalar length of voxel side, scalar
    #  \param[in] interpolator choose type of interpolator for resampling
    #  \param[in] extra_frame additional extra frame of zero intensities surrounding the stack in mm
    #  \return isotropically, resampled stack as Stack object
    def get_isotropically_resampled_stack(self, spacing_new_scalar=None, interpolator="Linear", extra_frame=0):

        ## Choose interpolator
        try:
            interpolator_str = interpolator
            interpolator = eval("sitk.sitk" + interpolator_str)
        except:
            raise ValueError("Error: interpolator is not known")
        

        ## Read original spacing (voxel dimension) and size of target stack:
        spacing = np.array(self.sitk.GetSpacing())
        size = np.array(self.sitk.GetSize()).astype("int")
        origin = np.array(self.sitk.GetOrigin())
        direction = self.sitk.GetDirection()

        if spacing_new_scalar is None:
            size_new = size
            spacing_new = spacing
            ## Update information according to isotropic resolution
            size_new[2] = np.round(spacing[2]/spacing[0]*size[2]).astype("int")
            spacing_new[2] = spacing[0]
        else:
            spacing_new = np.ones(3)*spacing_new_scalar
            size_new = np.round(spacing/spacing_new*size).astype("int")

        if extra_frame is not 0:

            ## Get extra_frame in voxel space
            extra_frame_vox = np.round(extra_frame/spacing_new[0]).astype("int")
            
            ## Compute size of resampled stack by considering additional extra_frame
            size_new = size_new + 2*extra_frame_vox

            ## Compute origin of resampled stack by considering additional extra_frame
            a_x = self.sitk.TransformIndexToPhysicalPoint((1,0,0)) - origin
            a_y = self.sitk.TransformIndexToPhysicalPoint((0,1,0)) - origin
            a_z = self.sitk.TransformIndexToPhysicalPoint((0,0,1)) - origin
            e_x = a_x/np.linalg.norm(a_x)
            e_y = a_y/np.linalg.norm(a_y)
            e_z = a_z/np.linalg.norm(a_z)

            translation = (e_x + e_y + e_z)*extra_frame_vox*spacing[0]

            origin = origin - translation


        ## Resample image and its mask to isotropic grid
        default_pixel_value = 0.0

        isotropic_resampled_stack_sitk =  sitk.Resample(
            self.sitk, 
            size_new, 
            sitk.Euler3DTransform(), 
            interpolator, 
            origin, 
            spacing_new,
            direction,
            default_pixel_value,
            self.sitk.GetPixelIDValue())

        isotropic_resampled_stack_sitk_mask =  sitk.Resample(
            self.sitk_mask, 
            size_new, 
            sitk.Euler3DTransform(), 
            sitk.sitkNearestNeighbor, 
            origin, 
            spacing_new,
            direction,
            default_pixel_value,
            self.sitk_mask.GetPixelIDValue())

        ## Create Stack instance
        stack = self.from_sitk_image(isotropic_resampled_stack_sitk, self._filename + "_" + interpolator_str + "Iso", isotropic_resampled_stack_sitk_mask)

        return stack


    ## Increase stack by adding zero voxels in respective directions
    #  \remark Used for MS project to add empty slices on top of (chopped) brain
    #  \param[in] spacing_new_scalar length of voxel side, scalar
    #  \param[in] interpolator choose type of interpolator for resampling
    #  \param[in] extra_frame additional extra frame of zero intensities surrounding the stack in mm
    #  \return isotropically, resampled stack as Stack object
    def get_increased_stack(self, extra_slices_z=0):

        interpolator = sitk.sitkNearestNeighbor

        ## Read original spacing (voxel dimension) and size of target stack:
        spacing = np.array(self.sitk.GetSpacing())
        size = np.array(self.sitk.GetSize()).astype("int")
        origin = np.array(self.sitk.GetOrigin())
        direction = self.sitk.GetDirection()

        ## Update information according to isotropic resolution
        size[2] += extra_slices_z
    
        ## Resample image and its mask to isotropic grid
        default_pixel_value = 0.0

        isotropic_resampled_stack_sitk =  sitk.Resample(
            self.sitk, 
            size, 
            sitk.Euler3DTransform(), 
            interpolator, 
            origin, 
            spacing,
            direction,
            default_pixel_value,
            self.sitk.GetPixelIDValue())

        isotropic_resampled_stack_sitk_mask =  sitk.Resample(
            self.sitk_mask, 
            size, 
            sitk.Euler3DTransform(), 
            sitk.sitkNearestNeighbor, 
            origin, 
            spacing,
            direction,
            default_pixel_value,
            self.sitk_mask.GetPixelIDValue())

        ## Create Stack instance
        stack = self.from_sitk_image(isotropic_resampled_stack_sitk, "zincreased_"+self._filename, isotropic_resampled_stack_sitk_mask)

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
