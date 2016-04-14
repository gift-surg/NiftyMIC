## \file Stack.py
#  \brief  Class containing a stack as sitk.Image object with additional helpers
# 
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date September 2015


## Import libraries
import os                       # used to execute terminal commands in python
import SimpleITK as sitk
import numpy as np

## Import modules from src-folder
import Slice as sl

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
    #  \return Stack object including its slices with corresponding masks
    @classmethod
    def from_nifti(cls, dir_input, filename):

        stack = cls()
        # stack = []

        stack.sitk = sitk.ReadImage(dir_input+filename+".nii.gz", sitk.sitkFloat64)
        stack._dir = dir_input
        stack._filename = filename

        stack._N_slices = stack.sitk.GetSize()[-1]
        stack._slices = [None]*stack._N_slices

        ## If mask is provided attach it to Stack
        if os.path.isfile(dir_input+filename+"_mask.nii.gz"):
            stack.sitk_mask = sitk.ReadImage(dir_input+filename+"_mask.nii.gz", sitk.sitkUInt8)

        ## If not: Generate binary mask consisting of ones
        else:
            stack.sitk_mask = stack._generate_binary_mask()

        ## Extract all slices and their masks from the stack and store them 
        stack._slices = stack._extract_slices()

        return stack


    ## Create Stack instance from exisiting sitk.Image instance. Slices are
    #  not extracted and stored separately in the object. The idea is to use
    #  this function when the stack is regarded as entire volume (like the 
    #  reconstructed HR volume). A mask can be added via self.add_mask then.
    #  \param[in] image_sitk sitk.Image created from nifti-file
    #  \param[in] name string containing the chosen name for the stack
    #  \return Stack object without slice information
    @classmethod
    def from_sitk_image(cls, image_sitk, name):
        stack = cls()
        
        stack.sitk = sitk.Image(image_sitk)
        stack._filename = name

        stack._N_slices = stack.sitk.GetSize()[-1]
        stack._slices = [None]*stack._N_slices

        stack.sitk_mask = None


    #         self.sitk = sitk.Image(image_sitk)
    #         self._filename = filename

    #         self._N_slices = None
    #         self._slices = None

    #         self.sitk_mask = None

        return stack


    ## Add a mask to a existing Stack instance with no existing mask yet.
    #  \param[in] image_sitk_mask sitk.Image representing the mask
    def add_mask(self, image_sitk_mask):

        try:
            if self.sitk_mask is None:
                self.sitk_mask = image_sitk_mask

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


    ## Get filename of read/assigned nifti file (Stack.from_nifti vs Stack.from_sitk_image)
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


    ## Burst the stack into its slices and return all slices of the stack
    #  return list of Slice objects
    def _extract_slices(self):

        slices = [None]*self._N_slices

        ## Extract slices and add masks
        for i in range(0, self._N_slices):
            slices[i] = sl.Slice(
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
