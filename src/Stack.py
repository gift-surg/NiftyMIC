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

## In addition to the stack as being stored as sitk.Image the class Stack
#  also contains additional variables helpful to work with the data 
class Stack:

    # The constructor
    # def __init__(self):
    #     self.sitk = None
    #     self.sitk_mask = None
    #     self._dir = None
    #     self._filename = None
    #     self._N_slices = None


    ## Create Stack from file
    #  \param dir_input string to input directory of nifti-file to read
    #  \param filename string of nifti-file to read
    @classmethod
    def from_nifti(cls, dir_input, filename):

        stack = cls()
        # stack = []

        stack.sitk = sitk.ReadImage(dir_input+filename+".nii.gz", sitk.sitkFloat64)
        stack._dir = dir_input
        stack._filename = filename

        stack._N_slices = stack.sitk.GetSize()[-1]
        stack._slices = [None]*stack._N_slices

        ## If mask is provided
        if os.path.isfile(dir_input+filename+"_mask.nii.gz"):
            stack.sitk_mask = sitk.ReadImage(dir_input+filename+"_mask.nii.gz", sitk.sitkUInt8)
            stack._slices_masks = [None]*stack._N_slices

        ## If not: Generate binary mask consisting of ones
        else:
            stack.sitk_mask = stack._generate_binary_mask()

        stack._extract_slices()

        return stack


    ## Create Stack from exisiting sitk.Image instance
    #  \param image_sitk sitk.Image created from nifti-file
    #  \param name string containing the chosen name
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


    ## Burst the stack into its slices and store them
    def _extract_slices(self):
        ## Extract slices and add masks
        if self.sitk_mask is not None:
            for i in range(0, self._N_slices):
                self._slices[i] = sl.Slice(
                    slice_sitk = self.sitk[:,:,i:i+1], 
                    dir_input = self._dir, 
                    filename = self._filename, 
                    slice_number = i,
                    slice_sitk_mask = self.sitk_mask[:,:,i:i+1])
        
        ## No masks available
        else:
            for i in range(0, self._N_slices):
                self._slices[i] = sl.Slice(
                    slice_sitk = self.sitk[:,:,i:i+1], 
                    dir_input = self._dir, 
                    filename = self._filename, 
                    slice_number = i)

        return None


    ## Add a mask to the existing Stack instance
    #  \param image_sitk_mask sitk.Image containing the mask
    def add_mask(self, image_sitk_mask):
        self.sitk_mask = image_sitk_mask
        return None


    ## Get all slices of current stack
    #  \return Array of sitk.Images containing slices in 3D space
    def get_slices(self):
        return self._slices


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
    #  \param show_segmentation display stack with or without associated segmentation (default=0)
    def show(self, show_segmentation=0):
        dir_output = "/tmp/"

        if show_segmentation:
            sitk.WriteImage(self.sitk, dir_output + self._filename + ".nii.gz")
            sitk.WriteImage(self.sitk_mask, dir_output + self._filename + "_mask.nii.gz")

            cmd = "itksnap " \
                    + "-g " + dir_output + self._filename + ".nii.gz " \
                    + "-s " +  dir_output + self._filename + "_mask.nii.gz " + \
                    "& "

        else:
            sitk.WriteImage(self.sitk, dir_output + self._filename + ".nii.gz")

            cmd = "itksnap " \
                    + "-g " + dir_output + self._filename + ".nii.gz " \
                    "& "

        # cmd = "fslview " + dir_output + filename_out + ".nii.gz & "
        os.system(cmd)

        return None

    ## Write the sitk.Image object of Stack.
    #  \param directory string specifying where the output will be written to (default="/tmp/")
    #  \param filename string specifying the filename. If not given the assigned one within Stack will be chosen.
    #  \param write_slices boolean indicating whether each Slice of the stack shall be written (default=False)
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

        return None


    # \return binary_mask consisting of ones
    def _generate_binary_mask(self):
        shape = sitk.GetArrayFromImage(self.sitk).shape
        nda = np.ones(shape, dtype=np.uint8)

        binary_mask = sitk.GetImageFromArray(nda)
        binary_mask.CopyInformation(self.sitk)

        return binary_mask
