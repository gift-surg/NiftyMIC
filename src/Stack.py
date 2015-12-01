## \file Stack.py
#  \brief  
# 
#  \author Michael Ebner
#  \date September 2015


## Import libraries
import os                       # used to execute terminal commands in python
import SimpleITK as sitk
# import numpy as np

## Import modules from src-folder
import Slice as slice

class Stack:

    # def __init__(self, dir_input, filename):
    #     self.sitk = sitk.ReadImage(dir_input+filename+".nii.gz", sitk.sitkFloat64)
    #     self._dir = dir_input
    #     self._filename = filename

    #     self._N_slices = self.sitk.GetSize()[-1]
    #     self._slices = [None]*self._N_slices

    #     self.sitk_mask = None

    #     if os.path.isfile(dir_input+filename+"_mask.nii.gz"):
    #         self.sitk_mask = sitk.ReadImage(dir_input+filename+"_mask.nii.gz", sitk.sitkUInt8)
    #         self._slices_masks = [None]*self._N_slices

    #     self._extract_slices()

    #     return None


    # ## Stack is not read from
    # @classmethod
    # def create_from_sitk_image(cls, image_sitk, filename):
    #     cls.sitk = sitk.Image(image_sitk)
    #     cls._filename = filename

    #     cls.sitk_mask = None

    #     return cls

    # def __getitem__(self, index):
    #     try:
    #         if(abs(index) > self._N_slices-1):
    #             raise ValueError("Error: Slice number %r > number of slices (%r)" %(index, self._N_slices-1))

    #         return self._slices[index].get_slice_sitk()

    #     except ValueError as err:
    #         print(err)


    def __init__(self, *args, **kwargs):

        ## Stack(dir_input, filename)
        if isinstance(args[0], str):
            dir_input = args[0]
            filename = args[1]

            self.sitk = sitk.ReadImage(dir_input+filename+".nii.gz", sitk.sitkFloat64)
            self._dir = dir_input
            self._filename = filename

            self._N_slices = self.sitk.GetSize()[-1]
            self._slices = [None]*self._N_slices

            self.sitk_mask = None

            if os.path.isfile(dir_input+filename+"_mask.nii.gz"):
                self.sitk_mask = sitk.ReadImage(dir_input+filename+"_mask.nii.gz", sitk.sitkUInt8)
                self._slices_masks = [None]*self._N_slices

            self._extract_slices()


        ## Stack(image_sitk, directory, filename)
        else:
            image_sitk = args[0]
            directory = args[1]
            filename = args[2]

            self.sitk = sitk.Image(image_sitk)
            self._dir = directory
            self._filename = filename

            self._N_slices = None
            self._slices = None

            self.sitk_mask = None


        return None


    def _extract_slices(self):
        ## Extract slices and add masks
        if self.sitk_mask is not None:
            for i in range(0, self._N_slices):
                self._slices[i] = slice.Slice(
                    slice_sitk = self.sitk[:,:,i:i+1], 
                    dir_input = self._dir, 
                    filename = self._filename, 
                    slice_number = i,
                    slice_sitk_mask = self.sitk_mask[:,:,i:i+1])
        
        ## No masks available
        else:
            for i in range(0, self._N_slices):
                self._slices[i] = slice.Slice(
                    slice_sitk = self.sitk[:,:,i:i+1], 
                    dir_input = self._dir, 
                    filename = self._filename, 
                    slice_number = i)

        return None


    def add_mask(self, image_sitk_mask):
        self.sitk_mask = image_sitk_mask
        return None


    def get_slices(self):
        return self._slices


    def get_directory(self):
        return self._dir


    def get_filename(self):
        return self._filename


    def get_number_of_slices(self):
        return self._N_slices


    def show_stack(self, display_segmentation=0):
        dir_output = "/tmp/"

        if display_segmentation:
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


    def write_stack(self, directory):
        full_file_name = os.path.join(directory, self._filename + "_" + str(self._slice_number) + ".nii.gz")
        sitk.WriteImage(self.sitk, full_file_name)

        print("Stack %s was successfully written to %s" %(self._filename, directory))

        return None