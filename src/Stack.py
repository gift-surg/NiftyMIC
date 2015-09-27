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

    def __init__(self, dir_input, filename):
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

        return None


    # def __getitem__(self, index):
    #     try:
    #         if(abs(index) > self._N_slices-1):
    #             raise ValueError("Error: Slice number %r > number of slices (%r)" %(index, self._N_slices-1))

    #         return self._slices[index].get_slice_sitk()

    #     except ValueError as err:
    #         print(err)


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


    def get_slices(self):
        return self._slices


    def get_directory(self):
        return self._dir


    def get_filename(self):
        return self._filename


    def get_number_of_slices(self):
        return self._N_slices