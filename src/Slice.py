## \file Slice.py
#  \brief  
# 
#  \author Michael Ebner
#  \date September 2015

## Import libraries
import os                       # used to execute terminal commands in python
import SimpleITK as sitk
import numpy as np


class Slice:

    def __init__(self, slice_sitk, dir_input, filename, slice_number):
        self.sitk = slice_sitk
        self._dir_input = dir_input
        self._filename = filename
        self._slice_number = slice_number
        self._affine_transform = sitk.AffineTransform(np.eye(3).flatten(),(0,0,0))

        return None


    def get_slice_sitk(self):
        return self.sitk


    def set_affine_transform(self, affine_transform):
        self._affine_transform = affine_transform
        return None


    def get_affine_transform(self):
        return self._affine_transform


    def write_results(self, directory):
        ## Write slice:
        full_file_name = os.path.join(directory, self._filename + "_" + str(self._slice_number) + ".nii.gz")
        sitk.WriteImage(self.sitk, full_file_name)

        ## Write transformation:
        full_file_name = os.path.join(directory, self._filename + "_" + str(self._slice_number) + ".tfm")
        sitk.WriteTransform(self._affine_transform, full_file_name)

        print("Slice %r of stack %s was successfully written to %s" %(self._slice_number, self._filename, directory))