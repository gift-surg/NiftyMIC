## \file Slice.py
#  \brief  
# 
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date September 2015

## Import libraries
import os                       # used to execute terminal commands in python
import SimpleITK as sitk
import SimpleITKHelper as sitkh
import numpy as np


class Slice:

    def __init__(self, slice_sitk, dir_input, filename, slice_number, slice_sitk_mask = None):
        self.sitk = slice_sitk
        self.sitk_mask = slice_sitk_mask
        self._dir_input = dir_input
        self._filename = filename
        self._slice_number = slice_number

        self._affine_transform_sitk = sitkh.get_sitk_affine_transform_from_sitk_image(self.sitk)

        self._registration_history_sitk = []
        self._registration_history_sitk.append(self._affine_transform_sitk)

        ## Get transform to align original (!) stack with axes of physical coordinate system
        self._T_PP = sitkh.get_3D_transform_to_align_stack_with_physical_coordinate_system(self.sitk)

        ## HACK (for current slice-to-volume registration)
        self._sitk_upsampled = None
        self._sitk_mask_upsampled = None

        return None


    def set_affine_transform(self, affine_transform_sitk):
        self._affine_transform_sitk = sitk.AffineTransform(affine_transform_sitk)

        self._registration_history_sitk.append(affine_transform_sitk)

        origin = sitkh.get_sitk_image_origin_from_sitk_affine_transform(affine_transform_sitk, self.sitk)
        direction = sitkh.get_sitk_image_direction_matrix_from_sitk_affine_transform(affine_transform_sitk, self.sitk)

        ## Update origin and direction of 3D sitk slice
        self.sitk.SetOrigin(origin)
        self.sitk.SetDirection(direction)

        ## Update origin and direction of 3D sitk slice mask
        if self.sitk_mask is not None:
            self.sitk_mask.SetOrigin(origin)
            self.sitk_mask.SetDirection(direction)

        ## HACK (for current slice-to-volume registration)
        if self._sitk_upsampled is not None:
            self._sitk_upsampled.SetOrigin(origin)
            self._sitk_upsampled.SetDirection(direction)

            self._sitk_mask_upsampled.SetOrigin(origin)
            self._sitk_mask_upsampled.SetDirection(direction)

        return None


    def get_slice_sitk(self):
        return self.sitk


    def get_filename(self):
        return self._filename


    def get_slice_number(self):
        return self._slice_number


    def get_directory(self):
        return self._dir_input


    def get_affine_transform(self):
        return self._affine_transform_sitk


    def get_registration_history(self):
        return self._registration_history_sitk


    def get_transform_to_align_with_physical_coordinate_system(self):
        return self._T_PP


    def write(self, directory="/tmp/", filename=None):
        if filename is None:
            filename_out = self._filename + "_" + str(self._slice_number)
        else:
            filename_out = filename + "_" + str(self._slice_number)
            

        ## Write slice:
        full_file_name = os.path.join(directory, filename_out + ".nii.gz")
        sitk.WriteImage(self.sitk, full_file_name)
        print("Slice %r of stack %s was successfully written to %s" %(self._slice_number, self._filename, full_file_name))

        ## Write transformation:
        full_file_name = os.path.join(directory, filename_out + ".tfm")
        sitk.WriteTransform(self._affine_transform_sitk, full_file_name)
        print("Transformation of slice %r of stack %s was successfully written to %s" %(self._slice_number, self._filename, full_file_name))


        return None