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


## In addition to the nifti-image as being stored as sitk.Image for a single
#  3D slice \f$ \in R^3 \times R^3 \times 1\f$ the class Slice
#  also contains additional variables helpful to work with the data
class Slice:

    ## Create Slice instance with additional information to actual slice
    #  \param[in] slice_sitk 3D slice in \R x \R x 1, sitk.Image object
    #  \param[in] dir_input directory where parent stack is stored, string
    #  \param[in] filename of parent stack, string
    #  \param[in] slice_number number of slice within parent stack, integer
    #  \param[in] slice_sitk_mask associated mask of slice, sitk.Image object (optional)
    def __init__(self, slice_sitk, dir_input, filename, slice_number, slice_sitk_mask = None):
        self.sitk = slice_sitk
        self.sitk_mask = slice_sitk_mask
        self._dir_input = dir_input
        self._filename = filename
        self._slice_number = slice_number

        ## Store current affine transform of image
        self._affine_transform_sitk = sitkh.get_sitk_affine_transform_from_sitk_image(self.sitk)

        ## Prepare history of spatial transformations in the course of the 
        #  registration/reconstruction process
        self._registration_history_sitk = []
        self._registration_history_sitk.append(self._affine_transform_sitk)

        ## Get transform to align original (!) stack with axes of physical coordinate system
        #  Used for in-plane registration, i.e. class InPlaneRigidRegistration
        self._T_PP = sitkh.get_3D_transform_to_align_stack_with_physical_coordinate_system(self.sitk)

        ## HACK (for current slice-to-volume registration)
        #  See class SliceToVolumeRegistration
        self._sitk_upsampled = None
        self._sitk_mask_upsampled = None


    ## Update slice with new affine transform, specifying updated spatial
    #  position of slice in physical space. The transform is obtained via
    #  slice-to-volume registration step, e.g.
    #  \param[in] affine_transform_sitk affine transform as sitk-object
    def set_affine_transform(self, affine_transform_sitk):

        ## Ensure correct object type
        self._affine_transform_sitk = sitk.AffineTransform(affine_transform_sitk)

        ## Append transform to registration history
        self._registration_history_sitk.append(affine_transform_sitk)

        ## Update origin and direction of 3D slice given the new spatial transform
        origin = sitkh.get_sitk_image_origin_from_sitk_affine_transform(affine_transform_sitk, self.sitk)
        direction = sitkh.get_sitk_image_direction_matrix_from_sitk_affine_transform(affine_transform_sitk, self.sitk)

        self.sitk.SetOrigin(origin)
        self.sitk.SetDirection(direction)

        ## Update origin and direction of 3D slice mask
        if self.sitk_mask is not None:
            self.sitk_mask.SetOrigin(origin)
            self.sitk_mask.SetDirection(direction)

        ## HACK (for current slice-to-volume registration)
        #  See class SliceToVolumeRegistration
        if self._sitk_upsampled is not None:
            self._sitk_upsampled.SetOrigin(origin)
            self._sitk_upsampled.SetDirection(direction)

            self._sitk_mask_upsampled.SetOrigin(origin)
            self._sitk_mask_upsampled.SetDirection(direction)


    ## Get filename of slice, e.g. name of parent stack
    #  \return filename, string
    def get_filename(self):
        return self._filename


    ## Get number of slice within parent stack
    #  \return slice number, integer
    def get_slice_number(self):
        return self._slice_number


    ## Get directory where parent stack is stored
    #  \return directory, string
    def get_directory(self):
        return self._dir_input


    ## Get current affine transformation defining the spatial position in 
    #  physical space of slice
    #  \return affine transformation, sitk.AffineTransform object
    def get_affine_transform(self):
        return self._affine_transform_sitk


    ## Get history of registrations the slice underwent in the course of
    #  registration/reconstruction steps
    #  \return list of sitk.AffineTransform objects
    def get_registration_history(self):
        return self._registration_history_sitk


    ## Get transform to align original (!) stack with axes of physical coordinate system
    #  Used for in-plane registration, i.e. class InPlaneRigidRegistration
    def get_transform_to_align_with_physical_coordinate_system(self):
        return self._T_PP


    ## Display slice with external viewer (ITK-Snap)
    #  \param[in] show_segmentation display slice with or without associated segmentation (default=0)
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


    ## Write information of Slice to HDD to given diretory:
    #  - sitk.Image object of slice
    #  - affine transformation describing physical space position of slice
    #  \param[in] directory string specifying where the output will be written to (default="/tmp/")
    #  \param[in] filename string specifyig the filename. If not given, filename of parent stack is used
    def write(self, directory="/tmp/", filename=None):
        if filename is None:
            filename_out = self._filename + "_" + str(self._slice_number)
        else:
            filename_out = filename + "_" + str(self._slice_number)

        ## Write slice:
        full_file_name = os.path.join(directory, filename_out + ".nii.gz")
        sitk.WriteImage(self.sitk, full_file_name)
        # print("Slice %r of stack %s was successfully written to %s" %(self._slice_number, self._filename, full_file_name))

        ## Write transformation:
        full_file_name = os.path.join(directory, filename_out + ".tfm")
        sitk.WriteTransform(self._affine_transform_sitk, full_file_name)
        # print("Transformation of slice %r of stack %s was successfully written to %s" %(self._slice_number, self._filename, full_file_name))

