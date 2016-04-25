## \file Slice.py
#  \brief  
# 
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date September 2015


## Import libraries
import os                       # used to execute terminal commands in python
import itk
import SimpleITK as sitk
import numpy as np

## Import modules from src-folder
import SimpleITKHelper as sitkh

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
    @classmethod
    def from_sitk_image(cls, slice_sitk, dir_input, filename, slice_number, slice_sitk_mask = None):

        slice = cls()

        slice._dir_input = dir_input
        slice._filename = filename
        slice._slice_number = slice_number

        ## Append stacks as SimpleITK and ITK Image objects
        slice.sitk = slice_sitk
        slice.itk = sitkh.convert_sitk_to_itk_image(slice_sitk)

        ## Append masks (if provided)
        if slice_sitk_mask is not None:
            slice.sitk_mask = slice_sitk_mask
            slice.itk_mask = sitkh.convert_sitk_to_itk_image(slice_sitk_mask)
        else:
            slice.sitk_mask = None
            slice.itk_mask = None

        slice._sitk_upsampled = None

        ## HACK (for current Slice-to-Volume Registration)
        #  See class SliceToVolumeRegistration
        # slice._sitk_upsampled = slice._get_upsampled_isotropic_resolution_slice(slice_sitk)
        # slice._itk_upsampled = sitkh.convert_sitk_to_itk_image(slice._sitk_upsampled)

        # if slice_sitk_mask is not None:
        #     slice._sitk_mask_upsampled = slice._get_upsampled_isotropic_resolution_slice(slice_sitk_mask)
        #     slice._itk_mask_upsampled = sitkh.convert_sitk_to_itk_image(slice._sitk_mask_upsampled)
        # else:
        #     slice._sitk_mask_upsampled = None
        #     slice._itk_mask_upsampled = None

        ## Store current affine transform of image
        slice._affine_transform_sitk = sitkh.get_sitk_affine_transform_from_sitk_image(slice.sitk)

        ## Prepare history of spatial transformations in the course of the 
        #  registration/reconstruction process
        slice._registration_history_sitk = []
        slice._registration_history_sitk.append(slice._affine_transform_sitk)

        ## Get transform to align original (!) stack with axes of physical coordinate system
        #  Used for in-plane registration, i.e. class InPlaneRigidRegistration
        slice._T_PP = sitkh.get_3D_transform_to_align_stack_with_physical_coordinate_system(slice.sitk)

        return slice


    ## Copy constructor
    #  \param[in] slice_to_copy Slice object to be copied
    #  \return copied Slice object
    # TODO: That's not really well done!
    @classmethod
    def from_slice(cls, slice_to_copy):
        slice = cls()
        
        ## Copy image slice and mask
        slice.sitk = sitk.Image(slice_to_copy.sitk)
        slice.itk = sitkh.convert_sitk_to_itk_image(slice.sitk)

        slice.sitk_mask = sitk.Image(slice_to_copy.sitk_mask)
        slice.itk_mask = sitkh.convert_sitk_to_itk_image(slice.sitk_mask)
        
        slice._filename = "copy_" + slice_to_copy.get_filename()
        slice._slice_number = slice_to_copy.get_slice_number()

        return slice


    ## Update slice with new affine transform, specifying updated spatial
    #  position of slice in physical space. The transform is obtained via
    #  slice-to-volume registration step, e.g.
    #  \param[in] affine_transform_sitk affine transform as sitk-object
    def update_affine_transform(self, affine_transform_sitk):

        ## Ensure correct object type
        self._affine_transform_sitk = sitk.AffineTransform(affine_transform_sitk)

        ## Append transform to registration history
        self._registration_history_sitk.append(affine_transform_sitk)

        ## Get origin and direction of transformed 3D slice given the new spatial transform
        origin = sitkh.get_sitk_image_origin_from_sitk_affine_transform(affine_transform_sitk, self.sitk)
        direction = sitkh.get_sitk_image_direction_matrix_from_sitk_affine_transform(affine_transform_sitk, self.sitk)

        ## Update image objects
        self.sitk.SetOrigin(origin)
        self.sitk.SetDirection(direction)

        self.itk.SetOrigin(origin)
        self.itk.SetDirection(sitkh.get_itk_direction_form_sitk_direction(direction))

        ## Update image mask objects
        if self.sitk_mask is not None:
            self.sitk_mask.SetOrigin(origin)
            self.sitk_mask.SetDirection(direction)

            self.itk_mask.SetOrigin(origin)
            self.itk_mask.SetDirection(sitkh.get_itk_direction_form_sitk_direction(direction))

        ## HACK (for current slice-to-volume registration)
        #  See class SliceToVolumeRegistration
        if self._sitk_upsampled is not None:
            self._sitk_upsampled.SetOrigin(origin)
            self._sitk_upsampled.SetDirection(direction)

            self._itk_upsampled.SetOrigin(origin)
            self._itk_upsampled.SetDirection(sitkh.get_itk_direction_form_sitk_direction(direction))

            if self.sitk_mask is not None:
                self._sitk_mask_upsampled.SetOrigin(origin)
                self._sitk_mask_upsampled.SetDirection(direction)

                self._itk_mask_upsampled.SetOrigin(origin)
                self._itk_mask_upsampled.SetDirection(sitkh.get_itk_direction_form_sitk_direction(direction))


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

        filename_out = self._filename + "_" + str(self._slice_number)
        if show_segmentation:
            sitk.WriteImage(self.sitk, dir_output + filename_out + ".nii.gz")
            sitk.WriteImage(self.sitk_mask, dir_output + filename_out + "_mask.nii.gz")

            cmd = "itksnap " \
                    + "-g " + dir_output + filename_out + ".nii.gz " \
                    + "-s " +  dir_output + filename_out + "_mask.nii.gz " + \
                    "& "

        else:
            sitk.WriteImage(self.sitk, dir_output + filename_out + ".nii.gz")

            cmd = "itksnap " \
                    + "-g " + dir_output + filename_out + ".nii.gz " \
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
            filename_out = filename + "_" + str(self._slice_number)
        else:
            filename_out = filename + "_" + str(self._slice_number)

        ## Define filename
        full_file_name = os.path.join(directory, filename_out)

        ## Write slice with mask and affine transform
        sitk.WriteImage(self.sitk, full_file_name + ".nii.gz")
        sitk.WriteImage(self.sitk_mask, full_file_name + "_mask.nii.gz")
        sitk.WriteTransform(self._affine_transform_sitk, full_file_name + ".tfm")

        # print("Slice %r of stack %s was successfully written to %s" %(self._slice_number, self._filename, full_file_name))
        # print("Transformation of slice %r of stack %s was successfully written to %s" %(self._slice_number, self._filename, full_file_name))


    ## Upsample slices in k-direction to in-plane resolution.
    #  \param[in] slice_sitk slice as sitk.Image object to be upsampled
    #  \return upsampled slice as sitk.Image object
    #  \warning only used for Slice-to-Volume Registration and shall me removed at some point
    def _get_upsampled_isotropic_resolution_slice(self, slice_sitk):
    
        ## Fetch info used for upsampling
        spacing = np.array(slice_sitk.GetSpacing())
        size = np.array(slice_sitk.GetSize())

        ## Set dimension of each slice in k-direction accordingly
        size[2] = np.round(spacing[2]/spacing[0])

        ## Update spacing in k-direction to be equal to in-plane spacing
        spacing[2] = spacing[0]

        ## Upsample slice to isotropic resolution
        default_pixel_value = 0

        slice_upsampled_sitk = sitk.Resample(
            slice_sitk, 
            size, 
            sitk.Euler3DTransform(), 
            sitk.sitkNearestNeighbor, 
            slice_sitk.GetOrigin(), 
            spacing, 
            slice_sitk.GetDirection(), 
            default_pixel_value,
            slice_sitk.GetPixelIDValue())

        return slice_upsampled_sitk

