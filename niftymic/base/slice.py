##
# \file Slice.py
# \brief      { item_description }
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       September 2015
#

import SimpleITK as sitk
import numpy as np
# Import libraries
import os

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh

import niftymic.base.exceptions as exceptions
from niftymic.definitions import DIR_TMP


# In addition to the nifti-image as being stored as sitk.Image for a single
#  3D slice \f$ \in R^3 \times R^3 \times 1\f$ the class Slice
#  also contains additional variables helpful to work with the data


class Slice:

    # Create Slice instance with additional information to actual slice
    #  \param[in] slice_sitk 3D slice in \R x \R x 1, sitk.Image object
    #  \param[in] filename of parent stack, string
    #  \param[in] slice_number number of slice within parent stack, integer
    #  \param[in] slice_sitk_mask associated mask of slice, sitk.Image object (optional)
    @classmethod
    def from_sitk_image(cls, slice_sitk, slice_number, filename="unknown", slice_sitk_mask=None):

        slice = cls()

        # Directory
        # dir_input = "/".join(filename.split("/")[0:-1]) + "/"

        # Filename without extension
        # filename = filename.split("/")[-1:][0].split(".")[0]

        slice._dir_input = None
        slice._filename = filename
        slice._slice_number = slice_number

        # Append stacks as SimpleITK and ITK Image objects
        slice.sitk = slice_sitk
        slice.itk = sitkh.get_itk_from_sitk_image(slice_sitk)

        # Append masks (if provided)
        if slice_sitk_mask is not None:
            slice.sitk_mask = slice_sitk_mask
            slice.itk_mask = sitkh.get_itk_from_sitk_image(slice_sitk_mask)
        else:
            slice.sitk_mask = slice._generate_identity_mask()
            slice.itk_mask = sitkh.get_itk_from_sitk_image(slice.sitk_mask)

        # slice._sitk_upsampled = None

        # HACK (for current Slice-to-Volume Registration)
        #  See class SliceToVolumeRegistration
        # slice._sitk_upsampled = slice._get_upsampled_isotropic_resolution_slice(slice_sitk)
        # slice._itk_upsampled = sitkh.get_itk_from_sitk_image(slice._sitk_upsampled)

        # if slice_sitk_mask is not None:
        #     slice._sitk_mask_upsampled = slice._get_upsampled_isotropic_resolution_slice(slice_sitk_mask)
        #     slice._itk_mask_upsampled = sitkh.get_itk_from_sitk_image(slice._sitk_mask_upsampled)
        # else:
        #     slice._sitk_mask_upsampled = None
        #     slice._itk_mask_upsampled = None

        # Store current affine transform of image
        slice._affine_transform_sitk = sitkh.get_sitk_affine_transform_from_sitk_image(
            slice.sitk)

        # Prepare history of affine transforms, i.e. encoded spatial
        #  position+orientation of slice, and rigid motion estimates of slice
        #  obtained in the course of the registration/reconstruction process
        slice._history_affine_transforms = []
        slice._history_affine_transforms.append(slice._affine_transform_sitk)

        slice._history_motion_corrections = []
        slice._history_motion_corrections.append(sitk.Euler3DTransform())

        return slice

    # Create Stack instance from file and add corresponding mask. Mask is
    #  either provided in the directory or created as binary mask consisting
    #  of ones.
    #  \param[in] dir_input string to input directory of nifti-file to read
    #  \param[in] filename string of nifti-file to read
    #  \param[in] stack_filename filename extension of parent stack, string
    #  \param[in] slice_number number of slice within parent stack, integer
    #  \param[in] suffix_mask extension of slice filename which indicates associated mask
    #  \return Stack object including its slices with corresponding masks
    @classmethod
    def from_filename(cls, file_path, slice_number, file_path_mask=None, verbose=False):

        slice = cls()

        if not ph.file_exists(file_path):
            raise exceptions.FileNotExistent(file_path)

        slice._dir_input = os.path.dirname(file_path)
        slice._filename = os.path.basename(file_path).split(".")[0]
        slice._slice_number = slice_number

        # Append stacks as SimpleITK and ITK Image objects
        slice.sitk = sitk.ReadImage(file_path, sitk.sitkFloat64)
        slice.itk = sitkh.get_itk_from_sitk_image(slice.sitk)

        # Append masks (if provided)
        if file_path_mask is None:
            slice.sitk_mask = slice._generate_identity_mask()
            if verbose:
                ph.print_info(
                    "Identity mask created for '%s'." % (file_path))

        else:
            if not ph.file_exists(file_path_mask):
                raise exceptions.FileNotExistent(file_path_mask)
            slice.sitk_mask = sitk.ReadImage(file_path_mask, sitk.sitkUInt8)

        slice.itk_mask = sitkh.get_itk_from_sitk_image(slice.sitk_mask)

        # Store current affine transform of image
        slice._affine_transform_sitk = sitkh.get_sitk_affine_transform_from_sitk_image(
            slice.sitk)

        # Prepare history of affine transforms, i.e. encoded spatial
        #  position+orientation of slice, and motion estimates of slice
        #  obtained in the course of the registration/reconstruction process
        slice._history_affine_transforms = []
        slice._history_affine_transforms.append(slice._affine_transform_sitk)

        slice._history_motion_corrections = []
        slice._history_motion_corrections.append(sitk.Euler3DTransform())

        return slice

    # Copy constructor
    #  \param[in] slice_to_copy Slice object to be copied
    #  \return copied Slice object
    # TODO: That's not really well done!
    @classmethod
    def from_slice(cls, slice_to_copy):
        slice = cls()

        # Copy image slice and mask
        slice.sitk = sitk.Image(slice_to_copy.sitk)
        slice.itk = sitkh.get_itk_from_sitk_image(slice.sitk)

        slice.sitk_mask = sitk.Image(slice_to_copy.sitk_mask)
        slice.itk_mask = sitkh.get_itk_from_sitk_image(slice.sitk_mask)

        slice._filename = slice_to_copy.get_filename()
        slice._slice_number = slice_to_copy.get_slice_number()
        slice._dir_input = slice_to_copy.get_directory()

        # slice._history_affine_transforms, slice._history_motion_corrections = slice_to_copy.get_registration_history()

        # Store current affine transform of image
        slice._affine_transform_sitk = sitkh.get_sitk_affine_transform_from_sitk_image(
            slice.sitk)

        # Prepare history of affine transforms, i.e. encoded spatial
        #  position+orientation of slice, and rigid motion estimates of slice
        #  obtained in the course of the registration/reconstruction process

        slice._history_affine_transforms, slice._history_motion_corrections = slice_to_copy.get_registration_history()

        return slice

    ##
    #       Motion correction update.
    # \date       2016-09-21 00:50:08+0100
    #
    # Update motion correction of slice and update its position in physical
    # space accordingly.
    #
    # \param      self                   The object
    # \param[in]  affine_transform_sitk  transform as sitk.AffineTransform
    #                                    object
    # \post       origin and direction of slice gets updated based on transform
    #
    def update_motion_correction(self, affine_transform_sitk):

        # Update rigid motion estimate
        current_rigid_motion_estimate = sitkh.get_composite_sitk_affine_transform(
            affine_transform_sitk, self._history_motion_corrections[-1])
        self._history_motion_corrections.append(current_rigid_motion_estimate)

        # New affine transform of slice after rigid motion correction
        affine_transform = sitkh.get_composite_sitk_affine_transform(
            affine_transform_sitk, self._affine_transform_sitk)

        # Update affine transform of slice, i.e. change image origin and
        # direction in physical space
        self._update_affine_transform(affine_transform)

    # ## Update rigid motion estimate of slice and update its position in
    # #  physical space accordingly.
    # #  \param[in] rigid_transform_sitk rigid transform as sitk object
    # #  \post origin and direction of slice gets updated based on rigid transform
    # def update_rigid_motion_estimate(self, rigid_transform_sitk):

    #     ## Update rigid motion estimate
    #     current_rigid_motion_estimate = sitkh.get_composite_sitk_euler_transform(rigid_transform_sitk, self._history_rigid_motion_estimates[-1])
    #     self._history_rigid_motion_estimates.append(current_rigid_motion_estimate)

    #     ## New affine transform of slice after rigid motion correction
    #     affine_transform = sitkh.get_composite_sitk_affine_transform(rigid_transform_sitk, self._affine_transform_sitk)

    #     ## Update affine transform of slice, i.e. change image origin and direction in physical space
    #     self._update_affine_transform(affine_transform)

    # Get filename of slice, e.g. name of parent stack
    #  \return filename, string
    def get_filename(self):
        return self._filename

    # Get number of slice within parent stack
    #  \return slice number, integer
    def get_slice_number(self):
        return self._slice_number

    # Get directory where parent stack is stored
    #  \return directory, string
    def get_directory(self):
        return self._dir_input

    # Get current affine transformation defining the spatial position in
    #  physical space of slice
    #  \return affine transformation, sitk.AffineTransform object
    def get_affine_transform(self):
        return self._affine_transform_sitk

    ##
    # Get applied motion correction transform to slice
    # \date       2017-08-08 13:16:28+0100
    #
    # \param      self  The object
    #
    # \return     The motion correction transform.
    #
    def get_motion_correction_transform(self):
        return self._history_motion_corrections[-1]

    # Get history history of affine transforms, i.e. encoded spatial
    #  position+orientation of slice, and rigid motion estimates of slice
    #  obtained in the course of the registration/reconstruction process
    #  \return list of sitk.AffineTransform and sitk.Euler3DTransform objects
    def get_registration_history(self):
        affine_transforms = list(self._history_affine_transforms)
        motion_corrections = list(self._history_motion_corrections)
        return affine_transforms, motion_corrections

    # Display slice with external viewer (ITK-Snap)
    #  \param[in] show_segmentation display slice with or without associated segmentation (default=0)
    def show(self, show_segmentation=0, dir_output=DIR_TMP):

        filename_out = self._filename + "_" + str(self._slice_number)
        if show_segmentation:
            sitkh.write_nifti_image_sitk(
                self.sitk, dir_output + filename_out + ".nii.gz")
            sitkh.write_nifti_image_sitk(self.sitk_mask, dir_output +
                                      filename_out + "_mask.nii.gz")

            cmd = "itksnap " \
                + "-g " + dir_output + filename_out + ".nii.gz " \
                + "-s " +  dir_output + filename_out + "_mask.nii.gz " + \
                "& "

        else:
            sitkh.write_nifti_image_sitk(
                self.sitk, dir_output + filename_out + ".nii.gz")

            cmd = "itksnap " \
                + "-g " + dir_output + filename_out + ".nii.gz " \
                "& "

        # cmd = "fslview " + dir_output + filename_out + ".nii.gz & "
        os.system(cmd)

    # Write information of Slice to HDD to given diretory:
    #  - sitk.Image object of slice
    #  - affine transformation describing physical space position of slice
    #  \param[in] directory string specifying where the output will be written to (default="/tmp/")
    #  \param[in] filename string specifyig the filename. If not given, filename of parent stack is used
    def write(self, directory, filename=None, write_transform=False, suffix_mask="_mask", prefix_slice="_slice"):

        # Create directory if not existing
        ph.create_directory(directory)

        # Construct filename
        if filename is None:
            filename_out = self._filename + \
                prefix_slice + str(self._slice_number)
        else:
            filename_out = filename + prefix_slice + str(self._slice_number)

        full_file_name = os.path.join(directory, filename_out)

        # Write slice and affine transform
        sitkh.write_nifti_image_sitk(self.sitk, full_file_name + ".nii.gz")
        if write_transform:
            sitk.WriteTransform(
                # self.get_affine_transform(),
                self.get_motion_correction_transform(),
                full_file_name + ".tfm")

        # Write mask to specified location if given
        if self.sitk_mask is not None:
            nda = sitk.GetArrayFromImage(self.sitk_mask)

            # Write mask if it does not consist of only ones
            if not np.all(nda):
                sitkh.write_nifti_image_sitk(self.sitk_mask, full_file_name +
                                          "%s.nii.gz" % (suffix_mask))

        # print("Slice %r of stack %s was successfully written to %s" %(self._slice_number, self._filename, full_file_name))
        # print("Transformation of slice %r of stack %s was successfully written to %s" %(self._slice_number, self._filename, full_file_name))

    # Update slice with new affine transform, specifying updated spatial
    #  position of slice in physical space. The transform is obtained via
    #  slice-to-volume registration step, e.g.
    #  \param[in] affine_transform_sitk affine transform as sitk-object
    def _update_affine_transform(self, affine_transform_sitk):

        # Ensure correct object type
        self._affine_transform_sitk = sitk.AffineTransform(
            affine_transform_sitk)

        # Append transform to registration history
        self._history_affine_transforms.append(affine_transform_sitk)

        # Get origin and direction of transformed 3D slice given the new
        # spatial transform
        origin = sitkh.get_sitk_image_origin_from_sitk_affine_transform(
            affine_transform_sitk, self.sitk)
        direction = sitkh.get_sitk_image_direction_from_sitk_affine_transform(
            affine_transform_sitk, self.sitk)

        # Update image objects
        self.sitk.SetOrigin(origin)
        self.sitk.SetDirection(direction)

        self.itk.SetOrigin(origin)
        self.itk.SetDirection(sitkh.get_itk_from_sitk_direction(direction))

        # Update image mask objects
        if self.sitk_mask is not None:
            self.sitk_mask.SetOrigin(origin)
            self.sitk_mask.SetDirection(direction)

            self.itk_mask.SetOrigin(origin)
            self.itk_mask.SetDirection(
                sitkh.get_itk_from_sitk_direction(direction))

    # ## Upsample slices in k-direction to in-plane resolution.
    # #  \param[in] slice_sitk slice as sitk.Image object to be upsampled
    # #  \return upsampled slice as sitk.Image object
    # #  \warning only used for Slice-to-Volume Registration and shall me removed at some point
    # def _get_upsampled_isotropic_resolution_slice(self, slice_sitk):

    #     ## Fetch info used for upsampling
    #     spacing = np.array(slice_sitk.GetSpacing())
    #     size = np.array(slice_sitk.GetSize())

    #     ## Set dimension of each slice in k-direction accordingly
    #     size[2] = np.round(spacing[2]/spacing[0])

    #     ## Update spacing in k-direction to be equal to in-plane spacing
    #     spacing[2] = spacing[0]

    #     ## Upsample slice to isotropic resolution
    #     default_pixel_value = 0

    #     slice_upsampled_sitk = sitk.Resample(
    #         slice_sitk,
    #         size,
    #         sitk.Euler3DTransform(),
    #         sitk.sitkNearestNeighbor,
    #         slice_sitk.GetOrigin(),
    #         spacing,
    #         slice_sitk.GetDirection(),
    #         default_pixel_value,
    #         slice_sitk.GetPixelIDValue())

    #     return slice_upsampled_sitk

    # Create a binary mask consisting of ones
    #  \return binary_mask as sitk.Image object consisting of ones
    def _generate_identity_mask(self):
        shape = sitk.GetArrayFromImage(self.sitk).shape
        nda = np.ones(shape, dtype=np.uint8)

        binary_mask = sitk.GetImageFromArray(nda)
        binary_mask.CopyInformation(self.sitk)

        return binary_mask
