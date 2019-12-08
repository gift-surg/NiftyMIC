##
# \file stack.py
# \brief      { item_description }
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       September 2015
#


import os
import re
import numpy as np
import SimpleITK as sitk

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh
import simplereg.resampler

import niftymic.base.slice as sl
import niftymic.base.exceptions as exceptions
import niftymic.base.data_writer as dw

from niftymic.definitions import ALLOWED_EXTENSIONS, VIEWER


##
# In addition to the nifti-image (stored as sitk.Image object) this class Stack
# also contains additional variables helpful to work with the data.
#
class Stack:

    def __init__(self):
        self._is_unity_mask = True
        self._deleted_slices = []
        self._history_affine_transforms = []
        self._history_motion_corrections = []

    ##
    # Create Stack instance from file and add corresponding mask. Mask is
    # either provided in the directory or created as binary mask consisting of
    # ones.
    # \param[in]  dir_input    string to input directory of nifti-file to read
    # \param[in]  filename     string of nifti-file to read
    # \param[in]  suffix_mask  extension of stack filename which indicates
    #                          associated mask
    # \return     Stack object including its slices with corresponding masks
    #
    @classmethod
    def from_filename(cls,
                      file_path,
                      file_path_mask=None,
                      extract_slices=True,
                      verbose=False,
                      slice_thickness=None,
                      ):

        stack = cls()

        if not ph.file_exists(file_path):
            raise exceptions.FileNotExistent(file_path)

            path_to_directory = os.path.dirname(file_path)

        # Strip extension from filename and remove potentially included "."
        filename = [re.sub("." + ext, "", os.path.basename(file_path))
                    for ext in ALLOWED_EXTENSIONS
                    if file_path.endswith(ext)][0]
        # filename = filename.replace(".", "p")

        stack._dir = os.path.dirname(file_path)
        stack._filename = filename

        # Append stacks as SimpleITK and ITK Image objects
        stack.sitk = sitkh.read_nifti_image_sitk(file_path, sitk.sitkFloat64)
        stack.itk = sitkh.get_itk_from_sitk_image(stack.sitk)

        # Set slice thickness of acquisition
        if slice_thickness is None:
            stack._slice_thickness = stack.sitk.GetSpacing()[-1]
        else:
            stack._slice_thickness = slice_thickness

        # Append masks (either provided or binary mask)
        if file_path_mask is None:
            stack.sitk_mask = stack._generate_identity_mask()
            if verbose:
                ph.print_info(
                    "Identity mask created for '%s'." % (file_path))

        else:
            if not ph.file_exists(file_path_mask):
                raise exceptions.FileNotExistent(file_path_mask)
            stack.sitk_mask = sitkh.read_nifti_image_sitk(
                file_path_mask, sitk.sitkUInt8)
            try:
                # ensure masks occupy same physical space
                stack.sitk_mask.CopyInformation(stack.sitk)
            except RuntimeError as e:
                raise IOError(
                    "Given image and its mask do not occupy the same space: %s" %
                    e.message)
            stack._is_unity_mask = False

        # Append itk object
        stack.itk_mask = sitkh.get_itk_from_sitk_image(stack.sitk_mask)

        # Store current affine transform of image
        stack._affine_transform_sitk = sitkh.get_sitk_affine_transform_from_sitk_image(
            stack.sitk)

        # Prepare history of affine transforms, i.e. encoded spatial
        #  position+orientation of stack, and motion estimates of stack
        #  obtained in the course of the registration/reconstruction process
        stack._history_affine_transforms = []
        stack._history_affine_transforms.append(stack._affine_transform_sitk)

        stack._history_motion_corrections = []
        stack._history_motion_corrections.append(sitk.Euler3DTransform())

        # Extract all slices and their masks from the stack and store them
        if extract_slices:
            dimenson = stack.sitk.GetDimension()
            if dimenson == 3:
                stack._N_slices = stack.sitk.GetSize()[-1]
                stack._slices = stack._extract_slices(
                    slice_thickness=stack.get_slice_thickness())
            elif dimenson == 2:
                stack._N_slices = 1
                stack._slices = [stack.sitk[:, :]]
        else:
            stack._N_slices = 0
            stack._slices = None

        if verbose:
            ph.print_info(
                "Stack (image + mask) associated to '%s' successfully read." %
                (file_path))

        return stack

    ##
    # Create Stack instance from stack slices in specified directory and add
    # corresponding mask.
    # \date       2017-08-15 19:18:56+0100
    #
    # \param      cls                  The cls
    # \param[in]  dir_input            string to input directory where bundle
    #                                  of slices are stored
    # \param[in]  prefix_stack         prefix indicating the corresponding
    #                                  stack
    # \param[in]  suffix_mask          extension of stack filename which
    #                                  indicates associated mask
    # \param      dic_slice_filenames  Dictionary linking slice number (int)
    #                                  with filename (without extension)
    # \param      prefix_slice         The prefix slice
    #
    # \return     Stack object including its slices with corresponding masks
    # \example    mask (suffix_mask) of slice j of stack i (prefix_stack)
    # reads: i_slicej_mask.nii.gz
    #
    # TODO: Code cleaning
    @classmethod
    def from_slice_filenames(cls,
                             dir_input,
                             prefix_stack,
                             suffix_mask=None,
                             dic_slice_filenames=None,
                             prefix_slice="_slice",
                             slice_thickness=None,
                             ):

        stack = cls()

        if dir_input[-1] is not "/":
            dir_input += "/"

        stack._dir = dir_input
        stack._filename = prefix_stack

        # Get 3D images
        stack.sitk = sitkh.read_nifti_image_sitk(
            dir_input + prefix_stack + ".nii.gz", sitk.sitkFloat64)
        stack.itk = sitkh.get_itk_from_sitk_image(stack.sitk)

        # Store current affine transform of image
        stack._affine_transform_sitk = sitkh.get_sitk_affine_transform_from_sitk_image(
            stack.sitk)

        # Prepare history of affine transforms, i.e. encoded spatial
        #  position+orientation of stack, and motion estimates of stack
        #  obtained in the course of the registration/reconstruction process
        stack._history_affine_transforms = []
        stack._history_affine_transforms.append(stack._affine_transform_sitk)

        stack._history_motion_corrections = []
        stack._history_motion_corrections.append(sitk.Euler3DTransform())

        # Set slice thickness of acquisition
        if slice_thickness is None:
            stack._slice_thickness = float(stack.sitk.GetSpacing()[-1])
        else:
            stack._slice_thickness = float(slice_thickness)

        # Append masks (either provided or binary mask)
        if suffix_mask is not None and \
            os.path.isfile(dir_input +
                           prefix_stack + suffix_mask + ".nii.gz"):
            stack.sitk_mask = sitkh.read_nifti_image_sitk(
                dir_input + prefix_stack + suffix_mask + ".nii.gz",
                sitk.sitkUInt8)
            stack.itk_mask = sitkh.get_itk_from_sitk_image(stack.sitk_mask)
            stack._is_unity_mask = False
        else:
            stack.sitk_mask = stack._generate_identity_mask()
            stack.itk_mask = sitkh.get_itk_from_sitk_image(stack.sitk_mask)
            stack._is_unity_mask = True

        # Get slices
        if dic_slice_filenames is None:
            stack._N_slices = stack.sitk.GetDepth()
            stack._slices = [None] * stack._N_slices

            # Append slices as Slice objects
            for i in range(0, stack._N_slices):
                path_to_slice = os.path.join(
                    dir_input,
                    prefix_stack + prefix_slice + str(i) + ".nii.gz")
                path_to_slice_mask = os.path.join(
                    dir_input,
                    prefix_stack + prefix_slice + str(i) + suffix_mask + ".nii.gz")

                if ph.file_exists(path_to_slice_mask):
                    stack._slices[i] = sl.Slice.from_filename(
                        file_path=path_to_slice,
                        slice_number=i,
                        file_path_mask=path_to_slice_mask)
                else:
                    stack._slices[i] = sl.Slice.from_filename(
                        file_path=path_to_slice,
                        slice_number=i)
        else:
            slice_numbers = sorted(dic_slice_filenames.keys())
            stack._N_slices = len(slice_numbers)
            stack._slices = [None] * stack._N_slices

            for i, slice_number in enumerate(slice_numbers):
                path_to_slice = os.path.join(
                    dir_input,
                    dic_slice_filenames[slice_number] + ".nii.gz")
                path_to_slice_mask = os.path.join(
                    dir_input, dic_slice_filenames[slice_number] + suffix_mask + ".nii.gz")

                if ph.file_exists(path_to_slice_mask):
                    stack._slices[i] = sl.Slice.from_filename(
                        file_path=path_to_slice,
                        slice_number=slice_number,
                        file_path_mask=path_to_slice_mask,
                        slice_thickness=stack.get_slice_thickness(),
                    )
                else:
                    stack._slices[i] = sl.Slice.from_filename(
                        file_path=path_to_slice,
                        slice_number=slice_number,
                        slice_thickness=stack.get_slice_thickness(),
                    )

        return stack

    # Create Stack instance from exisiting sitk.Image instance. Slices are
    #  not extracted and stored separately in the object. The idea is to use
    #  this function when the stack is regarded as entire volume (like the
    #  reconstructed HR volume).
    #  \param[in] image_sitk sitk.Image created from nifti-file
    #  \param[in] name string containing the chosen name for the stack
    #  \param[in] image_sitk_mask associated mask of stack, sitk.Image object (optional)
    #  \return Stack object without slice information
    @classmethod
    def from_sitk_image(cls,
                        image_sitk,
                        slice_thickness,
                        filename="unknown",
                        image_sitk_mask=None,
                        extract_slices=True,
                        slice_numbers=None,
                        ):
        stack = cls()

        # Explicit cast (+ creation of other image instance)
        stack.sitk = sitk.Cast(image_sitk, sitk.sitkFloat64)
        stack.itk = sitkh.get_itk_from_sitk_image(stack.sitk)

        # Set slice thickness of acquisition
        if not ph.is_float(slice_thickness):
            raise ValueError("Slice thickness must be of type float")
        stack._slice_thickness = float(slice_thickness)

        stack._filename = filename
        stack._dir = None

        # Append masks (if provided)
        if image_sitk_mask is not None:
            stack.sitk_mask = sitk.Cast(image_sitk_mask, sitk.sitkUInt8)
            try:
                # ensure mask occupies the same physical space
                stack.sitk_mask.CopyInformation(stack.sitk)
            except RuntimeError as e:
                raise IOError(
                    "Given image and its mask do not occupy the same space: %s" %
                    e.message)
            stack.itk_mask = sitkh.get_itk_from_sitk_image(stack.sitk_mask)
            if sitk.GetArrayFromImage(stack.sitk_mask).prod() == 1:
                stack._is_unity_mask = True
            else:
                stack._is_unity_mask = False
        else:
            stack.sitk_mask = stack._generate_identity_mask()
            stack.itk_mask = sitkh.get_itk_from_sitk_image(stack.sitk_mask)
            stack._is_unity_mask = True

        # Extract all slices and their masks from the stack and store them
        if extract_slices:
            stack._N_slices = stack.sitk.GetSize()[-1]
            stack._slices = stack._extract_slices(
                slice_numbers=slice_numbers,
                slice_thickness=slice_thickness,
            )
        else:
            stack._N_slices = 0
            stack._slices = None

        # Store current affine transform of image
        stack._affine_transform_sitk = sitkh.get_sitk_affine_transform_from_sitk_image(
            stack.sitk)

        stack._history_affine_transforms = []
        stack._history_affine_transforms.append(stack._affine_transform_sitk)

        stack._history_motion_corrections = []
        stack._history_motion_corrections.append(sitk.Euler3DTransform())

        return stack

    ##
    # Copy constructor
    # \date       2019-01-15 16:55:09+0000
    #
    # \param      cls            The cls
    # \param      stack_to_copy  Stack object to be copied
    # \param      filename       The filename
    #
    # \return     copied Stack object TODO: That's not really well done
    #
    @classmethod
    def from_stack(cls, stack_to_copy, filename=None):
        stack = cls()

        if not isinstance(stack_to_copy, Stack):
            raise ValueError("Input must be of type Stack. Given: %s" %
                             type(stack_to_copy))

        # Copy image stack and mask
        stack.sitk = sitk.Image(stack_to_copy.sitk)
        stack.itk = sitkh.get_itk_from_sitk_image(stack.sitk)

        stack._slice_thickness = stack_to_copy.get_slice_thickness()

        stack.sitk_mask = sitk.Image(stack_to_copy.sitk_mask)
        stack.itk_mask = sitkh.get_itk_from_sitk_image(stack.sitk_mask)
        stack._is_unity_mask = stack_to_copy.is_unity_mask()

        if filename is None:
            stack._filename = stack_to_copy.get_filename()
        else:
            stack._filename = filename
        stack._dir = stack_to_copy.get_directory()
        stack._deleted_slices = stack_to_copy.get_deleted_slice_numbers()

        # Store current affine transform of image
        stack.set_registration_history(
            stack_to_copy.get_registration_history())

        # Extract all slices and their masks from the stack and store them if
        # given
        if stack_to_copy.get_slices() is not None:
            stack._N_slices = stack_to_copy.get_number_of_slices()
            stack._slices = [None] * stack._N_slices
            slices_to_copy = stack_to_copy.get_slices()

            for j, slice_j in enumerate(slices_to_copy):
                stack._slices[j] = sl.Slice.from_slice(slice_j)
        else:
            stack._N_slices = 0
            stack._slices = None

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

    # Get all slices of current stack
    #  \return Array of sitk.Images containing slices in 3D space
    def get_slices(self):
        if self._slices is None:
            return None
        else:
            return [s for s in self._slices if s is not None]

    ##
    # Get one particular slice of current stack
    # \date       2018-04-18 22:06:38-0600
    #
    # \param      self   The object
    # \param      index  slice index as integer
    #
    # \return     requested 3D slice of stack as Slice object
    #
    def get_slice(self, index):

        index = int(index)
        if abs(index) > self._N_slices - 1:
            raise ValueError(
                "Enter a valid index between -%s and %s. Tried: %s" %
                (self._N_slices - 1, self._N_slices - 1, index))

        return self._slices[index]

    def get_slice_thickness(self):
        return float(self._slice_thickness)

    def get_inplane_resolution(self):
        return float(self.sitk.GetSpacing()[0])

    ##
    # Gets the deleted slice numbers, i.e. misregistered slice numbers detected
    # by robust outlier algorithm. Indices refer to slice numbers within
    # original stack
    # \date       2018-07-08 23:06:24-0600
    #
    # \param      self  The object
    #
    # \return     The deleted slice numbers as list of integers.
    #
    def get_deleted_slice_numbers(self):
        return list(self._deleted_slices)

    ##
    # Sets the slice.
    # \date       2018-04-18 22:05:28-0600
    #
    # \param      self   The object
    # \param      slice  slice as Slice object
    # \param      index  slice index as integer
    #
    def set_slice(self, slice, index):
        if not isinstance(slice, sl.Slice):
            raise IOError("Input must be of type Slice")

        index = int(index)
        if abs(index) > self._N_slices - 1:
            raise ValueError(
                "Enter a valid index between -%s and %s. Tried: %s" %
                (self._N_slices - 1, self._N_slices - 1, index))

        self._slices[index] = slice

    ##
    # Delete slice at given index
    # \date       2017-12-01 00:38:56+0000
    #
    # Note that index refers to list index of slices (0 ... N_slices_current) whereas
    # "deleted slice index" refers to actual slice number within original stack
    #
    # \param      self   The object
    # \param      index  slice as Slice object to be deleted
    #
    def delete_slice(self, slice):
        if not isinstance(slice, sl.Slice):
            raise IOError("Input must be of type Slice")

        # keep slice number (w.r.t. original stack)
        self._deleted_slices.append(int(slice.get_slice_number()))
        self._deleted_slices = sorted((self._deleted_slices))

        # delete slice
        index = self._slices.index(slice)
        self._slices[index] = None

    def get_deleted_slice_numbers(self):
        return list(self._deleted_slices)

    # Get name of directory where nifti was read from
    #  \return string of directory wher nifti was read from
    #  \bug Does not exist for all created instances! E.g. Stack.from_sitk_image
    def get_directory(self):
        return self._dir

    def set_filename(self, filename):
        self._filename = filename

        slices = self.get_slices()
        if slices is not None:
            for s in slices:
                s.set_filename(filename)

    # Get filename of read/assigned nifti file (Stack.from_filename vs Stack.from_sitk_image)
    #  \return string of filename
    def get_filename(self):
        return self._filename

    # Get history history of affine transforms, i.e. encoded spatial
    #  position+orientation of slice, and rigid motion estimates of slice
    #  obtained in the course of the registration/reconstruction process
    #  \return list of sitk.AffineTransform and sitk.Euler3DTransform objects
    def get_registration_history(self):
        affine_transforms = list(self._history_affine_transforms)
        motion_corrections = list(self._history_motion_corrections)
        return affine_transforms, motion_corrections

    def set_registration_history(self, registration_history):
        affine_transform_sitk = registration_history[0][-1]
        self._update_affine_transform(affine_transform_sitk)

        self._history_affine_transforms = [a for a in registration_history[0]]
        self._history_motion_corrections = [t for t in registration_history[1]]

    # Get number of slices of stack
    #  \return number of slices of stack
    def get_number_of_slices(self):
        return len(self.get_slices())

    def is_unity_mask(self):
        return self._is_unity_mask

    # Display stack with external viewer (ITK-Snap)
    #  \param[in][in] show_segmentation display stack with or without associated segmentation (default=0)
    def show(self, show_segmentation=0, label=None, viewer=VIEWER, verbose=True):

        if label is None:
            label = self._filename

        if show_segmentation:
            sitk_mask = self.sitk_mask
        else:
            sitk_mask = None

        sitkh.show_sitk_image(
            self.sitk,
            label=label,
            segmentation=sitk_mask,
            viewer=viewer,
            verbose=verbose)

    def show_slices(self):
        sitkh.plot_stack_of_slices(
            self.sitk, cmap="Greys_r", title=self.get_filename())

    # Write information of Stack to HDD to given directory:
    #  - sitk.Image object as entire volume
    #  - each single slice with its associated spatial transformation (optional)
    #  \param[in] directory string specifying where the output will be written to (default="/tmp/")
    #  \param[in] filename string specifying the filename. If not given the assigned one within Stack will be chosen.
    #  \param[in] write_slices boolean indicating whether each Slice of the stack shall be written (default=False)
    def write(self,
              directory,
              filename=None,
              write_stack=True,
              write_mask=False,
              write_slices=False,
              write_transforms=False,
              suffix_mask="_mask",
              write_transforms_history=False,
              ):

        # Create directory if not existing
        ph.create_directory(directory)

        # Construct filename
        if filename is None:
            filename = self._filename

        full_file_name = os.path.join(directory, filename)

        # Write file to specified location
        if write_stack:
            dw.DataWriter.write_image(self.sitk, "%s.nii.gz" % full_file_name)

        # Write mask to specified location if given
        if self.sitk_mask is not None:
            # nda = sitk.GetArrayFromImage(self.sitk_mask)

            # Write mask if it does not consist of only ones
            if not self._is_unity_mask and write_mask:
                dw.DataWriter.write_mask(
                    self.sitk_mask, "%s%s.nii.gz" % (full_file_name, suffix_mask))

        if write_transforms:
            stack_transform_sitk = self._history_motion_corrections[-1]
            sitk.WriteTransform(
                stack_transform_sitk,
                os.path.join(directory, self.get_filename() + ".tfm")
            )

        # Write each separate Slice of stack (if they exist)
        if write_slices or write_transforms:
            try:
                # Check whether variable exists
                # if 'self._slices' not in locals() or all(i is None for i in
                # self._slices):
                if not hasattr(self, '_slices'):
                    raise ValueError(
                        "Error occurred in attempt to write %s.nii.gz: "
                        "No separate slices of object Slice are found" %
                        full_file_name)

                # Write slices
                else:
                    if write_transforms and write_slices:
                        ph.print_info(
                            "Write %s image slices and slice transforms to %s ... " % (
                                self.get_filename(), directory),
                            newline=False)
                    elif write_transforms and not write_slices:
                        ph.print_info(
                            "Write %s slice transforms to %s ... " % (
                                self.get_filename(), directory),
                            newline=False)
                    else:
                        ph.print_info(
                            "Write %s image slices to %s ... " % (
                                self.get_filename(), directory),
                            newline=False)
                    for slice in self.get_slices():
                        slice.write(
                            directory=directory,
                            filename=filename,
                            write_transform=write_transforms,
                            write_slice=write_slices,
                            suffix_mask=suffix_mask,
                            write_transforms_history=write_transforms_history,
                        )
                    print("done")

            except ValueError as err:
                print(err.message)

    ##
    #       Apply transform on stack and all its slices
    # \date       2016-11-05 19:15:57+0000
    #
    # \param      self                   The object
    # \param      affine_transform_sitk  The affine transform sitk
    #
    def update_motion_correction(self, affine_transform_sitk):

        # Update rigid motion estimate
        current_rigid_motion_estimate = sitkh.get_composite_sitk_affine_transform(
            affine_transform_sitk, self._history_motion_corrections[-1])
        self._history_motion_corrections.append(current_rigid_motion_estimate)

        # New affine transform of slice after rigid motion correction
        affine_transform = sitkh.get_composite_sitk_affine_transform(
            affine_transform_sitk, self._affine_transform_sitk)

        # Update affine transform of stack, i.e. change image origin and
        # direction in physical space
        self._update_affine_transform(affine_transform)

        # Update slices
        if self.get_slices() is not None:
            for i in range(0, self._N_slices):
                self._slices[i].update_motion_correction(affine_transform_sitk)

    ##
    #       Apply transforms on all the slices of the stack. Stack itself
    #             is not getting transformed
    # \date       2016-11-05 19:16:33+0000
    #
    # \param      self                    The object
    # \param      affine_transforms_sitk  List of sitk transform instances
    #
    def update_motion_correction_of_slices(self, affine_transforms_sitk):
        if [type(affine_transforms_sitk) is list or type(affine_transforms_sitk) is np.array] \
                and len(affine_transforms_sitk) is self._N_slices:
            for i in range(0, self._N_slices):
                self._slices[i].update_motion_correction(
                    affine_transforms_sitk[i])

        else:
            raise ValueError("Number of affine transforms does not match the "
                             "number of slices")

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

        self.sitk_mask.SetOrigin(origin)
        self.sitk_mask.SetDirection(direction)

        self.itk.SetOrigin(origin)
        self.itk.SetDirection(sitkh.get_itk_from_sitk_direction(direction))

        self.itk_mask.SetOrigin(origin)
        self.itk_mask.SetDirection(
            sitkh.get_itk_from_sitk_direction(direction))

    ##
    #       Gets the resampled stack from slices.
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
    def get_resampled_stack_from_slices(self, resampling_grid=None, interpolator="NearestNeighbor", default_pixel_value=0.0, filename=None):

        # Choose interpolator
        try:
            interpolator_str = interpolator
            interpolator = eval("sitk.sitk" + interpolator_str)
        except:
            raise ValueError("Error: interpolator is not known")

        # Use resampling grid defined by original volumetric image
        if resampling_grid is None:
            resampling_grid = Stack.from_sitk_image(
                image_sitk=self.sitk,
                slice_thickness=self.get_slice_thickness(),
            )

        else:
            # Use resampling grid defined by first slice (which might be
            # shifted already)
            if resampling_grid in ["on_first_slice"]:
                stack_sitk = sitk.Image(self.sitk)
                foo_sitk = sitk.Image(self._slices[0].sitk)
                stack_sitk.SetDirection(foo_sitk.GetDirection())
                stack_sitk.SetOrigin(foo_sitk.GetOrigin())
                stack_sitk.SetSpacing(foo_sitk.GetSpacing())
                resampling_grid = Stack.from_sitk_image(stack_sitk)

            # Use resampling grid defined by given sitk.Image
            elif type(resampling_grid) is sitk.Image:
                resampling_grid = Stack.from_sitk_image(resampling_grid)

        # Get shape of image data array
        nda_shape = resampling_grid.sitk.GetSize()[::-1]

        # Create zero image and its mask aligned with sitk.Image
        nda = np.zeros(nda_shape)

        stack_resampled_sitk = sitk.GetImageFromArray(nda)
        stack_resampled_sitk.CopyInformation(resampling_grid.sitk)
        stack_resampled_sitk = sitk.Cast(
            stack_resampled_sitk, resampling_grid.sitk.GetPixelIDValue())

        stack_resampled_sitk_mask = sitk.GetImageFromArray(nda.astype("uint8"))
        stack_resampled_sitk_mask.CopyInformation(resampling_grid.sitk_mask)
        stack_resampled_sitk_mask = sitk.Cast(
            stack_resampled_sitk_mask, resampling_grid.sitk_mask.GetPixelIDValue())

        # Create helper used for normalization at the end
        nda_stack_covered_indices = np.zeros(nda_shape)

        for i in range(0, self._N_slices):
            slice = self._slices[i]

            # Resample slice and its mask to stack space (volume)
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
                0,
                resampling_grid.sitk_mask.GetPixelIDValue())

            # Add resampled slice and mask to stack space
            stack_resampled_sitk += stack_resampled_slice_sitk
            stack_resampled_sitk_mask += stack_resampled_slice_sitk_mask

            # Get indices which are updated in stack space
            nda_stack_resampled_slice_ind = sitk.GetArrayFromImage(
                stack_resampled_slice_sitk)
            ind = np.nonzero(nda_stack_resampled_slice_ind)

            # Increment counter for respective updated voxels
            nda_stack_covered_indices[ind] += 1

        # Set voxels with zero counter to 1 so as to have well-defined
        # normalization
        nda_stack_covered_indices[nda_stack_covered_indices == 0] = 1

        # Normalize resampled image
        stack_normalization = sitk.GetImageFromArray(nda_stack_covered_indices)
        stack_normalization.CopyInformation(resampling_grid.sitk)
        stack_normalization = sitk.Cast(
            stack_normalization, resampling_grid.sitk.GetPixelIDValue())
        stack_resampled_sitk /= stack_normalization

        # Get valid binary mask
        stack_resampled_slice_sitk_mask /= stack_resampled_slice_sitk_mask

        if filename is None:
            filename = self._filename + "_" + interpolator_str

        stack = self.from_sitk_image(
            image_sitk=stack_resampled_sitk,
            filename=filename,
            image_sitk_mask=stack_resampled_sitk_mask,
            slice_thickness=stack_resampled_sitk.GetSpacing()[-1],
        )

        return stack

    ##
    # Gets the resampled stack.
    # \date       2016-12-02 17:05:10+0000
    #
    # \param      self                 The object
    # \param      resampling_grid      The resampling grid as SimpleITK image
    # \param      interpolator         The interpolator
    # \param      default_pixel_value  The default pixel value
    #
    # \return     The resampled stack as Stack object
    #
    def get_resampled_stack(self, resampling_grid=None, spacing=None, interpolator="Linear", default_pixel_value=0.0, filename=None):

        if (resampling_grid is None and spacing is None) or \
                (resampling_grid is not None and spacing is not None):
            raise IOError(
                "Either 'resampling_grid' or 'spacing' must be specified")

        # Get SimpleITK-interpolator
        try:
            interpolator_str = interpolator
            interpolator = eval("sitk.sitk" + interpolator_str)
        except:
            raise ValueError(
                "Error: interpolator is not known. "
                "Must fit sitk.InterpolatorEnum format. "
                "Possible examples include "
                "'NearestNeighbor', 'Linear', or 'BSpline'.")

        if resampling_grid is not None:
            resampled_stack_sitk = sitk.Resample(
                self.sitk,
                resampling_grid,
                sitk.Euler3DTransform(),
                interpolator,
                default_pixel_value,
                self.sitk.GetPixelIDValue())

            resampled_stack_sitk_mask = sitk.Resample(
                self.sitk_mask,
                resampling_grid,
                sitk.Euler3DTransform(),
                sitk.sitkNearestNeighbor,
                0,
                self.sitk_mask.GetPixelIDValue())
        else:
            resampler = simplereg.resampler.Resampler
            resampled_stack_sitk = resampler.get_resampled_image_sitk(
                image_sitk=self.sitk,
                spacing=spacing,
                interpolator=interpolator,
                padding=default_pixel_value,
                add_to_grid_unit="mm",
            )
            resampled_stack_sitk_mask = resampler.get_resampled_image_sitk(
                image_sitk=self.sitk_mask,
                spacing=spacing,
                interpolator=sitk.sitkNearestNeighbor,
                padding=0,
                add_to_grid_unit="mm",
            )

        # Create Stack instance
        if filename is None:
            filename = self._filename + "_" + interpolator_str
        stack = self.from_sitk_image(
            image_sitk=resampled_stack_sitk,
            slice_thickness=resampled_stack_sitk.GetSpacing()[-1],
            filename=filename,
            image_sitk_mask=resampled_stack_sitk_mask,
        )

        return stack

    ##
    # Gets the stack multiplied with its mask. Rationale behind is to obtain
    # "cleaner" looking HR images after the SRR step where motion-correction
    # might have dispersed some slices
    # \date       2017-05-26 13:50:39+0100
    #
    # \param      self      The object
    # \param      filename  The filename
    #
    # \return     The stack multiplied with its mask.
    #
    def get_stack_multiplied_with_mask(self, filename=None, mask_sitk=None):

        if mask_sitk is None:
            mask_sitk = self.sitk_mask

        # Multiply stack with its mask
        image_sitk = self.sitk * \
            sitk.Cast(mask_sitk, self.sitk.GetPixelIDValue())

        if filename is None:
            filename = self.get_filename()

        return Stack.from_sitk_image(
            image_sitk=image_sitk,
            filename=filename,
            image_sitk_mask=mask_sitk,
            slice_thickness=self.get_slice_thickness(),
        )

    # Get stack resampled on isotropic grid based on the actual position of
    #  its slices
    #  \param[in] resolution length of voxel side, scalar
    #  \return isotropically, resampled stack as Stack object
    def get_isotropically_resampled_stack_from_slices(self, resolution=None, interpolator="NearestNeighbor", default_pixel_value=0.0, filename=None):
        resampled_stack = self.get_resampled_stack_from_slices()

        # Choose interpolator
        try:
            interpolator_str = interpolator
            interpolator = eval("sitk.sitk" + interpolator_str)
        except:
            raise ValueError("Error: interpolator is not known")

        # Read original spacing (voxel dimension) and size of target stack:
        spacing = np.array(resampled_stack.sitk.GetSpacing())
        size = np.array(resampled_stack.sitk.GetSize()).astype("int")

        if resolution is None:
            size_new = size
            spacing_new = spacing
            # Update information according to isotropic resolution
            size_new[2] = np.round(
                spacing[2] / spacing[0] * size[2]).astype("int")
            spacing_new[2] = spacing[0]
        else:
            spacing_new = np.ones(3) * resolution
            size_new = np.round(spacing / spacing_new * size).astype("int")

        # For Python3: sitk.Resample in Python3 does not like np.int types!
        size_new = [int(i) for i in size_new]

        # Resample image and its mask to isotropic grid
        isotropic_resampled_stack_sitk = sitk.Resample(
            resampled_stack.sitk,
            size_new,
            sitk.Euler3DTransform(),
            interpolator,
            resampled_stack.sitk.GetOrigin(),
            spacing_new,
            resampled_stack.sitk.GetDirection(),
            default_pixel_value,
            resampled_stack.sitk.GetPixelIDValue())

        isotropic_resampled_stack_sitk_mask = sitk.Resample(
            resampled_stack.sitk_mask,
            size_new,
            sitk.Euler3DTransform(),
            sitk.sitkNearestNeighbor,
            resampled_stack.sitk.GetOrigin(),
            spacing_new,
            resampled_stack.sitk.GetDirection(),
            0,
            resampled_stack.sitk_mask.GetPixelIDValue())

        # Create Stack instance
        if filename is None:
            filename = self._filename + "_" + interpolator_str + "Iso"
        stack = self.from_sitk_image(
            isotropic_resampled_stack_sitk, filename, isotropic_resampled_stack_sitk_mask)

        return stack

    ##
    # Gets the isotropically resampled stack.
    # \date       2017-02-03 16:34:24+0000
    #
    # \param      self                  The object
    # \param      resolution    length of voxel side, scalar
    # \param      interpolator          choose type of interpolator for
    #                                   resampling
    # \param      extra_frame           additional extra frame of zero
    #                                   intensities surrounding the stack in mm
    # \param      filename              Filename of resampled stack
    # \param      mask_dilation_radius  The mask dilation radius
    # \param      mask_dilation_kernel  The kernel in "Ball", "Box", "Annulus"
    #                                   or "Cross"
    #
    # \return     The isotropically resampled stack.
    #
    def get_isotropically_resampled_stack(self, resolution=None, interpolator="Linear", extra_frame=0, filename=None, mask_dilation_radius=0, mask_dilation_kernel="Ball"):

        # Choose interpolator
        try:
            interpolator_str = interpolator
            interpolator = eval("sitk.sitk" + interpolator_str)
        except:
            raise ValueError("Error: interpolator is not known")

        if resolution is None:
            spacing = self.sitk.GetSpacing()[0]
        else:
            spacing = resolution

        # Resample image and its mask to isotropic grid
        resampler = simplereg.resampler.Resampler
        isotropic_resampled_stack_sitk = resampler.get_resampled_image_sitk(
            image_sitk=self.sitk,
            spacing=spacing,
            interpolator=interpolator,
            padding=0.0,
            add_to_grid=extra_frame,
            add_to_grid_unit="mm",
        )
        isotropic_resampled_stack_sitk_mask = resampler.get_resampled_image_sitk(
            image_sitk=self.sitk_mask,
            spacing=spacing,
            interpolator=sitk.sitkNearestNeighbor,
            padding=0,
            add_to_grid=extra_frame,
            add_to_grid_unit="mm",
        )

        if mask_dilation_radius > 0:
            dilater = sitk.BinaryDilateImageFilter()
            dilater.SetKernelType(eval("sitk.sitk" + mask_dilation_kernel))
            dilater.SetKernelRadius(mask_dilation_radius)
            isotropic_resampled_stack_sitk_mask = dilater.Execute(
                isotropic_resampled_stack_sitk_mask)

        # Create Stack instance
        if filename is None:
            filename = self._filename + "_" + interpolator_str + "Iso"
        stack = self.from_sitk_image(
            image_sitk=isotropic_resampled_stack_sitk,
            filename=filename,
            slice_thickness=isotropic_resampled_stack_sitk.GetSpacing()[-1],
            image_sitk_mask=isotropic_resampled_stack_sitk_mask,
        )

        return stack

    # Increase stack by adding zero voxels in respective directions
    #  \remark Used for MS project to add empty slices on top of (chopped) brain
    #  \param[in] resolution length of voxel side, scalar
    #  \param[in] interpolator choose type of interpolator for resampling
    #  \param[in] extra_frame additional extra frame of zero intensities surrounding the stack in mm
    #  \return isotropically, resampled stack as Stack object
    def get_increased_stack(self, extra_slices_z=0):

        interpolator = sitk.sitkNearestNeighbor

        # Read original spacing (voxel dimension) and size of target stack:
        spacing = np.array(self.sitk.GetSpacing())
        size = np.array(self.sitk.GetSize()).astype("int")
        origin = np.array(self.sitk.GetOrigin())
        direction = self.sitk.GetDirection()

        # Update information according to isotropic resolution
        size[2] += extra_slices_z

        # Resample image and its mask to isotropic grid
        default_pixel_value = 0.0

        isotropic_resampled_stack_sitk = sitk.Resample(
            self.sitk,
            size,
            sitk.Euler3DTransform(),
            interpolator,
            origin,
            spacing,
            direction,
            default_pixel_value,
            self.sitk.GetPixelIDValue())

        isotropic_resampled_stack_sitk_mask = sitk.Resample(
            self.sitk_mask,
            size,
            sitk.Euler3DTransform(),
            sitk.sitkNearestNeighbor,
            origin,
            spacing,
            direction,
            0,
            self.sitk_mask.GetPixelIDValue())

        # Create Stack instance
        stack = self.from_sitk_image(
            isotropic_resampled_stack_sitk, "zincreased_" + self._filename, isotropic_resampled_stack_sitk_mask)

        return stack

    def get_cropped_stack_based_on_mask(self, boundary_i=0, boundary_j=0, boundary_k=0, unit="mm"):

        # Get rectangular region surrounding the masked voxels
        [x_range, y_range, z_range] = self._get_rectangular_masked_region(
            self.sitk_mask)

        if np.array([x_range, y_range, z_range]).all() is None:
            raise RuntimeError(
                "Cropping to bounding box of mask led to an empty image. "
                "Check the image stack to see whether the region of interest "
                "is presented in '%s'." % self._filename)

        if unit == "mm":
            spacing = self.sitk.GetSpacing()
            boundary_i = np.round(boundary_i / float(spacing[0]))
            boundary_j = np.round(boundary_j / float(spacing[1]))
            boundary_k = np.round(boundary_k / float(spacing[2]))

        shape = self.sitk.GetSize()
        x_range[0] = np.max([0, x_range[0] - boundary_i])
        x_range[1] = np.min([shape[0], x_range[1] + boundary_i])

        y_range[0] = np.max([0, y_range[0] - boundary_j])
        y_range[1] = np.min([shape[1], y_range[1] + boundary_j])

        z_range[0] = np.max([0, z_range[0] - boundary_k])
        z_range[1] = np.min([shape[2], z_range[1] + boundary_k])

        # Crop to image region defined by rectangular mask
        image_crop_sitk = self._crop_image_to_region(
            self.sitk, x_range, y_range, z_range)
        mask_crop_sitk = self._crop_image_to_region(
            self.sitk_mask, x_range, y_range, z_range)

        slice_numbers = range(z_range[0], z_range[1])
        stack = self.from_sitk_image(
            image_sitk=image_crop_sitk,
            slice_thickness=self.get_slice_thickness(),
            filename=self._filename,
            image_sitk_mask=mask_crop_sitk,
            slice_numbers=slice_numbers)

        return stack

    # Return rectangular region surrounding masked region.
    #  \param[in] mask_sitk sitk.Image representing the mask
    #  \return range_x pair defining x interval of mask in voxel space
    #  \return range_y pair defining y interval of mask in voxel space
    #  \return range_z pair defining z interval of mask in voxel space
    def _get_rectangular_masked_region(self, mask_sitk):

        spacing = np.array(mask_sitk.GetSpacing())

        # Get mask array
        nda = sitk.GetArrayFromImage(mask_sitk)

        # Return in case no masked pixel available
        if np.sum(abs(nda)) == 0:
            return None, None, None

        # Get shape defining the dimension in each direction
        shape = nda.shape

        # Compute sum of pixels of each slice along specified directions
        sum_xy = np.sum(nda, axis=(0, 1))  # sum within x-y-plane
        sum_xz = np.sum(nda, axis=(0, 2))  # sum within x-z-plane
        sum_yz = np.sum(nda, axis=(1, 2))  # sum within y-z-plane

        # Find masked regions (non-zero sum!)
        range_x = np.zeros(2)
        range_y = np.zeros(2)
        range_z = np.zeros(2)

        # Non-zero elements of numpy array nda defining x_range
        ran = np.nonzero(sum_yz)[0]
        range_x[0] = np.max([0,         ran[0]])
        range_x[1] = np.min([shape[0], ran[-1] + 1])

        # Non-zero elements of numpy array nda defining y_range
        ran = np.nonzero(sum_xz)[0]
        range_y[0] = np.max([0,         ran[0]])
        range_y[1] = np.min([shape[1], ran[-1] + 1])

        # Non-zero elements of numpy array nda defining z_range
        ran = np.nonzero(sum_xy)[0]
        range_z[0] = np.max([0,         ran[0]])
        range_z[1] = np.min([shape[2], ran[-1] + 1])

        # Numpy reads the array as z,y,x coordinates! So swap them accordingly
        return range_z.astype(int), range_y.astype(int), range_x.astype(int)

    # Crop given image to region defined by voxel space ranges
    #  \param[in] image_sitk image which will be cropped
    #  \param[in] range_x pair defining x interval in voxel space for image cropping
    #  \param[in] range_y pair defining y interval in voxel space for image cropping
    #  \param[in] range_z pair defining z interval in voxel space for image cropping
    #  \return image cropped to defined region
    def _crop_image_to_region(self, image_sitk, range_x, range_y, range_z):

        image_cropped_sitk = image_sitk[
            range_x[0]:range_x[1],
            range_y[0]:range_y[1],
            range_z[0]:range_z[1]
        ]

        return image_cropped_sitk

    # Burst the stack into its slices and return all slices of the stack
    #  return list of Slice objects
    def _extract_slices(self, slice_thickness, slice_numbers=None):

        slices = [None] * self._N_slices

        if slice_numbers is None:
            slice_numbers = range(0, self._N_slices)

        if len(slice_numbers) != self._N_slices:
            raise ValueError(
                "slice_numbers must correspond to the number of slices "
                "of the image volume")

        # Extract slices and add masks
        for i in range(0, self._N_slices):
            slices[i] = sl.Slice.from_sitk_image(
                slice_sitk=self.sitk[:, :, i:i + 1],
                filename=self._filename,
                slice_number=slice_numbers[i],
                slice_sitk_mask=self.sitk_mask[:, :, i:i + 1],
                slice_thickness=slice_thickness,
            )

        return slices

    # Create a binary mask consisting of ones
    #  \return binary_mask as sitk.Image object consisting of ones
    def _generate_identity_mask(self):
        shape = sitk.GetArrayFromImage(self.sitk).shape
        nda = np.ones(shape, dtype=np.uint8)

        binary_mask = sitk.GetImageFromArray(nda)
        binary_mask.CopyInformation(self.sitk)

        return binary_mask
