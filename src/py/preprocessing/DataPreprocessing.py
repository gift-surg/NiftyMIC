# \file DataPreprocessing.py
#  \brief Performs preprocessing steps
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date May 2017

# Import libraries
import os                       # used to execute terminal commands in python
import sys
import SimpleITK as sitk
import numpy as np

# Import modules from src-folder
import base.Stack as st
import utilities.PythonHelper as ph
import utilities.SimpleITKHelper as sitkh
import preprocessing.IntensityCorrection as ic
import preprocessing.N4BiasFieldCorrection as n4bfc
import utilities.Exceptions as Exceptions


##
# Class implementing data preprocessing steps
#
class DataPreprocessing:

    ##
    # Constructor
    # \date       2017-05-12 00:48:43+0100
    #
    # \param      self                      The object
    # \param      use_N4BiasFieldCorrector  Use N4 bias field corrector, bool
    # \param      segmentation_propagator   None or SegmentationPropagation
    #                                       instance
    # \param      target_stack_index        Index of template stack. Template
    #                                       stack will be put on first position
    #                                       of Stack list after preprocessing
    # \param      use_cropping_to_mask          The use crop to mask
    # \param      boundary_i                added value to first coordinate
    #                                       (can also be negative)
    # \param      boundary_j                added value to second coordinate
    #                                       (can also be negative)
    # \param      boundary_k                added value to third coordinate
    #                                       (can also be negative)
    # \param      unit                      Unit can either be "mm" or "voxel"
    #
    def __init__(self,
                 use_N4BiasFieldCorrector=False,
                 use_intensity_correction=False,
                 segmentation_propagator=None,
                 target_stack_index=0,
                 use_cropping_to_mask=True,
                 boundary_i=0,
                 boundary_j=0,
                 boundary_k=0,
                 unit="mm"
                 ):

        self._use_N4BiasFieldCorrector = use_N4BiasFieldCorrector
        self._use_intensity_correction = use_intensity_correction
        self._segmentation_propagator = segmentation_propagator
        self._target_stack_index = target_stack_index
        self._use_cropping_to_mask = use_cropping_to_mask
        self._boundary_i = boundary_i
        self._boundary_j = boundary_j
        self._boundary_k = boundary_k
        self._unit = unit

    ##
    # Initialize data preprocessing class based on filenames and directory
    # \date       2017-05-12 00:44:27+0100
    #
    # \param      cls                       The cls
    # \param      dir_input                 Input directory
    # \param      filenames                 List of stack filenames to read
    #                                       (without ".nii.gz")
    # \param      suffix_mask               extension of stack filename which
    #                                       indicates associated mask
    # \param      use_N4BiasFieldCorrector  Use N4 bias field corrector, bool
    # \param      segmentation_propagator   None or SegmentationPropagation
    #                                       instance
    # \param      target_stack_index        Index of template stack. Template
    #                                       stack will be put on first position
    #                                       of Stack list after preprocessing
    # \param      use_cropping_to_mask          The use crop to mask
    # \param      boundary_i                added value to first coordinate
    #                                       (can also be negative)
    # \param      boundary_j                added value to second coordinate
    #                                       (can also be negative)
    # \param      boundary_k                added value to third coordinate
    #                                       (can also be negative)
    # \param      unit                      Unit can either be "mm" or "voxel"
    #
    @classmethod
    def from_filenames(cls,
                       filenames,
                       suffix_mask="_mask",
                       use_N4BiasFieldCorrector=False,
                       use_intensity_correction=False,
                       segmentation_propagator=None,
                       target_stack_index=0,
                       use_cropping_to_mask=True,
                       boundary_i=0,
                       boundary_j=0,
                       boundary_k=0,
                       unit="mm",
                       ):
        self = cls(use_N4BiasFieldCorrector=use_N4BiasFieldCorrector,
                   use_intensity_correction=use_intensity_correction,
                   segmentation_propagator=segmentation_propagator,
                   target_stack_index=target_stack_index,
                   use_cropping_to_mask=use_cropping_to_mask,
                   boundary_i=boundary_i,
                   boundary_j=boundary_j,
                   boundary_k=boundary_k,
                   unit=unit)

        # Read input data
        self._read_input_data(dir_input, filenames, suffix_mask)

        return self

    ##
    # Initialize data preprocessing class based on all nifti files in directory
    # \date       2017-05-18 16:34:22+0100
    #
    # \param      cls                       The cls
    # \param      dir_input                 Input directory
    # \param      suffix_mask               extension of stack filename which
    #                                       indicates associated mask
    # \param      use_N4BiasFieldCorrector  Use N4 bias field corrector, bool
    # \param      segmentation_propagator   None or SegmentationPropagation
    #                                       instance
    # \param      target_stack_index        Index of template stack. Template
    #                                       stack will be put on first position
    #                                       of Stack list after preprocessing
    # \param      use_cropping_to_mask          The use crop to mask
    # \param      boundary_i                added value to first coordinate
    #                                       (can also be negative)
    # \param      boundary_j                added value to second coordinate
    #                                       (can also be negative)
    # \param      boundary_k                added value to third coordinate
    #                                       (can also be negative)
    # \param      unit                      Unit can either be "mm" or "voxel"
    #
    @classmethod
    def from_directory(cls,
                       dir_input,
                       suffix_mask="_mask",
                       use_N4BiasFieldCorrector=False,
                       use_intensity_correction=False,
                       segmentation_propagator=None,
                       target_stack_index=0,
                       use_cropping_to_mask=True,
                       boundary_i=0,
                       boundary_j=0,
                       boundary_k=0,
                       unit="mm"):

        self = cls(use_N4BiasFieldCorrector=use_N4BiasFieldCorrector,
                   use_intensity_correction=use_intensity_correction,
                   segmentation_propagator=segmentation_propagator,
                   target_stack_index=target_stack_index,
                   use_cropping_to_mask=use_cropping_to_mask,
                   boundary_i=boundary_i,
                   boundary_j=boundary_j,
                   boundary_k=boundary_k,
                   unit=unit)

        # Get data filenames without filename extension
        filenames_tmp = []

        for file in os.listdir(dir_input):
            for filename_extension in [".nii.gz", ".nii"]:
                if file.endswith(filename_extension) and not file.endswith(suffix_mask + filename_extension):
                    filenames_tmp.append(file)
        filenames = [f.split(".")[0] for f in sorted(filenames_tmp)]

        # Read input data
        self._read_input_data(dir_input, filenames, suffix_mask)

        # # Number of stacks to be read
        # self._N_stacks = len(filenames)

        # # Read stacks and their masks (if no mask is found a binary image is
        # # created automatically)
        # self._stacks_preprocessed = [None]*self._N_stacks

        # for i in range(0, self._N_stacks):
        #     self._stacks_preprocessed[i] = st.Stack.from_filename(
        #         dir_input, filenames[i], suffix_mask)
        #     txt = "Image '%s' was read for further processing." % (filenames[
        #                                                            i])
        #     if i is self._target_stack_index:
        #         ph.print_debug_info(txt + " [Selected target stack]")
        #     else:
        #         ph.print_debug_info(txt)

        # # ph.print_debug_info("%s stacks were read for data preprocessing." %(self._N_stacks))

        return self

    ##
    # Initialize data preprocessing class based on list of Stacks
    # \date       2017-05-12 00:49:43+0100
    #
    # \param      cls                       The cls
    # \param      stacks                    List of Stack instances
    # \param      use_N4BiasFieldCorrector  Use N4 bias field corrector, bool
    # \param      segmentation_propagator   None or SegmentationPropagation
    #                                       instance
    # \param      target_stack_index        Index of template stack. Template
    #                                       stack will be put on first position
    #                                       of Stack list after preprocessing
    # \param      use_cropping_to_mask          The use crop to mask
    # \param      boundary_i                added value to first coordinate
    #                                       (can also be negative)
    # \param      boundary_j                added value to second coordinate
    #                                       (can also be negative)
    # \param      boundary_k                added value to third coordinate
    #                                       (can also be negative)
    # \param      unit                      Unit can either be "mm" or "voxel"
    #
    @classmethod
    def from_stacks(cls, stacks, use_N4BiasFieldCorrector=False, segmentation_propagator=None, target_stack_index=0, use_cropping_to_mask=True, boundary_i=0, boundary_j=0, boundary_k=0, unit="mm"):

        self = cls(use_N4BiasFieldCorrector=use_N4BiasFieldCorrector, segmentation_propagator=segmentation_propagator, target_stack_index=target_stack_index,
                   use_cropping_to_mask=use_cropping_to_mask, boundary_i=boundary_i, boundary_j=boundary_j, boundary_k=boundary_k, unit=unit)

        # Number of stacks
        self._N_stacks = len(stacks)

        # Use stacks provided
        self._stacks_preprocessed = [None]*self._N_stacks
        for i in range(0, self._N_stacks):
            self._stacks_preprocessed[i] = st.Stack.from_stack(stacks[i])

        ph.print_debug_info(
            "%s stacks were loaded for data preprocessing." % (self._N_stacks))

        return self

    # Specify whether bias field correction based on N4 Bias Field Correction
    #  Filter shall be used
    #  \param[in] flag
    def use_N4BiasFieldCorrector(self, flag):
        self._use_N4BiasFieldCorrector = flag

    # Specify prefix which will be used for naming the stacks
    #  param[in] prefix as string
    def set_filename_prefix(self, prefix):
        self._filename_prefix = prefix

    # Perform data preprocessing step by reading images from files
    #  \param[in] mask_template_number relevant in case not all masks are given (optional). Indicates stack for mask propagation.
    #  \param[in] boundary additional boundary surrounding mask in mm (optional). Capped by image domain.
    def run_preprocessing(self):

        time_start = ph.start_timing()

        # Segmentation propagation
        all_masks_provided = 0

        if self._segmentation_propagator is not None:

            stacks_to_propagate_indices = []
            for i in range(0, self._N_stacks):
                if self._stacks_preprocessed[i].is_unity_mask():
                    stacks_to_propagate_indices.append(i)

            stacks_to_propagate_indices = list(
                set(stacks_to_propagate_indices) - set([self._target_stack_index]))

            # Set target mask
            target = self._stacks_preprocessed[self._target_stack_index]

            # Propagate masks
            self._segmentation_propagator.set_template(target)
            for i in stacks_to_propagate_indices:
                ph.print_debug_info("Propagate mask from stack '%s' to '%s'" % (
                    target.get_filename(), self._stacks_preprocessed[i].get_filename()))
                self._segmentation_propagator.set_stack(
                    self._stacks_preprocessed[i])
                self._segmentation_propagator.run_segmentation_propagation()
                self._stacks_preprocessed[
                    i] = self._segmentation_propagator.get_segmented_stack()

                # self._stacks_preprocessed[i].show(1)

        # Crop to mask
        if self._use_cropping_to_mask:
            ph.print_debug_info("Crop stacks to their masks")

            for i in range(0, self._N_stacks):
                self._stacks_preprocessed[i] = self._stacks_preprocessed[i].get_cropped_stack_based_on_mask(
                    boundary_i=self._boundary_i, boundary_j=self._boundary_j, boundary_k=self._boundary_k, unit=self._unit)

        # N4 Bias Field Correction
        if self._use_N4BiasFieldCorrector:
            ph.print_debug_info(
                "Perform N4 Bias Field Correction for all stacks ... ")
            bias_field_corrector = n4bfc.N4BiasFieldCorrection()

            for i in range(0, self._N_stacks):
                bias_field_corrector.set_stack(self._stacks_preprocessed[i])
                bias_field_corrector.run_bias_field_correction()
                self._stacks_preprocessed[
                    i] = bias_field_corrector.get_bias_field_corrected_stack()

            print("done")

        # Linear Intensity Correction
        if self._use_intensity_correction:
            stacks_to_intensity_correct = list(
                set(range(0, self._N_stacks)) - set([self._target_stack_index]))

            intensity_corrector = ic.IntensityCorrection()
            intensity_corrector.use_individual_slice_correction(False)
            intensity_corrector.use_reference_mask(True)
            intensity_corrector.use_verbose(True)

            for i in stacks_to_intensity_correct:
                stack = self._stacks_preprocessed[i]
                intensity_corrector.set_stack(stack)
                intensity_corrector.set_reference(
                    target.get_resampled_stack(resampling_grid=stack.sitk))
                # intensity_corrector.run_affine_intensity_correction()
                intensity_corrector.run_linear_intensity_correction()
                self._stacks_preprocessed[
                    i] = intensity_corrector.get_intensity_corrected_stack()
        self._computational_time = ph.stop_timing(time_start)

    # Get preprocessed stacks
    #  \return preprocessed stacks as list of Stack objects
    def get_preprocessed_stacks(self):

        # Return a copy of preprocessed stacks
        stacks_copy = [None]*self._N_stacks

        # Move target stack to first position
        stacks_copy[0] = st.Stack.from_stack(
            self._stacks_preprocessed[self._target_stack_index])
        remaining_indices = list(
            set(range(0, self._N_stacks)) - set([self._target_stack_index]))

        i_ctr = 1
        for i in remaining_indices:
            stacks_copy[i_ctr] = st.Stack.from_stack(
                self._stacks_preprocessed[i])
            i_ctr = i_ctr + 1
        return stacks_copy

    def get_computational_time(self):
        return self._computational_time

    # Write preprocessed data to specified output directory
    #  \param[in] dir_output output directory
    def write_preprocessed_data(self, dir_output):
        if all(x is None for x in self._stacks_preprocessed):
            raise Exceptions.ObjectNotCreated("run_preprocessing")

        # Write all slices
        for i in range(0, self._N_stacks):
            slices = self._stacks_preprocessed[i].write(
                directory=dir_output, write_mask=True, write_slices=False)

    ##
    # Reads an input data.
    # \date       2017-06-14 11:03:57+0100
    #
    # \param      self         The object
    # \param      dir_input    The dir input
    # \param      filenames    The filenames
    # \param      suffix_mask  The suffix mask
    #
    # \post       updated _N_stacks and _stacks_preprocessed
    #
    def _read_input_data(self, dir_input, filenames, suffix_mask):

        # Number of stacks to be read
        self._N_stacks = len(filenames)

        ph.print_debug_info(
            "%d stacks are going to be read ..." % (self._N_stacks))

        if self._N_stacks == 0:
            raise Exceptions.InputFilesNotValid(dir_input)

        # Read stacks and their masks (if no mask is found a binary image is
        # created automatically)
        self._stacks_preprocessed = [None]*self._N_stacks

        for i in range(0, self._N_stacks):
            self._stacks_preprocessed[i] = st.Stack.from_filename(
                dir_input, filenames[i], suffix_mask)
            txt = "Image '%s' was read for further processing." % (filenames[
                                                                   i])
            if i is self._target_stack_index:
                ph.print_debug_info(txt + " [Selected target stack]")
            else:
                ph.print_debug_info(txt)

        # ph.print_debug_info("%s stacks were read for data preprocessing." %(self._N_stacks))
