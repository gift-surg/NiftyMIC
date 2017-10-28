# \file DataPreprocessing.py
#  \brief Performs preprocessing steps
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date May 2017

# Import libraries

import niftymic.base.stack as st
import niftymic.utilities.intensity_correction as ic
import niftymic.utilities.n4_bias_field_correction as n4bfc
import niftymic.base.exceptions as exceptions
import pysitk.python_helper as ph


##
# Class implementing data preprocessing steps
#
class DataPreprocessing:

    ##
    # Initialize data preprocessing class based on list of Stacks
    # \date       2017-05-12 00:49:43+0100
    #
    # \param      self                      The object
    # \param      stacks                    List of Stack instances
    # \param      use_N4BiasFieldCorrector  Use N4 bias field corrector, bool
    # \param      use_intensity_correction  Use linear intensity correction
    # \param      segmentation_propagator   None or SegmentationPropagation
    #                                       instance
    # \param      target_stack_index        Index of template stack.
    # \param      use_cropping_to_mask      The use crop to mask
    # \param      boundary_i                added value to first coordinate
    #                                       (can also be negative)
    # \param      boundary_j                added value to second coordinate
    #                                       (can also be negative)
    # \param      boundary_k                added value to third coordinate
    #                                       (can also be negative)
    # \param      unit                      Unit can either be "mm" or "voxel"
    # \param      cls   The cls
    #
    def __init__(self,
                 stacks,
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

        self._use_N4BiasFieldCorrector = use_N4BiasFieldCorrector
        self._use_intensity_correction = use_intensity_correction
        self._segmentation_propagator = segmentation_propagator
        self._target_stack_index = target_stack_index
        self._use_cropping_to_mask = use_cropping_to_mask
        self._boundary_i = boundary_i
        self._boundary_j = boundary_j
        self._boundary_k = boundary_k
        self._unit = unit

        # Number of stacks
        self._N_stacks = len(stacks)

        # Use stacks provided
        self._stacks = [st.Stack.from_stack(s) for s in stacks]

        ph.print_info(
            "%s stacks were loaded for data preprocessing." % (self._N_stacks))

    # Specify whether bias field correction based on N4 Bias Field Correction
    #  Filter shall be used
    #  \param[in] flag
    def use_N4BiasFieldCorrector(self, flag):
        self._use_N4BiasFieldCorrector = flag

    #
    # Perform data preprocessing
    # \date       2017-07-25 21:13:19+0100
    #
    # \param      self  The object
    #
    def run(self):

        time_start = ph.start_timing()

        # Segmentation propagation
        all_masks_provided = 0

        if self._segmentation_propagator is not None:

            stacks_to_propagate_indices = []
            for i in range(0, self._N_stacks):
                if self._stacks[i].is_unity_mask():
                    stacks_to_propagate_indices.append(i)

            stacks_to_propagate_indices = \
                list(set(stacks_to_propagate_indices) -
                     set([self._target_stack_index]))

            # Set target mask
            target = self._stacks[self._target_stack_index]

            # Propagate masks
            self._segmentation_propagator.set_template(target)
            for i in stacks_to_propagate_indices:
                ph.print_info("Propagate mask from stack '%s' to '%s'" % (
                    target.get_filename(),
                    self._stacks[i].get_filename()))
                self._segmentation_propagator.set_stack(
                    self._stacks[i])
                self._segmentation_propagator.run_segmentation_propagation()
                self._stacks[i] = \
                    self._segmentation_propagator.get_segmented_stack()

                # self._stacks[i].show(1)

        # Crop to mask
        if self._use_cropping_to_mask:
            ph.print_info("Crop stacks to their masks")

            for i in range(0, self._N_stacks):
                self._stacks[i] = self._stacks[i].get_cropped_stack_based_on_mask(
                    boundary_i=self._boundary_i,
                    boundary_j=self._boundary_j,
                    boundary_k=self._boundary_k,
                    unit=self._unit)

        # N4 Bias Field Correction
        if self._use_N4BiasFieldCorrector:
            bias_field_corrector = n4bfc.N4BiasFieldCorrection()

            for i in range(0, self._N_stacks):
                ph.print_info(
                    "Perform N4 Bias Field Correction for stack %d ... "
                    % (i+1), newline=False)
                bias_field_corrector.set_stack(self._stacks[i])
                bias_field_corrector.run_bias_field_correction()
                self._stacks[i] = \
                    bias_field_corrector.get_bias_field_corrected_stack()
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
                stack = self._stacks[i]
                intensity_corrector.set_stack(stack)
                intensity_corrector.set_reference(
                    target.get_resampled_stack(resampling_grid=stack.sitk))
                # intensity_corrector.run_affine_intensity_correction()
                intensity_corrector.run_linear_intensity_correction()
                self._stacks[
                    i] = intensity_corrector.get_intensity_corrected_stack()
        self._computational_time = ph.stop_timing(time_start)

    # Get preprocessed stacks
    #  \return preprocessed stacks as list of Stack objects
    def get_preprocessed_stacks(self):

        # Return a copy of preprocessed stacks
        return [st.Stack.from_stack(stack) for stack in self._stacks]

    def get_computational_time(self):
        return self._computational_time

    # Write preprocessed data to specified output directory
    #  \param[in] dir_output output directory
    def write_preprocessed_data(self, dir_output):
        if all(x is None for x in self._stacks):
            raise exceptions.ObjectNotCreated("run")

        # Write all slices
        for i in range(0, self._N_stacks):
            slices = self._stacks[i].write(
                directory=dir_output, write_mask=True, write_slices=False)
