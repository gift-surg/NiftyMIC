##
# \file volumetric_reconstruction_pipeline.py
# \brief      Collection of modules useful for registration and
#             reconstruction tasks.
#
# E.g. Volume-to-Volume Registration, Slice-to-Volume registration,
# Multi-component Reconstruction.
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Aug 2017
#

import numpy as np
import SimpleITK as sitk
from abc import ABCMeta, abstractmethod

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh

import niftymic.base.stack as st
import niftymic.validation.motion_evaluator as me


##
# Class which holds basic interface for all modules
# \date       2017-08-08 02:20:40+0100
#
class Pipeline(object):
    __metaclass__ = ABCMeta

    def __init__(self, stacks, verbose):
        self._stacks = stacks
        self._verbose = verbose

        self._computational_time = ph.get_zero_time()

    def set_stacks(self, stacks):
        self._stacks = stacks

    def get_stacks(self):
        return [st.Stack.from_stack(stack) for stack in self._stacks]

    def set_verbose(self, verbose):
        self._verbose = verbose

    def get_verbose(self):
        return self._verbose

    def get_computational_time(self):
        return self._computational_time

    def run(self):

        time_start = ph.start_timing()

        self._run()

        self._computational_time = ph.stop_timing(time_start)

        if self._verbose:
            ph.print_info("Required computational time: %s" %
                          (self.get_computational_time()))

    @abstractmethod
    def _run(self):
        pass


##
# Class which holds basic interface for all registration associated modules
# \date       2017-08-08 02:21:17+0100
#
class RegistrationPipeline(Pipeline):
    __metaclass__ = ABCMeta

    ##
    # Store variables relevant to register stacks to a certain reference volume
    # \date       2017-08-08 02:21:56+0100
    #
    # \param      self                 The object
    # \param      verbose              Verbose output, bool
    # \param      stacks               List of Stack objects
    # \param      reference            Reference as Stack object
    # \param      registration_method  Registration method, e.g.
    #                                  CppItkRegistration
    #
    def __init__(self, verbose, stacks, reference, registration_method):

        Pipeline.__init__(self, stacks=stacks, verbose=verbose)

        self._reference = reference
        self._registration_method = registration_method

    def set_reference(self, reference):
        self._reference = reference

    def get_reference(self):
        return st.Stack.from_stack(self._reference)


##
# Class to perform Volume-to-Volume registration
# \date       2017-08-08 02:28:56+0100
#
class VolumeToVolumeRegistration(RegistrationPipeline):

    ##
    # Store relevant information to perform Volume-to-Volume registration
    # \date       2017-08-08 02:29:13+0100
    #
    # \param      self                 The object
    # \param      stacks               The stacks
    # \param      reference            The reference
    # \param      registration_method  The registration method
    # \param      verbose              The verbose
    #
    def __init__(self,
                 stacks,
                 reference,
                 registration_method,
                 verbose=1,
                 ):
        RegistrationPipeline.__init__(
            self,
            stacks=stacks,
            reference=reference,
            registration_method=registration_method,
            verbose=verbose)

    def _run(self):

        ph.print_title("Volume-to-Volume Registration")

        self._registration_method.set_moving(self._reference)

        for i in range(0, len(self._stacks)):
            txt = "Volume-to-Volume Registration -- " \
                "Stack %d/%d" % (i+1, len(self._stacks))
            if self._verbose:
                ph.print_subtitle(txt)
            else:
                ph.print_info(txt)

            self._registration_method.set_fixed(self._stacks[i])
            self._registration_method.run()

            # Update position of stack
            transform_sitk = \
                self._registration_method.\
                get_registration_transform_sitk()
            self._stacks[i].update_motion_correction(transform_sitk)


##
# Class to perform Slice-To-Volume registration
# \date       2017-08-08 02:30:03+0100
#
class SliceToVolumeRegistration(RegistrationPipeline):

    ##
    # { constructor_description }
    # \date       2017-08-08 02:30:18+0100
    #
    # \param      self                 The object
    # \param      stacks               The stacks
    # \param      reference            The reference
    # \param      registration_method  Registration method, e.g.
    #                                  CppItkRegistration
    # \param      verbose              The verbose
    # \param      print_prefix         Print at each iteration at the
    #                                  beginning, string
    #
    def __init__(self,
                 stacks,
                 reference,
                 registration_method,
                 verbose=1,
                 print_prefix="",
                 ):
        RegistrationPipeline.__init__(
            self,
            stacks=stacks,
            reference=reference,
            registration_method=registration_method,
            verbose=verbose)
        self._print_prefix = print_prefix

    def set_print_prefix(self, print_prefix):
        self._print_prefix = print_prefix

    def _run(self):

        ph.print_title("Slice-to-Volume Registration")

        self._registration_method.set_moving(self._reference)

        for i in range(0, len(self._stacks)):
            stack = self._stacks[i]
            slices = stack.get_slices()

            transforms_sitk = [None] * len(slices)
            for j, slice in enumerate(slices):

                txt = "%sSlice-to-Volume Registration -- " \
                    "Stack %d/%d -- Slice %d/%d" % (
                        self._print_prefix,
                        i+1, len(self._stacks),
                        j+1, len(slices))
                if self._verbose:
                    ph.print_subtitle(txt)
                else:
                    ph.print_info(txt)

                self._registration_method.set_fixed(slice)
                self._registration_method.run()

                # Store information on registration transform
                transform_sitk = \
                    self._registration_method.\
                    get_registration_transform_sitk()
                transforms_sitk[j] = transform_sitk

            dir_output = "/tmp/foo/verbose"
            motion_evaluator = me.MotionEvaluator(transforms_sitk)
            motion_evaluator.run()
            title = "%s%s" % (self._print_prefix, stack.get_filename())
            motion_evaluator.display(dir_output=dir_output, title=title)
            motion_evaluator.show(dir_output=dir_output, title=title)


            # Update position of slice
            for j, slice in enumerate(slices):
                slice.update_motion_correction(transforms_sitk[j])


##
# Class to perform registration for the stack based on a specified set of
# slices
# \date       2017-10-16 12:52:18+0100
#
class SliceSetToVolumeRegistration(RegistrationPipeline):

    ##
    # { constructor_description }
    # \date       2017-10-16 12:53:04+0100
    #
    # \param      self                        The object
    # \param      stacks                      The stacks
    # \param      reference                   The reference
    # \param      registration_method         The registration method
    # \param      slice_index_sets_of_stacks  Dictionary specifying the slice
    #                                         index sets for all stacks
    # \param      verbose                     The verbose
    # \param      print_prefix                The print prefix
    #
    def __init__(self,
                 stacks,
                 reference,
                 registration_method,
                 slice_index_sets_of_stacks,
                 verbose=1,
                 print_prefix="",
                 ):
        RegistrationPipeline.__init__(
            self,
            stacks=stacks,
            reference=reference,
            registration_method=registration_method,
            verbose=verbose)

        self._print_prefix = print_prefix
        self._slice_index_sets_of_stacks = slice_index_sets_of_stacks

    def _run(self, debug=0):

        for i, stack in enumerate(self._stacks):
            slices = stack.get_slices()
            for indices in self._slice_index_sets_of_stacks[i]:
                txt = "%sSliceSet-to-Volume Registration -- " \
                    "Stack %d/%d -- Slices %s" % (
                        self._print_prefix,
                        i+1, len(self._stacks),
                        str(indices))
                if self._verbose:
                    ph.print_subtitle(txt)
                else:
                    ph.print_info(txt)

                image = self._get_stack_subgroup(stack, indices)

                if debug:
                    ph.killall_itksnap()
                    image.show()
                    stack.get_slice(indices[1]).show()

                self._registration_method.set_fixed(image)
                self._registration_method.run()
                transform_sitk = self._registration_method.\
                    get_registration_transform_sitk()

                for j in indices:
                    slices[j].update_motion_correction(transform_sitk)

                # Debug:
                # image.update_motion_correction(transform_sitk)
                # foo = [slices[j] for j in indices]
                # title = ["%s_%s" % (slices[j].get_filename(), slices[
                #                     j].get_slice_number()) for j in indices]
                # foo.insert(0, image)
                # title.insert(0, image.get_filename())
                # sitkh.show_stacks(foo, title)

    ##
    # Gets the bundled stack of selected slices.
    # \date       2017-10-16 13:10:32+0100
    #
    # \param      self     The object
    # \param      stack    Stack as Stack object
    # \param      indices  Indices of slices as list
    #
    # \return     Stack object holding image of selected slices.
    #
    def _get_stack_subgroup(self, stack, indices):

        # For some reason simple element indexing does not work for sitk
        # Problem: indices = [ 8 10 12 14]; but only 3 (!) slices are indexed!
        # But this happens quite irregularly!?
        # print all_[indices[0]:indices[-1]+1:self._interleave]
        # print all_
        # image_sitk = stack.sitk[
        #     :,
        #     :,
        #     indices[0]:indices[-1]+self._interleave:self._interleave]
        # image_sitk_mask = stack.sitk_mask[
        #     :,
        #     :,
        #     indices[0]:indices[-1]+self._interleave:self._interleave]

        # Build image from selected slices
        nda = sitk.GetArrayFromImage(stack.sitk)
        nda_mask = sitk.GetArrayFromImage(stack.sitk_mask)

        image_sitk = sitk.GetImageFromArray(nda[indices, :, :])
        image_sitk_mask = sitk.GetImageFromArray(nda_mask[indices, :, :])

        # Update stack/slice subgroup position in space according to first
        # slice which has undergone same motion as all remaining slices in the
        # list
        slice_sitk = stack.get_slice(indices[0]).sitk

        direction = slice_sitk.GetDirection()
        origin = slice_sitk.GetOrigin()
        spacing = np.array(slice_sitk.GetSpacing())

        # Update slice spacing according to selected interleave
        if len(indices) > 1:
            spacing[2] *= (indices[1] - indices[0])

        # Update information for image and its mask
        image_sitk.SetSpacing(spacing)
        image_sitk.SetDirection(direction)
        image_sitk.SetOrigin(origin)
        image_sitk_mask.CopyInformation(image_sitk)

        filename = stack.get_filename() + "-"
        filename += ("_").join([str(j) for j in indices])
        image = st.Stack.from_sitk_image(
            image_sitk=image_sitk,
            filename=filename,
            image_sitk_mask=image_sitk_mask,
            extract_slices=False)

        return image


class ReconstructionRegistrationPipeline(RegistrationPipeline):
    __metaclass__ = ABCMeta

    ##
    # Store variables relevant for two-step registration-reconstruction
    # pipeline.
    # \date       2017-10-16 10:30:39+0100
    #
    # \param      self                   The object
    # \param      verbose                Verbose output, bool
    # \param      stacks                 List of Stack objects
    # \param      reference              Reference as Stack object
    # \param      registration_method    Registration method, e.g.
    #                                    CppItkRegistration
    # \param      reconstruction_method  Reconstruction method, e.g. TK1
    # \param      alpha_range            Specify regularization parameter
    #                                    range, i.e. list [alpha_min,
    #                                    alpha_max]
    #
    def __init__(self,
                 verbose,
                 stacks,
                 reference,
                 registration_method,
                 reconstruction_method,
                 alpha_range,
                 ):

        RegistrationPipeline.__init__(
            self,
            verbose=verbose,
            stacks=stacks,
            reference=reference,
            registration_method=registration_method)

        self._reconstruction_method = reconstruction_method
        self._alpha_range = alpha_range

        self._reconstructions = [st.Stack.from_stack(
            self._reference,
            filename="Iter0_" + self._reference.get_filename())]
        self._computational_time_reconstruction = ph.get_zero_time()
        self._computational_time_registration = ph.get_zero_time()

    def get_iterative_reconstructions(self):
        return self._reconstructions

    def get_computational_time_reconstruction(self):
        return self._computational_time_reconstruction

    def get_computational_time_registration(self):
        return self._computational_time_registration


##
# Class to perform the two-step Slice-to-Volume registration and volumetric
# reconstruction iteratively
# \date       2017-08-08 02:30:43+0100
#
class TwoStepSliceToVolumeRegistrationReconstruction(
        ReconstructionRegistrationPipeline):

    ##
    # Store information to perform the two-step S2V reg and recon
    # \date       2017-08-08 02:31:24+0100
    #
    # \param      self                   The object
    # \param      stacks                 The stacks
    # \param      reference              The reference
    # \param      registration_method    Registration method, e.g.
    #                                    CppItkRegistration
    # \param      reconstruction_method  Reconstruction method, e.g. TK1
    # \param      alpha_range            Specify regularization parameter range
    #                                    used for each individual cycle, list
    #                                    or array
    # \param      cycles                 Number of cycles, int
    # \param      verbose                The verbose
    #
    def __init__(self,
                 stacks,
                 reference,
                 registration_method,
                 reconstruction_method,
                 alpha_range,
                 cycles,
                 verbose=1,
                 ):

        ReconstructionRegistrationPipeline.__init__(
            self,
            stacks=stacks,
            reference=reference,
            registration_method=registration_method,
            reconstruction_method=reconstruction_method,
            alpha_range=alpha_range,
            verbose=verbose)

        self._cycles = cycles

    def _run(self):

        ph.print_title("Two-step S2V-Registration and SRR Reconstruction")

        # Use linear spacing for alphas excluding the last alpha reserved
        # for the final SRR step
        alphas = np.linspace(
            self._alpha_range[0], self._alpha_range[1], self._cycles)
        alphas = alphas[0:self._cycles]

        s2vreg = SliceToVolumeRegistration(
            stacks=self._stacks,
            reference=self._reference,
            registration_method=self._registration_method,
            verbose=self._verbose)

        reference = self._reference

        for cycle in range(0, self._cycles):

            # Slice-to-volume registration step
            s2vreg.set_reference(reference)
            s2vreg.set_print_prefix("Cycle %d/%d: " % (cycle+1, self._cycles))
            s2vreg.run()

            self._computational_time_registration += \
                s2vreg.get_computational_time()

            # SRR step
            if cycle < self._cycles - 1:
                self._reconstruction_method.set_alpha(alphas[cycle])
                self._reconstruction_method.run()

                self._computational_time_reconstruction += \
                    self._reconstruction_method.get_computational_time()

                reference = self._reconstruction_method.get_reconstruction()

                # Store SRR
                filename = "Iter%d_%s" % (
                    cycle+1,
                    self._reconstruction_method.get_setting_specific_filename()
                )
                self._reconstructions.insert(0, st.Stack.from_stack(
                    reference, filename=filename))

                if self._verbose:
                    sitkh.show_stacks(self._reconstructions,
                                      segmentation=self._reference)


##
# Class to perform hierarchical slice alignment.
#
# Given an interleave, associated subpackages with a decreasing number of
# slices are jointly registered to reference volume
# \date       2017-10-16 10:17:09+0100
#
class HieararchicalSliceSetRegistrationReconstruction(
        ReconstructionRegistrationPipeline):

    ##
    # Store relevant variables
    # \date       2017-10-16 10:18:58+0100
    #
    # \param      self                   The object
    # \param      stacks                 List of stacks to be registered
    # \param      reference              Reference image as Stack object.
    # \param      registration_method    method, e.g. CppItkRegistration
    # \param      reconstruction_method  Reconstruction method, e.g. TK1
    # \param      alpha_range            Specify regularization parameter range
    #                                    used for each individual cycle, list
    #                                    or array
    # \param      interleave             Interleave of scans, integer
    # \param      verbose                The verbose
    #
    def __init__(self,
                 stacks,
                 reference,
                 registration_method,
                 reconstruction_method,
                 alpha_range,
                 interleave,
                 verbose=1,
                 ):

        ReconstructionRegistrationPipeline.__init__(
            self,
            stacks=stacks,
            reference=reference,
            registration_method=registration_method,
            reconstruction_method=reconstruction_method,
            alpha_range=alpha_range,
            verbose=verbose)
        self._interleave = interleave

    def _run(self, debug=1):
        ph.print_title(
            "Hierarchical SliceSet2V-Registration and SRR Reconstruction")

        N_stacks = len(self._stacks)

        # Minimum number of stacks at which no further splitting performed
        N_min = 1
        slice_sets_indices = [None] * N_stacks
        for i, stack in enumerate(self._stacks):
            slice_sets_indices[i] = \
                self._get_slice_set_indices_per_cycle(stack, N_min=N_min)

        # Debug
        if debug:
            for i, stack in enumerate(self._stacks):
                print("Stack %d/%d:" % (i+1, N_stacks))
                for k, v in slice_sets_indices[i].iteritems():
                    print("\tCycle %d: arrays = %s" % (k+1, str(v)))

        N_cycles = np.max([len(slice_sets_indices[i])
                           for i in range(N_stacks)])

        reference = st.Stack.from_stack(self._reference)
        alphas = np.linspace(
            self._alpha_range[0], self._alpha_range[1], N_cycles + 1)
        alphas = alphas[0:N_cycles]

        ctr_iter = [0]
        for i_cycle in range(0, N_cycles):
            self._registration_method.set_moving(reference)

            slice_index_sets_of_stacks = {
                i: (slice_sets_indices[i][i_cycle] if
                    i_cycle in slice_sets_indices[i] else [])
                for i in range(len(self._stacks))
            }

            ss2vreg = SliceSetToVolumeRegistration(
                print_prefix="Cycle %d/%d -- " % (i_cycle+1, N_cycles),
                stacks=self._stacks,
                reference=reference,
                registration_method=self._registration_method,
                slice_index_sets_of_stacks=slice_index_sets_of_stacks,
                verbose=self._verbose,
            )
            ss2vreg.run()
            self._computational_time_registration += \
                ss2vreg.get_computational_time()

            # SRR step
            self._reconstruction_method.set_alpha(alphas[i_cycle])
            self._reconstruction_method.run()

            self._computational_time_reconstruction += \
                self._reconstruction_method.get_computational_time()

            reference = self._reconstruction_method.get_reconstruction()

            # Store SRR
            filename = "Iter%d_%s" % (
                ph.add_one(ctr_iter),
                self._reconstruction_method.get_setting_specific_filename())
            self._reconstructions.insert(0, st.Stack.from_stack(
                reference, filename=filename))
            if self._verbose:
                sitkh.show_stacks(self._reconstructions)

        # Run slice-to-volume registration in case last hierarchical run was
        # not based on individual slices
        if N_min > 1:
            s2vreg = SliceToVolumeRegistration(
                stacks=self._stacks,
                reference=reference,
                registration_method=self._registration_method,
                verbose=self._verbose)
            s2vreg.run()
            self._computational_time_registration += \
                s2vreg.get_computational_time()

            # SRR step
            self._reconstruction_method.set_alpha(alphas[-1])
            self._reconstruction_method.run()

            self._computational_time_reconstruction += \
                self._reconstruction_method.get_computational_time()

            # Store SRR
            filename = "Iter%d_%s" % (
                ph.add_one(ctr_iter),
                self._reconstruction_method.get_setting_specific_filename())
            self._reconstructions.insert(0, st.Stack.from_stack(
                reference, filename=filename))

            if self._verbose:
                sitkh.show_stacks(self._reconstructions)

    ##
    # Gets the slice set indices per cycle.
    # \date       2017-10-16 11:43:29+0100
    #
    # \param      self   The object
    # \param      stack  The stack
    # \param      N_min  Minimum number of stacks at which no further splitting
    #                    shall be performed
    #
    # \return     The slice set indices per cycle as dictionary
    #
    def _get_slice_set_indices_per_cycle(self, stack, N_min):
        N_slices = stack.get_number_of_slices()

        # Separate in packages according to scan interleave
        interleaved_acquisitions = {
            0: [np.arange(i, N_slices, self._interleave)
                for i in range(self._interleave)]
        }

        finished = False
        i = 0

        # Split into smaller subpackages
        while not finished:
            i = i+1

            # Get list of indices based on interleaved acquisition
            interleaved_acquisitions[i] = self._get_array_list_split(
                interleaved_acquisitions[i-1], N_min)

            # Stop if number of elements smaller than N_min. Remark, single
            # index splits can still occur. E.g. [1,3,5] is split into [1,3]
            # and [5] in case of N_min = 2
            if all(len(item) <= N_min for item in interleaved_acquisitions[i]):
                finished = True

        return interleaved_acquisitions

    ##
    # Split list of arrays into halfs.
    # \date       2017-10-16 13:16:50+0100
    #
    # \param      self        The object
    # \param      array_list  The array list
    # \param      N_min       Minimum number of elements at which no further
    #                         split shall be performed
    #
    # \return     List of arrays holding slice indices.
    #
    def _get_array_list_split(self, array_list, N_min):
        new_array_list = []
        for lst in np.atleast_1d(array_list):
            if len(lst) > N_min:
                a = np.array_split(lst, 2)
                new_array_list.extend(np.array_split(lst, 2))
            else:
                new_array_list.extend(np.array([lst]))
                a = np.atleast_1d(lst)
        return new_array_list


##
# Class to perform multi-component reconstruction
#
# Each stack is individually reconstructed at a given reconstruction space
# \date       2017-08-08 02:34:40+0100
#
class MultiComponentReconstruction(Pipeline):

    ##
    # Store information relevant for multi-component reconstruction
    # \date       2017-08-08 02:37:40+0100
    #
    # \param      self                   The object
    # \param      stacks                 The stacks
    # \param      reconstruction_method  The reconstruction method
    # \param      suffix                 Suffix added to filenames of each
    #                                    individual stack, string
    # \param      verbose                The verbose
    #
    def __init__(self,
                 stacks,
                 reconstruction_method,
                 suffix="_recon",
                 verbose=0):

        Pipeline.__init__(self, stacks=stacks, verbose=verbose)

        self._reconstruction_method = reconstruction_method
        self._reconstructions = None
        self._suffix = suffix

    def set_reconstruction_method(self, reconstruction_method):
        self._reconstruction_method = reconstruction_method

    def set_suffix(self, suffix):
        self._suffix = suffix

    def get_suffix(self):
        return self._suffix

    def get_reconstructions(self):
        return [st.Stack.from_stack(stack) for stack in self._reconstructions]

    def _run(self):

        ph.print_title("Multi-Component Reconstruction")

        self._reconstructions = [None] * len(self._stacks)

        for i in range(0, len(self._stacks)):
            ph.print_subtitle("Multi-Component Reconstruction -- "
                              "Stack %d/%d" % (i+1, len(self._stacks)))
            stack = self._stacks[i]
            self._reconstruction_method.set_stacks([stack])
            self._reconstruction_method.run()
            self._reconstructions[i] = st.Stack.from_stack(
                self._reconstruction_method.get_reconstruction())
            self._reconstructions[i].set_filename(
                stack.get_filename() + self._suffix)
