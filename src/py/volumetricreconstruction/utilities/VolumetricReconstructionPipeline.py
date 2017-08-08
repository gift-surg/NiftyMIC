##
# \file VolumetricReconstructionPipeline.py
# \brief      Collection of modules useful for registration and
#             reconstruction tasks.
#
# E.g. Volume-to-Volume Registration, Slice-to-Volume registration,
# Multi-component Reconstruction.
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Aug 2017
#

import datetime
from abc import ABCMeta, abstractmethod

import pythonhelper.PythonHelper as ph

import volumetricreconstruction.base.Stack as st


##
# Class which holds basic interface for all modules
# \date       2017-08-08 02:20:40+0100
#
class Pipeline(object):
    __metaclass__ = ABCMeta

    def __init__(self, stacks, verbose):
        self._stacks = stacks
        self._verbose = verbose

        self._computational_time = datetime.timedelta(seconds=0)

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
    #                                  RegistrationITK
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
            ph.print_subtitle("Volume-to-Volume Registration -- "
                              "Stack %d/%d" % (i+1, len(self._stacks)))

            self._registration_method.set_fixed(self._stacks[i])
            self._registration_method.run_registration()

            # Update position of stack
            transform_sitk = \
                self._registration_method.\
                get_registration_transform_sitk()
            # transform_sitk = eval(
            #     "sitk." + transform_sitk.GetName() +
            #     "(transform_sitk.GetInverse())")
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
    #                                  RegistrationITK
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
            for j in range(0, len(slices)):

                ph.print_subtitle(
                    "%sSlice-to-Volume Registration -- "
                    "Stack %d/%d -- Slice %d/%d" % (
                        self._print_prefix,
                        i+1, len(self._stacks),
                        j+1, len(slices)))

                self._registration_method.set_fixed(slices[j])
                self._registration_method.run_registration()

                # Update position of slice
                transform_sitk = \
                    self._registration_method.\
                    get_registration_transform_sitk()
                slices[j].update_motion_correction(transform_sitk)


##
# Class to perform the two-step Slice-to-Volume registration and volumetric
# reconstruction iteratively
# \date       2017-08-08 02:30:43+0100
#
class TwoStepSliceToVolumeRegistrationReconstruction(RegistrationPipeline):

    ##
    # Store information to perform the two-step S2V reg and recon
    # \date       2017-08-08 02:31:24+0100
    #
    # \param      self                   The object
    # \param      stacks                 The stacks
    # \param      reference              The reference
    # \param      registration_method    Registration method, e.g.
    #                                    RegistrationITK
    # \param      reconstruction_method  Reconstruction method, e.g. TK1
    # \param      alphas                 Specify regularization parameter used
    #                                    for each individual cycle, list or
    #                                    array
    # \param      cycles                 Number of cycles, int
    # \param      verbose                The verbose
    #
    def __init__(self,
                 stacks,
                 reference,
                 registration_method,
                 reconstruction_method,
                 alphas,
                 cycles,
                 verbose=1,
                 ):

        RegistrationPipeline.__init__(
            self,
            stacks=stacks,
            reference=reference,
            registration_method=registration_method,
            verbose=verbose)

        self._reconstruction_method = reconstruction_method
        self._cycles = cycles
        self._alphas = alphas

        self._reconstructions = []
        self._computational_time_registration = datetime.timedelta(seconds=0)
        self._computational_time_reconstruction = datetime.timedelta(seconds=0)

    def get_iterative_reconstructions(self):
        return self._reconstructions

    def get_computational_time_registration(self):
        return self._computational_time_registration

    def get_computational_time_reconstruction(self):
        return self._computational_time_reconstruction

    def _run(self):

        if len(self._alphas) != self._cycles:
            raise ValueError("Number of regularization parameters must match "
                             "number of cycles")
        ph.print_title("Two-step S2V-Registration and SRR Reconstruction")

        self._reconstructions.append(
            st.Stack.from_stack(
                self._reference,
                filename="Iter0_" + self._reference.get_filename()))

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

            self._computational_time_registration = ph.add_times(
                self._computational_time_registration,
                s2vreg.get_computational_time())

            # SRR step
            self._reconstruction_method.set_alpha(self._alphas[cycle])
            self._reconstruction_method.run_reconstruction()

            self._computational_time_reconstruction = ph.add_times(
                self._computational_time_reconstruction,
                self._reconstruction_method.get_computational_time())

            reference = self._reconstruction_method.get_reconstruction()

            # Store SRR
            filename = "Iter%d_%s" % (
                cycle+1,
                self._reconstruction_method.get_setting_specific_filename())
            self._reconstructions.insert(0, st.Stack.from_stack(
                reference, filename=filename))


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
            self._reconstruction_method.run_reconstruction()
            self._reconstructions[i] = st.Stack.from_stack(
                self._reconstruction_method.get_reconstruction())
            self._reconstructions[i].set_filename(
                stack.get_filename() + self._suffix)
