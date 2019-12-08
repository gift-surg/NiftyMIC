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

import six
import numpy as np
import SimpleITK as sitk
from abc import ABCMeta, abstractmethod

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh

import niftymic.base.stack as st
import niftymic.validation.motion_evaluator as me
import niftymic.utilities.outlier_rejector as outre
import niftymic.utilities.robust_motion_estimator as rme
import niftymic.registration.transform_initializer as tinit
import niftymic.reconstruction.scattered_data_approximation as sda
import niftymic.utilities.binary_mask_from_mask_srr_estimator as bm

from niftymic.definitions import VIEWER


##
# Class which holds basic interface for all modules
# \date       2017-08-08 02:20:40+0100
#
class Pipeline(object):
    __metaclass__ = ABCMeta

    def __init__(self, stacks, verbose, viewer):
        self._stacks = stacks
        self._verbose = verbose
        self._viewer = viewer

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
    #                                  SimpleItkRegistration
    #
    def __init__(self, verbose, stacks, reference, registration_method, viewer):

        Pipeline.__init__(self, stacks=stacks, verbose=verbose, viewer=viewer)

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
                 viewer=VIEWER,
                 robust=False,
                 ):
        RegistrationPipeline.__init__(
            self,
            stacks=stacks,
            reference=reference,
            registration_method=registration_method,
            viewer=viewer,
            verbose=verbose,
        )
        self._robust = robust

    def _run(self):

        ph.print_title("Volume-to-Volume Registration")

        for i in range(0, len(self._stacks)):
            txt = "Volume-to-Volume Registration -- " \
                "Stack %d/%d" % (i + 1, len(self._stacks))
            if self._verbose:
                ph.print_subtitle(txt)
            else:
                ph.print_info(txt)

            if self._robust:
                transform_initializer = tinit.TransformInitializer(
                    fixed=self._reference,
                    moving=self._stacks[i],
                    similarity_measure="NCC",
                    refine_pca_initializations=True,
                )
                transform_initializer.run()
                transform_sitk = transform_initializer.get_transform_sitk()
                transform_sitk = sitk.AffineTransform(
                    transform_sitk.GetInverse())

            else:
                self._registration_method.set_moving(self._reference)
                self._registration_method.set_fixed(self._stacks[i])
                self._registration_method.run()
                transform_sitk = self._registration_method.get_registration_transform_sitk()

            # Update position of stack
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
    #                                  SimpleItkRegistration
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
                 s2v_smoothing=None,
                 interleave=2,
                 viewer=VIEWER,
                 ):
        RegistrationPipeline.__init__(
            self,
            stacks=stacks,
            reference=reference,
            registration_method=registration_method,
            verbose=verbose,
            viewer=viewer,
        )
        self._print_prefix = print_prefix
        self._s2v_smoothing = s2v_smoothing
        self._interleave = interleave

    def set_print_prefix(self, print_prefix):
        self._print_prefix = print_prefix

    def set_s2v_smoothing(self, s2v_smoothing):
        self._s2v_smoothing = s2v_smoothing

    def get_s2v_smoothing(self):
        return self._s2v_smoothing

    def _run(self):

        ph.print_title("Slice-to-Volume Registration")

        self._registration_method.set_moving(self._reference)

        for i, stack in enumerate(self._stacks):
            slices = stack.get_slices()

            transforms_sitk = {}

            for j, slice_j in enumerate(slices):

                txt = "%sSlice-to-Volume Registration -- " \
                    "Stack %d/%d (%s) -- Slice %d/%d" % (
                        self._print_prefix,
                        i + 1, len(self._stacks), stack.get_filename(),
                        j + 1, len(slices))
                if self._verbose:
                    ph.print_subtitle(txt)
                else:
                    ph.print_info(txt)

                self._registration_method.set_fixed(slice_j)
                self._registration_method.run()

                # Store information on registration transform
                transform_sitk = \
                    self._registration_method.get_registration_transform_sitk()
                transforms_sitk[slice_j.get_slice_number()] = transform_sitk

            # Avoid slice misregistrations
            if self._s2v_smoothing is not None:
                # import os
                # for slice_number in transforms_sitk.keys():
                #     path_to_file = os.path.join(
                #         "/tmp/fetal_brain", "%s_slice%d.tfm" % (
                #             stack.get_filename(), slice_number))
                #     sitk.WriteTransform(
                #         transforms_sitk[slice_number], path_to_file)
                ph.print_subtitle(
                    "Robust slice motion estimation "
                    "(GP smoothing = %g, interleave = %d)" % (
                        self._s2v_smoothing, self._interleave))
                robust_motion_estimator = rme.RobustMotionEstimator(
                    transforms_sitk=transforms_sitk,
                    interleave=self._interleave)
                robust_motion_estimator.run(self._s2v_smoothing)
                transforms_sitk = \
                    robust_motion_estimator.get_robust_transforms_sitk()

                # Update position of slice
                for slice in slices:
                    slice_number = slice.get_slice_number()
                    slice.update_motion_correction(
                        transforms_sitk[slice_number])

                # Run s2v-reg again
                for j, slice_j in enumerate(slices):
                    txt = "%sSlice-to-Volume Registration -- " \
                        "Stack %d/%d -- Slice %d/%d (after GP init)" % (
                            self._print_prefix,
                            i + 1, len(self._stacks),
                            j + 1, len(slices))
                    if self._verbose:
                        ph.print_subtitle(txt)
                    else:
                        ph.print_info(txt)

                    self._registration_method.set_fixed(slice_j)
                    self._registration_method.run()

                    # Store information on registration transform
                    transform_sitk = \
                        self._registration_method.get_registration_transform_sitk()
                    transforms_sitk[
                        slice_j.get_slice_number()] = transform_sitk

                # Export figures
                # title = "%s_Stack%d%s" % (
                #     self._print_prefix, i, stack.get_filename())
                # title = ph.replace_string_for_print(title)
                # robust_motion_estimator.show_estimated_transform_parameters(
                #     dir_output="/tmp/fetal_brain/figs", title=title)

            # dir_output = "/tmp/fetal/figs"
            # motion_evaluator = me.MotionEvaluator(transforms_sitk)
            # motion_evaluator.run()
            # motion_evaluator.display(dir_output=dir_output, title=title)
            # motion_evaluator.show(dir_output=dir_output, title=title)

            # Update position of slice
            for slice in slices:
                slice_number = slice.get_slice_number()
                slice.update_motion_correction(transforms_sitk[slice_number])


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
                 stack,
                 reference,
                 registration_method,
                 slice_set_indices,
                 verbose=1,
                 print_prefix="",
                 viewer=VIEWER,
                 ):
        RegistrationPipeline.__init__(
            self,
            stacks=[stack],
            reference=reference,
            registration_method=registration_method,
            verbose=verbose,
            viewer=viewer,
        )

        self._print_prefix = print_prefix
        self._slice_set_indices = slice_set_indices

    def _run(self, debug=1):

        stack = self._stacks[0]
        slices = stack.get_slices()
        for i, indices in enumerate(self._slice_set_indices):
            txt = "%s Split %d/%d -- Slices %s" % (
                self._print_prefix, i + 1,
                len(self._slice_set_indices), str(indices))
            if self._verbose:
                ph.print_subtitle(txt)
            else:
                ph.print_info(txt)

                image = self._get_stack_subgroup(indices)

                if debug:
                    first = np.linalg.norm(
                        stack.get_slice(indices[0]).sitk.GetOrigin() -
                        np.array(image.sitk[:, :, 0:1].GetOrigin()))
                    last = np.linalg.norm(
                        stack.get_slice(indices[-1]).sitk.GetOrigin() -
                        np.array(image.sitk[:, :, -1:].GetOrigin()))
                    if first > 1e-6:
                        raise RuntimeError(
                            "Hierarchical S2V: first slice position flawed")
                    if last > 1e-6:
                        raise RuntimeError(
                            "Hierarchical S2V: last slice position flawed")

                self._registration_method.set_fixed(image)
                self._registration_method.run()
                transform_sitk = self._registration_method.\
                    get_registration_transform_sitk()

                for j in indices:
                    slices[j].update_motion_correction(transform_sitk)

                # if debug:
                #     image_after = self._get_stack_subgroup(indices)
                #     ph.killall_itksnap()
                #     print(stack.get_filename())
                #     sitkh.show_stacks(
                #         [self._reference, image, image_after],
                #         label=["reference", "before", "after"]
                #         # segmentation=image,
                #     )

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
    def _get_stack_subgroup(self, indices):

        stack = self._stacks[0]

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
            extract_slices=False,
            slice_thickness=stack.get_slice_thickness(),
        )

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
    #                                    SimpleItkRegistration
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
                 alphas,
                 viewer,
                 ):

        RegistrationPipeline.__init__(
            self,
            verbose=verbose,
            stacks=stacks,
            reference=reference,
            registration_method=registration_method,
            viewer=viewer,
        )

        self._reconstruction_method = reconstruction_method
        self._alphas = alphas

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
    # \param      self                           The object
    # \param      stacks                         The stacks
    # \param      reference                      The reference
    # \param      registration_method            Registration method, e.g.
    #                                            SimpleItkRegistration
    # \param      reconstruction_method          Reconstruction method, e.g.
    #                                            TK1
    # \param      alphas                         List of alphas
    #                                            array
    # \param      verbose                        The verbose
    # \param      cycles                         Number of cycles, int
    # \param      outlier_rejection              The outlier rejection
    # \param      threshold_measure              The threshold measure
    # \param      thresholds                     The threshold range
    # \param      use_robust_registration        The use robust registration
    # \param      use_hierarchical_registration  The use hierarchical registration
    # \param      s2v_smoothing                  The s 2 v smoothing
    # \param      interleave                     The interleave
    # \param      viewer                         The viewer
    # \param      sigma_sda_mask                 The sigma sda mask
    #
    def __init__(self,
                 stacks,
                 reference,
                 registration_method,
                 reconstruction_method,
                 alphas,
                 verbose=1,
                 cycles=3,
                 outlier_rejection=False,
                 threshold_measure="NCC",
                 thresholds=[0.6, 0.7, 0.8],
                 use_robust_registration=False,
                 use_hierarchical_registration=False,
                 s2v_smoothing=0.5,
                 interleave=3,
                 viewer=VIEWER,
                 sigma_sda_mask=1.,
                 ):

        # Last volumetric reconstruction step is performed outside
        if len(alphas) != cycles - 1:
            raise ValueError(
                "Elements in alpha list must correspond to cycles-1")

        if outlier_rejection and len(thresholds) != cycles:
            raise ValueError(
                "Elements in outlier rejection threshold list must "
                "correspond to the number of cycles")

        ReconstructionRegistrationPipeline.__init__(
            self,
            stacks=stacks,
            reference=reference,
            registration_method=registration_method,
            reconstruction_method=reconstruction_method,
            alphas=alphas,
            viewer=viewer,
            verbose=verbose,
        )

        self._sigma_sda_mask = sigma_sda_mask

        self._cycles = cycles
        self._outlier_rejection = outlier_rejection
        self._threshold_measure = threshold_measure
        self._thresholds = thresholds
        self._use_robust_registration = use_robust_registration
        self._use_hierarchical_registration = use_hierarchical_registration
        self._s2v_smoothing = s2v_smoothing
        self._interleave = interleave

    def _run(self):

        ph.print_title("Two-step S2V-Registration and SRR Reconstruction")

        s2vreg = SliceToVolumeRegistration(
            stacks=self._stacks,
            reference=self._reference,
            registration_method=self._registration_method,
            verbose=False,
            interleave=self._interleave,
        )

        reference = self._reference

        for cycle in range(0, self._cycles):

            if cycle == 0 and self._use_hierarchical_registration:
                hs2vreg = HieararchicalSliceSetRegistration(
                    stacks=self._stacks,
                    reference=reference,
                    registration_method=self._registration_method,
                    interleave=self._interleave,
                    viewer=self._viewer,
                    min_slices=1,
                    verbose=False,
                )
                hs2vreg.run()
                self._computational_time_registration += \
                    hs2vreg.get_computational_time()
            else:
                # Slice-to-volume registration step
                s2vreg.set_reference(reference)
                s2vreg.set_print_prefix("Cycle %d/%d: " %
                                        (cycle + 1, self._cycles))
                if self._use_robust_registration and cycle == 0:
                    s2vreg.set_s2v_smoothing(self._s2v_smoothing)
                else:
                    s2vreg.set_s2v_smoothing(None)
                s2vreg.run()

            self._computational_time_registration += \
                s2vreg.get_computational_time()

            # Reject misregistered slices
            if self._outlier_rejection:
                ph.print_subtitle("Slice Outlier Rejection (%s < %g)" % (
                    self._threshold_measure, self._thresholds[cycle]))
                outlier_rejector = outre.OutlierRejector(
                    stacks=self._stacks,
                    reference=self._reference,
                    threshold=self._thresholds[cycle],
                    measure=self._threshold_measure,
                    verbose=True,
                )
                outlier_rejector.run()
                self._reconstruction_method.set_stacks(
                    outlier_rejector.get_stacks())

                if len(self._stacks) == 0:
                    raise RuntimeError(
                        "All slices of all stacks were rejected "
                        "as outliers. Volumetric reconstruction is aborted.")

            # SRR step
            if cycle < self._cycles - 1:
                # ---------------- Perform Image Reconstruction ---------------
                ph.print_subtitle("Volumetric Image Reconstruction")
                if isinstance(
                    self._reconstruction_method,
                    sda.ScatteredDataApproximation
                ):
                    self._reconstruction_method.set_sigma(self._alphas[cycle])
                else:
                    self._reconstruction_method.set_alpha(self._alphas[cycle])
                self._reconstruction_method.run()

                self._computational_time_reconstruction += \
                    self._reconstruction_method.get_computational_time()

                reference = self._reconstruction_method.get_reconstruction()

                # ------------------ Perform Image Mask SDA -------------------
                ph.print_subtitle("Volumetric Image Mask Reconstruction")
                SDA = sda.ScatteredDataApproximation(
                    self._stacks,
                    reference,
                    sigma=self._sigma_sda_mask,
                    sda_mask=True,
                )
                SDA.run()

                # reference contains updated mask based on SDA
                reference = SDA.get_reconstruction()

                # -------------------- Store Reconstruction -------------------
                filename = "Iter%d_%s" % (
                    cycle + 1,
                    self._reconstruction_method.get_setting_specific_filename()
                )
                self._reconstructions.insert(0, st.Stack.from_stack(
                    reference, filename=filename))

                if self._verbose:
                    sitkh.show_stacks(self._reconstructions,
                                      segmentation=self._reference,
                                      viewer=self._viewer)


##
# Class to perform hierarchical slice alignment.
#
# Given an interleave, associated subpackages with a decreasing number of
# slices are jointly registered to reference volume
# \date       2017-10-16 10:17:09+0100
#
class HieararchicalSliceSetRegistration(RegistrationPipeline):

    ##
    # Store relevant variables
    # \date       2017-10-16 10:18:58+0100
    #
    # \param      self                 The object
    # \param      stacks               List of stacks to be registered
    # \param      reference            Reference image as Stack object.
    # \param      registration_method  method, e.g. SimpleItkRegistration
    # \param      interleave           Interleave of scans, integer
    # \param      min_slices           The minimum slices
    # \param      verbose              The verbose
    # \param      viewer               The viewer
    #
    def __init__(self,
                 stacks,
                 reference,
                 registration_method,
                 interleave,
                 min_slices=1,
                 verbose=1,
                 viewer=VIEWER,
                 ):

        RegistrationPipeline.__init__(
            self,
            stacks=stacks,
            reference=reference,
            registration_method=registration_method,
            verbose=verbose,
            viewer=VIEWER,
        )
        self._interleave = interleave
        self._min_slices = min_slices

    def _run(self, debug=0):
        ph.print_title(
            "Hierarchical SliceSet2V-Registration")

        N_stacks = len(self._stacks)

        self._registration_method.set_moving(self._reference)

        for i_stack, stack in enumerate(self._stacks):
            n_slices = stack.get_number_of_slices()
            for i in range(self._interleave):
                package = list(np.arange(i, n_slices, self._interleave))
                if len(package) / 2 >= self._min_slices:
                    indices_splits = self._recursive_split(
                        package, [], self._min_slices)
                else:
                    indices_splits = [package]

                prefix = "Hierarchical S2V-Reg: " \
                    "Stack %d/%d (%s) -- Interleave %d/%d --" % (
                        i_stack + 1, len(self._stacks), stack.get_filename(),
                        i + 1, self._interleave,
                    )
                if debug:
                    ph.print_subtitle(
                        "%s %d splits: %s" % (
                            prefix, len(indices_splits), indices_splits),
                    )

                ss2vreg = SliceSetToVolumeRegistration(
                    print_prefix=prefix,
                    stack=stack,
                    reference=self._reference,
                    registration_method=self._registration_method,
                    slice_set_indices=indices_splits,
                    verbose=self._verbose,
                )
                ss2vreg.run()

    ##
    # Split list of arrays into halfs.
    # \date       2017-10-16 13:16:50+0100
    #
    # \param      self           The object
    # \param      indices        The indices
    # \param      indices_split  The indices split
    # \param      N_min          Minimum number of elements at which no further
    #                            split shall be performed
    #
    # \return     List of arrays holding slice indices.
    #
    def _recursive_split(self, indices, indices_split, N_min):
        mid = int(len(indices) / 2)

        a = indices[0:mid]
        b = indices[mid:]

        indices_split.append(a)
        indices_split.append(b)

        if len(a) / 2 >= N_min:
            self._recursive_split(a, indices_split, N_min)

        if len(b) / 2 >= N_min:
            self._recursive_split(b, indices_split, N_min)

        return indices_split


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
                 verbose=0,
                 viewer=VIEWER,
                 ):

        Pipeline.__init__(self, stacks=stacks, verbose=verbose, viewer=viewer)

        self._reconstruction_method = reconstruction_method
        self._reconstructions = None
        self._suffix = suffix

    def set_reconstruction_method(self, reconstruction_method):
        self._reconstruction_method = reconstruction_method

    def get_reconstruction_method(self):
        return self._reconstruction_method

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
                              "Stack %d/%d" % (i + 1, len(self._stacks)))
            stack = self._stacks[i]
            self._reconstruction_method.set_stacks([stack])
            self._reconstruction_method.run()
            self._reconstructions[i] = st.Stack.from_stack(
                self._reconstruction_method.get_reconstruction())
            self._reconstructions[i].set_filename(
                stack.get_filename() + self._suffix)
