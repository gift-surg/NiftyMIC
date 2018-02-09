##
# \file residual_evaluator.py
# \brief      Class to evaluate computed residuals between a
#             simulated/projected and original/acquired slices of stacks
#
# Should help to assess the registration accuracy.
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       January 2018
#


# Import libraries
import os
import re
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

from nsol.similarity_measures import SimilarityMeasures as \
    SimilarityMeasures
import pysitk.python_helper as ph
import pysitk.statistics_helper as sh

import niftymic.reconstruction.linear_operators as lin_op
import niftymic.base.exceptions as exceptions


##
# Class to evaluate computed residuals between a simulated/projected and
# original/acquired slices of stacks
#
# \date       2018-01-19 17:24:35+0000
#
class ResidualEvaluator(object):

    ##
    # { constructor_description }
    # \date       2018-01-19 17:24:46+0000
    #
    # \param      self       The object
    # \param      stacks     List of Stack objects
    # \param      reference  Reference as Stack object. Used to simulate slices
    #                        at the position of the slices in stacks
    # \param      use_masks  Turn on/off using masks for the residual
    #                        evaluation
    # \param      measures   Similarity measures as given in
    #                        nsol.similarity_measures, list of strings
    #
    def __init__(
            self,
            stacks=None,
            reference=None,
            use_slice_masks=True,
            use_reference_mask=True,
            measures=["NCC", "NMI", "PSNR", "SSIM", "RMSE"],
            verbose=True,
    ):
        self._stacks = stacks
        self._reference = reference
        self._measures = measures
        self._use_slice_masks = use_slice_masks
        self._use_reference_mask = use_reference_mask
        self._verbose = verbose

        self._slice_projections = None
        self._similarities = None
        self._slice_similarities = None

    ##
    # Sets the stacks.
    # \date       2018-01-19 17:26:04+0000
    #
    # \param      self    The object
    # \param      stacks  List of Stack objects
    #
    def set_stacks(self, stacks):
        self._stacks = stacks

    ##
    # Sets the reference from which the slices shall be simulated/projected.
    # \date       2018-01-19 17:26:14+0000
    #
    # \param      self       The object
    # \param      reference  The reference
    #
    def set_reference(self, reference):
        self._reference = reference

    def get_measures(self):
        return self._measures

    ##
    # Gets the slice similarities computed between simulated/projected and
    # original/acquired slices.
    # \date       2018-01-19 17:26:44+0000
    #
    # \param      self  The object
    #
    # \return     The slice similarities for all slices and measures as
    #             dictionary. E.g. {
    #             fetal_brain_1: {'NCC': 1D-array[...], 'NMI': 1D-array[..]},
    #             ...
    #             fetal_brain_N: {'NCC': 1D-array[...], 'NMI': 1D-array[..]}
    #             }
    #
    def get_slice_similarities(self):
        return self._slice_similarities

    ##
    # Gets the slice projections.
    # \date       2018-01-19 17:27:41+0000
    #
    # \param      self  The object
    #
    # \return     The slice projections as list of lists. E.g. [
    #             [stack1_slice1_sim, stack1_slice2_sim, ...], ...
    #             [stackN_slice1_sim, stackN_slice2_sim, ...]
    #             ]
    #
    def get_slice_projections(self):
        return self._slice_projections

    ##
    # Calculates the slice simulations/projections from the reference given the
    # assumed slice acquisition protocol.
    # \date       2018-01-19 17:29:20+0000
    #
    # \param      self  The object
    #
    # \return     The slice projections.
    #
    def compute_slice_projections(self):

        linear_operators = lin_op.LinearOperators()
        self._slice_projections = [None] * len(self._stacks)

        for i_stack, stack in enumerate(self._stacks):
            slices = stack.get_slices()
            self._slice_projections[i_stack] = [None] * len(slices)

            if self._verbose:
                ph.print_info(
                    "Stack %d/%d: Compute slice projections ... " % (
                        i_stack + 1, len(self._stacks)),
                    newline=False)

            # Compute slice projections based on assumed slice acquisition
            # protocol
            for i_slice, slice in enumerate(slices):
                self._slice_projections[i_stack][i_slice] = linear_operators.A(
                    self._reference, slice)

            if self._verbose:
                print("done")

    ##
    # Evaluate slice similarities for all simulated slices of all stacks for
    # all similarity measures.
    # \date       2018-01-19 17:30:37+0000
    #
    # \param      self  The object
    #
    def evaluate_slice_similarities(self):
        if self._slice_projections is None:
            raise exceptions.ObjectNotCreated("compute_slice_projections")

        self._slice_similarities = {
            stack.get_filename(): {} for stack in self._stacks
        }

        similarity_measures = {
            m: SimilarityMeasures.similarity_measures[m]
            for m in self._measures
        }

        for i_stack, stack in enumerate(self._stacks):
            slices = stack.get_slices()
            stack_name = stack.get_filename()
            self._slice_similarities[stack_name] = {
                m: np.zeros(len(slices)) for m in self._measures
            }
            if self._verbose:
                ph.print_info(
                    "Stack %d/%d: Compute similarity measures ... " % (
                        i_stack + 1, len(self._stacks)),
                    newline=False)

            for i_slice, slice in enumerate(slices):

                slice_nda = np.squeeze(sitk.GetArrayFromImage(slice.sitk))
                slice_proj_nda = np.squeeze(sitk.GetArrayFromImage(
                    self._slice_projections[i_stack][i_slice].sitk))

                mask_nda = np.ones_like(slice_nda)

                if self._use_slice_masks:
                    mask_nda *= np.squeeze(
                        sitk.GetArrayFromImage(slice.sitk_mask))
                if self._use_reference_mask:
                    mask_nda *= np.squeeze(
                        sitk.GetArrayFromImage(
                            self._slice_projections[i_stack][i_slice].sitk_mask))
                indices = np.where(mask_nda > 0)

                if len(indices[0]) > 0:
                    for m in self._measures:
                        try:
                            self._slice_similarities[stack_name][m][i_slice] = \
                                similarity_measures[m](
                                    slice_nda[indices], slice_proj_nda[indices])
                        except ValueError as e:
                            # Error in case only a few/to less non-zero entries
                            # exist
                            if m == "SSIM":
                                self._slice_similarities[
                                    stack_name][m][i_slice] = \
                                    SimilarityMeasures.UNDEF[m]
                            else:
                                raise ValueError(e.message)
                else:
                    for m in self._measures:
                        self._slice_similarities[
                            stack_name][m][i_slice] = \
                            SimilarityMeasures.UNDEF[m]
            if self._verbose:
                print("done")

    ##
    # Writes the computed slice similarities for all stacks to output directory
    # \date       2018-01-19 17:42:27+0000
    #
    # \param      self       The object
    # \param      directory  path to output directory, string
    #
    def write_slice_similarities(self, directory):
        for i_stack, stack in enumerate(self._stacks):
            stack_name = stack.get_filename()
            path_to_file = os.path.join(
                directory, "%s.txt" % stack_name)

            # Write header info
            header = "# %s, %s\n" % (stack.get_filename(), ph.get_time_stamp())
            header += "# %s\n" % ("\t").join(self._measures)
            ph.write_to_file(path_to_file, header, verbose=self._verbose)

            # Write array information
            array = np.zeros(
                (stack.get_number_of_slices(), len(self._measures)))
            for i_m, m in enumerate(self._measures):
                array[:, i_m] = self._slice_similarities[stack_name][m]
            ph.write_array_to_file(path_to_file, array, verbose=self._verbose)

    ##
    # Reads computed slice similarities for all files in directory.
    # \date       2018-01-19 17:42:54+0000
    #
    # \param      self       The object
    # \param      directory  The directory
    # \param      ext        The extent
    # \post       self._slice_similarities updated
    #
    def read_slice_similarities(self, directory, ext="txt"):

        if not ph.directory_exists(directory):
            raise IOError("Given directory '%s' does not exist" % (
                directory))

        pattern = "([a-zA-Z0-9_\+\-]+)[.]%s" % ext
        p = re.compile(pattern)

        stack_names = [
            p.match(f).group(1)
            for f in os.listdir(directory) if p.match(f)
        ]

        self._slice_similarities = {
            stack_name: {} for stack_name in stack_names
        }

        for stack_name in stack_names:
            path_to_file = os.path.join(directory, "%s.%s" % (stack_name, ext))

            # Read computed measures
            self._measures = ph.read_file_line_by_line(path_to_file)[1]
            self._measures = re.sub("# ", "", self._measures)
            self._measures = re.sub("\n", "", self._measures)
            self._measures = self._measures.split("\t")

            # Read array
            array = np.loadtxt(path_to_file, skiprows=2)
            if array.ndim == 1:
                array = array.reshape(len(array), 1)

            for i_m, m in enumerate(self._measures):
                self._slice_similarities[stack_name][m] = array[:, i_m]

    ##
    # Shows the slice similarities in plots.
    # \date       2018-02-09 18:28:45+0000
    #
    # \param      self       The object
    # \param      directory  The directory
    # \param      title      The title
    # \param      measures   The measures
    # \param      threshold  The threshold
    #
    def show_slice_similarities(
            self,
            directory=None,
            title=None,
            measures=["NCC"],
            threshold=0.8,
            ):

        for i_m, measure in enumerate(measures):
            fig = plt.figure(measure)
            fig.clf()
            if title is not None:
                title = "%s: %s" % (title, measure)
            else:
                title = measure
            plt.suptitle(title)

            stack_names = self._slice_similarities.keys()
            for i_name, stack_name in enumerate(stack_names):
                ax = plt.subplot(np.ceil(len(stack_names) / 2.), 2, i_name + 1)
                nda = self._slice_similarities[stack_name][measure]

                nda = np.nan_to_num(nda)

                indices_in = np.where(nda >= threshold)
                indices_out = np.where(nda < threshold)

                plt.plot(indices_in[0], nda[indices_in],
                    color=ph.COLORS_TABLEAU20[0],
                    markerfacecolor="w",
                    marker=ph.MARKERS[0],
                    linestyle="",
                    )
                plt.plot(indices_out[0], nda[indices_out],
                    color=ph.COLORS_TABLEAU20[6],
                    markerfacecolor="w",
                    marker=ph.MARKERS[2],
                    linestyle="",
                    )

                plt.xlabel("Slice")
                # plt.ylabel(measure)
                plt.title(stack_name)

                x = np.arange(nda.size)
                ax.set_xticks(x)
                # ax.set_xticklabels(x + 1)
                ax.set_ylim([0, 1])

            sh.make_figure_fullscreen()
            plt.show(block=False)

            if directory is not None:
                filename = "slice_similarities_%s.pdf" % measure
                ph.save_fig(fig, directory, filename)
