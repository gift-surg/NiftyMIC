##
# \file image_similarity_evaluator.py
# \brief      Class to evaluate image similarity between stacks and a reference
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       February 2018
#


# Import libraries
import os
import re
import numpy as np
import SimpleITK as sitk

import nsol.observer as obs
from nsol.similarity_measures import SimilarityMeasures as \
    SimilarityMeasures
import pysitk.python_helper as ph

import niftymic.reconstruction.linear_operators as lin_op
import niftymic.base.exceptions as exceptions


##
# Class to evaluate image similarity between stacks and a reference
# \date       2018-02-08 16:16:08+0000
#
class ImageSimilarityEvaluator(object):

    ##
    # { constructor_description }
    # \date       2018-02-08 14:13:19+0000
    #
    # \param      self                The object
    # \param      stacks              List of Stack objects
    # \param      reference           Reference as Stack object
    # \param      use_reference_mask  The use reference mask
    # \param      measures            Similarity measures as given in
    #                                 nsol.similarity_measures, list of strings
    # \param      verbose             The verbose
    #
    def __init__(
            self,
            stacks=None,
            reference=None,
            use_reference_mask=True,
            measures=["NCC", "NMI", "PSNR", "SSIM", "RMSE" ,"MAE"],
            verbose=True,
    ):
        self._stacks = stacks
        self._reference = reference
        self._measures = measures
        self._use_reference_mask = use_reference_mask
        self._verbose = verbose

        self._similarities = None

        self._filename_filenames = "filenames.txt"
        self._filename_similarities = "similarities.txt"

    ##
    # Sets the stacks.
    # \date       2018-02-08 14:13:27+0000
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

    ##
    # Gets the computed similarities.
    # \date       2018-02-08 14:36:15+0000
    #
    # \param      self  The object
    #
    # \return     The similarities as dictionary. E.g. { "NCC": np.array()}
    #
    def get_similarities(self):
        return self._similarities

    def get_measures(self):
        return self._measures

    ##
    # Calculates the similarities. Outcome can be fetched using
    # 'get_similarities'
    # \date       2018-02-08 16:16:41+0000
    #
    # \param      self  The object
    #
    # \post       self._similarities updated.
    #
    def compute_similarities(self):
        for stack in self._stacks:
            try:
                stack.sitk - self._reference.sitk
            except RuntimeError as e:
                raise IOError(
                    "All provided images must be at the same image space")

        x_ref = sitk.GetArrayFromImage(self._reference.sitk)

        x_ref_mask = np.ones_like(x_ref)
        if self._use_reference_mask:
            x_ref_mask *= sitk.GetArrayFromImage(self._reference.sitk_mask)
        indices = np.where(x_ref_mask > 0)

        if len(indices[0]) == 0:
            raise RuntimeError(
                "Support to evaluate similarity measures is zero")

        # Define similarity measures as dic
        measures_dic = {
            m: lambda x, m=m:
            SimilarityMeasures.similarity_measures[m](
                x[indices], x_ref[indices])
            # SimilarityMeasures.similarity_measures[m](x, x_ref)
            for m in self._measures
        }

        # Compute similarities
        observer = obs.Observer()
        observer.set_measures(measures_dic)
        for stack in self._stacks:
            nda = sitk.GetArrayFromImage(stack.sitk)
            observer.add_x(nda)
        observer.compute_measures()
        self._similarities = observer.get_measures()

        # Add filenames to dictionary
        image_names = [s.get_filename() for s in self._stacks]
        self._similarities["filenames"] = image_names

    ##
    # Writes the evaluated similarities to two files; one containing the
    # similarity information, the other the filename information.
    # \date       2018-02-08 14:58:29+0000
    #
    # \param      self       The object
    # \param      directory  The directory
    #
    def write_similarities(self, directory):

        # Store information in array
        similarities_nda = np.zeros((len(self._stacks), len(self._measures)))
        filenames = []
        for i_stack, stack in enumerate(self._stacks):
            similarities_nda[i_stack, :] = np.array(
                [self._similarities[m][i_stack] for m in self._measures])
            filenames.append(stack.get_filename())

        # Build header of files
        header = "# Ref: %s, Ref-Mask: %d, %s \n" % (
            self._reference.get_filename(),
            self._use_reference_mask,
            ph.get_time_stamp(),
        )
        header += "# %s\n" % ("\t").join(self._measures)

        # Get filename paths
        path_to_file_filenames, path_to_file_similarities = self._get_filename_paths(
            directory)

        # Write similarities
        ph.write_to_file(path_to_file_similarities, header)
        ph.write_array_to_file(
            path_to_file_similarities, similarities_nda, verbose=self._verbose)

        # Write stack filenames
        text = header
        text += "%s\n" % "\n".join(filenames)
        ph.write_to_file(path_to_file_filenames, text, verbose=self._verbose)

    ##
    # Reads similarities.
    # \date       2018-02-08 15:32:04+0000
    #
    # \param      self       The object
    # \param      directory  The directory
    #
    def read_similarities(self, directory):
        
        if not ph.directory_exists(directory):
            raise IOError("Directory '%s' does not exist." % directory)

        # Get filename paths
        path_to_file_filenames, path_to_file_similarities = self._get_filename_paths(
            directory)

        for f in [path_to_file_filenames, path_to_file_similarities]:
            if not ph.file_exists(path_to_file_filenames):
                raise IOError("File '%s' does not exist" % f)

        lines = ph.read_file_line_by_line(path_to_file_filenames)

        # Get image filenames
        image_names = [re.sub("\n", "", f) for f in lines[2:]]

        # Get computed measures
        measures = lines[1]
        measures = re.sub("# ", "", measures)
        measures = re.sub("\n", "", measures)
        self._measures = measures.split("\t")

        # Get computed similarities
        similarities_nda = np.loadtxt(path_to_file_similarities, skiprows=2)

        # Reconstruct similarity dictionary
        self._similarities = {}
        self._similarities["filenames"] = image_names
        for i_m, m in enumerate(self._measures):
            self._similarities[m] = similarities_nda[:, i_m]

    def _get_filename_paths(self, directory):

        # Define filename paths
        path_to_file_filenames = os.path.join(
            directory, self._filename_filenames)
        path_to_file_similarities = os.path.join(
            directory, self._filename_similarities)

        return path_to_file_filenames, path_to_file_similarities
