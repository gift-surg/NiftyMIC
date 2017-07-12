# \file DataReader.py
#  \brief Reads data and returns Stack objects
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date July 2017


# Import libraries
from abc import ABCMeta, abstractmethod
import os                       # used to execute terminal commands in python
import sys
import SimpleITK as sitk
import numpy as np
import re
import natsort

# Import modules from src-folder
import base.Stack as st
import utilities.PythonHelper as ph
import utilities.Exceptions as Exceptions

from definitions import REGEX_FILENAMES
from definitions import REGEX_FILENAME_EXTENSIONS

##
# DataReader is an abstract class to read 3D images.
# \date       2017-07-12 11:38:07+0100
#
class DataReader(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self._stacks = None

    @abstractmethod
    def read_data(self):
        pass

    ##
    # Returns the read data as list of Stack objects
    # \date       2017-07-12 11:38:52+0100
    #
    # \param      self  The object
    #
    # \return     The stacks.
    #
    def get_stacks(self):

        if type(self._stacks) is not list:
            raise Exceptions.ObjectNotCreated("read_data")

        return self._stacks


##
# DirectoryReader reads images and their masks from a given directory and
# returns them as a list of Stack objects.
# \date       2017-07-12 11:36:22+0100
#
class DirectoryReader(DataReader):

    ##
    # Store relevant information to images and their potential masks from a
    # specified directory
    # \date       2017-07-11 19:04:25+0100
    #
    # \param      self               The object
    # \param      path_to_directory  String to specify the path to input
    #                                directory
    # \param      suffix_mask        extension of stack filename as string
    #                                indicating associated mask, e.g. "_mask"
    #                                for "A_mask.nii".
    # \param      extract_slices     Boolean to indicate whether given 3D image
    #                                shall be split into its slices along the
    #                                k-direction.
    #
    def __init__(self,
                 path_to_directory,
                 suffix_mask="_mask",
                 extract_slices=True):

        super(self.__class__, self).__init__()

        self._path_to_directory = path_to_directory
        self._suffix_mask = suffix_mask
        self._extract_slices = extract_slices

    ##
    # Reads the image data from the given folder.
    # \date       2017-07-11 17:10:40+0100
    #
    def read_data(self):

        if not ph.directory_exists(self._path_to_directory):
            raise Exceptions.DirectoryNotExistent(self._path_to_directory)

        abs_path_to_directory = os.path.abspath(self._path_to_directory)

        # Get data filenames of images without filename extension
        pattern = "(" + REGEX_FILENAMES + ")[.]" + REGEX_FILENAME_EXTENSIONS
        pattern_mask = "(" + REGEX_FILENAMES + ")" + self._suffix_mask + \
            "[.]" + REGEX_FILENAME_EXTENSIONS
        p = re.compile(pattern)
        p_mask = re.compile(pattern_mask)

        # Exclude potential mask filenames
        # TODO: If folder contains A.nii and A.nii.gz that ambiguity will not
        #       be detected
        dic_filenames = {p.match(f).group(1): p.match(f).group(0)
                         for f in os.listdir(abs_path_to_directory)
                         if p.match(f) and not p_mask.match(f)}

        dic_filenames_mask = {p_mask.match(f).group(1):
                              p_mask.match(f).group(0)
                              for f in os.listdir(abs_path_to_directory)
                              if p_mask.match(f)}

        # Filenames without filename ending as sorted list
        filenames = natsort.natsorted(
            dic_filenames.keys(), key=lambda y: y.lower())

        self._stacks = [None] * len(filenames)
        for i, filename in enumerate(filenames):

            abs_path_image = os.path.join(abs_path_to_directory,
                                          dic_filenames[filename])

            if filename in dic_filenames_mask.keys():
                abs_path_mask = os.path.join(abs_path_to_directory,
                                             dic_filenames_mask[filename])
            else:
                ph.print_debug_info("No mask found for '%s'." %
                                    (abs_path_image))
                abs_path_mask = None

            self._stacks[i] = st.Stack.from_filename(
                abs_path_image,
                abs_path_mask,
                extract_slices=self._extract_slices)


##
# MultipleImagesReader reads multiple nifti images and returns them as a list
# of Stack objects.
# \date       2017-07-12 11:28:10+0100
#
class MultipleImagesReader(DataReader):

    ##
    # Store relevant information to read multiple images and their potential
    # masks.
    # \date       2017-07-11 19:04:25+0100
    #
    # \param      self            The object
    # \param      file_paths      The paths to filenames as single string
    #                             separated by white spaces, e.g.
    #                             "A.nii.gz B.nii C.nii.gz"
    # \param      suffix_mask     extension of stack filename as string
    #                             indicating associated mask, e.g. "_mask" for
    #                             "A_mask.nii".
    # \param      extract_slices  Boolean to indicate whether given 3D image
    #                             shall be split into its slices along the
    #                             k-direction.
    #
    def __init__(self, file_paths, suffix_mask="_mask", extract_slices=True):

        super(self.__class__, self).__init__()

        # Get list of paths to image
        self._file_paths = file_paths.split(" ")
        self._suffix_mask = suffix_mask
        self._extract_slices = extract_slices

    ##
    # Reads the data of multiple images.
    # \date       2017-07-12 11:30:35+0100
    #
    def read_data(self):

        self._stacks = [None] * len(self._file_paths)

        for i, file_path in enumerate(self._file_paths):

            # Build absolute path to directory of image
            path_to_directory = os.path.dirname(file_path)
            if not ph.directory_exists(path_to_directory):
                raise Exceptions.DirectoryNotExistent(path_to_directory)
            abs_path_to_directory = os.path.abspath(path_to_directory)

            # Get absolute path mask to image
            filename = os.path.basename(file_path).split(".")[0]
            pattern_mask = filename + self._suffix_mask + "[.]" + \
                REGEX_FILENAME_EXTENSIONS
            p_mask = re.compile(pattern_mask)
            filename_mask = [p_mask.match(f).group(0)
                             for f in os.listdir(abs_path_to_directory)
                             if p_mask.match(f)]

            if len(filename_mask) == 0:
                abs_path_mask = None
            else:
                abs_path_mask = os.path.join(abs_path_to_directory,
                                             filename_mask[0])
            self._stacks[i] = st.Stack.from_filename(
                file_path,
                abs_path_mask,
                extract_slices=self._extract_slices)
