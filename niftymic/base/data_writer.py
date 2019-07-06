##
# \file DataWriter.py
# \brief      Writes data to HDD
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Aug 2017
#

import os
import sys
import numpy as np
import SimpleITK as sitk
from abc import ABCMeta, abstractmethod

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh

import niftymic


class DataWriter(object):


    ##
    # Gets the header update.
    #
    # aux_file: 23 characters max
    # descrip: 79 characters max
    #
    # \see        https://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields
    # \date       2019-07-05 18:09:40+0100
    #
    # \param      description  description of NIfTI image (max 79 chars)
    #
    # \return     dictionary that carries NIfTI header information updates.
    #
    @staticmethod
    def _get_header_update(description=None):

        header_update = {
            "aux_file": "NiftyMIC-v%s" % niftymic.__version__
        }

        if description is None:
            header_update["descrip"] = ""
        else:
            header_update["descrip"] = description


        return header_update

    @staticmethod
    def write_image(
        image_sitk,
        path_to_file,
        compress=True,
        verbose=True,
        description=None,
    ):
        info = "Write image to %s" % path_to_file
        if compress:
            image_sitk = sitk.Cast(image_sitk, sitk.sitkFloat32)
            info += " (float32)"
        if verbose:
            ph.print_info("%s ... " % info, newline=False)
        header_update = DataWriter._get_header_update(description=description)

        sitkh.write_nifti_image_sitk(
            image_sitk, path_to_file, header_update=header_update)
        if verbose:
            print("done")

    @staticmethod
    def write_mask(
        mask_sitk,
        path_to_file,
        compress=True,
        verbose=True,
        description=None,
    ):
        info = "Write mask to %s" % path_to_file
        if compress:
            mask_sitk = sitk.Cast(mask_sitk, sitk.sitkUInt8)
            info += " (uint8)"
        if verbose:
            ph.print_info("%s ... " % info, newline=False)
        header_update = DataWriter._get_header_update(description=description)

        sitkh.write_nifti_image_sitk(
            mask_sitk, path_to_file, header_update=header_update)
        if verbose:
            print("done")


class StacksWriter(object):
    __metaclass__ = ABCMeta

    def __init__(self, stacks):
        self._stacks = stacks

    def set_stacks(self, stacks):
        self._stacks = stacks

    @abstractmethod
    def write_data(self):
        pass


class MultipleStacksWriter(StacksWriter):

    def __init__(self,
                 stacks,
                 directory,
                 write_mask=False,
                 write_slices=False,
                 write_transforms=False,
                 suffix_mask="_mask"):

        StacksWriter.__init__(self, stacks=stacks)
        self._directory = directory
        self._write_mask = write_mask
        self._write_slices = write_slices
        self._write_transforms = write_transforms
        self._suffix_mask = suffix_mask

    def set_directory(self, directory):
        self._directory = directory

    def write_data(self):
        for stack in self._stacks:
            stack.write(self._directory,
                        write_mask=self._write_mask,
                        write_slices=self._write_slices,
                        write_transforms=self._write_transforms,
                        suffix_mask=self._suffix_mask)


class MultiComponentImageWriter(StacksWriter):

    def __init__(self,
                 stacks,
                 filename=None,
                 write_mask=False,
                 suffix_mask="_mask",
                 ):

        StacksWriter.__init__(self, stacks=stacks)
        self._filename = filename
        self._write_mask = write_mask
        self._suffix_mask = suffix_mask

    def set_filename(self, filename):
        self._filename = filename

    def write_data(self):

        if self._filename is None:
            raise ValueError("Filename is not set")

        ph.create_directory(os.path.dirname(self._filename))

        vector_image_sitk = sitkh.get_sitk_vector_image_from_components(
            [stack.sitk for stack in self._stacks])
        sitkh.write_sitk_vector_image(vector_image_sitk, self._filename)

        if self._write_mask:
            filename_split = (self._filename).split(".")
            filename = filename_split[0]
            filename += self._suffix_mask + "." + \
                (".").join(filename_split[1:])
            vector_image_sitk = sitkh.get_sitk_vector_image_from_components(
                [stack.sitk_mask for stack in self._stacks])
            sitkh.write_sitk_vector_image(vector_image_sitk, filename)
