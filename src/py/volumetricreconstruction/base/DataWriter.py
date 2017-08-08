# \file DataWriter.py
#  \brief Writes data to HDD
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date Aug 2017


# Import libraries
from abc import ABCMeta, abstractmethod
import os
import sys
import SimpleITK as sitk
import numpy as np

import pythonhelper.PythonHelper as ph
import pythonhelper.SimpleITKHelper as sitkh


class DataWriter(object):
    __metaclass__ = ABCMeta

    def __init__(self, stacks):
        self._stacks = stacks

    def set_stacks(self, stacks):
        self._stacks = stacks

    @abstractmethod
    def write_data(self):
        pass


class MultipleStacksWriter(DataWriter):

    def __init__(self,
                 stacks,
                 directory,
                 write_mask=False,
                 write_slices=False,
                 write_transforms=False):

        DataWriter.__init__(self, stacks=stacks)
        self._directory = directory
        self._write_mask = write_mask
        self._write_slices = write_slices
        self._write_transforms = write_transforms

    def set_directory(self, directory):
        self._directory = directory

    def write_data(self):
        for stack in self._stacks:
            stack.write(self._directory,
                        write_mask=self._write_mask,
                        write_slices=self._write_slices,
                        write_transforms=self._write_transforms)


class MultiComponentImageWriter(DataWriter):

    def __init__(self,
                 stacks,
                 filename=None,
                 write_mask=False,
                 suffix_mask="_mask"):

        DataWriter.__init__(self, stacks=stacks)
        self._filename = filename
        self._write_mask = write_mask
        self._suffix_mask = suffix_mask

    def set_filename(self, filename):
        self._filename = filename

    def write_data(self):
        
        if self._filename is None:
            raise ValueError("Filename is not set")

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
