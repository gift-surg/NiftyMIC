##
# \file data_reader_test.py
#  \brief  Unit tests for data reader
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date January 2018


import os
import unittest
import numpy as np
import re
import SimpleITK as sitk

import pysitk.python_helper as ph
import niftymic.base.data_reader as dr

from niftymic.definitions import DIR_TMP, DIR_TEST


class DataReaderTest(unittest.TestCase):

    def setUp(self):
        self.precision = 7
        self.dir_data = os.path.join(DIR_TEST, "case-studies", "fetal-brain")
        self.filenames = [
            os.path.join(self.dir_data,
                         "input-data",
                         "%s.nii.gz" % f)
            for f in ["axial", "coronal", "sagittal"]]
        self.dir_output = os.path.join(DIR_TMP, "case-studies", "fetal-brain")
        self.suffix_mask = "_mask"

    ##
    # Check that the same number of stacks (and slices therein) are read
    # \date       2018-01-31 23:03:52+0000
    #
    # \param      self  The object
    #
    def test_read_transformations(self):

        directory_motion_correction = os.path.join(
            DIR_TEST,
            "case-studies",
            "fetal-brain",
            "reconstruct_volume",
            "motion_correction",
        )

        data_reader = dr.MultipleImagesReader(
            file_paths=self.filenames,
            dir_motion_correction=directory_motion_correction)
        data_reader.read_data()
        stacks = data_reader.get_data()

        data_reader = dr.SliceTransformationDirectoryReader(
            directory_motion_correction)
        data_reader.read_data()
        transformations_dic = data_reader.get_data()

        self.assertEqual(len(stacks) - len(transformations_dic.keys()), 0)

        for stack in stacks:
            N_slices = stack.get_number_of_slices()
            N_slices2 = len(transformations_dic[stack.get_filename()].keys())
            self.assertEqual(N_slices - N_slices2, 0)


