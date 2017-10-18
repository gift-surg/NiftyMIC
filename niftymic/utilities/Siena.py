##
# \file Siena.py
# \brief
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Nov 2016
#


# Import libraries
import SimpleITK as sitk
import numpy as np
import sys
import os
import re
from skimage.measure import compare_ssim as ssim

# Import modules
import pysitk.simple_itk_helper as sitkh
import pysitk.python_helper as ph

import niftymic.base.Stack as st
from niftymic.definitions import DIR_TMP


class Siena(object):

    def __init__(self,
                 stack1,
                 stack2,
                 dir_output="./siena/",
                 options='-B "-B -f 0.1" -2',
                 dir_tmp=os.path.join(DIR_TMP, "siena/")):

        self._stack1 = st.Stack.from_stack(stack1)
        self._stack2 = st.Stack.from_stack(stack2)
        self._dir_output = dir_output
        self._dir_tmp = dir_tmp
        self._options = options

    def run(self):
        ph.create_directory(dir_tmp, delete_files=True)

        # Write images
        sitk.WriteImage(self._stack1.sitk, self._dir_tmp +
                        self._stack1.get_filename() + ".nii.gz")
        sitk.WriteImage(self._stack2.sitk, self._dir_tmp +
                        self._stack2.get_filename() + ".nii.gz")

        cmd = "siena "
        cmd += self._dir_tmp + self._stack1.get_filename() + ".nii.gz "
        cmd += self._dir_tmp + self._stack2.get_filename() + ".nii.gz "
        cmd += "-o " + self._dir_output + " "
        cmd += self._options

        time_start = ph.start_timing()
        ph.execute_command(cmd)
        self._elapsed_time = ph.stop_timing(time_start)

        # Extract measures from report
        self._extract_percentage_brain_volume_change()

    def print_statistics(self):
        print("\tElapsed time: %s" % (self._elapsed_time))
        print("\tPercentage Brain Volume Change (PBVC): %.2f%%" %
              (self._percentage_brain_volume_change))

    ##
    # Percentage Brain Volume Change
    # \date       2016-11-27 17:42:55+0000
    #
    # \param      self  The object
    #
    def _extract_percentage_brain_volume_change(self):
        datafile = file(self._dir_output + "report.siena")

        for line in datafile:
            if "finalPBVC" in line:
                parts = line.split(" ")
                break
        self._percentage_brain_volume_change = float(parts[1])

    def get_percentage_brain_volume_change(self):
        return self._percentage_brain_volume_change
