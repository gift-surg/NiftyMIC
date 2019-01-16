##
# \file target_stack_estimator.py
# \brief      Class to estimate target stack automatically
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       January 2018
#


import os
import re
import numpy as np
import SimpleITK as sitk

import pysitk.simple_itk_helper as sitkh


##
# Class to estimate target stack automatically
# \date       2018-01-26 16:32:11+0000
#
class TargetStackEstimator(object):

    def __init__(self):
        self._target_stack_index = None

    def get_target_stack_index(self):
        return self._target_stack_index

    ##
    # Use stack with largest mask volume as target stack
    # \date       2018-01-26 16:52:39+0000
    #
    # \param      cls               The cls
    # \param      file_paths_masks  paths to image masks as list of strings
    #
    @classmethod
    def from_masks(cls, file_paths_masks):
        target_stack_estimator = cls()

        masks_sitk = [sitkh.read_nifti_image_sitk(str(f), sitk.sitkUInt8)
                      for f in file_paths_masks]

        # Compute volumes of all masks
        volumes = np.zeros(len(masks_sitk))
        for i, mask_sitk in enumerate(masks_sitk):
            mask_nda = sitk.GetArrayFromImage(mask_sitk)
            spacing = np.array(mask_sitk.GetSpacing())
            volumes[i] = np.sum(mask_nda) * spacing.prod()

        # Get index corresponding to maximum volume stack mask
        target_stack_estimator._target_stack_index = np.argmax(volumes)

        return target_stack_estimator
