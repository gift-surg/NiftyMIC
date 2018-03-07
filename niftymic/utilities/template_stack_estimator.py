##
# \file template_stack_estimator.py
# \brief      Class to estimate template stack automatically
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       January 2018
#


import os
import re
import numpy as np
import json
import SimpleITK as sitk

import pysitk.simple_itk_helper as sitkh

from niftymic.definitions import DIR_TEMPLATES, TEMPLATES_INFO


##
# Class to estimate template stack automatically
# \date       2018-01-26 16:32:11+0000
#
class TemplateStackEstimator(object):

    def __init__(self):
        self._template_path = None

    ##
    # Gets the path to estimated template.
    # \date       2018-01-27 02:14:53+0000
    #
    # \param      self  The object
    #
    # \return     The path to template.
    #
    def get_path_to_template(self):
        return self._template_path

    ##
    # Select template with similar brain volume
    # \date       2018-01-26 16:52:39+0000
    #
    # \param      cls               The cls
    # \param      file_paths_masks  paths to image masks as list of strings
    #
    @classmethod
    def from_mask(cls, file_path_mask):
        template_stack_estimator = cls()

        mask_sitk = sitkh.read_nifti_image_sitk(
            file_path_mask, sitk.sitkUInt8)
        mask_nda = sitk.GetArrayFromImage(mask_sitk)
        spacing = np.array(mask_sitk.GetSpacing())
        volume = len(np.where(mask_nda > 0)[0]) * spacing.prod()

        # Read in template info
        path_to_template_info = os.path.join(DIR_TEMPLATES, TEMPLATES_INFO)
        with open(path_to_template_info) as json_file:
            dic = json.load(json_file)

        # Get gestational ages as list of integers
        gestational_ages = sorted([int(gw) for gw in dic.keys()])

        # # Get matching gestational age
        # template_volumes = np.array([dic[str(k)]["volume_mask"]
        #                              for k in gestational_ages])
        # index = np.argmin(np.abs(template_volumes - volume))
        
        # # Ensure valid index after correction
        # index = np.max([0, index - 1])
        # # index = np.min([index + 1, len(template_volumes)-1])
        
        # # Matching gestational age/week
        # gw_match = str(gestational_ages[index])

        # template_stack_estimator._template_path = os.path.join(
        #     DIR_TEMPLATES, dic[gw_match]["image"])
        
        # return template_stack_estimator

        # Find template which has slightly smaller mask volume
        for k in gestational_ages:
            if dic[str(k)]["volume_mask_dil"] > volume:
                key = str(np.max([gestational_ages[0], k - 1]))
                template_stack_estimator._template_path = os.path.join(
                    DIR_TEMPLATES, dic[key]["image"])
                return template_stack_estimator

        # Otherwise, return path to oldest template image available
        template_stack_estimator._template_path = os.path.join(
            DIR_TEMPLATES, dic[str(gestational_ages[-1])]["image"])
        return template_stack_estimator
