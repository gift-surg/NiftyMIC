#!/usr/bin/python

## \file FilenameParser.py
#  \brief Reading filenames from directory and other useful parsing functions
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date Nov 2016


## Import libraries
import sys
import itk
import SimpleITK as sitk
import numpy as np
import os
import re

## Add directories to import modules
# DIR_SRC_ROOT = "../"
# sys.path.append(DIR_SRC_ROOT)

## Import modules from src-folder
import utilities.SimpleITKHelper as sitkh
import base.PSF as psf
import base.Slice as sl


##-----------------------------------------------------------------------------
# \brief      Class for parsing operations
# \date       2016-11-02 00:15:47+0000
#
class FilenameParser(object):

    ##-----------------------------------------------------------------------------
    # \brief      Gets the filenames of particular type in directory.
    # \date       2016-08-05 17:06:30+0100
    #
    # \param[in]  directory           directory as string
    # \param[in]  filename_extension  extension of filename as string
    #
    # \return     filenames in directory without filename extension as list of
    #             strings
    #
    def get_filenames_which_match_pattern_in_directory(self, directory, pattern, filename_extension=".dcm"):
        
        ## List of all files in directory
        all_files = os.listdir(directory)

        ## Find names which match pattern
        filenames = [f for f in all_files if re.match(pattern,f)]

        return filenames


    ##-------------------------------------------------------------------------
    # \brief      Crop extensions from filenames
    # \date       2016-11-02 00:40:01+0000
    #
    # \param      self       The object
    # \param      filenames  Either string or list of strings
    #
    # \return     { description_of_the_return_value }
    #
    def crop_filename_extension(self, filenames):

        if type(filenames) is list:
            filenames = [f.split(".")[0] for f in filenames]

        else:
            filenames = filenames.split(".")[0]

        ## Subtract the (known) filename-extension
        # filenames = [re.sub(filename_extension,"",f) for f in filenames]

        return filenames
