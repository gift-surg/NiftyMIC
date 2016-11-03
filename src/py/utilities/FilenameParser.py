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
import fnmatch

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

    ##-------------------------------------------------------------------------
    # \brief      Gets the filenames in directory (of certain type) which match
    #             pattern(s).
    # \date       2016-08-05 17:06:30+0100
    #
    # \param      self                The object
    # \param[in]  directory           directory as string
    # \param      patterns            patterns as string or list of strings
    # \param[in]  filename_extension  extension of filename as string
    #
    # \return     filenames in directory without filename extension as list of
    #             strings
    #
    def get_filenames_which_match_pattern_in_directory(self, directory, patterns, filename_extension=None):
        
        if type(patterns) is not list:
            patterns = [patterns]

        ## List of all files in directory
        filenames = os.listdir(directory)

        ## Only consider files with given filename_extension
        if filename_extension is not None:
            filenames = [f for f in filenames if fnmatch.fnmatch(f, "*"+filename_extension)]

        ## Get filenames which match all patterns
        for i in range(0, len(patterns)):
            filenames = [f for f in filenames if fnmatch.fnmatch(f, "*"+patterns[i]+"*")]

        ## Crop filename extension
        filenames = self.crop_filename_extension(filenames)

        return filenames


    ##-------------------------------------------------------------------------
    # \brief      Exclude filenames which match pattern(s)
    # \date       2016-11-03 16:26:20+0000
    #
    # \param      self       The object
    # \param      filenames  The filenames
    # \param      patterns   The patterns
    #
    # \return     Get filenames reduced by the ones matching the patterns
    #
    def exclude_filenames_which_match_pattern(self, filenames, patterns):

        if type(patterns) is not list:
            patterns = [patterns]

        ## Exclude files which match pattern
        for i in range(0, len(patterns)):
            filenames_tmp = np.array(filenames)
            for f in filenames_tmp:
                if fnmatch.fnmatch(f, "*"+patterns[i]+"*"):
                    filenames.remove(f)

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
