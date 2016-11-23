##
# \file FilenamesSelector.py
# \brief      Class containing functions to correct for intensities
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Nov 2016
# 
# \remark needs to be very flexible! For the moment I stick to the file 


## Import libraries 
import SimpleITK as sitk
import numpy as np
import sys
import os

## Import modules
import utilities.SimpleITKHelper as sitkh
import utilities.PythonHelper as ph
import utilities.FilenameParser as fp

##
# Class to get filenames in certain dir_input
# \date       2016-11-23 14:56:06+0000
#
class FilenamesSelector(object):

    def __init__(self, dir_input, dir_input_existing_volumes):
        self._dir_input = dir_input
        self._dir_input_existing_volumes = dir_input_existing_volumes

        self._stack_type_combination = "baseline_and_1yr_and_20yr_and_30yr"
        self._get_filenames_and_subfolders = {
              "baseline_and_1yr_and_20yr_and_30yr" : self._get_filenames_and_subfolders_baseline_and_1yr_and_20yr_and_30yr
            , "5yr_where_electronic_version" : self._get_filenames_and_subfolders_5yr_where_electronic_version
            , "10yr_where_electronic_version" : self._get_filenames_and_subfolders_10yr_where_electronic_version
            
            ## Not done yet!
            # , "5yr_and_10yr_and_ge2scans" : self._get_filenames_and_subfolders_5yr_and_10yr_and_ge2scans 
        }

        self._filename_parser = fp.FilenameParser()


    def set_possible_stack_type_combinations(self, stack_type_combination):
        if stack_type_combination not in self._get_filenames_and_subfolders.keys():
            raise ErrorValue("Stack type combination not possible.")

        self._stack_type_combination = stack_type_combination

    def get_possible_stack_type_combinations(self):
        return self._get_filenames_and_subfolders.keys()


    def get_filenames_and_subfolders(self, stack_type_combination=None):

        if stack_type_combination is None:
            stack_type_combination = self._stack_type_combination

        return self._get_filenames_and_subfolders[stack_type_combination]()


    def _get_filenames_and_subfolders_baseline_and_1yr_and_20yr_and_30yr(self):

        filenames_1yr = self._filename_parser.get_filenames_which_match_pattern_in_directory(self._dir_input + "1yr_3x3/", "1yr")
        filenames_1yr_short =  self._filename_parser.get_dash_partitioned_filename(filenames_1yr)
        
        filenames_Baseline = self._filename_parser.get_filenames_which_match_pattern_in_directory(self._dir_input + "Baseline_3x3/", "Baseline")
        filenames_Baseline_short =  self._filename_parser.get_dash_partitioned_filename(filenames_Baseline)

        filenames_ref_20 = self._filename_parser.get_filenames_which_match_pattern_in_directory(self._dir_input, "20yr-0-PD")
        filenames_ref_20_short =  self._filename_parser.get_dash_partitioned_filename(filenames_ref_20)

        filenames_ref_30 = self._filename_parser.get_filenames_which_match_pattern_in_directory(self._dir_input, "30yr-0-T1")
        filenames_ref_30_short =  self._filename_parser.get_dash_partitioned_filename(filenames_ref_30)

        ## Intersection of lists to figure out the amount of files with common features
        timepoints = list(set(filenames_1yr_short).intersection(filenames_Baseline_short))
        timepoints_m20 = list(set(timepoints).intersection(filenames_ref_20_short))
        timepoints_m20_m30 = list(set(timepoints_m20).intersection(filenames_ref_30_short))

        print len(timepoints)
        print len(timepoints_m20)
        print len(timepoints_m20_m30)

        return sorted(timepoints_m20_m30), ["1yr_3x3/", "Baseline_3x3/"]

    def _get_filenames_and_subfolders_5yr_where_electronic_version(self):
        
        subfolder = "5yr_5x3/"
        
        ## Get filenames of 5yr scans with more than two films
        filenames_5yr = self._filename_parser.get_filenames_which_match_pattern_in_directory(self._dir_input + "5yr_5x3/", "5yr")
        filenames_5yr_short =  self._filename_parser.get_dash_partitioned_filename(filenames_5yr)
        
        ## Get filenames of existing, electronic stacks of 5yr, PD slices
        filenames_electronic_5yr_PD = self._filename_parser.get_filenames_which_match_pattern_in_directory(self._dir_input_existing_volumes, patterns=["5yr", "PD"])
        filenames_electronic_5yr_PD_short =  self._filename_parser.get_dash_partitioned_filename(filenames_electronic_5yr_PD)
        print("\n%2d filenames where 5yr PD electronic versions are available" %(len(filenames_electronic_5yr_PD_short)))
        # print filenames_electronic_5yr_PD_short

        filenames_common = list(set(filenames_5yr_short).intersection(filenames_electronic_5yr_PD_short))
        print("\n%2d filenames where for both 5yr film and electronic version are available" %(len(filenames_common)))

        return sorted(filenames_common), [subfolder]


    def _get_filenames_and_subfolders_10yr_where_electronic_version(self):

        subfolder = "10yr_5x3/"
        
        ## Get filenames of 10yr scans with more than two films
        filenames_10yr = self._filename_parser.get_filenames_which_match_pattern_in_directory(self._dir_input + "10yr_5x3/", "10yr")
        filenames_10yr_short =  self._filename_parser.get_dash_partitioned_filename(filenames_10yr)
        
        ## Get filenames of existing, electronic stacks of 10yr, PD slices
        filenames_electronic_10yr_PD = self._filename_parser.get_filenames_which_match_pattern_in_directory(self._dir_input_existing_volumes, patterns=["10yr", "PD"])
        filenames_electronic_10yr_PD_short =  self._filename_parser.get_dash_partitioned_filename(filenames_electronic_10yr_PD)
        print("\n%2d filenames where 10yr PD electronic versions are available" %(len(filenames_electronic_10yr_PD_short)))
        # print filenames_electronic_10yr_PD_short

        filenames_common = list(set(filenames_10yr_short).intersection(filenames_electronic_10yr_PD_short))
        print("\n%2d filenames where for both 10yr film and electronic version are available" %(len(filenames_common)))

        ## Get full extension
        filenames_10yr_common = [f + "-10yr" for f in filenames_common]

        return sorted(filenames_common), [subfolder], sorted(filenames_10yr_common)


    ## NOT DONE YET! (will it ever be?)
    # Gets the filenames and subfolders 5 yr and 10 yr and ge 2 scans.
    # \date       2016-11-23 15:40:27+0000
    #
    # \param      self  The object
    #
    # \return     The filenames and subfolders 5 yr and 10 yr and ge 2 scans.
    #
    def _get_filenames_and_subfolders_5yr_and_10yr_and_ge2scans(self):
        ## Get filenames of 5yr scans with more than two films (then brains/skulls are not merged!)
        filenames_5yr = self._filename_parser.get_filenames_which_match_pattern_in_directory(self._dir_input + "5yr_5x3/", "5yr")
        filenames_5yr_short =  self._filename_parser.get_dash_partitioned_filename(filenames_5yr)
        filenames_5yr_ge2_scans = []
        for i in range(0, len(filenames_5yr_short)):
            filenames_tmp = self._filename_parser.get_filenames_which_match_pattern_in_directory(self._dir_input + "5yr_5x3/", filenames_5yr_short[i])
            if len(filenames_tmp) > 1:
                filenames_5yr_ge2_scans.append(filenames_5yr_short[i])

        print("\n%2d filenames where 5yr scans more than one film is available" %(len(filenames_5yr_ge2_scans)))
        # print filenames_5yr_ge2_scans

        ## Get filenames of 5yr scans with more than two films (then brains/skulls are not merged!)
        filenames_10yr = self._filename_parser.get_filenames_which_match_pattern_in_directory(self._dir_input + "10yr_5x3/", "10yr")
        filenames_10yr_short =  self._filename_parser.get_dash_partitioned_filename(filenames_10yr)
        filenames_10yr_ge2_scans = []
        for i in range(0, len(filenames_10yr_short)):
            filenames_tmp = self._filename_parser.get_filenames_which_match_pattern_in_directory(self._dir_input + "10yr_5x3/", filenames_10yr_short[i])
            if len(filenames_tmp) > 1:
                filenames_10yr_ge2_scans.append(filenames_10yr_short[i])

        print("\n%2d filenames where 10yr scans more than one film is available" %(len(filenames_10yr_ge2_scans)))
        # print filenames_10yr_ge2_scans
        
        ## Get filenames of existing stacks of slices
        filenames_electronic_5yr_PD = self._filename_parser.get_filenames_which_match_pattern_in_directory(DIR_INPUT_STACKED_VOLUMES, patterns=["5yr", "PD"])
        filenames_electronic_5yr_PD_short =  self._filename_parser.get_dash_partitioned_filename(filenames_electronic_5yr_PD)
        print("\n%2d filenames where 5yr PD electronic versions are available" %(len(filenames_electronic_5yr_PD_short)))
        # print filenames_electronic_5yr_PD_short

        ## Intersection of lists to figure out the amount of files with common features
        filenames_common = list(set(filenames_5yr_ge2_scans).intersection(filenames_10yr_ge2_scans))
        print("\n%2d filenames where for both 5yr and 10yr scans more than one film is available" %(len(filenames_common)))

        # filenames_common = list(set(filenames_5yr_ge2_scans).intersection(filenames_electronic_5yr_PD_short))
        filenames_common = list(set(filenames_5yr_short).intersection(filenames_electronic_5yr_PD_short))
        print("\n%2d filenames where for both 5yr film and electronic version are available" %(len(filenames_common)))

        print(sorted(filenames_common))
