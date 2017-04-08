#!/usr/bin/python

##
# \file Number02VolunteerSRRUpperGI.py
# \brief SRR Upper GI Scan #2 (WL, 13 December 2016)
# 
# Summary:  o Oblique acquisitions and BFFE sequence
#           o Standard and T2 ref 3D MRCP (heavily T2w vol) did not work out;
#             Probably, the gating was not set up properly
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       April 2017
#


## Import libraries
import os                       # used to execute terminal commands in python

## Add directories to import modules
# dir_src_root = "../src/"
# sys.path.append( dir_src_root )

from studies.SRRUpperGI.VolunteerSRRUpperGI import VolunteerSRRUpperGI


class Number02VolunteerSRRUpperGI(VolunteerSRRUpperGI):

    def __init__(self, select_anatomy="upper_gi"):

        self._directory_upper_gi = "/Users/mebner/UCL/Data/Data_SRRUpperGIAnatomyStudy/SRR02/nifti/"
        self._directory_brain = None

        VolunteerSRRUpperGI.__init__(self, select_anatomy=select_anatomy)


    ## ************************************************************************
    ## Upper GI Anatomy

    ## Getter for single-shot T2-weighted filenames
    def _get_filenames_single_shot_t2_weighted_standard_upper_gi(self):
        
        filenames = []
        labels = []

        ## Standard anatomical planes (0.78 x 0.78 x 5)
        filenames.append("T2CorBHSENSEs901a1009")
        filenames.append("T2AxBHSENSEs1001a1010")
        filenames.append("T2SagBHSENSEs1101a1011")
        
        labels.append(self._label_SST2W_standard + "_" + self._label_coronal)
        labels.append(self._label_SST2W_standard + "_" + self._label_transverse)
        labels.append(self._label_SST2W_standard + "_" + self._label_sagittal)
        
        return filenames, labels


    def _get_filenames_single_shot_t2_weighted_oblique_upper_gi(self):
        
        filenames = []
        labels = []
        
        ## Oblique planes (0.78 x 0.78 x 5)
        filenames.append("T2CorBHSENSEs1201a1012") ## oblique axial
        filenames.append("T2CorBHSENSEs1301a1013") ## oblique sagittal
        filenames.append("T2CorBHSENSEs1401a1014") ## oblique axial
        
        filenames.append("T2CorBHSENSEs1501a1015") ## oblique axial; ITK-Snap cannot open it

        for i in range(0, len(filenames)):
            labels.append(self._label_SST2W_oblique + "_" + str(i))

        return filenames, labels


    def _get_filenames_single_shot_t2_weighted_standard_offset_upper_gi(self):
        pass 
       

    ## Getter for BFFE filename
    def _get_filename_balanced_fast_field_echo_upper_gi(self):
        ## 0.73 x 0.73 x 1.5
        filename = "bFFECORSENSEs401a1004"

        return filename, self._label_BFFE


    ## Getter for heavily T2-weighted volumetric filename
    def _get_filename_heavily_t2_weighted_volume_upper_gi(self):
        
        ## Very poor: Standard MRCP 3D HR volume (heavily T2w vol)
        filename = "sMRCP3DHRSENSEs701a1007" # 0.65 x 0.65 x 0.9

        return filename, self._label_heavilyT2w


    ## Getter for T2-weighted volumetric filename
    def _get_filename_t2_weighted_volume_upper_gi(self):
        
        ## Very poor: 3D volume with SST2W contrast
        filename = "sMRCPte803DHRSENSEs801a1008" # 0.65 x 0.65 x 0.9

        return filename, self._label_T2w


    ## ************************************************************************
    ## Brain
    
    ## Getter for single-shot T2-weighted filenames
    def _get_filenames_single_shot_t2_weighted_standard_brain(self):
        pass
    def _get_filenames_single_shot_t2_weighted_oblique_brain(self):
        pass
    def _get_filenames_single_shot_t2_weighted_standard_offset_brain(self):
        pass

    ## Getter for BFFE filename
    def _get_filename_balanced_fast_field_echo_brain(self):
        pass

    ## Getter for heavily T2-weighted volumetric filename
    def _get_filename_heavily_t2_weighted_volume_brain(self):
        pass

    ## Getter for T2-weighted volumetric filename
    def _get_filename_t2_weighted_volume_brain(self):
        pass
