#!/usr/bin/python

## \file TestSessionVolunteerSRRUpperGI.py
#  \brief
#
# SRR Upper GI: Sequence development scan (MC, 16 March 2017) 
#           Does not count to our volunteer scan
#           
# Summary:  o Idea was to test whether the oblique acquisitions towards
#               the four diagonals of the cube are working
#           o Respiratory-gated 3D acquisition with same contrast as SST2W
#               acquisitions by using both bellow and navigator (are quite blurry)
#               
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date April 2017


## Import libraries
import os                       # used to execute terminal commands in python

## Add directories to import modules
# dir_src_root = "../src/"
# sys.path.append( dir_src_root )

from studies.SRRUpperGI.VolunteerSRRUpperGI import VolunteerSRRUpperGI


class TestSessionVolunteerSRRUpperGI(VolunteerSRRUpperGI):

    def __init__(self, select_anatomy="upper_gi"):

        self._directory_upper_gi = "/Users/mebner/UCL/Data/Data_SRRUpperGIAnatomyStudy/SRR03_SequenceDevelopment_16Mar2017/nifti/"
        self._directory_brain = None

        VolunteerSRRUpperGI.__init__(self, select_anatomy=select_anatomy)


    ## ************************************************************************
    ## Upper GI Anatomy

    ## Getter for single-shot T2-weighted filenames
    def _get_filenames_single_shot_t2_weighted_standard_upper_gi(self):
        
        filenames = []
        labels = []

        ## Standard anatomical planes (0.78 x 0.78 x 5)
        filenames.append("T2WTSECorBHSENSEs1001a1010")   # 0.78 x 0.78 x 6 !!
        filenames.append("T2WTSEtraBHSENSEs1201a1012")   # 0.78 x 0.78 x 6 !!
        filenames.append("T2WTSEsagBHSENSEs1101a1011")   # 0.78 x 0.78 x 6 !!
        
        labels.append(self._label_SST2W_standard + "_" + self._label_coronal)
        labels.append(self._label_SST2W_standard + "_" + self._label_transverse)
        labels.append(self._label_SST2W_standard + "_" + self._label_sagittal)
        
        return filenames, labels


    def _get_filenames_single_shot_t2_weighted_oblique_upper_gi(self):
        
        filenames = []
        labels = []
        
        ## Oblique planes (0.78 x 0.78 x 5)
        filenames.append("T2WTSECorBHSENSEs601a1006")   # 0.78 x 0.78 x 6 !!
        filenames.append("T2WTSECorBHSENSEs701a1007")   # 0.78 x 0.78 x 6 !!
        filenames.append("T2WTSECorBHSENSEs801a1008")   # 0.78 x 0.78 x 6 !!
        filenames.append("T2WTSECorBHSENSEs901a1009")   # 0.78 x 0.78 x 6 !!

        for i in range(0, len(filenames)):
            labels.append(self._label_SST2W_oblique + "_" + str(i))


        return filenames, labels


    def _get_filenames_single_shot_t2_weighted_standard_offset_upper_gi(self):
       pass 
       

    ## Getter for BFFE filename
    def _get_filename_balanced_fast_field_echo_upper_gi(self):
        
        ## 0.73 x 0.73 x 1.5
        filename = "bFFECORSENSEs1301a1013"

        return filename, self._label_BFFE


    ## Getter for heavily T2-weighted volumetric filename
    def _get_filename_heavily_t2_weighted_volume_upper_gi(self):
        
        ## Standard MRCP 3D HR volume (heavily T2w vol)
        filename = "sMRCP3DHRSENSEs1401a1014" # 0.65 x 0.65 x 0.9

        return filename, self._label_heavilyT2w


    ## Getter for T2-weighted volumetric filename
    def _get_filename_t2_weighted_volume_upper_gi(self):
        
        ## Bellow-gated 3D volume with SST2W contrast (blurry)
        # filename = "sMRCP3DHRSENSEs401a1004" # 0.65 x 0.65 x 0.9

        ## Navigator-gated 3D volume with SST2W contrast (blurry)
        filename = "sMRCP3DHRnavSENSEs501a1005" # 0.65 x 0.65 x 0.9

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
