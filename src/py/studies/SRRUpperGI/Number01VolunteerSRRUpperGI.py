#!/usr/bin/python

##
# \file Number01VolunteerSRRUpperGI.py
# \brief SRR Upper GI Scan #1 (LF, 4 November 2016)
# 
# Summary:  o Test of oblique acquisitions but did not go that well
#           o Test to get some "ground-truth" references for SRR
#           o No standard 3D MRCP volume has been acquired
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


class Number01VolunteerSRRUpperGI(VolunteerSRRUpperGI):

    def __init__(self, select_anatomy="upper_gi"):

        self._directory_upper_gi = "/Users/mebner/UCL/Data/Data_SRRUpperGIAnatomyStudy/SRR01/nifti/"
        self._directory_brain = None

        VolunteerSRRUpperGI.__init__(self, select_anatomy=select_anatomy)


    ## ************************************************************************
    ## Upper GI Anatomy

    ## Getter for single-shot T2-weighted filenames
    def _get_filenames_single_shot_t2_weighted_standard_upper_gi(self):
        
        filenames = []
        labels = []

        ## Standard anatomical planes (0.78 x 0.78 x 5)
        filenames.append("T2WTSECorBHSENSEs401a1004")
        filenames.append("T2WTSEAxBHSENSEs501a1005")        # 6mm slice thickness
        filenames.append("T2WTSESagBHSENSEs601a1006")
        
        filenames.append("T2WTSEaCorBHSENSEs901a1009")
        filenames.append("T2WTSEaSagBHSENSEs1001a1010")
        
        labels.append(self._label_SST2W_standard + "_" + self._label_coronal)
        labels.append(self._label_SST2W_standard + "_" + self._label_transverse)
        labels.append(self._label_SST2W_standard + "_" + self._label_sagittal)
        
        labels.append(self._label_SST2W_standard + "_" + self._label_coronal)
        labels.append(self._label_SST2W_standard + "_" + self._label_sagittal)

        return filenames, labels


    def _get_filenames_single_shot_t2_weighted_oblique_upper_gi(self):
        
        filenames = []
        labels = []
        
        ## Oblique planes (0.78 x 0.78 x 5)
        ## Oblique planes (coronal, sagittal, axial)
        filenames.append("T2WTSEaCorBHSENSEs1101a1011")
        filenames.append("T2WTSEaSagBHSENSEs1201a1012")     # very dark intensity
        filenames.append("T2WTSEaCorBHSENSEs1301a1013")     # ITK-Snap cannot open it

        ## "Rotated towards the in-plane" but no offset (?) (coronal, sagittal, axial)
        filenames.append("T2WTSErCorBHSENSEs1401a1014")     # ITK-Snap cannot open it
        filenames.append("T2WTSErSagBHSENSEs1501a1015")     # ITK-Snap cannot open it
        filenames.append("T2WTSErAxBHSENSEs1601a1016")      # ITK-Snap cannot open it; 6mm slice thickness

        for i in range(0, len(filenames)):
            labels.append(self._label_SST2W_oblique + "_" + str(i))

        return filenames, labels


    def _get_filenames_single_shot_t2_weighted_standard_offset_upper_gi(self):
        pass 
       

    ## Getter for BFFE filename
    def _get_filename_balanced_fast_field_echo_upper_gi(self):
        pass 


    ## Getter for heavily T2-weighted volumetric filename
    def _get_filename_heavily_t2_weighted_volume_upper_gi(self):
        pass 


    ## Getter for T2-weighted volumetric filename
    def _get_filename_t2_weighted_volume_upper_gi(self):
        
        ## Volume acquisitions
        # filename = "WIP3DT2WTSECorBHSENSEs701a1007"  # noisy + very dark
        # filename = "WIP3DT2WTSECorBHSENSEs801a1008"  # noisy
        # filename = "WIP3DT2WTSECorBHSENSEs1701a1017" # noisy
        # filename = "WIP3DT2WTSECorBHSENSEs1801a1018" # noisy

        ## Single-shot acquisitions
        # filename = "T2WTSECorBHSENSEs1901a1019" # 0.78 x 0.78 x 2.5
        filename = "T2WTSECorBHSENSEs2001a1020" # 0.78 x 0.78 x 1.5

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
