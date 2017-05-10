#!/usr/bin/python

##
# \file Number03VolunteerSRRUpperGI.py
# \brief      { item_description }
#
# SRR Upper GI Scan #4 (LH, 10 May 2017) 
# Summary:  o Final acquisition protocol was used
#           o First time, all sequences for both abdomen and brain 
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date May 2017


## Import libraries
import os                       # used to execute terminal commands in python

## Add directories to import modules
# dir_src_root = "../src/"
# sys.path.append( dir_src_root )

from studies.SRRUpperGI.VolunteerSRRUpperGI import VolunteerSRRUpperGI


class Number04VolunteerSRRUpperGI(VolunteerSRRUpperGI):

    def __init__(self, select_anatomy="upper_gi"):

        self._directory_upper_gi = "/Users/mebner/UCL/Data/Data_SRRUpperGIAnatomyStudy/SRR04/nifti/"
        self._directory_brain = "/Users/mebner/UCL/Data/Data_SRRUpperGIAnatomyStudy/SRR04/nifti/brain/"

        VolunteerSRRUpperGI.__init__(self, select_anatomy=select_anatomy)


    ## ************************************************************************
    ## Upper GI Anatomy

    ## Getter for single-shot T2-weighted filenames
    def _get_filenames_single_shot_t2_weighted_standard_upper_gi(self):
        
        filenames = []
        labels = []

        ## Standard anatomical planes (0.78 x 0.78 x 5)
        filenames.append("T2WCorBHSENSEs801a1008")
        filenames.append("T2WtraBHSENSEs901a1009")
        filenames.append("T2WsagBHSENSEs1001a1010")
        
        labels.append(self._label_SST2W_standard + "_" + self._label_coronal)
        labels.append(self._label_SST2W_standard + "_" + self._label_transverse)
        labels.append(self._label_SST2W_standard + "_" + self._label_sagittal)
        
        return filenames, labels


    def _get_filenames_single_shot_t2_weighted_oblique_upper_gi(self):
        
        filenames = []
        labels = []
        
        ## Oblique planes (0.78 x 0.78 x 5)
        filenames.append("T2W354530BHSENSEs1101a1011")  # -35 +45 +30
        filenames.append("T2W354530BHSENSEs1201a1012")  # +35 +45 -30
        filenames.append("T2W354530BHSENSEs1301a1013")  # -35 -45 -30
        filenames.append("T2W354530BHSENSEs1401a1014")  # +35 -45 +30

        for i in range(0, len(filenames)):
            labels.append(self._label_SST2W_oblique + "_" + str(i))


        return filenames, labels



    def _get_filenames_single_shot_t2_weighted_standard_offset_upper_gi(self):
        
        filenames = []
        labels = []

        ## Offset standard anatomical planes (0.78 x 0.78 x 5)
        filenames.append("T2WOFFCorBHSENSEs1501a1015")
        filenames.append("T2WOFFtraBHSENSEs1601a1016")
        filenames.append("T2WOFFsagBHSENSEs1701a1017")

        labels.append(self._label_SST2W_standard_offset + "_" + self._label_coronal)
        labels.append(self._label_SST2W_standard_offset + "_" + self._label_transverse)
        labels.append(self._label_SST2W_standard_offset + "_" + self._label_sagittal)

        return filenames, labels


    ## Getter for BFFE filename
    def _get_filename_balanced_fast_field_echo_upper_gi(self):
        
        ## Standard MRCP BTFE sequence (0.9 x 0.9 x 6)
        # filename = "BTFEBHSENSEs401a1004"

        ## 0.73 x 0.73 x 1.5
        filename = "BFFECORBHSENSEs701a1007"

        return filename, self._label_BFFE


    ## Getter for heavily T2-weighted volumetric filename
    def _get_filename_heavily_t2_weighted_volume_upper_gi(self):
        
        ## Standard MRCP 3D HR volume, i.e. heavily T2w vol (0.65 x 0.65 x 0.9)
        filename = "sMRCP3DHRSENSEs601a1006"

        return filename, self._label_heavilyT2w


    ## Getter for T2-weighted volumetric filename
    def _get_filename_t2_weighted_volume_upper_gi(self):
        pass

    ## ************************************************************************
    ## Brain     

    def _get_filenames_single_shot_t2_weighted_standard_brain(self):

        filenames = []
        labels = []

        ## Standard anatomical planes (0.78 x 0.78 x 5)
        filenames.append("T2WCorBHSENSEs2801a1028")
        filenames.append("T2WtraBHSENSEs2901a1029")
        filenames.append("T2WsagBHSENSEs3001a1030")
        
        labels.append(self._label_SST2W_standard + "_" + self._label_coronal)
        labels.append(self._label_SST2W_standard + "_" + self._label_transverse)
        labels.append(self._label_SST2W_standard + "_" + self._label_sagittal)
        
        return filenames, labels


    def _get_filenames_single_shot_t2_weighted_oblique_brain(self):
        
        filenames = []
        labels = []
        
        ## Oblique planes (0.78 x 0.78 x 5)
        filenames.append("T2W354530BHSENSEs3101a1031")  # -35 +45 +30
        filenames.append("T2W354530BHSENSEs3201a1032")  # +35 +45 -30
        filenames.append("T2W354530BHSENSEs3301a1033")  # -35 -45 -30
        filenames.append("T2W354530BHSENSEs3401a1034")  # +35 -45 +30

        for i in range(0, len(filenames)):
            labels.append(self._label_SST2W_oblique + "_" + str(i))

        return filenames, labels
    

    def _get_filenames_single_shot_t2_weighted_standard_offset_brain(self):
        
        filenames = []
        labels = []

        ## Offset standard anatomical planes (0.78 x 0.78 x 5)
        filenames.append("T2WOFFCorBHSENSEs3501a1035")
        filenames.append("T2WOFFtraBHSENSEs3601a1036")
        filenames.append("T2WOFFsagBHSENSEs3701a1037")

        labels.append(self._label_SST2W_standard_offset + "_" + self._label_coronal)
        labels.append(self._label_SST2W_standard_offset + "_" + self._label_transverse)
        labels.append(self._label_SST2W_standard_offset + "_" + self._label_sagittal)

        return filenames, labels


    ## Getter for BFFE filename
    def _get_filename_balanced_fast_field_echo_brain(self):
        
        ## Standard MRCP BTFE sequence (0.9 x 0.9 x 6)
        # filename = "BTFEBHSENSEs2401a1024"

        ## 0.73 x 0.73 x 1.5
        filename = "BFFECORBHSENSEs2701a1027"

        return filename, self._label_BFFE


    ## Getter for heavily T2-weighted volumetric filename
    def _get_filename_heavily_t2_weighted_volume_brain(self):
        
        ## Standard MRCP (0.65 x 0.65 x 0.9)
        filename = "sMRCP3DHRSENSEs2601a1026"

        return filename, self._label_heavilyT2w

    ## Getter for T2-weighted volumetric filename
    def _get_filename_t2_weighted_volume_brain(self):
        
        ## 3D T2 Brain (0.98 x 0.98 x 0.5)
        filename = "3DBrainVIEWT2SENSEs3801a1038"

        return filename, self._label_T2w
        
