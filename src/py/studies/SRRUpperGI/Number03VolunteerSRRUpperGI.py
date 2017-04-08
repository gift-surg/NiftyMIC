#!/usr/bin/python

##
# \file Number03VolunteerSRRUpperGI.py
# \brief      { item_description }
#
# SRR Upper GI Scan #3 (LP, 6 April 2017) 
# Summary:  o First scan after having set abdominal protocol 
#           o Same protocol was acquired at brain (but no 3D HR ref was done) 
#           It still was a bit chaotic ...
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date April 2017


## Import libraries
import os                       # used to execute terminal commands in python

## Add directories to import modules
# dir_src_root = "../src/"
# sys.path.append( dir_src_root )

from studies.SRRUpperGI.VolunteerSRRUpperGI import VolunteerSRRUpperGI


class Number03VolunteerSRRUpperGI(VolunteerSRRUpperGI):

    def __init__(self, select_anatomy="upper_gi"):

        self._directory_upper_gi = "/Users/mebner/UCL/Data/Data_SRRUpperGIAnatomyStudy/SRR03/nifti/"
        self._directory_brain = "/Users/mebner/UCL/Data/Data_SRRUpperGIAnatomyStudy/SRR03/nifti/brain/"

        VolunteerSRRUpperGI.__init__(self, select_anatomy=select_anatomy)


    ## ************************************************************************
    ## Upper GI Anatomy

    ## Getter for single-shot T2-weighted filenames
    def _get_filenames_single_shot_t2_weighted_standard_upper_gi(self):
        
        filenames = []
        labels = []

        ## Standard anatomical planes (0.78 x 0.78 x 5)
        filenames.append("T2WTSECorBHSENSEs1201a1012")   # 0.78 x 0.78 x 6 !!
        # filenames.append("T2WTSEtraBHSENSEs1301a1013") # Double breath-hold!! -> exclude!?
        filenames.append("T2WTSEtraBHSENSEs1501a1015")
        filenames.append("T2WTSEsagBHSENSEs1401a1014")
        
        labels.append(self._label_SST2W_standard + "_" + self._label_coronal)
        labels.append(self._label_SST2W_standard + "_" + self._label_transverse)
        labels.append(self._label_SST2W_standard + "_" + self._label_sagittal)
        
        return filenames, labels


    def _get_filenames_single_shot_t2_weighted_oblique_upper_gi(self):
        
        filenames = []
        labels = []
        
        ## Oblique planes (0.78 x 0.78 x 5)
        filenames.append("T2WTSECorBHSENSEs801a1008")
        filenames.append("T2WTSECorBHSENSEs901a1009")
        filenames.append("T2WTSECorBHSENSEs1001a1010")
        filenames.append("T2WTSECorBHSENSEs1101a1011")

        for i in range(0, len(filenames)):
            labels.append(self._label_SST2W_oblique + "_" + str(i))


        return filenames, labels



    def _get_filenames_single_shot_t2_weighted_standard_offset_upper_gi(self):
        
        filenames = []
        labels = []

        ## Offset standard anatomical planes (0.78 x 0.78 x 5)
        filenames.append("T2WTSECorBHSENSEs1701a1017")   # 0.78 x 0.78 x 6 !!
        filenames.append("T2WTSEtraBHSENSEs1901a1019")
        filenames.append("T2WTSEsagBHSENSEs1801a1018")

        labels.append(self._label_SST2W_standard_offset + "_" + self._label_coronal)
        labels.append(self._label_SST2W_standard_offset + "_" + self._label_transverse)
        labels.append(self._label_SST2W_standard_offset + "_" + self._label_sagittal)

        return filenames, labels


    ## Getter for BFFE filename
    def _get_filename_balanced_fast_field_echo_upper_gi(self):
        
        ## Standard MRCP BTFE sequence: 0.9 x 0.9 x 6
        # filename = "BTFEBHSENSEs401a1004"

        ## 0.73 x 0.73 x 1.5
        filename = "bFFECORSENSEs1601a1016"

        return filename, self._label_BFFE


    ## Getter for heavily T2-weighted volumetric filename
    def _get_filename_heavily_t2_weighted_volume_upper_gi(self):
        
        ## Standard MRCP 3D HR volume (heavily T2w vol)
        filename = "sMRCP3DHRSENSEs701a1007" # 0.65 x 0.65 x 0.9

        return filename, self._label_heavilyT2w


    ## Getter for T2-weighted volumetric filename
    def _get_filename_t2_weighted_volume_upper_gi(self):
        
        ## Bellow-gated 3D volume with SST2W contrast (blurry)
        filename = "sMRCP3DHRSENSEs601a1006" # 0.65 x 0.65 x 0.9

        return filename, self._label_T2w


    ## ************************************************************************
    ## Brain     

    def _get_filenames_single_shot_t2_weighted_standard_brain(self):

        filenames = []
        labels = []

        ## Standard anatomical planes (0.78 x 0.78 x 5)
        filenames.append("T2WTSECorBHSENSEs2701a1027")  # 0.78 x 0.78 x 6 !!
        filenames.append("T2WTSEtraBHSENSEs2801a1028")
        filenames.append("T2WTSEsagBHSENSEs3201a1032")
        
        labels.append(self._label_SST2W_standard + "_" + self._label_coronal)
        labels.append(self._label_SST2W_standard + "_" + self._label_transverse)
        labels.append(self._label_SST2W_standard + "_" + self._label_sagittal)
        
        return filenames, labels


    def _get_filenames_single_shot_t2_weighted_oblique_brain(self):
        
        filenames = []
        labels = []
        
        ## Oblique planes (0.78 x 0.78 x 5)
        filenames.append("T2WTSECorBHSENSEs2301a1023")
        # filenames.append("T2WTSECorBHSENSEs2401a1024")  ## something went wrong here! angulation and position is not correct!
        filenames.append("T2WTSECorBHSENSEs2501a1025")
        filenames.append("T2WTSECorBHSENSEs2601a1026")
        # filenames.append("T2WTSECorBHSENSEs3601a1036") ## same direction as 1023!

        for i in range(0, len(filenames)):
            labels.append(self._label_SST2W_oblique + "_" + str(i))

        return filenames, labels
    

    def _get_filenames_single_shot_t2_weighted_standard_offset_brain(self):
        
        filenames = []
        labels = []

        ## Offset standard anatomical planes (0.78 x 0.78 x 5)
        filenames.append("T2WTSECorBHSENSEs3001a1030")   # 0.78 x 0.78 x 6 !!
        filenames.append("T2WTSEtraBHSENSEs3101a1031")
        filenames.append("T2WTSEsagBHSENSEs3301a1033")

        labels.append(self._label_SST2W_standard_offset + "_" + self._label_coronal)
        labels.append(self._label_SST2W_standard_offset + "_" + self._label_transverse)
        labels.append(self._label_SST2W_standard_offset + "_" + self._label_sagittal)

        return filenames, labels


    ## Getter for BFFE filename
    def _get_filename_balanced_fast_field_echo_brain(self):
        ## 0.73 x 0.73 x 1.5
        filename = "bFFECORSENSEs2901a1029"

        return filename, self._label_BFFE


    ## Getter for heavily T2-weighted volumetric filename
    def _get_filename_heavily_t2_weighted_volume_brain(self):
        
        ## Standard as in abdominal but worse FOV selection
        # filename = "sMRCP3DHRSENSEs3401a1034" # 0.65 x 0.65 x 0.9

        ## Rotated such that better coverage of brain
        filename = "sMRCP3DHRSENSEs3501a1035" # 0.65 x 0.65 x 0.9

        return filename, self._label_heavilyT2w

    ## Getter for T2-weighted volumetric filename
    def _get_filename_t2_weighted_volume_brain(self):
        # return None, None
        pass
