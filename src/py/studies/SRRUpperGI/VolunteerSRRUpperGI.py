#!/usr/bin/python

## \file VolunteerSRRUpperGI.py
#  \brief
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date April 2017


## Import libraries 
from abc import ABCMeta, abstractmethod


##
# { class_description }
# \date       2017-04-08 14:29:11+0100
#
class VolunteerSRRUpperGI(object):
    __metaclass__ = ABCMeta

    def __init__(self, select_anatomy):

        self._select_anatomy = select_anatomy

        self._label_SST2W_standard = "SST2W_std"
        self._label_SST2W_oblique = "SST2W_ang"
        self._label_SST2W_standard_offset = "SST2W_stdoff"

        self._label_BFFE = "BFFE"
        self._label_heavilyT2w = "HT2W3D"
        self._label_T2w = "T2W3D"

        self._label_transverse = "ax"
        self._label_sagittal = "sag"
        self._label_coronal = "cor"


        self._directory = {
            "upper_gi"  : self._directory_upper_gi,
            "brain"  : self._directory_brain,
        }
    
        self._get_filenames_single_shot_t2_weighted_standard = {
            "upper_gi"  : self._get_filenames_single_shot_t2_weighted_standard_upper_gi,
            "brain"  : self._get_filenames_single_shot_t2_weighted_standard_brain,
        }

        self._get_filenames_single_shot_t2_weighted_oblique = {
            "upper_gi"  : self._get_filenames_single_shot_t2_weighted_oblique_upper_gi,
            "brain"  : self._get_filenames_single_shot_t2_weighted_oblique_brain,
        }

        self._get_filenames_single_shot_t2_weighted_standard_offset = {
            "upper_gi"  : self._get_filenames_single_shot_t2_weighted_standard_offset_upper_gi,
            "brain"  : self._get_filenames_single_shot_t2_weighted_standard_offset_brain,
        }

        self._get_filename_balanced_fast_field_echo = {
            "upper_gi"  : self._get_filename_balanced_fast_field_echo_upper_gi,
            "brain"  : self._get_filename_balanced_fast_field_echo_brain,
        }

        self._get_filename_heavily_t2_weighted_volume = {
            "upper_gi"  : self._get_filename_heavily_t2_weighted_volume_upper_gi,
            "brain"  : self._get_filename_heavily_t2_weighted_volume_brain,
        }

        self._get_filename_t2_weighted_volume = {
            "upper_gi"  : self._get_filename_t2_weighted_volume_upper_gi,
            "brain"  : self._get_filename_t2_weighted_volume_brain,
        }


    def get_directory(self):
        return self._directory[self._select_anatomy]


    def get_filenames_all_acquisitions(self):
        
        filenames, labels = self.get_filenames_single_shot_t2_weighted(standard=1, oblique=1, offset=1)

        tmp = self._get_filename_balanced_fast_field_echo[self._select_anatomy]()
        if tmp is not None:
            filenames.append(tmp[0])
            labels.append(tmp[1])

        tmp = self._get_filename_heavily_t2_weighted_volume[self._select_anatomy]()
        if tmp is not None:
            filenames.append(tmp[0])
            labels.append(tmp[1])

        tmp = self._get_filename_t2_weighted_volume[self._select_anatomy]()
        if tmp is not None:
            filenames.append(tmp[0])
            labels.append(tmp[1])

        return filenames, labels


    ## Getter for single-shot T2-weighted filenames
    def get_filenames_single_shot_t2_weighted(self, standard=1, oblique=1, offset=1):

        filenames = []
        labels = []

        if standard:
            tmp = self._get_filenames_single_shot_t2_weighted_standard[self._select_anatomy]()
            if tmp is not None:
                for i in range(0, len(tmp[0])):
                    filenames.append(tmp[0][i])
                    labels.append(tmp[1][i])
            
        if oblique:
            tmp = self._get_filenames_single_shot_t2_weighted_oblique[self._select_anatomy]()
            if tmp is not None:
                for i in range(0, len(tmp[0])):
                    filenames.append(tmp[0][i])
                    labels.append(tmp[1][i])

        if offset:
            tmp = self._get_filenames_single_shot_t2_weighted_standard_offset[self._select_anatomy]()

            if tmp is not None:
                for i in range(0, len(tmp[0])):
                    filenames.append(tmp[0][i])
                    labels.append(tmp[1][i])

        return filenames, labels


    def get_filenames_single_shot_t2_weighted_standard(self):
        return self._get_filenames_single_shot_t2_weighted_standard[self._select_anatomy]()

    def get_filenames_single_shot_t2_weighted_oblique(self):
        return self._get_filenames_single_shot_t2_weighted_oblique[self._select_anatomy]()

    def get_filenames_single_shot_t2_weighted_standard_offset(self):
        return self._get_filenames_single_shot_t2_weighted_standard_offset[self._select_anatomy]()

    def get_filename_balanced_fast_field_echo(self):
        return self._get_filename_balanced_fast_field_echo[self._select_anatomy]()

    def get_filename_heavily_t2_weighted_volume(self):
        return self._get_filename_heavily_t2_weighted_volume[self._select_anatomy]()

    def get_filename_t2_weighted_volume(self):
        return self._get_filename_t2_weighted_volume[self._select_anatomy]()


    ## ************************************************************************
    ## Upper GI Anatomy
    
    ## Getter for single-shot T2-weighted filenames
    @abstractmethod
    def _get_filenames_single_shot_t2_weighted_standard_upper_gi(self):
        pass
    @abstractmethod
    def _get_filenames_single_shot_t2_weighted_oblique_upper_gi(self):
        pass
    @abstractmethod
    def _get_filenames_single_shot_t2_weighted_standard_offset_upper_gi(self):
        pass

    ## Getter for BFFE filename
    @abstractmethod
    def _get_filename_balanced_fast_field_echo_upper_gi(self):
        pass

    ## Getter for heavily T2-weighted volumetric filename
    @abstractmethod
    def _get_filename_heavily_t2_weighted_volume_upper_gi(self):
        pass

    ## Getter for T2-weighted volumetric filename
    @abstractmethod
    def _get_filename_t2_weighted_volume_upper_gi(self):
        pass


    ## ************************************************************************
    ## Brain

    ## Getter for single-shot T2-weighted filenames
    @abstractmethod
    def _get_filenames_single_shot_t2_weighted_standard_brain(self):
        pass
    @abstractmethod
    def _get_filenames_single_shot_t2_weighted_oblique_brain(self):
        pass
    @abstractmethod
    def _get_filenames_single_shot_t2_weighted_standard_offset_brain(self):
        pass

    ## Getter for BFFE filename
    @abstractmethod
    def _get_filename_balanced_fast_field_echo_brain(self):
        pass

    ## Getter for heavily T2-weighted volumetric filename
    @abstractmethod
    def _get_filename_heavily_t2_weighted_volume_brain(self):
        pass

    ## Getter for T2-weighted volumetric filename
    @abstractmethod
    def _get_filename_t2_weighted_volume_brain(self):
        pass