## \file fetalImage.py
#  \brief  
# 
#  \author Michael Ebner
#  \date June 2015

import nibabel as nib           # nifti files
import numpy as np


class FetalImage:

    def __init__(self, dir_nifti, nifti_filename):
        self._dir_nifti = dir_nifti
        self._nifti_filename = nifti_filename
        self._img_nifti = nib.load(dir_nifti + nifti_filename + ".nii.gz")

        self.affine = self._img_nifti.affine

    @classmethod
    def directly_from_nifti(self, img_nifti):
        # Initialize directly from nifti file without loading it from HDD
        
        self._img_nifti = img_nifti

        self.affine = self._img_nifti.affine
        self._dir_nifti = None
        self._nifti_filename = None

    def get_data(self):
        return self._img_nifti.get_data()

    def get_shape(self):
        return self._img_nifti.get_data().shape


    def get_dir(self):
        return self._dir_nifti

    def get_filename(self):
        return self._nifti_filename

    def get_header(self):
        return self._img_nifti.header

    